"""Build a frame-per-row Lance store for the 10-video Walking Tours set.

Convention: 15 fps stored rate (per-video stride off the native fps), short
edge 384, JPEG quality 90, consecutive non-overlapping 128-frame episodes
(~8.5 s each). Training then samples 16-frame clips every `frame_stride`-th
row (default stride 4 -> 3.75 fps, ~4.3 s per clip).

The stride is derived per video rather than fixed, because the walks are not
all the same frame rate: 9 are 60 fps but Wildlife is 30 fps. See `stride_for`.

Schema is (episode_idx, step_idx, frame, h, w, label); `label` is a constant
-1 because Walking Tours is unlabeled. Ordering matters: rows must come out
sorted by episode_idx because LanceDataset._load_episode_index detects episode
boundaries by watching that column change, so a shuffled write would fuse or
split episodes. One video is processed at a time and pool.imap preserves order
within it.

Usage (after `bash scripts/download_walking_tours.sh`):
  python scripts/build_lance_walking_tours.py \
      --videos data/walking_tours/videos \
      --out data/walking_tours/train.lance

Requires decord (installed by `uv sync --extra data`).
"""

import argparse
import io
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

SHORT_EDGE = 384
JPEG_Q = 90
TARGET_FPS = 15.0  # stored rate; training subsamples further via frame_stride
EP_LEN = 128  # frames per episode, ~8.5 s at 15 fps
DECODE_CHUNK = 32  # output frames decoded at once, caps worker RSS


def stride_for(fps: float) -> int:
    """Per-video stride hitting TARGET_FPS.

    Not a constant: 9 of the 10 walks are 60 fps but Wildlife is 30 fps, so a
    flat stride of 4 would store Wildlife at 7.5 fps and everything else at 15,
    i.e. half the intended temporal rate for that video.
    """
    return max(1, round(fps / TARGET_FPS))

_READER = {}


def _reader(path):
    """One VideoReader per (worker, video); building the index is expensive."""
    from decord import VideoReader, cpu

    if path not in _READER:
        _READER.clear()  # only ever one video in flight
        _READER[path] = VideoReader(str(path), ctx=cpu(0), num_threads=1)
    return _READER[path]


def encode_episode(task):
    """Decode + JPEG-encode one episode. Returns (ep_idx, [(step, bytes, h, w)])."""
    from PIL import Image

    ep_idx, path, first_frame, stride = task
    try:
        vr = _reader(path)
        out = []
        for base in range(0, EP_LEN, DECODE_CHUNK):
            n = min(DECODE_CHUNK, EP_LEN - base)
            idx = [first_frame + (base + i) * stride for i in range(n)]
            arrs = vr.get_batch(idx).asnumpy()
            for j, arr in enumerate(arrs):
                img = Image.fromarray(arr)
                w, h = img.size
                if h < w:
                    nh, nw = SHORT_EDGE, round(w * SHORT_EDGE / h)
                else:
                    nh, nw = round(h * SHORT_EDGE / w), SHORT_EDGE
                img = img.resize((nw, nh), Image.BILINEAR)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=JPEG_Q)
                out.append((base + j, buf.getvalue(), img.size[1], img.size[0]))
        return ep_idx, out
    except Exception as exc:  # noqa: BLE001 - one bad episode must not kill the build
        print(f"  episode {ep_idx} failed: {exc}", flush=True)
        return ep_idx, None


def main():
    import lance
    import pyarrow as pa
    from decord import VideoReader, cpu

    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="data/walking_tours/videos")
    ap.add_argument("--out", default="data/walking_tours/train.lance")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 16)))
    args = ap.parse_args()

    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"{out_path} already exists; remove it first")

    videos = sorted(Path(args.videos).glob("*.mp4"))
    if not videos:
        raise SystemExit(f"no .mp4 under {args.videos}")
    print(f"{len(videos)} videos: {[v.stem for v in videos]}", flush=True)

    schema = pa.schema([
        pa.field("episode_idx", pa.int32()),
        pa.field("step_idx", pa.int32()),
        pa.field("frame", pa.binary()),
        pa.field("h", pa.int16()),
        pa.field("w", pa.int16()),
        pa.field("label", pa.int16()),
    ])

    # Plan episodes per video up front so the total is known before encoding.
    plan = []  # (video_path, first_frame, stride)
    for v in videos:
        vr = VideoReader(str(v), ctx=cpu(0), num_threads=1)
        n_raw, fps = len(vr), vr.get_avg_fps()
        stride = stride_for(fps)
        n_out = n_raw // stride
        n_eps = n_out // EP_LEN
        print(
            f"  {v.stem:14s} {n_raw:>8} frames @ {fps:5.1f} fps "
            f"({n_raw/fps/60:6.1f} min) stride {stride} -> {fps/stride:4.1f} fps "
            f"-> {n_out:>7} stored -> {n_eps:>5} episodes",
            flush=True,
        )
        for e in range(n_eps):
            plan.append((v, e * EP_LEN * stride, stride))
        del vr
    print(f"total {len(plan)} episodes, {len(plan) * EP_LEN} rows", flush=True)

    t0 = time.time()
    written = {"eps": 0, "rows": 0, "failed": 0}

    def batches():
        ep_counter = 0
        ctx = get_context("spawn")
        # One video at a time: each worker then holds exactly one VideoReader,
        # and imap keeps episodes of that video in order.
        for v in videos:
            starts = [(f, s) for p, f, s in plan if p == v]
            if not starts:
                continue
            tasks = [(i, str(v), f, s) for i, (f, s) in enumerate(starts)]
            print(f"=== {v.stem}: {len(tasks)} episodes", flush=True)
            with ctx.Pool(args.workers) as pool:
                for _, frames in pool.imap(encode_episode, tasks, chunksize=1):
                    if frames is None:
                        written["failed"] += 1
                        continue
                    ep = ep_counter
                    ep_counter += 1
                    written["eps"] += 1
                    written["rows"] += len(frames)
                    yield pa.RecordBatch.from_arrays(
                        [
                            pa.array([ep] * len(frames), type=pa.int32()),
                            pa.array([s for s, _, _, _ in frames], type=pa.int32()),
                            pa.array([b for _, b, _, _ in frames], type=pa.binary()),
                            pa.array([h for _, _, h, _ in frames], type=pa.int16()),
                            pa.array([w for _, _, _, w in frames], type=pa.int16()),
                            pa.array([-1] * len(frames), type=pa.int16()),
                        ],
                        schema=schema,
                    )
                    if written["eps"] % 100 == 0:
                        el = time.time() - t0
                        rate = written["eps"] / el
                        eta = (len(plan) - written["eps"]) / rate / 60
                        print(
                            f"  {written['eps']}/{len(plan)} episodes, "
                            f"{written['rows']/1e6:.2f}M rows, "
                            f"{rate:.1f} ep/s, eta {eta:.0f} min",
                            flush=True,
                        )

    lance.write_dataset(batches(), str(out_path), schema=schema, mode="create")
    print(
        f"wrote {out_path}: {written['eps']} episodes, {written['rows']} rows, "
        f"{written['failed']} failed, in {(time.time()-t0)/60:.1f} min",
        flush=True,
    )

    ds = lance.dataset(str(out_path))
    print(f"verify: {ds.count_rows()} rows in store")
    assert ds.count_rows() == written["rows"], "row count mismatch"


if __name__ == "__main__":
    sys.exit(main())
