import multiprocessing as mp
import hashlib
import io
import os
from pathlib import Path

import lance
import numpy as np
import pyarrow as pa
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import v2


class LanceDataset:
    """Minimal Lance-backed video dataset compatible with the old loader API."""

    def __init__(
        self,
        path,
        num_steps,
        frameskip=1,
        keys_to_load=None,
        transform=None,
    ):
        self.path = str(path)
        self.num_steps = num_steps
        self.frameskip = frameskip
        self.span = num_steps * frameskip
        self.transform = transform
        self._fetch_columns = keys_to_load or ["frame"]
        self._dataset = None
        self.lengths, self.offsets = self._load_episode_index()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_dataset"] = None
        return state

    def _ensure_open(self):
        if self._dataset is None:
            self._dataset = lance.dataset(self.path)
        return self._dataset

    def _index_cache_path(self, row_count: int) -> Path:
        key = hashlib.sha1(str(Path(self.path).resolve()).encode()).hexdigest()[:16]
        cache_dir = Path(
            os.environ.get(
                "LE_VJEPA_INDEX_CACHE",
                Path.home() / ".cache" / "le-vjepa" / "lance_index",
            )
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{key}-{row_count}-episodes.npz"

    def _load_episode_index(self) -> tuple[np.ndarray, np.ndarray]:
        ds = self._ensure_open()
        row_count = ds.count_rows()
        cache_path = self._index_cache_path(row_count)
        if cache_path.exists():
            cached = np.load(cache_path)
            return cached["lengths"], cached["offsets"]

        lengths: list[int] = []
        offsets: list[int] = []
        current_ep = None
        current_offset = 0
        current_length = 0
        rows_seen = 0

        for batch in ds.to_batches(columns=["episode_idx"], batch_size=1 << 20):
            episodes = batch.column(0).to_numpy(zero_copy_only=False)
            if len(episodes) == 0:
                continue
            starts = np.concatenate(
                ([0], np.flatnonzero(episodes[1:] != episodes[:-1]) + 1)
            )
            ends = np.concatenate((starts[1:], [len(episodes)]))

            for start, end in zip(starts, ends):
                ep = int(episodes[start])
                run_length = int(end - start)
                if current_ep is None:
                    current_ep = ep
                    current_offset = rows_seen + int(start)
                    current_length = run_length
                elif ep == current_ep:
                    current_length += run_length
                else:
                    offsets.append(current_offset)
                    lengths.append(current_length)
                    current_ep = ep
                    current_offset = rows_seen + int(start)
                    current_length = run_length
            rows_seen += len(episodes)

        if current_ep is not None:
            offsets.append(current_offset)
            lengths.append(current_length)

        lengths_arr = np.asarray(lengths, dtype=np.int64)
        offsets_arr = np.asarray(offsets, dtype=np.int64)
        tmp_path = cache_path.with_suffix(f".{os.getpid()}.tmp.npz")
        np.savez(tmp_path, lengths=lengths_arr, offsets=offsets_arr)
        os.replace(tmp_path, cache_path)
        return lengths_arr, offsets_arr

    def _take_rows(self, rows: list[int]) -> pa.Table:
        return self._ensure_open().take(rows, columns=self._fetch_columns)

    def _decode_frame(self, encoded) -> torch.Tensor:
        if hasattr(encoded, "as_py"):
            encoded = encoded.as_py()
        with Image.open(io.BytesIO(encoded)) as img:
            array = np.array(img.convert("RGB"), copy=True)
        return torch.from_numpy(array).permute(2, 0, 1)


class VJEPAClipDataset(LanceDataset):
    """One sample per episode with a random (or center) temporal crop.

    Mirrors V-JEPA's `VideoDataset` semantics: per epoch each video yields
    `clips_per_video` clips, each a `num_frames`-frame window sampled every
    `frame_stride`-th stored row from a fresh random offset. Only the selected
    rows are fetched (no span read amplification).

    ``pad_short=True`` keeps episodes shorter than the clip span and pads them
    by repeating their last frame (upstream's ``filter_short_videos: false``
    behaviour); with the default ``False`` such episodes are dropped.
    """

    def __init__(
        self,
        path,
        num_frames,
        frame_stride,
        clips_per_video=1,
        random_crop=True,
        transform=None,
        pad_short=False,
        name=None,
    ):
        super().__init__(
            path=path,
            num_steps=num_frames,
            frameskip=frame_stride,
            keys_to_load=["frame"],
            transform=transform,
        )
        self.num_frames = num_frames
        self.clips_per_video = clips_per_video
        self.random_crop = random_crop
        self.pad_short = pad_short
        self.name = name or Path(path).parent.name
        if pad_short:
            # Every non-empty episode is usable; short ones get clamped.
            self.valid_eps = np.flatnonzero(self.lengths > 0).astype(np.int64)
        else:
            self.valid_eps = np.flatnonzero(self.lengths >= self.span).astype(np.int64)
        # Precomputed per-episode length/offset for the valid set, so the hot
        # path indexes two small arrays instead of going through valid_eps.
        self._ep_len = self.lengths[self.valid_eps].astype(np.int64)
        self._ep_off = self.offsets[self.valid_eps].astype(np.int64)
        self._frame_offsets = np.arange(num_frames, dtype=np.int64) * frame_stride

    def __len__(self) -> int:
        return int(len(self.valid_eps) * self.clips_per_video)

    def _sample_start(self, ep_pos: int) -> int:
        max_start = int(self._ep_len[ep_pos]) - self.span
        if max_start <= 0:
            return 0
        if self.random_crop:
            return int(torch.randint(0, max_start + 1, (1,)).item())
        return max_start // 2

    def _clip_rows(self, ep_pos: int) -> np.ndarray:
        """Global row indices for one clip; for short episodes the tail
        indices saturate at the last row (repeat-the-last-frame padding)."""
        length = int(self._ep_len[ep_pos])
        rows = self._frame_offsets + self._sample_start(ep_pos)
        if rows[-1] >= length:
            np.clip(rows, 0, length - 1, out=rows)
        return rows + int(self._ep_off[ep_pos])

    def _decode_clip(self, rows: np.ndarray, byte_lookup) -> dict:
        """Decode one clip, decoding each distinct row only once (padded clips
        repeat rows and JPEG decode is the bottleneck)."""
        uniq, inverse = np.unique(rows, return_inverse=True)
        decoded = [self._decode_frame(byte_lookup(int(r))) for r in uniq]
        frames = torch.stack([decoded[i] for i in inverse])
        return {"frame": frames}

    def _fetch(self, all_rows: np.ndarray):
        """Fetch every distinct row of a batch in one Lance take.

        Returns a callable mapping a global row index to its JPEG bytes.
        """
        self._ensure_open()
        uniq = np.unique(all_rows)
        table = self._take_rows(uniq.tolist())
        values = table["frame"].to_pylist()
        lookup = {int(r): v for r, v in zip(uniq, values)}
        return lookup.__getitem__

    def __getitem__(self, idx: int) -> dict:
        ep_pos = idx // self.clips_per_video
        rows = self._clip_rows(ep_pos)
        steps = self._decode_clip(rows, self._fetch(rows))
        return self.transform(steps) if self.transform else steps

    def __getitems__(self, indices: list[int]) -> list[dict]:
        per_sample = [self._clip_rows(idx // self.clips_per_video) for idx in indices]
        if not per_sample:
            return []
        byte_lookup = self._fetch(np.concatenate(per_sample))

        results: list[dict] = []
        for rows in per_sample:
            steps = self._decode_clip(rows, byte_lookup)
            if self.transform:
                steps = self.transform(steps)
            results.append(steps)
        return results


class MixtureClipDataset(torch.utils.data.Dataset):
    """Concatenate any number of `VJEPAClipDataset` sources into one index.

    The mixing ratio is set per source via `clips_per_video`, which keeps the
    epoch length deterministic and survives DistributedSampler. Unlike
    `ConcatDataset`, `__getitems__` buckets a batch's indices by source, so a
    batch costs one Lance take per source rather than one per sample. The
    global index space is shuffled as a whole, so batches are proportional
    draws from all sources -- required because the projector uses BatchNorm1d.
    """

    def __init__(self, sources: list[VJEPAClipDataset]):
        if not sources:
            raise ValueError("MixtureClipDataset needs at least one source")
        num_frames = {s.num_frames for s in sources}
        if len(num_frames) != 1:
            raise ValueError(
                f"all sources must yield the same clip length, got {num_frames}"
            )
        self.sources = list(sources)
        self.sizes = [len(s) for s in self.sources]
        if any(n == 0 for n in self.sizes):
            empty = [s.name for s, n in zip(self.sources, self.sizes) if n == 0]
            raise ValueError(f"source(s) contributed zero clips: {empty}")
        # cum[i]..cum[i+1] is source i's slice of the global index space.
        self.cum = np.concatenate(([0], np.cumsum(self.sizes))).astype(np.int64)

    def __len__(self) -> int:
        return int(self.cum[-1])

    def describe(self) -> str:
        total = len(self)
        parts = [
            f"{s.name}: {n} clips ({100.0 * n / total:.1f}%), "
            f"{len(s.valid_eps)} episodes x{s.clips_per_video}, "
            f"stride {s.frameskip}, pad_short={s.pad_short}"
            for s, n in zip(self.sources, self.sizes)
        ]
        return f"MixtureClipDataset: {total} clips/epoch\n  " + "\n  ".join(parts)

    def _route(self, indices: np.ndarray) -> np.ndarray:
        """Map global indices to source ids (vectorized over the batch)."""
        return np.searchsorted(self.cum, indices, side="right") - 1

    def __getitem__(self, idx: int) -> dict:
        src = int(self._route(np.asarray([idx], dtype=np.int64))[0])
        return self.sources[src][int(idx - self.cum[src])]

    def __getitems__(self, indices: list[int]) -> list[dict]:
        idx = np.asarray(indices, dtype=np.int64)
        src_ids = self._route(idx)
        local = idx - self.cum[src_ids]

        results: list[dict | None] = [None] * len(indices)
        for src in np.unique(src_ids):
            positions = np.flatnonzero(src_ids == src)
            batch = self.sources[int(src)].__getitems__(local[positions].tolist())
            for pos, sample in zip(positions, batch):
                results[int(pos)] = sample
        return results  # type: ignore[return-value]


class VJEPAMultiCropTransform:
    def __init__(
        self,
        global_size=224,
        local_size=96,
        local_crops_number=4,
        local_crops_scale=(0.05, 0.4),
        global_crops_scale=(0.6, 1.0),
        hflip=True,
        color_jitter_prob=0.8,
        grayscale_prob=0.2,
        gaussian_blur_prob=0.0,
        color_jitter_hue=0.1,
        normalize_on_gpu=False,
    ):
        self.local_crops_number = local_crops_number
        # ``normalize_on_gpu`` drops only the trailing ToDtype+Normalize pair,
        # which then run on the GPU instead (main.to_float_normalized).
        # Numerically identical, but the workers ship uint8 instead of float32,
        # cutting host RAM and shared-memory IPC 4x.
        self.normalize_on_gpu = normalize_on_gpu
        tail = []
        if not normalize_on_gpu:
            tail = [
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        photometric = self._photometric(
            hflip, color_jitter_prob, grayscale_prob, gaussian_blur_prob,
            color_jitter_hue,
        )

        # Clean target global: random resized crop at near-full scale + normalize.
        self.global_aug = v2.Compose(
            [
                v2.RandomResizedCrop(
                    size=(global_size, global_size),
                    scale=global_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                *tail,
            ]
        )

        self.local_aug = v2.Compose(
            [
                v2.RandomResizedCrop(
                    size=(local_size, local_size),
                    scale=local_crops_scale,
                    interpolation=v2.InterpolationMode.BICUBIC,
                    antialias=True,
                ),
                *photometric,
                *tail,
            ]
        )

    @staticmethod
    def _photometric(hflip, color_jitter_prob, grayscale_prob, gaussian_blur_prob,
                     color_jitter_hue=0.1):
        """Photometric aug stack shared by the local views.

        Each entry is skipped entirely when its probability is 0, so
        augmentation types can be toggled from config. ``color_jitter_hue=0``
        drops the hue term only: the transform gets ~3x cheaper (CPU
        ``adjust_hue`` dominates ColorJitter cost, pytorch/vision#6619), but
        hue jitter is what stops the encoder shortcutting on colour
        histograms, so runs with and without it are not comparable.
        """
        transforms = []
        if hflip:
            transforms.append(v2.RandomHorizontalFlip(p=0.5))
        if color_jitter_prob > 0:
            transforms.append(
                v2.RandomApply(
                    [
                        v2.ColorJitter(
                            brightness=0.4,
                            contrast=0.4,
                            saturation=0.2,
                            hue=color_jitter_hue,
                        )
                    ],
                    p=color_jitter_prob,
                )
            )
        if grayscale_prob > 0:
            transforms.append(v2.RandomGrayscale(p=grayscale_prob))
        if gaussian_blur_prob > 0:
            transforms.append(
                v2.RandomApply(
                    [v2.GaussianBlur(kernel_size=9, sigma=(0.1, 2.0))],
                    p=gaussian_blur_prob,
                )
            )
        return transforms

    def __call__(self, sample):
        frames = sample["frame"]
        local_frames = torch.stack(
            [self.local_aug(frames) for _ in range(self.local_crops_number)]
        )
        output = {k: v for k, v in sample.items() if k != "frame"}
        output["global_frame"] = self.global_aug(frames)
        output["local_frames"] = local_frames
        return output


def _coerce_sources(spec, frame_stride, clips_per_video):
    """Normalize the `data.<split>` config into a list of source dicts.

    Accepts a bare path or a list of per-source mappings; a single-source list
    behaves exactly like the bare path.
    """
    if isinstance(spec, (str, os.PathLike)):
        spec = [{"path": spec}]
    sources = []
    for entry in spec:
        if isinstance(entry, (str, os.PathLike)):
            entry = {"path": entry}
        else:
            entry = dict(entry)
        if "path" not in entry:
            raise ValueError(f"data source is missing 'path': {entry}")
        entry.setdefault("frame_stride", frame_stride)
        entry.setdefault("clips_per_video", clips_per_video)
        entry.setdefault("pad_short", False)
        entry.setdefault("name", Path(str(entry["path"])).parent.name)
        sources.append(entry)
    return sources


def build_loader(
    lance_path,
    batch_size=48,
    num_workers=32,
    num_frames=16,
    frame_stride=4,
    crop_size=224,
    local_size=96,
    local_crops_number=4,
    local_crops_scale=(0.05, 0.4),
    global_crops_scale=(0.8, 1.0),
    hflip=True,
    color_jitter_prob=0.8,
    grayscale_prob=0.2,
    gaussian_blur_prob=0.0,
    color_jitter_hue=0.1,
    normalize_on_gpu=False,
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
    clips_per_video=1,
    random_crop=True,
):
    transform = VJEPAMultiCropTransform(
        global_size=crop_size,
        local_size=local_size,
        local_crops_number=local_crops_number,
        local_crops_scale=local_crops_scale,
        global_crops_scale=global_crops_scale,
        hflip=hflip,
        color_jitter_prob=color_jitter_prob,
        grayscale_prob=grayscale_prob,
        gaussian_blur_prob=gaussian_blur_prob,
        color_jitter_hue=color_jitter_hue,
        normalize_on_gpu=normalize_on_gpu,
    )
    sources = _coerce_sources(lance_path, frame_stride, clips_per_video)
    built = [
        VJEPAClipDataset(
            path=src["path"],
            num_frames=num_frames,
            frame_stride=src["frame_stride"],
            clips_per_video=src["clips_per_video"],
            random_crop=random_crop,
            transform=transform,
            pad_short=src["pad_short"],
            name=src["name"],
        )
        for src in sources
    ]
    ds = built[0] if len(built) == 1 else MixtureClipDataset(built)
    if isinstance(ds, MixtureClipDataset):
        print(ds.describe(), flush=True)
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "shuffle": shuffle,
        "drop_last": drop_last,
    }
    if num_workers > 0:
        loader_kwargs.update(
            multiprocessing_context=mp.get_context("spawn"),
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
    return DataLoader(ds, **loader_kwargs)
