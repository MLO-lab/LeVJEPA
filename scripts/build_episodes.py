"""Build the episode index a lance frame store needs for VariableLengthBatchSampler.

Scans the `episode_idx` column once and writes `<store>.episodes.npz` next to the
store (or to --out) with three arrays in the order data.py unpacks them:
starts (first global row of each episode), lengths (rows per episode), episode_idx.
Rows of one episode must be contiguous.

    python scripts/build_episodes.py data/kinetics/train.lance
"""
import argparse
import lance
import numpy as np


def build(store, out=None, column="episode_idx"):
    ds = lance.dataset(store)
    ep = np.concatenate([b.column(0).to_numpy(zero_copy_only=False)
                         for b in ds.to_batches(columns=[column], batch_size=1 << 20)])
    starts = np.concatenate(([0], np.flatnonzero(ep[1:] != ep[:-1]) + 1)).astype(np.int64)
    lengths = np.diff(np.append(starts, len(ep))).astype(np.int64)
    episode_idx = ep[starts].astype(np.int64)
    assert len(np.unique(episode_idx)) == len(episode_idx), "episodes are not contiguous in this store"

    out = out or str(store).rstrip("/").removesuffix(".lance") + ".episodes.npz"
    np.savez(out, starts=starts, lengths=lengths, episode_idx=episode_idx)
    print(f"{out}: {len(starts)} episodes, {len(ep)} rows, length min/median/max "
          f"{lengths.min()}/{np.median(lengths):.0f}/{lengths.max()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("store")
    ap.add_argument("--out", default=None)
    ap.add_argument("--column", default="episode_idx")
    args = ap.parse_args()
    build(args.store, args.out, args.column)


if __name__ == "__main__":
    main()
