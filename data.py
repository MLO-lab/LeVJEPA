import io
import math

import lance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from PIL import Image
from torchvision.io import decode_image, ImageReadMode
from torchvision.ops import roi_align

def decode(jpeg, height, width):
    """Image bytes -> uint8 (height, width, 3). libjpeg-turbo decode + antialiased bilinear resize; torch's uint8
    channels-last resize is ~4x faster than PIL's and matches it to +-1 (about half the CPU of PIL decode + resize).
    Not every stored "JPEG" is one (ImageNet has PNGs and CMYK files): decode_image sniffs the format, PIL is the fallback."""
    try:
        x = decode_image(torch.frombuffer(bytearray(jpeg), dtype=torch.uint8), mode=ImageReadMode.RGB)
    except RuntimeError:
        x = torch.from_numpy(np.array(Image.open(io.BytesIO(jpeg)).convert("RGB"))).permute(2, 0, 1)
    x = F.interpolate(x[None].contiguous(memory_format=torch.channels_last), size=(height, width), mode="bilinear",
                      antialias=True, align_corners=False)
    return x[0].permute(1, 2, 0).numpy()

class Episodes:
    """Episode table of one or more lance stores, concatenated in global row order (store i's rows come after store
    i-1's). Each source is a dict: path, episodes_path (from scripts/build_episodes.py), and optionally
      stride: rows skipped between two frames of a window (2 on a 6 fps store gives 3 fps)
      pad:    keep episodes shorter than a window and repeat their last row (ImageNet stills, short SSv2 clips)
      weight: share of the samples drawn from this source; by default its episode count, so every episode (video,
              image) is about equally likely, as when V-JEPA concatenates its datasets"""
    def __init__(self, sources):
        self.sources = [dict(s) for s in sources]
        starts, lengths, source, offset = [], [], [], 0
        for i, s in enumerate(self.sources):
            s_starts, s_lengths, _ = np.load(s["episodes_path"], allow_pickle=True).values()
            starts.append(s_starts + offset); lengths.append(s_lengths); source.append(np.full(len(s_starts), i))
            s["offset"] = offset
            offset += int(s_starts[-1] + s_lengths[-1])
        self.starts, self.lengths, self.source = (np.concatenate(a) for a in (starts, lengths, source))
        self.offsets = np.array([s["offset"] for s in self.sources])
        self.stride = np.array([s.get("stride", 1) for s in self.sources])
        self.pad = np.array([s.get("pad", False) for s in self.sources])
        counts = np.bincount(self.source, minlength=len(self.sources))
        self.weights = np.array([s.get("weight") or counts[i] for i, s in enumerate(self.sources)], dtype=float)
        self.weights /= self.weights.sum()

    def span(self, episodes, length):
        """Rows a `length`-frame window covers in each of `episodes`."""
        return (length - 1) * self.stride[self.source[episodes]] + 1

    def windows(self, episodes, first, length):
        """(n, length) global rows of the windows starting at rows `first` of `episodes`; past the end of a short
        episode the last frame on the stride repeats, as upstream V-JEPA pads."""
        stride = self.stride[self.source[episodes]]
        last = self.starts[episodes] + self.lengths[episodes] - 1
        max_step = (last - first) // stride * stride
        return first[:, None] + np.minimum(np.arange(length) * stride[:, None], max_step[:, None])

class WindowedLanceDataset:
    def __init__(self, paths, offsets, height, width, column_name="frame"):
        self.paths = list(paths)
        self.offsets = np.asarray(offsets)
        self.height = height
        self.width = width
        self.column_name = column_name
        self.ds = None

    def __getitems__(self, row_indices):
        # padded windows repeat rows: every distinct row is read and decoded once, into its first slot
        flat = row_indices.reshape(-1)
        rows, first, inverse = np.unique(flat, return_index=True, return_inverse=True)
        which = np.searchsorted(self.offsets, rows, side="right") - 1
        blobs = [None] * len(rows)
        for i in np.unique(which):
            sel = np.flatnonzero(which == i)
            table = self._lance(i).take((rows[sel] - self.offsets[i]).tolist(), columns=[self.column_name])
            for j, b in zip(sel, table.column(self.column_name).to_pylist()):
                blobs[j] = b
        # decode straight into shared memory: skips the np.stack copy, and the DataLoader then hands the storage to the
        # main process as-is instead of copying the whole batch into shm again (together ~25% of a worker's CPU time)
        clip = torch.empty((len(flat), self.height, self.width, 3), dtype=torch.uint8).share_memory_()
        frames = clip.numpy()
        for u, b in enumerate(blobs):
            frames[first[u]] = decode(b, self.height, self.width)
        repeats = np.flatnonzero(first[inverse] != np.arange(len(flat)))
        frames[repeats] = frames[first[inverse[repeats]]]
        return {"clip": rearrange(clip, "(b t) h w c -> b t h w c", b=row_indices.shape[0])}

    def _lance(self, i):
        if self.ds is None:
            self.ds = [None] * len(self.paths)
        if self.ds[i] is None:
            self.ds[i] = lance.dataset(self.paths[i])
        return self.ds[i]

def holdout_episodes(episodes, num, length, seed=0):
    """Indices of `num` episodes (long enough for an unpadded `length`-frame window) kept out of training for the
    holdout loss; the same on every rank."""
    eligible = np.flatnonzero(episodes.lengths >= episodes.span(np.arange(len(episodes.lengths)), length))
    return np.sort(np.random.default_rng(seed).choice(eligible, size=num, replace=False))

def holdout_batches(episodes, holdout, length, batch_size, rank=0, world_size=1):
    """This rank's batches of fixed holdout clips: the centre `length`-frame window of each holdout episode, as
    (batch, length) row indices like VariableLengthBatchSampler. Every rank gets the same number of batches."""
    first = episodes.starts[holdout] + (episodes.lengths[holdout] - episodes.span(holdout, length)) // 2
    rows = episodes.windows(holdout, first, length)
    batches = [rows[i:i + batch_size] for i in range(0, len(rows) - batch_size + 1, batch_size)]
    per_rank = len(batches) // world_size
    assert per_rank > 0, f"{len(rows)} holdout clips give fewer than one batch of {batch_size} per rank"
    return batches[rank * per_rank:(rank + 1) * per_rank]

class CachedBatches:
    """Wraps a dataset for a fixed list of batches (the holdout): each batch is decoded once, then served from RAM.
    Use with num_workers=0, the cache lives in the training process."""
    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def __getitems__(self, row_indices):
        key = int(row_indices.flat[0])
        if key not in self.cache:
            self.cache[key] = self.dataset.__getitems__(row_indices)
        # a fresh dict: the forward writes the loss into the batch; `seed` fixes the batch's view / mask randomness
        return {**self.cache[key], "seed": key}

class VariableLengthBatchSampler:
    """Every sample picks a source by `episodes.weights`, then a window uniformly among that source's windows."""
    def __init__(self, episodes, batch_size, probs, batches_per_epoch, seed=0, rank=0, world_size=1, exclude=None):
        self.episodes = episodes
        self.batch_size = batch_size
        self.lengths = list(probs)
        self.probs = np.array([probs[length] for length in self.lengths])
        self.probs /= self.probs.sum()

        self.batches_per_epoch = batches_per_epoch

        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0

        keep = np.ones(len(episodes.starts), dtype=bool)
        if exclude is not None:  # holdout episodes never enter a training window
            keep[exclude] = False
        # per window length and source: its usable episodes and the cumulative count of window starts in them, so a
        # uniform draw over starts is one searchsorted (a table of every start would be ~100M entries per length)
        self.tables = {length: self.window_tables(keep, length) for length in self.lengths}

    def set_epoch(self, epoch):
        self.epoch = epoch

    @property
    def sampler(self):
        # Lightning only calls set_epoch on `loader.sampler` / `loader.batch_sampler.sampler`,
        # never on the batch sampler itself, so expose ourselves there.
        return self

    def __len__(self):
        return self.batches_per_epoch

    def __iter__(self):
        rng_length = np.random.default_rng(self.seed + self.epoch)
        rng = np.random.default_rng(self.seed + self.epoch * self.world_size + self.rank + 1)
        for _ in range(self.batches_per_epoch):
            length = int(rng_length.choice(self.lengths, p=self.probs))
            tables = self.tables[length]
            source = rng.choice(len(tables), size=self.batch_size, p=self.episodes.weights)
            eps, first = np.empty(self.batch_size, dtype=np.int64), np.empty(self.batch_size, dtype=np.int64)
            for i in np.unique(source):
                sel = np.flatnonzero(source == i)
                usable, cum = tables[i]
                # without replacement within the source
                draw = rng.choice(cum[-1], size=len(sel), replace=False)
                k = np.searchsorted(cum, draw, side="right")
                eps[sel] = usable[k]
                first[sel] = self.episodes.starts[usable[k]] + draw - np.concatenate(([0], cum))[k]
            yield self.episodes.windows(eps, first, length)

    def window_tables(self, keep, length):
        e = self.episodes
        tables = []
        for i in range(len(e.sources)):
            eps = np.flatnonzero(keep & (e.source == i))
            starts = e.lengths[eps] - e.span(eps, length) + 1
            if e.pad[i]:
                starts = np.maximum(starts, 1)  # a short episode has one window, its tail padded
            usable = eps[starts > 0]
            assert len(usable), f"source {e.sources[i]['path']} has no episode for a {length}-frame window"
            tables.append((usable, np.cumsum(starts[starts > 0])))
        return tables

class ViewBuilder(nn.Module):
    """uint8 clip (B, T, H, W, C) on GPU -> global (B, T, 3, G, G), locals (V, B, T, 3, L, L).

    All randomness is drawn once per (view, sample) and shared across the T frames.
    """

    def __init__(
        self,
        num_local=4,
        global_size=224,
        local_size=96,
        global_scale=(0.3, 1.0),
        local_scale=(0.05, 0.3),
        ratio=(3 / 4, 4 / 3),
        flip_p=0.5,
        jitter=(0.4, 0.4, 0.2),  # brightness, contrast, saturation
        jitter_p=0.8,
        gray_p=0.2,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.num_local = num_local
        self.global_size, self.local_size = global_size, local_size
        self.global_scale, self.local_scale, self.ratio = global_scale, local_scale, ratio
        self.flip_p, self.jitter_strength, self.jitter_p, self.gray_p = flip_p, jitter, jitter_p, gray_p
        self.register_buffer("mean", torch.tensor(mean).view(3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(3, 1, 1))

    @torch.no_grad()
    def forward(self, clip):
        B, T, H, W, _ = clip.shape
        frames = rearrange(clip, "b t h w c -> (b t) c h w").float().div_(255)
        g = self.crop(frames, B, T, H, W, 1, self.global_scale, self.global_size)[0]
        l = self.crop(frames, B, T, H, W, self.num_local, self.local_scale, self.local_size)
        l = self.jitter(l)
        return self.normalize(g), self.normalize(l)

    # ---- crops -----------------------------------------------------------
    def sample_boxes(self, n, H, W, scale, device):
        area = H * W * torch.empty(n, device=device).uniform_(*scale)
        r = torch.empty(n, device=device).uniform_(math.log(self.ratio[0]), math.log(self.ratio[1])).exp()
        w = (area * r).sqrt().clamp(max=W)
        h = (area / r).sqrt().clamp(max=H)
        x1 = torch.rand(n, device=device) * (W - w)
        y1 = torch.rand(n, device=device) * (H - h)
        return torch.stack([x1, y1, x1 + w, y1 + h], dim=1)  # (n, 4)

    def crop(self, frames, B, T, H, W, V, scale, size):
        device = frames.device
        boxes = self.sample_boxes(V * B, H, W, scale, device)  # row v*B + b
        b = torch.arange(V * B, device=device) % B
        frame_idx = (b[:, None] * T + torch.arange(T, device=device)[None]).reshape(-1)  # (V*B*T,)
        rois = torch.cat([frame_idx[:, None].float(), boxes.repeat_interleave(T, dim=0)], dim=1)
        max_side = math.sqrt(scale[1] * H * W * self.ratio[1])
        sampling_ratio = max(1, math.ceil(max_side / size))  # average when shrinking, avoids aliasing
        out = roi_align(frames, rois, output_size=size, spatial_scale=1.0, sampling_ratio=sampling_ratio, aligned=True)
        out = rearrange(out, "(v b t) c h w -> v b t c h w", v=V, b=B)
        flip = (torch.rand(V, B, device=device) < self.flip_p)[..., None, None, None, None]
        return torch.where(flip, out.flip(-1), out)

    # ---- photometric (locals only) ---------------------------------------
    @staticmethod
    def luminance(x):
        return 0.299 * x[..., 0:1, :, :] + 0.587 * x[..., 1:2, :, :] + 0.114 * x[..., 2:3, :, :]

    def jitter(self, x):
        V, B = x.shape[:2]
        device = x.device
        shape = (V, B, 1, 1, 1, 1)

        def factor(strength):
            return torch.empty(V, B, device=device).uniform_(1 - strength, 1 + strength).view(shape)

        apply = (torch.rand(V, B, device=device) < self.jitter_p).view(shape)
        one = torch.ones(shape, device=device)
        bri, con, sat = (torch.where(apply, factor(s), one) for s in self.jitter_strength)

        x = (x * bri).clamp_(0, 1)
        m = x.mean(dim=(-3, -2, -1), keepdim=True)
        x = ((x - m) * con + m).clamp_(0, 1)
        lum = self.luminance(x)
        x = ((x - lum) * sat + lum).clamp_(0, 1)

        to_gray = (torch.rand(V, B, device=device) < self.gray_p).view(shape)
        return torch.where(to_gray, self.luminance(x).expand_as(x), x)

    def normalize(self, x):
        return (x - self.mean) / self.std

def identity_collate(x):
    return x