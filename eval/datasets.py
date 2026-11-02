import io
import sys

import lance
import numpy as np
import timm.data
import torch
import torchvision.transforms as T

from PIL import Image

def episodes(ds):
    episode = np.concatenate([b.column(0).to_numpy() for b in ds.to_batches(columns=["episode_idx"], batch_size=1 << 20)])
    starts = np.concatenate(([0], np.flatnonzero(episode[1:] != episode[:-1]) + 1))
    lengths = np.diff(np.append(starts, len(episode)))
    labels = ds.take(starts.tolist(), columns=["label"])["label"].to_numpy().astype(np.int64)
    return starts, lengths, labels

class ProbeDataset:
    """Labelled lance store with one frame per row (columns episode_idx, frame, label), rows of an episode contiguous."""

    def __init__(self, path, train, num_frames, size):
        self.path = path
        self.train = train
        self.num_frames = num_frames
        self.size = size
        self.ds = None

        self.starts, self.lengths, labels = episodes(lance.dataset(path))
        self.labels = torch.from_numpy(labels)

    def _lance(self):
        if self.ds is None:
            self.ds = lance.dataset(self.path)
        return self.ds

    def __len__(self):
        return len(self.starts)

# V-JEPA's frozen evals (jepa/evals/{image,video}_classification_frozen), reproduced on lance stores. The video
# transforms are imported unmodified from a checkout of https://github.com/facebookresearch/jepa.
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

def decode_native(jpeg, short_side=None):
    """Image bytes -> PIL RGB at stored resolution (ImageNet has PNGs and CMYK files, so PIL rather than decode_image),
    or resized to `short_side` on the shorter edge."""
    image = Image.open(io.BytesIO(jpeg)).convert("RGB")
    if short_side is not None and min(image.size) != short_side:
        scale = short_side / min(image.size)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.BILINEAR)
    return image

class ImageProbeDataset(ProbeDataset):
    """V-JEPA's image protocol (image_classification_frozen/eval.py make_dataloader): the stored image at full resolution
    through timm's train transform (RandomResizedCrop 0.08-1 bicubic, flip, AutoAugment 'original', RandomErasing 0.25)
    or Resize(256) + CenterCrop(224) for val, normalised. One image per episode; the model repeats it over the frames."""

    def __init__(self, path, train, num_frames, size):
        super().__init__(path, train, num_frames, size)
        if train:
            self.transform = timm.data.create_transform(input_size=size, is_training=True, auto_augment="original", interpolation="bicubic",
                                                        re_prob=0.25, re_mode="pixel", re_count=1, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        else:
            self.transform = T.Compose([T.Resize(int(size * 256 / 224)), T.CenterCrop(size), T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    def __getitems__(self, indices):
        rows, inverse = np.unique(self.starts[indices], return_inverse=True)
        jpegs = self._lance().take(rows.tolist(), columns=["frame"]).column("frame").to_pylist()
        # bf16 halves the shared-memory batches; the encoder's autocast rounds its input to bf16 anyway
        images = torch.stack([self.transform(decode_native(jpegs[j])) for j in inverse]).to(torch.bfloat16)
        return {"image": images, "label": self.labels[indices], "index": [int(i) for i in indices]}

def segment_indices(length, frames, step, segments):
    """Frame indices of each segment, ported from VideoDataset.loadvideo_decord (jepa/src/datasets/video_dataset.py)
    with random_clip_sampling and allow_clip_overlap on, as the eval sets them for train and val alike.
    Copyright (c) Meta Platforms, Inc. and affiliates; CC BY-NC 4.0 like the rest of facebookresearch/jepa."""
    clip_len = frames * step
    partition_len = length // segments
    out = []
    for i in range(segments):
        if partition_len > clip_len:
            # a random clip_len window inside each of the equal partitions
            end = np.random.randint(clip_len, partition_len)
            idx = np.clip(np.linspace(end - clip_len, end, num=frames), end - clip_len, end - 1).astype(np.int64) + i * partition_len
        else:
            # from frame 0, every step frames, the last frame repeated up to `frames`; later segments shifted so the
            # last one ends at the end of the video (all equal when the video is not longer than clip_len)
            sample_len = min(clip_len, length) - 1
            idx = np.linspace(0, sample_len, num=sample_len // step)
            idx = np.clip(np.concatenate((idx, np.ones(frames - sample_len // step) * sample_len)), 0, sample_len - 1).astype(np.int64)
            idx = idx + i * ((length - clip_len) // (segments - 1) if length > clip_len else 0)
        out.append(idx)
    return np.stack(out)

class VideoProbeDataset(ProbeDataset):
    """V-JEPA's video protocol (video_classification_frozen, the 16x2x3 SSv2 config): `segments` clips of `num_frames`
    frames, `step` native frames apart, each through upstream's VideoTransform (RandAugment rand-m7-n4-mstd0.5-inc1,
    RandomResizedCrop 0.08-1, RandomErasing 0.25, no flip) for train, or upstream's EvalVideoTransform for val (short
    side to 224, `views` crops along the long side). The store must hold every native frame (step counts rows).
    `short_side` resizes frames to the source resolution upstream decodes (240 for SSv2 webm) if they were stored larger.
    `jepa_root` is the path of the jepa checkout."""

    def __init__(self, path, train, num_frames, size, jepa_root, step=4, segments=2, views=3, short_side=None):
        super().__init__(path, train, num_frames, size)
        self.step, self.segments, self.views, self.short_side = step, segments, 1 if train else views, short_side
        self.jepa_root = jepa_root
        self.transform = None

    def _transform(self):
        if self.transform is None:
            if self.jepa_root not in sys.path:
                sys.path.insert(0, self.jepa_root)
            from evals.video_classification_frozen.utils import make_transforms
            self.transform = make_transforms(training=self.train, num_views_per_clip=self.views, random_horizontal_flip=False,
                                             random_resize_aspect_ratio=(3 / 4, 4 / 3), random_resize_scale=(0.08, 1.0), reprob=0.25,
                                             auto_augment=True, motion_shift=False, crop_size=self.size)
        return self.transform

    def __getitems__(self, indices):
        idx = np.stack([self.starts[i] + segment_indices(int(self.lengths[i]), self.num_frames, self.step, self.segments) for i in indices])
        rows, inverse = np.unique(idx, return_inverse=True)
        frames = [np.asarray(decode_native(b, self.short_side)) for b in self._lance().take(rows.tolist(), columns=["frame"]).column("frame").to_pylist()]
        inverse = inverse.reshape(idx.shape)   # (B, segments, T); frame sizes differ between videos, so stacked per segment
        # each segment is transformed on its own, like VideoDataset; a transform returns `views` float (C, T, H, W) tensors,
        # cast to bf16 right away so a worker never holds a whole float32 batch (3 views x 2 segments of 16 frames each)
        clips = torch.stack([torch.stack([torch.stack(self._transform()(np.stack([frames[j] for j in seg]))).to(torch.bfloat16) for seg in video], dim=1)
                             for video in inverse])
        return {"segments": clips, "label": self.labels[indices], "index": [int(i) for i in indices]}   # (B, V, S, C, T, H, W)
