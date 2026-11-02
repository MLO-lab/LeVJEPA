# LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics

Official implementation of **LeVJEPA**, the first video encoder trained under LeJEPA's collapse-free objective.

Lukas Kuhn, Lucas Maes, Giuseppe Serra, Quentin Le Lidec, Yann LeCun, Randall Balestriero, Florian Buettner

[![Paper on arXiv](assets/badge-arxiv.svg)](https://arxiv.org/abs/2608.27395)
[![Project Page](assets/badge-project-page.svg)](https://levjepa.github.io)
[![Model on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/galilai-group/LeVJEPA-1.5-Large)

## What is LeVJEPA?

Self-supervised video pretraining has so far relied on architectural machinery to avoid representation collapse:
- an EMA target encoder, a stop-gradient and a capacity-limited predictor (V-JEPA), or
- pixel-space reconstruction with a dedicated decoder (VideoMAE).

LeVJEPA dispenses with all of it. A **single encoder** is trained with an invariance loss over global and local views of a clip. The loss is regularized by **SIGReg**, the Sketched Isotropic Gaussian Regularizer from LeJEPA, which constrains the embedding distribution to an isotropic Gaussian and so rules out collapse with a provable guarantee. The trainable architecture reduces to an encoder and a small projector, and the objective has a single hyperparameter, λ.

Because there is no target-encoder forward pass and no predictor, the cost of a training step depends only on the number of tokens the encoder sees. The paper develops two consequences of this:

- **Aggressive token dropping is an augmentation, not an approximation.** Dropping 95% of patch tokens uniformly at random raises ImageNet probing accuracy *monotonically*, from 33.9% with all tokens to 47.6% at 95% drop, while cutting per-step cost by up to 20×. Structured tube masking, which masked-prediction methods need, actively hurts here.
- **Causal attention across frames comes for free.** No asymmetry between branches is required, so the encoder can be causal across frames at no measurable accuracy cost. Every frame's representation then depends only on past observations, which is what streaming inference and autoregressive world models need.

## Results

[LeVJEPA-1.5-Large](https://huggingface.co/galilai-group/LeVJEPA-1.5-Large) was trained on ~5,300 unique video hours with 151 exaflops of compute. That is 25.7× less data and 5.6× less compute than V-JEPA 1, and 321× less data and ~28.5× less compute than V-JEPA 2. All three encoders are evaluated frozen with attentive probing. The table shows mean accuracy (%):

| | LeVJEPA 1.5 Large | V-JEPA 1 | V-JEPA 2 |
|---|---:|---:|---:|
| Image (15 datasets) | 83.34 | 83.03 | **84.90** |
| Video (7 datasets) | 68.94 | 72.79 | **74.61** |
| **Overall (22 datasets)** | 78.76 | 79.77 | **81.63** |

Per-dataset results and the training budget are on the [model card](https://huggingface.co/galilai-group/LeVJEPA-1.5-Large).

## Versions

This repository holds the second version of the code. The encoder of the paper and the released checkpoint (**v1**) uses a `[cls]` token over the whole clip and block-causal attention. The v1 code remains available as `legacy/vision_transformer.py` so that v1 checkpoints load and evaluate here (see [Loading v1 models](#loading-v1-models)).

The **v2** encoder ([model.py](model.py)) is built for streaming:
- Each frame attends only to itself and the `attention_window - 1` frames before it, i.e. causal attention with a sliding window over frames.
- A single readout token at the last frame summarises the clip.
- Positions use 3-D RoPE over (time, row, column) and the blocks use QK-norm and SwiGLU.

Pretraining is otherwise the paper's recipe:
- random token dropping (by default 10 of 196 tokens per frame for the global view and 2 of 36 for the local views)
- one global and several local views per clip
- an invariance loss on the projected readout plus λ · SIGReg.

## Pretrained checkpoint

A ViT-L/16 is released as [**LeVJEPA-1.5-Large**](https://huggingface.co/galilai-group/LeVJEPA-1.5-Large). It is a v1 encoder with 303M parameters, trained on 1.8M clips from Kinetics-710, Something-Something v2, Walking Tours and PE-Video. The released tensors are the EMA copy of the encoder.

It loads without this repository through `transformers`:

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained("galilai-group/LeVJEPA-1.5-Large", trust_remote_code=True).eval()

video = torch.randn(1, 3, 16, 224, 224)   # (B, C, T, H, W), ImageNet-normalised, 16 frames at ~7.5 fps
with torch.no_grad():
    out = model(pixel_values=video)
out.last_hidden_state   # (1, 3137, 1024): [cls] + 16*14*14 patch tokens
out["pooler_output"]    # (1, 1024): the [cls] token
```

For a single image, repeat it along the time axis: `image.unsqueeze(2).repeat(1, 1, 16, 1, 1)`. To see the features rather than just extract them, [notebooks/feature_visualization.ipynb](notebooks/feature_visualization.ipynb) reproduces the paper's dense-feature visualizations: patch-token PCA on images and query-patch cosine similarity on videos. Leave `config.attn_mode` at `"block_causal"`. The weights were trained causal, and full attention returns worse features without raising an error. The checkpoint is meant for frozen feature extraction and has no classification head. Like the v1 encoder code, the weights are under CC BY-NC 4.0.

### Loading v1 models

[legacy/encoder.py](legacy/encoder.py) wraps the v1 encoder behind the v2 interface. It loads:
- the Hugging Face release, by repo id
- a `.safetensors` file
- a Lightning `.ckpt` written by the v1 `main.py` (its EMA copy by default).

```python
from legacy.encoder import load_legacy_encoder

encoder = load_legacy_encoder("galilai-group/LeVJEPA-1.5-Large", name="vit_large", num_frames=16)
cls, patches, _ = encoder(clip)   # clip (B, T, C, H, W) -> (B, D), (B, T, 196, D)
```

The evaluation below runs on v1 models with [configs/model/levjepa_v1_vitl.yaml](configs/model/levjepa_v1_vitl.yaml):

```bash
python -m eval.eval model=levjepa_v1_vitl checkpoint=galilai-group/LeVJEPA-1.5-Large
python -m eval.eval model=levjepa_v1_vitl checkpoint=/path/to/v1/last.ckpt   # ema=false probes the student
```

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/) and runs from the repository root:

```bash
uv sync                   # training environment (CUDA 12.8 wheels on Linux)
uv sync --extra data      # + decord / yt-dlp, only to build the Walking Tours store (decord: Linux and Windows)
uv sync --extra notebook  # + matplotlib / jupyterlab / PyAV, for the notebooks
```

## Data

Training and evaluation read [Lance](https://lancedb.github.io/lance/) stores with one frame per row:

| column        | type   | used by                       |
|---------------|--------|-------------------------------|
| `frame`       | binary | encoded image (JPEG or PNG)   |
| `episode_idx` | int    | the video a row belongs to    |
| `label`       | int    | class label (evaluation only) |

All rows of one episode must be contiguous and in temporal order. Images are episodes of length 1. Stores written for v1, with schema `(episode_idx, step_idx, frame, h, w, label)`, work unchanged.

Every training store needs an episode index, which you build once:

```bash
python scripts/build_episodes.py data/kinetics/train.lance   # writes data/kinetics/train.episodes.npz
```

### Walking Tours

[Walking Tours](https://huggingface.co/datasets/shawshankvkt/Walking_Tours) (Venkataramanan et al., *Is ImageNet worth 1 video?*, ICLR 2024) consists of 10 long (1–7 h), continuous, first-person city walks. It lets you reproduce video pretraining end to end from public data. The Hugging Face dataset ships only the YouTube URLs, so the build has two steps:
1. Download the videos at 720p60 (~25 GB).
2. Encode them into a Lance store (~33 GB) at 15 fps, short edge 384, JPEG quality 90, in 128-frame episodes. The episode index is written at the end.

```bash
bash scripts/download_walking_tours.sh                                       # network-bound
uv run python scripts/build_lance_walking_tours.py --workers 16             # CPU-bound, use many cores
```

Both default to `data/walking_tours/`. To keep the data elsewhere, pass the paths explicitly and set `LEVJEPA_DATA_ROOT=/path/to/data` for training; the store is then read from `$LEVJEPA_DATA_ROOT/walking_tours/train.lance`.

## Pretraining

```bash
python train.py dataset=walking_tours             # ViT-B on Walking Tours, clips at 7.5 fps
python train.py                                   # configs/config.yaml: ViT-B on configs/dataset/single.yaml
python train.py dataset=mixture                   # several stores, see configs/dataset/mixture.yaml
torchrun --nproc_per_node=4 train.py trainer.devices=4
```

If you'd rather start small, [notebooks/training_workshop.ipynb](notebooks/training_workshop.ipynb) walks through the whole method in miniature. It builds a ViT-Tiny from this repository's components and trains it inside the notebook on five minutes of a Walking Tours video, using the same views and loss as `train.py`. It runs on a CUDA GPU, an Apple Silicon Mac (MPS) or plain CPU.

Any config value can be overridden on the command line with Hydra syntax, for example `data.batch_size=32 wandb.mode=offline`. Under Slurm, launch one task per GPU with `srun`. Ranks are read from `SLURM_PROCID` / `SLURM_NTASKS` (or `RANK` / `WORLD_SIZE` under torchrun).

The main settings in [configs/config.yaml](configs/config.yaml):

- **`masking`**: the fraction of patch tokens kept per frame for the global and local views, rounded up to a whole token. The default of 0.05 keeps 10 of 196 global and 2 of 36 local tokens. With `iid: true`, `T * keep` tokens are drawn uniformly over the whole clip instead of `keep` from every frame.
- **`loss.lam`**: the SIGReg weight. The paper uses 0.02; the v2 default is 0.06.
- **`data.probs`**: maps clip length to probability. One length is drawn per batch.
- **`ema`**: keeps an exponential moving average of the encoder, which evaluation probes by default. It plays no role in the objective. Set it to `null` to disable it.
- **`scheduler`**: `WSD` (warmup-stable-decay) warms up linearly, holds the learning rate flat, then decays it as `1 - sqrt` over the last `decay_epochs`. To train longer, resume a stable-phase checkpoint (`resume=...`) and set `trainer.max_epochs` to its epoch plus `decay_epochs`.
- **`muon`**: optional, see below.
- **`holdout`**: optionally keeps a fixed set of training videos out of training and logs their loss as `val/*`.

### Muon

When `muon` is set, the encoder's transformer blocks are trained with [Muon](https://kellerjordan.github.io/posts/muon/). These are the attention `qkv`/`proj` matrices and the SwiGLU `gate`/`up`/`down` matrices, selected by `MUON_MODULES` in [train.py](train.py). Muon keeps a momentum buffer per weight matrix and orthogonalises it with a few Newton–Schulz iterations before each update. The update therefore has the same strength in every direction, rather than being dominated by a few large singular directions. Muon is only defined for 2-D matrices, so every other parameter stays on AdamW: the patch embedding, readout token, norms and projector. Both optimizers use the same learning-rate schedule. With `adjust_lr_fn: match_rms_adamw`, Muon's update is rescaled to the RMS of an AdamW update, so the AdamW learning rate and weight decay can be reused:

```yaml
muon: {lr: 4e-4, weight_decay: 0.04, momentum: 0.95, nesterov: true, adjust_lr_fn: match_rms_adamw}
```

## Evaluation

The encoder stays frozen while an attentive probe is trained on its patch tokens. The probe is V-JEPA 2's `AttentiveClassifier` with depth 4: three self-attention blocks over the tokens, then one learned query that cross-attends to them, then a linear classifier. The datasets follow V-JEPA's frozen-evaluation protocols:

```bash
python -m eval.eval checkpoint=/path/to/last.ckpt                    # ImageNet-1k
python -m eval.eval checkpoint=/path/to/last.ckpt benchmark=ssv2     # Something-Something v2, 16x2x3
```

- On ImageNet, each image goes through timm's train transform (or resize + centre crop at validation) and is repeated over `num_frames` frames.
- On SSv2, the probe sees 2 segments of 16 frames and validation averages 3 spatial views. The video transforms are imported from a checkout of [facebookresearch/jepa](https://github.com/facebookresearch/jepa); set `benchmark.jepa_root` to its path.

When a checkpoint was trained with an EMA, the probe uses the EMA weights unless you set `ema=false`.

**Learning rate and training length.** The default learning rate is 3e-5. For the best numbers, search over learning rates when compute allows:

```bash
python -m eval.eval checkpoint=/path/to/last.ckpt "optimizer.lr=[3e-5,1e-4,3e-4,1e-3]"
```

Each learning rate trains its own probe on the same frozen tokens. The encoder still runs once per batch, so a search costs only the extra probe compute. Each probe logs `val/top1@<lr>`, and `val/best_top1` is the best value over all learning rates and epochs so far.

Probes train for up to 50 epochs with early stopping: the run ends once no probe has improved its `val/top1` for 10 epochs. Change this with `trainer.max_epochs=...` and `early_stopping.patience=...`, or turn early stopping off with `early_stopping=null`.

## Repository layout

```
train.py                     pretraining entry point (optimizers, schedule, loss step)
model.py                     v2 encoder (sliding-window causal ViT, readout token) and projector
loss.py                      SIGReg
data.py                      lance loading, episode / mixture sampling, GPU view augmentation
eval/eval.py                 frozen attentive-probe evaluation
eval/datasets.py             V-JEPA image and video evaluation datasets
eval/probe.py                V-JEPA 2 attentive probe (depth 4) and its lr / weight-decay schedule
legacy/vision_transformer.py v1 encoder (the paper's ViT), for v1 checkpoints
legacy/encoder.py            v1 checkpoint loading behind the v2 encoder interface
scripts/build_episodes.py    episode index of a lance store
scripts/*walking_tours*      Walking Tours download and Lance store build
configs/                     Hydra configs (model, dataset, benchmark)
notebooks/                   training workshop and feature visualization of the released checkpoint
```

## Citation

```bibtex
@misc{kuhn2026levjepaefficientscalable,
      title={LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics},
      author={Lukas Kuhn and Lucas Maes and Giuseppe Serra and Quentin Le Lidec and Yann LeCun and Randall Balestriero and Florian Buettner},
      year={2026},
      eprint={2608.27395},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2608.27395},
}
```

## License

MIT (see [LICENSE](LICENSE)). [eval/probe.py](eval/probe.py) is adapted from Meta's [V-JEPA 2](https://github.com/facebookresearch/vjepa2), which is also MIT-licensed. Code adapted from Meta's [V-JEPA](https://github.com/facebookresearch/jepa) remains under CC BY-NC 4.0:
- [legacy/vision_transformer.py](legacy/vision_transformer.py)
- `segment_indices` in [eval/datasets.py](eval/datasets.py).

The released checkpoint is also under CC BY-NC 4.0.
