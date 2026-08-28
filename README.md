# LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics

Official implementation of **LeVJEPA**, the first video encoder trained under LeJEPA's collapse-free objective.

Lukas Kuhn, Lucas Maes, Giuseppe Serra, Quentin Le Lidec, Yann LeCun, Randall Balestriero, Florian Buettner

[![arXiv](https://img.shields.io/badge/arXiv-2608.27395-b31b1b.svg)](https://arxiv.org/abs/2608.27395)
[![Project Page](https://img.shields.io/badge/Project-Page-1f6feb.svg)](https://levjepa.github.io)
[![Checkpoint on Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large)

## What is LeVJEPA?

Self-supervised video pretraining has so far relied on architectural machinery to avoid representation collapse: an EMA target encoder, a stop-gradient, a capacity-limited predictor (V-JEPA), or pixel-space reconstruction with a dedicated decoder (VideoMAE). LeVJEPA dispenses with all of it. A **single encoder** is trained with an invariance loss over global and local views of a clip, regularized by **SIGReg** (the Sketched Isotropic Gaussian Regularizer from LeJEPA), which constrains the embedding distribution to an isotropic Gaussian and thereby excludes collapse with a provable guarantee. The trainable architecture reduces to an encoder and a small projector, and the objective carries a single hyperparameter (λ = 0.02).

Because every operation contributes directly to the objective — there is no target-encoder forward pass and no predictor — the cost of a training step is governed only by the number of tokens the encoder observes. The paper develops two consequences of this:

- **Aggressive token dropping is an augmentation, not an approximation.** Dropping 95% of patch tokens uniformly at random raises ImageNet probing accuracy *monotonically* (33.9% with all tokens → 47.6% at 95% drop) while cutting per-step cost by up to 20×. Structured tube masking, necessary in masked-prediction methods to keep the imputation task non-trivial, actively hurts here (50.7% → 39.6%): when nothing is imputed, the retained tokens are the encoder's only observation of the clip, and a spatio-temporally distributed random sample preserves far more of its content.
- **Block-causal attention comes for free.** Since no asymmetry between branches is required, the encoder can be trained with attention that is bidirectional within a frame and causal across frames, at no measurable accuracy cost (51.2% vs. 50.7% bidirectional). Every frame's representation is then a function of past observations alone, so the encoder itself provides temporally ordered state that extends to incoming frames without re-encoding — what autoregressive world models and streaming inference need, without fitting a separate temporal model after pretraining.

Two further simplifications fall out of the same analysis: temporal patch aggregation at the input (tubelet 2) is unnecessary — per-frame tokenization matches or beats it at an equal token budget, including on motion-centric SSv2 — and dense, semantically organized patch representations emerge even though only the `[cls]` token is supervised, without the auxiliary patch-level objectives of prior work.

## Results

All comparisons retrain the baselines on identical data with their official implementations, evaluated with frozen attentive probing following V-JEPA's protocol.

- **Epoch-matched:** LeVJEPA matches or surpasses V-JEPA 2 across ViT-S/B/L at **5.6–20.8× less total pretraining compute**. The LeVJEPA ViT-L costs less than half the compute of the V-JEPA 2 ViT-S.
- **FLOP-matched (ViT-B):** granted an equal total FLOP budget, LeVJEPA leads the strongest video baseline by **+7.6 points on ImageNet-1K** (61.0%), attains the best Kinetics-400 linear probe (44.6%), and stays within 3.2 points on Something-Something-v2.
- **Vs. image pretraining:** against a compute-matched DINOv2 trained on frames of the same videos, LeVJEPA comes within 3.1 points on ImageNet while **nearly doubling** its motion-centric SSv2 accuracy (30.4% vs. 16.9%) — to our knowledge the first FLOP-matched comparison in which video pretraining reaches near-parity with a state-of-the-art image method on appearance-centric evaluation.
- **Consumer hardware:** a ViT-Tiny pretrained for 12 hours on a single RTX 5080 (16 GB) on eight unlabeled Walking Tours videos goes from 8.9% to 25.2% frozen ImageNet accuracy; sparse token sequences keep batch size 128 under 8 GB where an equivalent V-JEPA setup saturates the card at batch 28.

This repository contains the full pretraining recipe. The default configuration trains a ViT-B/16 on the [Walking Tours](https://huggingface.co/datasets/shawshankvkt/Walking_Tours) dataset (Venkataramanan et al., *Is ImageNet worth 1 video?*, ICLR 2024), so you can reproduce video pretraining end to end from public data.

## Pretrained checkpoint

If you just want features, skip the pretraining pipeline entirely: the ViT-L/16 from the data-scaling experiments is released as [**LeVJEPA-VideoMix-Large**](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large) — 303M parameters, trained on **VideoMix** (1.8M clips from Kinetics-710, Something-Something v2, Walking Tours, and PE-Video) with tubelet 1, RoPE, and block-causal attention. It reaches 69.5% top-1 on ImageNet-1K and 55.0% on Something-Something-v2 under frozen attentive probing. The released tensors are the EMA copy of the encoder, which is what we evaluate.

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "galilai-group/LeVJEPA-VideoMix-Large", trust_remote_code=True
).eval()

# (B, C, T, H, W), ImageNet-normalised, 16 frames at 224px
video = torch.randn(1, 3, 16, 224, 224)

with torch.no_grad():
    out = model(pixel_values=video)

out.last_hidden_state   # (1, 3137, 1024) -- CLS + 16*14*14 patch tokens
out["pooler_output"]    # (1, 1024)       -- the CLS token
```

`trust_remote_code=True` is required because the modeling code (RoPE + block-causal attention) ships with the weights. For a single image, repeat it along the temporal axis: `image.unsqueeze(2).repeat(1, 1, 16, 1, 1)`. Two things to keep in mind: normalise with the ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`; training sampled frames at ~7.5 fps, so 16 frames cover about two seconds), and leave `config.attn_mode` at its `"block_causal"` default — the weights were trained causal, and running them under full attention won't error but will quietly return worse features. The [model card](https://huggingface.co/galilai-group/LeVJEPA-VideoMix-Large) has the full usage guide and training details.

The checkpoint is intended for frozen feature extraction — attentive or linear probing, retrieval, or as a backbone for downstream heads; it is a self-supervised encoder with no classification head. Like [module.py](module.py), the released weights are under CC BY-NC 4.0.

To see the features rather than just extract them, [notebooks/feature_visualization.ipynb](notebooks/feature_visualization.ipynb) loads the checkpoint and reproduces the paper's dense-feature visualizations — patch-token PCA on images and query-patch cosine similarity on videos (`uv sync --extra notebook`, then `uv run jupyter lab notebooks/feature_visualization.ipynb`).

## Setup

The project is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync               # training environment (CUDA 12.8 wheels on Linux)
uv sync --extra data  # + decord/yt-dlp, needed only to build the dataset
```

Everything runs from the repo root; there is nothing to install as a package.

## Getting Walking Tours

Walking Tours is 10 long (1-7 h) continuous first-person city walk videos. The HuggingFace dataset ships only the YouTube URLs, so the videos are downloaded from YouTube at 720p60 (~25 GB) and then encoded into a frame-per-row [Lance](https://lancedb.github.io/lance/) store at 15 fps, short edge 384, JPEG q90, in 128-frame episodes (~33 GB):

```bash
# 1. download the 10 videos (needs outbound HTTPS; a login or data-transfer
#    node is fine, the step is network-bound)
bash scripts/download_walking_tours.sh

# 2. encode the Lance store (CPU-bound; use many cores)
uv run python scripts/build_lance_walking_tours.py --workers 16
```

On a slurm cluster, the same two steps as batch jobs (add your `--partition`/`--account` to the headers first):

```bash
dl=$(sbatch --parsable slurm/download_walking_tours.slurm)
sbatch --dependency=afterok:$dl slurm/build_walking_tours_lance.slurm
```

Both default to `data/walking_tours/` under the repo root. To keep the data elsewhere, pass the paths explicitly and point training at it with `LEVJEPA_DATA_ROOT=/path/to/data` (the loader expects `$LEVJEPA_DATA_ROOT/walking_tours/train.lance`).

## Training

If you'd rather start small, [notebooks/training_workshop.ipynb](notebooks/training_workshop.ipynb) walks through the whole method in miniature — building the model from this repo's components and training it inside the notebook on a few minutes of a Walking Tours video, with the same multi-crop transform and forward pass as the full pipeline. Both notebooks need only `uv sync --extra notebook` and run on a CUDA GPU, an Apple Silicon Mac (MPS), or plain CPU.

The full recipe lives in [conf/config.yaml](conf/config.yaml) and follows our large-scale ViT-L video-mixture pretraining runs: ViT-B/16, 16-frame clips at stride 2 (7.5 fps), per-frame tokenization (tubelet 1) with 95% random token drop, block-causal attention, one global 224 crop + 10 local 96 crops, projector 256, SIGReg λ = 0.02 on the CLS token, effective batch 3072 clips (16 GPUs x 96 x grad-accum 2), lr 4e-4 with 1200-step warmup then flat, constant weight decay 0.04, and an EMA of the encoder saved into every checkpoint under `state_dict_ema` (evaluation checkpoint only — it plays no role in the objective). That is ~395 optimizer steps per epoch, so the default 26 epochs is ~10k steps; since the schedule is flat after warmup, `trainer.max_epochs` can be raised to train longer without changing anything else.

Token drop applies during training only — at 95% the encoder sees ~158 of the clip's 3137 tokens per step, which is what makes tubelet 1 affordable — while evaluation and inference always run on the full token set.

Slurm (2 nodes x 8 GPUs, matching the config defaults):

```bash
sbatch slurm/train_walking_tours_vitb.slurm
```

Single node x 8 GPUs — keep the effective batch by doubling the accumulation:

```bash
sbatch --nodes=1 slurm/train_walking_tours_vitb.slurm \
  trainer.num_nodes=1 accumulate_grad_batches=4
```

Locally (e.g. one GPU, for a smoke run — this changes the effective batch, so it is not the paper recipe):

```bash
uv run python main.py \
  trainer.devices=1 trainer.num_nodes=1 \
  loader.batch_size=8 loader.num_workers=6 accumulate_grad_batches=1
```

Any config key can be overridden the same way (`optimizer.lr=2e-4`, `model.name=vit_large`, ...). The encoder trains with the paper's block-causal attention by default; pass `model.attn_mode=full` for a fully bidirectional encoder. Hydra writes each run into `outputs/<date>/<time>/`, with checkpoints under `checkpoints/` inside it. Logging goes to Weights & Biases when enabled: `wandb.enabled=true wandb.config.entity=<you>`.

To fine-tune or continue from a checkpoint with a fresh schedule, pass `resume.ckpt_path=/path/to/last.ckpt` (with the default `resume.weights_only=true` only the weights are loaded); set `resume.weights_only=false` to resume optimizer state and schedule as well.

## Repository layout

- [main.py](main.py) — hydra entry point: builds loaders, encoder, projector, loss, and the Lightning trainer.
- [module.py](module.py) — SIGReg loss and the VisionTransformer (adapted from [facebookresearch/jepa](https://github.com/facebookresearch/jepa), with RoPE, random token drop, and block-causal attention options).
- [data/loader.py](data/loader.py) — Lance-backed clip datasets (single store or weighted mixtures) and the multi-crop transform.
- [callbacks.py](callbacks.py) — weight-decay scheduler and weight EMA callbacks.
- [conf/](conf/) — hydra configs; add a yaml under `conf/data/` to train on another Lance store with the same `(episode_idx, step_idx, frame, h, w, label)` schema.
- [notebooks/feature_visualization.ipynb](notebooks/feature_visualization.ipynb) — load the pretrained checkpoint and visualize its features (patch-token PCA, cosine-similarity heatmaps).
- [notebooks/training_workshop.ipynb](notebooks/training_workshop.ipynb) — guided workshop: build a tiny LeVJEPA and train it in the notebook on a slice of a Walking Tours video.

## Citation

```bibtex
@article{kuhn2026levjepa,
  title   = {LeVJEPA: Efficient \& Scalable Video Pretraining without the Heuristics},
  author  = {Kuhn, Lukas and Maes, Lucas and Serra, Giuseppe and Le Lidec, Quentin
             and LeCun, Yann and Balestriero, Randall and Buettner, Florian},
  year    = {2026}
}
```

## License

MIT (see [LICENSE](LICENSE)), except [module.py](module.py), which is adapted from Meta's V-JEPA and remains under CC BY-NC 4.0.
