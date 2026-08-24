# LeVJEPA

Self-supervised video pretraining with [LeJEPA](https://arxiv.org/abs/2511.08544)-style SIGReg regularization on a V-JEPA vision transformer backbone. A multi-crop prediction objective (global clip embedding predicts local crop embeddings through a shared projector) is regularized with the Sketched Isotropic Gaussian Regularizer, so the method needs no EMA teacher, no stop-gradient, and no masking-based target network.

The default configuration trains a ViT-B/16 on the [Walking Tours](https://huggingface.co/datasets/shawshankvkt/Walking_Tours) dataset (Venkataramanan et al., *Is ImageNet worth 1 video?*, ICLR 2024).

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

The full recipe lives in [conf/config.yaml](conf/config.yaml) and follows our large-scale ViT-L video-mixture pretraining runs: ViT-B/16, 16-frame clips at stride 2 (7.5 fps), tubelet 1 with 95% token drop, one global 224 crop + 10 local 96 crops, projector 256, SIGReg 0.02 on the CLS token, effective batch 3072 clips (16 GPUs x 96 x grad-accum 2), lr 4e-4 with 1200-step warmup then flat, constant weight decay 0.04, and an EMA of the encoder saved into every checkpoint under `state_dict_ema`. That is ~395 optimizer steps per epoch, so the default 26 epochs is ~10k steps; since the schedule is flat after warmup, `trainer.max_epochs` can be raised to train longer without changing anything else.

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

Any config key can be overridden the same way (`optimizer.lr=2e-4`, `model.name=vit_large`, `model.attn_mode=block_causal`, ...). Hydra writes each run into `outputs/<date>/<time>/`, with checkpoints under `checkpoints/` inside it. Logging goes to Weights & Biases when enabled: `wandb.enabled=true wandb.config.entity=<you>`.

To fine-tune or continue from a checkpoint with a fresh schedule, pass `resume.ckpt_path=/path/to/last.ckpt` (with the default `resume.weights_only=true` only the weights are loaded); set `resume.weights_only=false` to resume optimizer state and schedule as well.

## Repository layout

- [main.py](main.py) — hydra entry point: builds loaders, encoder, projector, loss, and the Lightning trainer.
- [module.py](module.py) — SIGReg loss and the VisionTransformer (adapted from [facebookresearch/jepa](https://github.com/facebookresearch/jepa), with RoPE, token drop, and block-causal attention options).
- [data/loader.py](data/loader.py) — Lance-backed clip datasets (single store or weighted mixtures) and the multi-crop transform.
- [callbacks.py](callbacks.py) — weight-decay scheduler and weight EMA callbacks.
- [conf/](conf/) — hydra configs; add a yaml under `conf/data/` to train on another Lance store with the same `(episode_idx, step_idx, frame, h, w, label)` schema.

## License

MIT (see [LICENSE](LICENSE)), except [module.py](module.py), which is adapted from Meta's V-JEPA and remains under CC BY-NC 4.0.
