#!/bin/bash
#SBATCH --job-name=levjepa_wt_vitb
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=13
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# LeVJEPA ViT-B/16 on Walking Tours -- the recipe is fully specified by
# conf/config.yaml (tubelet 1, 95% token drop, 10 local crops, effective batch
# 3072 = 16 GPUs x 96 x accum 2, lr 4e-4 warmup->flat, ~10k optimizer steps
# over 26 epochs). Any hydra key can be overridden at submit time:
#   sbatch slurm/train_walking_tours_vitb.slurm            # defaults
#   sbatch slurm/train_walking_tours_vitb.slurm optimizer.lr=2e-4
#
# Running on 1 node x 8 GPUs instead: keep the effective batch by doubling
# the accumulation --
#   sbatch --nodes=1 slurm/train_walking_tours_vitb.slurm \
#     trainer.num_nodes=1 accumulate_grad_batches=4
#
# Add your site's --partition/--account before submitting. --mem=0 requests
# all node memory; replace with an explicit value if your site rejects it.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_DEVICE_ORDER=PCI_BUS_ID

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true

srun .venv/bin/python main.py \
  trainer.num_nodes="${SLURM_JOB_NUM_NODES}" \
  "$@"
