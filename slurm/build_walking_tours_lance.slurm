#!/bin/bash
#SBATCH --job-name=build_wtours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Encode the 10 Walking Tours videos into a frame-per-row Lance store at
# 15 fps (short edge 384, JPEG q90, 128-frame episodes), ~33 GB.
#
# Add your site's --partition/--account, then either submit standalone or with
# a dependency on the download job so it starts by itself:
#   sbatch --dependency=afterok:<dl_jobid> slurm/build_walking_tours_lance.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

WT_DIR="${WT_DIR:-data/walking_tours}"

echo "=== videos on disk"
ls -la "$WT_DIR/videos"

.venv/bin/python -u scripts/build_lance_walking_tours.py \
  --videos "$WT_DIR/videos" \
  --out "$WT_DIR/train.lance" \
  --workers "${SLURM_CPUS_PER_TASK:-16}"

du -sh "$WT_DIR/train.lance"
echo "done"
