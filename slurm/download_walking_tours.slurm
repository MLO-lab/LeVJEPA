#!/bin/bash
#SBATCH --job-name=dl_wtours
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Download the 10 Walking Tours videos (~25 GB). Requires outbound HTTPS from
# the compute node; if your cluster blocks that, run the script on a login /
# data-transfer node instead:  bash scripts/download_walking_tours.sh
#
# Add your site's --partition/--account before submitting:
#   sbatch slurm/download_walking_tours.slurm

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
mkdir -p logs

bash scripts/download_walking_tours.sh "${VIDEOS_DIR:-data/walking_tours/videos}"
