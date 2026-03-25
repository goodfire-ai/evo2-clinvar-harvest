#!/bin/bash
# SLURM wrapper for pipeline/harvest.py (chromosome-window harvest).
#
# Usage:
#   sbatch --array=0-15 pipeline/harvest.sh --preset deconfounded-full \
#       --storage /path/to/output [extra_args...]
#
# All arguments are forwarded to harvest.py. The script injects --shard-id
# and --n-shards from SLURM_ARRAY_TASK_ID / SLURM_ARRAY_TASK_COUNT.
#
# Examples:
#   # Full pipeline: harvest + finalize (recommended)
#   STORAGE=/path/to/output
#   HARVEST=$(sbatch --parsable --array=0-127 pipeline/harvest.sh \
#       --preset deconfounded-full --storage $STORAGE)
#   sbatch --dependency=afterok:${HARVEST} pipeline/finalize.sh $STORAGE
#
#   # From CSV manifest, 32 shards, 20B model
#   STORAGE=/path/to/output
#   HARVEST=$(sbatch --parsable --array=0-31 pipeline/harvest.sh \
#       --manifest /path/to/manifest.csv --storage $STORAGE \
#       --model-name evo2_20b --block 20)
#   sbatch --dependency=afterok:${HARVEST} pipeline/finalize.sh $STORAGE

#SBATCH --job-name=harvest
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=logs/harvest/%x_%A_%a.out
#SBATCH --error=logs/harvest/%x_%A_%a.err

set -euo pipefail

# SBATCH copies scripts to a tmp dir, so $0 is unreliable.
# Use SLURM_SUBMIT_DIR (the directory where sbatch was invoked).
cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/harvest

N_SHARDS="${SLURM_ARRAY_TASK_COUNT:-1}"

echo "=== Harvest ==="
echo "Shard: ${SLURM_ARRAY_TASK_ID} / ${N_SHARDS}"
echo "Node:  $(hostname)"
echo "Start: $(date)"
echo "Args:  $@"
echo ""

uv run python pipeline/harvest.py \
    --shard-id "${SLURM_ARRAY_TASK_ID}" \
    --n-shards "${N_SHARDS}" \
    "$@"

echo ""
echo "End: $(date)"
