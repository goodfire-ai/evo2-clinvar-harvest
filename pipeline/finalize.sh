#!/bin/bash
# Finalize partitioned harvest output into unified datasets.
#
# Merges partition_N/ directories into a single dataset with merged metadata
# and SQLite index. Validates all datasets have matching item counts and
# sequence IDs. Should run as an afterok dependency on the harvest array job.
#
# Usage:
#   # Chained after harvest (recommended)
#   HARVEST=$(sbatch --parsable --array=0-127 pipeline/harvest.sh ...)
#   sbatch --dependency=afterok:${HARVEST} pipeline/finalize.sh /path/to/storage-dir

#SBATCH --job-name=finalize
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/harvest/%x_%j.out
#SBATCH --error=logs/harvest/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR}"
mkdir -p logs/harvest

export STORAGE_DIR="${1:?Usage: finalize.sh <storage-dir>}"

echo "=== Finalize ==="
echo "Storage: ${STORAGE_DIR}"
echo "Node:    $(hostname)"
echo "Start:   $(date)"
echo ""

uv run python -c "
import os
from goodfire_core.storage import ActivationDataset, FilesystemStorage
from goodfire_core.storage.finalizer import finalize_partitions

storage = FilesystemStorage(os.environ['STORAGE_DIR'])
datasets = ('activations', 'positions', 'mean', 'loss')

# Finalize each dataset
results = {}
for name in datasets:
    print(f'Finalizing {name}...')
    meta = finalize_partitions(storage, name, cleanup=True, build_index=True)
    results[name] = meta['total_items']
    print(f'  {meta[\"total_items\"]:,} items, {meta[\"num_chunks\"]} chunks')

# Validate: all datasets must have the same item count
counts = set(results.values())
assert len(counts) == 1, (
    f'ITEM COUNT MISMATCH across datasets! '
    + ', '.join(f'{k}={v:,}' for k, v in results.items())
)
print(f'\nValidation passed: {results[\"activations\"]:,} items in all {len(datasets)} datasets')

# Validate: sequence IDs match between activations and mean
ds_acts = ActivationDataset(storage, 'activations', include_provenance=True)
ds_pool = ActivationDataset(storage, 'mean', include_provenance=True)
ids_acts = set(ds_acts.activation_index.list_sequence_ids())
ids_pool = set(ds_pool.activation_index.list_sequence_ids())
assert ids_acts == ids_pool, (
    f'SEQUENCE ID MISMATCH: '
    f'{len(ids_acts - ids_pool)} in activations only, '
    f'{len(ids_pool - ids_acts)} in mean only'
)
print(f'Sequence ID validation passed: {len(ids_acts):,} unique IDs match')
print('\nAll datasets finalized and validated.')
"

echo ""
echo "End: $(date)"
