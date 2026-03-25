# evo2-clinvar-harvest

End-to-end pipeline for predicting variant pathogenicity using [Evo2](https://arcinstitute.org/research/evo2) bidirectional activations on [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) variants.

## Overview

The pipeline extracts activation diffs from Evo2 (forward and backward passes on variant vs. reference sequences), trains a covariance probe on those diffs, and evaluates pathogenicity prediction on held-out genes.

```
setup-gencode    Download GRCh38 gene coordinates + sequences from GENCODE
setup-clinvar    Download ClinVar VCF + build variant presets
     |
harvest.py       Extract Evo2 bidirectional activation diffs (SLURM array)
finalize.sh      Merge sharded harvest output
     |
train.py         Train CovarianceProbe on unified diffs (DDP)
     |
embed.py         Stream diffs through probe -> embeddings + scores (SLURM array)
finalize_embed   Merge shard scores
     |
eval.py          AUROC/AUPRC with bootstrap CIs, stratified by consequence
```

## Quickstart

```bash
# Clone and install
git clone https://github.com/tdooms-goodfire/evo2-clinvar-harvest.git
cd evo2-clinvar-harvest
uv sync

# Download reference data (~10 min, needs internet)
uv run setup-gencode        # GENCODE v49 GTF + GRCh38 FASTA -> data/gencode/
uv run setup-clinvar        # ClinVar VCF -> data/clinvar/variants.feather

# Build a specific preset
uv run setup-clinvar deconfounded-full

# Harvest Evo2 activations (requires GPU + evo2 extra)
uv sync --extra evo2
sbatch pipeline/harvest.sh --preset deconfounded-full --model evo2-7b --output /path/to/activations

# Train probe
torchrun --nproc_per_node=4 pipeline/train.py \
    --activations /path/to/activations \
    --preset deconfounded-full \
    --output /path/to/probe

# Embed + evaluate
sbatch pipeline/embed.py --probe /path/to/probe --output /path/to/embeddings
uv run python pipeline/eval.py --scores /path/to/embeddings/scores.feather --preset deconfounded-full
```

## Data flow

All downloaded and generated data lives under `data/`:
- `data/_downloads/` -- raw VCF, GTF, FASTA files
- `data/gencode/genes.feather` -- 20k protein-coding genes with sequences
- `data/gencode/chromosomes.feather` -- full GRCh38 chromosome sequences
- `data/clinvar/variants.feather` -- ~2M parsed ClinVar variants
- `data/clinvar/{preset}/metadata.feather` -- filtered/sampled variant subsets

Activation datasets and probe outputs are stored at user-specified paths (CLI args).

## ClinVar presets

| Preset | Description | Size |
|--------|-------------|------|
| `all` | All ClinVar variants (including VUS, conflicting) | ~2M |
| `labeled` | Pathogenic + benign, stars >= 1 | ~450k |
| `deconfounded-full` | CADD-balanced, all variant types, stars >= 1 | ~184k |
| `deconfounded` | CADD-balanced, SNVs only, genes <= 100kb | ~50k |
| `broad` | Consequence-stratified sample | ~100k |
| `confident` | Stars >= 3 only | ~35k |
| `pilot` | 8 well-studied genes | ~400 |

## Dependencies

- [goodfire-core](https://github.com/goodfire-ai/goodfire-core) -- activation storage, probe training, streaming
- [Evo2](https://github.com/ArcInstitute/evo2) -- genomic foundation model (optional, for harvesting)
- PyTorch, Polars, cyvcf2, gffutils, pyfaidx
