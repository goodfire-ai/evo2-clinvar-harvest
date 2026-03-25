# evo2-clinvar-harvest

End-to-end pipeline for predicting variant pathogenicity using [Evo2](https://arcinstitute.org/research/evo2) bidirectional activations on [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) variants.

## Quickstart

```bash
git clone https://github.com/tdooms-goodfire/evo2-clinvar-harvest.git
cd evo2-clinvar-harvest
uv sync

# Download GENCODE + ClinVar, build all 3 presets (~15 min)
uv run setup
```

This downloads GRCh38 from GENCODE and the ClinVar VCF from NCBI, then builds three presets:

| Preset | Description | Size |
|--------|-------------|------|
| `pilot` | CADD-deconfounded SNVs in small genes, for fast iteration | ~8k |
| `labeled` | All pathogenic + benign variants (SNVs + indels, stars >= 1) | ~450k |
| `unlabeled` | VUS + conflicting + other + low-confidence (for inference) | ~1.5M |

## Pipeline

```
uv run setup         Download GENCODE + ClinVar, build presets
     |
harvest.py           Extract Evo2 bidirectional activation diffs (SLURM array)
finalize_harvest.sh  Merge sharded harvest output
     |
train.py             Train CovarianceProbe on unified diffs (DDP)
     |
embed.py             Stream diffs through probe -> embeddings + scores (SLURM array)
finalize_embed.sh    Merge shard scores
     |
eval.py              AUROC/AUPRC with bootstrap CIs, stratified by consequence
```

```bash
# Install evo2 for GPU harvesting
uv sync --extra evo2

# Harvest activations
sbatch pipeline/harvest.sh --preset labeled --model evo2-7b --storage /path/to/activations
bash pipeline/finalize_harvest.sh /path/to/activations

# Train probe (4 GPUs)
torchrun --nproc_per_node=4 pipeline/train.py \
    --activations /path/to/activations --preset labeled --output /path/to/probe

# Embed + evaluate
sbatch pipeline/embed.py --probe /path/to/probe --output /path/to/embeddings
bash pipeline/finalize_embed.sh /path/to/embeddings
uv run python pipeline/eval.py --scores /path/to/embeddings/scores.feather --preset labeled
```

## Dependencies

- [goodfire-core](https://github.com/goodfire-ai/goodfire-core) -- activation storage, probe training, streaming
- [Evo2](https://github.com/ArcInstitute/evo2) -- genomic foundation model (optional, for harvesting)
- PyTorch, Polars, cyvcf2, gffutils, pyfaidx
