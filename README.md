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
# Harvest activations (8 GPUs, pilot preset for fast test)
STORAGE=/path/to/output
HARVEST=$(sbatch --parsable --array=0-7 pipeline/harvest.sh \
    --preset pilot --storage $STORAGE)
sbatch --dependency=afterok:${HARVEST} pipeline/finalize_harvest.sh $STORAGE

# Train probe (4 GPUs)
torchrun --nproc_per_node=4 pipeline/train.py \
    --activations $STORAGE --preset pilot --output $STORAGE/cov64

# Embed + evaluate
EMBED=$(sbatch --parsable --array=0-7 --gres=gpu:1 --wrap \
    "cd \${SLURM_SUBMIT_DIR} && uv run python pipeline/embed.py \
     --probe $STORAGE/cov64 --activations $STORAGE --preset pilot \
     --shard-id \${SLURM_ARRAY_TASK_ID} --n-shards \${SLURM_ARRAY_TASK_COUNT}")
sbatch --dependency=afterok:${EMBED} pipeline/finalize_embed.sh $STORAGE/cov64
uv run python pipeline/eval.py --probe $STORAGE/cov64 --preset pilot
```

## Agent setup instructions

If you are an AI agent setting up this repo on a new machine, follow these steps:

1. **Clone and install** — `git clone` the repo, `cd` into it, run `uv sync`. This installs all dependencies including Evo2, goodfire-core, PyTorch (CUDA 12.8), and data tools (cyvcf2, gffutils, pyfaidx). Requires Python 3.12 and a CUDA GPU.

2. **Run setup** — `uv run setup`. This downloads GENCODE v49 (GTF + GRCh38 FASTA, ~1GB compressed) and ClinVar VCF (~300MB) from public FTP/HTTPS, then builds `data/gencode/genes.feather`, `data/gencode/chromosomes.feather`, `data/clinvar/variants.feather`, and 3 preset metadata files under `data/clinvar/{pilot,labeled,unlabeled}/`. Takes ~15 min on first run, cached thereafter. Pass `--refresh` to force rebuild.

3. **Harvest** — Submit `pipeline/harvest.sh` as a SLURM array job. Each shard processes a slice of variants: reads a genomic window from the chromosome, runs Evo2 bidir, selects top-K divergent positions, and writes activation diffs to goodfire-core chunked format. Key flags: `--preset` (which ClinVar variants), `--storage` (output dir), `--model-name` (evo2_7b or evo2_20b), `--block` (transformer layer, default 27 for 7B). After all shards finish, run `pipeline/finalize_harvest.sh <storage>` to merge partitions.

4. **Train** — Run `pipeline/train.py` with torchrun for DDP. Trains a SequenceCovarianceProbe from goodfire-core on unified activation diffs. Saves `weights.pt`, `split.feather` (gene holdout), and `config.json` to the output dir.

5. **Embed** — Run `pipeline/embed.py` as a SLURM array. Streams activations through the trained probe, writes embeddings + pathogenic scores. Run `pipeline/finalize_embed.sh` to merge shard scores into a single `scores.feather`.

6. **Eval** — Run `pipeline/eval.py --probe <dir> --preset <preset>`. Computes AUROC/AUPRC with bootstrap CIs, stratified by consequence type.

**Key conventions**: All pipeline scripts take paths via CLI flags — no hardcoded paths anywhere. SLURM scripts use `${SLURM_SUBMIT_DIR}` for portability. Don't specify `--partition` (cluster default). For Evo2-7B use `--block 27 --d-model 4096`; for 20B use `--block 20 --d-model 8192`.

## Dependencies

- [goodfire-core](https://github.com/goodfire-ai/goodfire-core) — activation storage, probe training, streaming
- [Evo2](https://github.com/ArcInstitute/evo2) — genomic foundation model
- PyTorch (CUDA 12.8), Polars, cyvcf2, gffutils, pyfaidx
