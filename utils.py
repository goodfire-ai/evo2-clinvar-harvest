"""Shared utilities: paths, downloads, metrics, transforms, and data helpers."""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path

import polars as pl
import torch
from goodfire_core.storage import ActivationDataset, FilesystemStorage
from loguru import logger
from torch import Tensor

# ── Paths ────────────────────────────────────────────────────────────────

OUTPUTS = Path("data")
DOWNLOADS = Path("data/_downloads")

CLINVAR_DIR = OUTPUTS / "clinvar"
GENCODE_DIR = OUTPUTS / "gencode"

CLINVAR_VARIANTS = CLINVAR_DIR / "variants.feather"
GENCODE_GENES = GENCODE_DIR / "genes.feather"
GENCODE_CHROMOSOMES = GENCODE_DIR / "chromosomes.feather"

# ── Biological constants ─────────────────────────────────────────────────

_DNA_COMPLEMENT = str.maketrans("ATGCatgcNn", "TACGtacgNn")

CHROMOSOMES = (*tuple(str(i) for i in range(1, 23)), "X", "Y", "M")

STAR_MAPPING = {
    "practice_guideline": 4,
    "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_conflicting_interpretations": 2,
    "criteria_provided,_single_submitter": 1,
}

CODING_CONSEQUENCES: frozenset[str] = frozenset({
    "frameshift_variant", "inframe_deletion", "inframe_insertion",
    "stop_gained", "nonsense", "stop_lost", "start_lost",
    "missense_variant", "synonymous_variant", "coding_sequence_variant",
    "protein_altering_variant", "incomplete_terminal_codon_variant",
})

SPLICE_CONSEQUENCES: frozenset[str] = frozenset({
    "splice_donor_variant", "splice_acceptor_variant",
    "splice_region_variant", "splice_donor_region_variant",
    "splice_polypyrimidine_tract_variant", "splice_donor_5th_base_variant",
})


def reverse_complement(seq: str) -> str:
    """Reverse complement a DNA sequence."""
    return seq.translate(_DNA_COMPLEMENT)[::-1]


def strip_version(gene_id: str) -> str:
    """Strip Ensembl version suffix: ENSG00000139618.19 -> ENSG00000139618."""
    return gene_id.split(".")[0] if gene_id.startswith("ENS") and "." in gene_id else gene_id


# ── Path / download helpers ──────────────────────────────────────────────


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, dest: Path, compressed: bool = False) -> None:
    """Download file with optional .gz decompression."""
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"}
    req = urllib.request.Request(url, headers=headers)

    raw_dest = dest.with_suffix(dest.suffix + ".gz") if compressed else dest
    with urllib.request.urlopen(req) as response, open(raw_dest, "wb") as out_file:
        shutil.copyfileobj(response, out_file)

    if compressed:
        with gzip.open(raw_dest, "rb") as f_in, open(dest, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
        raw_dest.unlink()


# ── Polars coordinate helpers ────────────────────────────────────────────


def strand_aware_seq_pos() -> pl.Expr:
    """0-based genomic pos -> 0-based sequence index, strand-aware."""
    return (
        pl.when(pl.col("strand") == "+")
        .then(pl.col("pos") - pl.col("start"))
        .otherwise(pl.col("end") - pl.col("pos") - 1)
        .cast(pl.Int64)
    )


# ── Variant data helpers ─────────────────────────────────────────────────


def finalize_variants(df: pl.DataFrame, columns: tuple[str, ...]) -> pl.DataFrame:
    """Add variant_id, shuffle deterministically, and enforce column order."""
    return (
        df
        .with_columns(variant_id=pl.format("{}:{}:{}:{}", "chrom", "pos", "ref", "alt"))
        .select(*columns)
        .sample(fraction=1.0, seed=42, shuffle=True)
    )


def enrich_variants(
    df: pl.DataFrame,
    genes: pl.DataFrame,
    unique_on: tuple[str, ...] = ("variant_id",),
) -> pl.DataFrame:
    """Add gene coordinates and strand-aware offset."""
    import gencode

    n_before = len(df)
    if "gene_id" in df.columns:
        df = df.drop("gene_id")
    gene_cols = ("gene_name", "gene_id", "start", "end", "strand", "length")
    result = (
        df
        .with_row_index("_order")
        .join(genes.select(gene_cols), on="gene_name", how="inner")
        .with_columns(strand_aware_seq_pos().alias("gene_offset"))
        .unique(subset=unique_on)
        .sort("_order")
        .drop("_order")
        .rename({"start": "gene_start", "end": "gene_end", "strand": "gene_strand", "length": "gene_length"})
    )
    logger.info(f"Enriched {len(result):,} / {n_before:,} entries ({n_before - len(result):,} dropped)")
    return result


def with_cache(module: str, preset: str, builder: Callable[[], pl.DataFrame] | None) -> pl.DataFrame:
    """Load DataFrame from cache or build and cache."""
    path = OUTPUTS / module / preset / "metadata.feather"

    if path.exists():
        logger.debug(f"Cache hit: {path}")
        return pl.read_ipc(path)

    assert builder is not None, f"No cache for frozen preset {module}/{preset}. Run setup first."

    logger.info(f"Computing {module}/{preset}")
    df = builder()

    ensure_parent(path)
    tmp_path = path.with_suffix(".feather.tmp")
    df.write_ipc(tmp_path)
    tmp_path.rename(path)
    logger.info(f"Saved {path}")
    return df


# ── Activation streaming ─────────────────────────────────────────────────


def iter_dataset(
    storage: FilesystemStorage,
    dataset_name: str,
    target_ids: set[str],
    transform: Callable[[Tensor], Tensor] | None = None,
    *,
    batch_size: int = 512,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> Iterator[tuple[Tensor, list[str]]]:
    """Stream one dataset with optional per-batch transform."""
    ds = ActivationDataset(storage, dataset_name, batch_size=batch_size, include_provenance=True)
    for batch in ds.training_iterator(
        device=device, n_epochs=1, shuffle=False, drop_last=False, sequence_ids=list(target_ids),
    ):
        x = batch.acts.to(dtype=dtype)
        if transform is not None:
            x = transform(x)
        yield x, batch.sequence_ids


def unified_diff(x: Tensor) -> Tensor:
    """[B, 2, 3, K, d] -> [B, K, 2*d]: var-ref diff, concat fwd+bwd."""
    diff = x[:, :, 0] - x[:, :, 1]
    return torch.cat([diff[:, 0], diff[:, 1]], dim=-1)


# ── Metrics ──────────────────────────────────────────────────────────────


@torch.inference_mode()
def auroc(probs: Tensor, labels: Tensor, reduction: str = "mean") -> Tensor:
    """AUROC (Area Under ROC Curve)."""
    n_samples = probs.shape[-2]
    n_labels = probs.shape[-1]
    batch_shape = probs.shape[:-2]

    sorted_idx = probs.argsort(dim=-2, descending=True)
    y_sorted = labels.gather(-2, sorted_idx).bool()

    n_pos = labels.sum(-2)
    n_neg = n_samples - n_pos

    tps = y_sorted.cumsum(-2).float()
    fps = (~y_sorted).cumsum(-2).float()

    tpr = tps / n_pos.unsqueeze(-2).clamp(min=1)
    fpr = fps / n_neg.unsqueeze(-2).clamp(min=1)

    zeros = torch.zeros(*batch_shape, 1, n_labels, device=probs.device, dtype=fpr.dtype)
    fpr_diff = torch.diff(fpr, dim=-2, prepend=zeros)
    aucs = (fpr_diff * tpr).sum(-2)

    valid = (n_pos > 0) & (n_neg > 0)
    aucs = torch.where(valid, aucs, torch.full_like(aucs, float("nan")))

    return aucs.mean(-1) if reduction == "mean" else aucs


@torch.inference_mode()
def aupr(probs: Tensor, labels: Tensor, reduction: str = "mean") -> Tensor:
    """AUPR (Area Under Precision-Recall Curve)."""
    n_samples = probs.shape[-2]

    sorted_idx = probs.argsort(dim=-2, descending=True)
    y_sorted = labels.gather(-2, sorted_idx).float()

    n_pos = labels.sum(-2)

    tps = y_sorted.cumsum(-2)
    ranks = torch.arange(1, n_samples + 1, device=probs.device, dtype=torch.float32)
    ranks = ranks.view(*([1] * (probs.dim() - 2)), n_samples, 1)
    precision = tps / ranks
    aps = (precision * y_sorted).sum(-2) / n_pos.clamp(min=1)

    valid = n_pos > 0
    aps = torch.where(valid, aps, torch.zeros_like(aps))

    return aps.mean(-1) if reduction == "mean" else aps


# ── Evaluation ───────────────────────────────────────────────────────────


def gene_split(
    metadata: pl.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    gene_col: str = "gene_name",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split metadata by genes -- no gene appears in both sets."""
    genes = metadata.select(gene_col).unique().sort(gene_col)
    n_test = int(len(genes) * test_size)
    shuffled = genes.sample(fraction=1.0, seed=seed, shuffle=True)
    test_genes = set(shuffled[gene_col][:n_test].to_list())

    train_df = metadata.filter(~pl.col(gene_col).is_in(test_genes))
    test_df = metadata.filter(pl.col(gene_col).is_in(test_genes))
    return train_df, test_df


def bootstrap_ci(scores: Tensor, labels: Tensor, metric_fn, n_bootstrap: int = 1000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a binary metric."""
    generator = torch.Generator(device=scores.device).manual_seed(seed)
    n = labels.shape[0]

    indices = torch.randint(0, n, (n_bootstrap, n), generator=generator, device=scores.device)
    boot_scores = scores[indices].unsqueeze(-1)
    boot_labels = labels[indices].unsqueeze(-1)

    values = metric_fn(boot_scores, boot_labels)
    if values.dim() > 1:
        values = values.squeeze(-1)
    values = values[~torch.isnan(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"))

    return (values.quantile(alpha / 2).item(), values.quantile(1 - alpha / 2).item())


STRATA_DEFS: dict[str, pl.Expr] = {
    "overall": pl.lit(True),
    "frameshift": pl.col("is_frameshift"),
    "inframe_coding": ~pl.col("is_frameshift") & pl.col("is_coding"),
    "noncoding": ~pl.col("is_coding"),
    "splice": pl.col("is_splice"),
    "non_splice": ~pl.col("is_splice"),
}


def annotate_strata(metadata: pl.DataFrame, consequence_col: str = "consequence", ref_col: str = "ref", alt_col: str = "alt") -> pl.DataFrame:
    """Add stratification columns (is_frameshift, is_coding, is_splice) to metadata."""
    result = metadata.with_columns(
        pl.col(consequence_col).str.contains("(?i)frameshift").alias("is_frameshift"),
        pl.col(consequence_col).is_in(CODING_CONSEQUENCES).alias("is_coding"),
        pl.col(consequence_col).is_in(SPLICE_CONSEQUENCES).alias("is_splice"),
    )
    if ref_col in metadata.columns and alt_col in metadata.columns:
        result = result.with_columns(
            (pl.col(alt_col).str.len_chars().cast(pl.Int32) - pl.col(ref_col).str.len_chars().cast(pl.Int32)).alias("indel_len"),
        ).with_columns(pl.col("indel_len").abs().alias("indel_size")).with_columns(
            pl.when(pl.col("indel_size") <= 1).then(pl.lit("1bp"))
            .when(pl.col("indel_size") <= 5).then(pl.lit("2-5bp"))
            .when(pl.col("indel_size") <= 20).then(pl.lit("6-20bp"))
            .otherwise(pl.lit(">20bp"))
            .alias("size_bin"),
        )
    return result


def compute_stratified_metrics(
    scores: Tensor, labels: Tensor, metadata: pl.DataFrame,
    strata: dict[str, pl.Expr] | None = None, bootstrap: bool = True,
    n_bootstrap: int = 1000, seed: int = 42, min_positive: int = 5, min_negative: int = 5,
) -> pl.DataFrame:
    """Compute AUROC and AUPRC for each stratum of variants."""
    if strata is None:
        strata = STRATA_DEFS
    rows: list[dict] = []
    for name, expr in strata.items():
        mask_t = torch.from_numpy(metadata.with_columns(expr.alias("_m"))["_m"].to_numpy())
        s, y = scores[mask_t], labels[mask_t]
        n_pos, n_neg = int(y.sum().item()), len(y) - int(y.sum().item())
        if n_pos < min_positive or n_neg < min_negative:
            continue
        s_2d, y_2d = s.unsqueeze(-1), y.unsqueeze(-1)
        row: dict = {"stratum": name, "n_variants": len(y), "n_pathogenic": n_pos, "n_benign": n_neg,
                     "auroc": auroc(s_2d, y_2d).item(), "auprc": aupr(s_2d, y_2d).item()}
        if bootstrap:
            auroc_lo, auroc_hi = bootstrap_ci(s, y, auroc, n_bootstrap=n_bootstrap, seed=seed)
            auprc_lo, auprc_hi = bootstrap_ci(s, y, aupr, n_bootstrap=n_bootstrap, seed=seed)
            row.update(auroc_ci_lo=auroc_lo, auroc_ci_hi=auroc_hi, auprc_ci_lo=auprc_lo, auprc_ci_hi=auprc_hi)
        rows.append(row)
    return pl.DataFrame(rows)


# ── Unified setup ────────────────────────────────────────────────────────


def _cached_download(url: str, dest: Path, compressed: bool = False, refresh: bool = False) -> None:
    if dest.exists() and not refresh:
        logger.info(f"Using cached: {dest}")
        return
    ensure_parent(dest)
    logger.info(f"Downloading: {url}")
    download_file(url, dest, compressed)
    logger.info(f"Saved {dest}")


def _cached_build(output: Path, builder: Callable, name: str, refresh: bool = False) -> None:
    if output.exists() and not refresh:
        logger.info(f"Using cached: {output}")
        return
    ensure_parent(output)
    logger.info(f"Building {name}...")
    builder()
    logger.info(f"Saved {output}")


def setup() -> None:
    """Download GENCODE + ClinVar and build all presets."""
    from clinvar import PRESETS, metadata
    from clinvar.setup import parse_vcf
    from gencode.setup import build_chromosomes, build_genes

    parser = argparse.ArgumentParser(description="Download GENCODE + ClinVar and build all presets")
    parser.add_argument("--refresh", action="store_true", help="Force redownload and rebuild")
    args = parser.parse_args()

    # GENCODE
    base = "ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49"
    gtf = DOWNLOADS / "gencode.v49.basic.annotation.gtf"
    fasta = DOWNLOADS / "GRCh38.primary_assembly.genome.fa"
    _cached_download(f"{base}/gencode.v49.basic.annotation.gtf.gz", gtf, compressed=True, refresh=args.refresh)
    _cached_download(f"{base}/GRCh38.primary_assembly.genome.fa.gz", fasta, compressed=True, refresh=args.refresh)
    _cached_build(GENCODE_GENES, lambda: build_genes(gtf, fasta, GENCODE_GENES), "genes", args.refresh)
    _cached_build(GENCODE_CHROMOSOMES, lambda: build_chromosomes(fasta, GENCODE_CHROMOSOMES), "chromosomes", args.refresh)

    # ClinVar
    vcf_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"
    vcf = DOWNLOADS / "clinvar.vcf.gz"
    _cached_download(vcf_url, vcf, refresh=args.refresh)
    _cached_download(f"{vcf_url}.tbi", vcf.with_suffix(".gz.tbi"), refresh=args.refresh)
    _cached_build(CLINVAR_VARIANTS, lambda: parse_vcf(vcf, CLINVAR_VARIANTS), "clinvar variants", args.refresh)

    # Presets
    for name in PRESETS:
        output = OUTPUTS / "clinvar" / name / "metadata.feather"
        _cached_build(output, lambda n=name: metadata(n), f"preset {name}", args.refresh)

    logger.info("Setup complete")
