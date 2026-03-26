"""Shared utilities: paths, downloads, transforms, variant helpers, and setup."""

from __future__ import annotations

import argparse
import gzip
import shutil
import urllib.request
from collections.abc import Callable
from pathlib import Path

import polars as pl
import torch
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


def reverse_complement(seq: str) -> str:
    """Reverse complement a DNA sequence."""
    return seq.translate(_DNA_COMPLEMENT)[::-1]


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


# ── Activation transform ─────────────────────────────────────────────────


def unified_diff(x: Tensor) -> Tensor:
    """[B, 2, 3, K, d] -> [B, K, 2*d]: var-ref diff, concat fwd+bwd."""
    diff = x[:, :, 0] - x[:, :, 1]
    return torch.cat([diff[:, 0], diff[:, 1]], dim=-1)


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
    base = "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_49"
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
