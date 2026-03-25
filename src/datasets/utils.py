"""Dataset utilities: biological constants, sequence manipulation, and data helpers."""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
from loguru import logger

from src.datasets.paths import OUTPUTS
from src.utils import ensure_dir, ensure_parent, run_setup

__all__ = (
    "CHROMOSOMES",
    "STAR_MAPPING",
    "enrich_variants",
    "ensure_dir",
    "ensure_parent",
    "finalize_variants",
    "mutate_single",
    "mutate_variants",
    "reverse_complement",
    "run_setup",
    "strand_aware_alt",
    "strand_aware_seq_pos",
    "strip_version",
    "with_cache",
)

# --- Biological constants ---

_DNA_COMPLEMENT = str.maketrans("ATGCatgcNn", "TACGtacgNn")

CHROMOSOMES = (*tuple(str(i) for i in range(1, 23)), "X", "Y", "M")

STAR_MAPPING = {
    "practice_guideline": 4,
    "reviewed_by_expert_panel": 3,
    "criteria_provided,_multiple_submitters,_no_conflicts": 2,
    "criteria_provided,_conflicting_interpretations": 2,
    "criteria_provided,_single_submitter": 1,
}

# --- Sequence manipulation ---

def reverse_complement(seq: str) -> str:
    """Reverse complement a DNA sequence."""
    return seq.translate(_DNA_COMPLEMENT)[::-1]


def mutate_single() -> pl.Expr:
    """Polars expression: splice alt into sequence at seq_pos, replacing ref.

    Supports SNVs and indels. Requires columns: sequence, seq_pos, ref, alt.
    """
    return (
        pl.col("sequence").str.slice(0, pl.col("seq_pos"))
        + pl.col("alt")
        + pl.col("sequence").str.slice(pl.col("seq_pos") + pl.col("ref").str.len_bytes())
    ).alias("sequence")


# --- Coordinate helpers ---

def strand_aware_alt() -> pl.Expr:
    """Polars expression: reverse complement alt for minus strand."""
    return (
        pl.when(pl.col("strand") == "-")
        .then(pl.col("alt").map_elements(reverse_complement, return_dtype=pl.String))
        .otherwise(pl.col("alt"))
    )


def strand_aware_seq_pos() -> pl.Expr:
    """0-based genomic pos -> 0-based sequence index, strand-aware.

    Requires columns: pos (0-based), start (0-based), end (exclusive), strand.
    """
    return (
        pl.when(pl.col("strand") == "+")
        .then(pl.col("pos") - pl.col("start"))
        .otherwise(pl.col("end") - pl.col("pos") - 1)
        .cast(pl.Int64)
    )


def strip_version(gene_id: str) -> str:
    """Strip Ensembl version suffix: ENSG00000139618.19 -> ENSG00000139618."""
    return gene_id.split(".")[0] if gene_id.startswith("ENS") and "." in gene_id else gene_id


# --- Variant utilities ---

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
    """Add gene coordinates and strand-aware offset.

    Joins with genes DataFrame and computes the variant's position within the gene
    (0-based, 5'->3'). Adds columns: gene_id, gene_start, gene_end, gene_strand,
    gene_length, gene_offset.
    """
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


def mutate_variants(
    variants: pl.DataFrame,
    genes: pl.DataFrame,
    unique_on: tuple[str, ...] = ("variant_id",),
) -> pl.DataFrame:
    """Apply strand-aware mutations to gene sequences."""
    n_before = len(variants)
    if "gene_id" in variants.columns:
        variants = variants.drop("gene_id")
    gene_cols = ("gene_name", "gene_id", "start", "end", "strand", "sequence")

    result = (
        variants
        .with_row_index("_order")
        .join(genes.select(gene_cols), on="gene_name", how="inner")
        .with_columns(seq_pos=strand_aware_seq_pos())
        .with_columns(alt_orig=pl.col("alt"), alt=strand_aware_alt())
        .with_columns(mutate_single())
        .with_columns(alt=pl.col("alt_orig"))
        .drop("alt_orig", "start", "end", "strand", "seq_pos")
        .sort("_order", "gene_id")
        .unique(subset=unique_on, keep="first")
        .sort("_order")
        .drop("_order")
    )
    logger.info(f"Mutated {len(result):,} / {n_before:,} variants ({n_before - len(result):,} dropped)")
    return result


# --- Data utilities ---

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
