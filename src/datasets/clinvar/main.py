"""Variant sampling from ClinVar for pathogenicity evaluation."""

from __future__ import annotations

import polars as pl

from .. import gencode, variant
from ..paths import CLINVAR_VARIANTS
from ..utils import finalize_variants
from .sampling import (
    COMPACT_GENES,
    PILOT_GENES,
    _build_comprehensive,
    _build_deconfounded,
    _build_label_diverse,
    _per_gene_limit,
    _sample_balanced,
    _sample_stratified,
)

_COLUMNS = (
    "chrom", "pos", "ref", "alt", "variant_type", "variant_id",
    "gene_name", "label", "clinical_significance", "stars",
    "consequence", "allele_id", "disease_name", "rs_id",
)


def _finalize(df: pl.DataFrame) -> pl.DataFrame:
    """Add variant_id and enforce column order."""
    return finalize_variants(df, _COLUMNS)


def _snvs(df: pl.DataFrame, min_stars: int = 1) -> pl.DataFrame:
    """Filter to SNVs with minimum star rating."""
    return df.filter((pl.col("stars") >= min_stars) & (pl.col("variant_type") == "snv"))


def _small_gene_snvs(df: pl.DataFrame, min_stars: int = 1, max_length: int = 100_000) -> pl.DataFrame:
    """Filter to SNVs in genes <= max_length with minimum star rating."""
    gene_lengths = gencode.metadata(sequences=False).select("gene_name", "length").unique(subset=["gene_name"])
    return (
        _snvs(df, min_stars)
        .join(gene_lengths, on="gene_name", how="inner")
        .filter(pl.col("length") <= max_length)
        .drop("length")
    )


_BINARY_LABELS = {
    "pathogenic": "pathogenic",
    "likely_pathogenic": "pathogenic",
    "benign": "benign",
    "likely_benign": "benign",
}


def _source() -> pl.DataFrame:
    """Load ALL variants (7 label values)."""
    return pl.read_ipc(CLINVAR_VARIANTS)


def _labeled_source() -> pl.DataFrame:
    """Pathogenic/benign variants for binary classification."""
    return _source().filter(pl.col("label").is_in(_BINARY_LABELS)).with_columns(
        pl.col("label").replace_strict(_BINARY_LABELS)
    )


# Consequence -> integer encoding (deterministic order by frequency in deconfounded-full)
CONSEQUENCE_CLASSES = (
    "missense_variant", "intron_variant", "synonymous_variant", "nonsense",
    "frameshift_variant", "non-coding_transcript_variant", "splice_donor_variant",
    "splice_acceptor_variant", "5_prime_UTR_variant", "3_prime_UTR_variant",
    "splice_region_variant", "start_lost", "inframe_deletion", "inframe_insertion",
    "inframe_indel", "stop_lost", "genic_downstream_transcript_variant",
    "genic_upstream_transcript_variant", "no_sequence_alteration",
    "initiator_codon_variant",
)


PRESETS = {
    # All variants (all labels including VUS, conflicting, other)
    "all":             lambda: _finalize(_source()),
    # All B/LB/LP/P variants (SNVs + indels, stars >= 1), binarized to pathogenic/benign
    "labeled":         lambda: _finalize(_labeled_source().filter(pl.col("stars") >= 1)),
    # Complement of labeled: VUS + conflicting + other + stars=0 (unlabeled for inference)
    "unlabeled":       lambda: _finalize(_source().filter(
        ~pl.col("label").is_in(["pathogenic", "likely_pathogenic", "benign", "likely_benign"])
        | (pl.col("stars") < 1)
    )),
    # Legacy: SNV-only labeled (frozen for reproducibility of prior experiments)
    "labeled-snv":     lambda: _finalize(_labeled_source().filter((pl.col("variant_type") == "snv") & (pl.col("stars") >= 1))),
    # Binary classification presets (pathogenic + benign only)
    "confident":       lambda: _finalize(_labeled_source().filter(pl.col("stars") >= 3)),
    "natural":         lambda: _finalize(_sample_balanced(_per_gene_limit(_labeled_source(), 50), 50_000)),
    "broad":           lambda: _finalize(_sample_stratified(_per_gene_limit(_labeled_source(), 50), 100_000)),
    "comprehensive":   lambda: _finalize(_build_comprehensive(_labeled_source().filter(pl.col("stars") >= 1))),
    # SNVs only, genes <= 100kb, CADD-balanced (~50k variants, fast ablations)
    "deconfounded":    lambda: _finalize(_build_deconfounded(_small_gene_snvs(_labeled_source()))),
    # All variant types (SNVs + indels), stars >= 1, CADD-balanced (~184k, production)
    "deconfounded-full": lambda: _finalize(_build_deconfounded(_labeled_source().filter(pl.col("stars") >= 1))),
    "pilot": lambda: _finalize(_labeled_source().filter(
        pl.col("gene_name").is_in(PILOT_GENES) & (pl.col("stars") >= 2)
    )),
    "compact": lambda: _finalize(_labeled_source().filter(
        pl.col("gene_name").is_in(COMPACT_GENES) & (pl.col("stars") >= 2)
    )),
    # Indel presets (binary classification)
    "indels": lambda: _finalize(_labeled_source().filter(
        (pl.col("variant_type") != "snv") & (pl.col("stars") >= 2)
    )),
    "with-indels":     lambda: _finalize(_labeled_source().filter(pl.col("stars") >= 2)),
    "labeled-indels":  lambda: _finalize(_labeled_source().filter(
        (pl.col("variant_type") != "snv") & (pl.col("stars") >= 1)
    )),
    # Tiny deconfounded for fast ablations: SNVs, genes <= 100kb, ~8k variants
    "deconfounded-small": lambda: _finalize(_build_deconfounded(_small_gene_snvs(_labeled_source()), n=8_000)),
    # Single-label presets (gene-diverse round-robin, SNVs, stars >= 1)
    "vus":             lambda: _finalize(_build_label_diverse(_small_gene_snvs(_source().filter(pl.col("label") == "vus")))),
    "conflicting":     lambda: _finalize(_build_label_diverse(_small_gene_snvs(
        _source().filter(pl.col("label") == "conflicting"), min_stars=0,
    ))),
    # All variants for a specific gene (regardless of label)
    "ldlr":            lambda: _finalize(_source().filter(pl.col("gene_name") == "LDLR")),
    # v1 presets: frozen snapshots for reproducibility of published experiments.
    "confident-v1":     None,
    "broad-v1":         None,
    "comprehensive-v1": None,
}


def metadata(preset: str = "all") -> pl.DataFrame:
    """Load enriched ClinVar variant metadata.

    The cached artifact contains 14 source-level columns. Gene coordinates
    (gene_id, gene_start, gene_end, gene_strand, gene_length, gene_offset)
    are added at read time via GENCODE join. All positions are 0-based.

    Args:
        preset: See PRESETS dict for all options.

    Returns:
        DataFrame with variant metadata + gene coordinates (20 columns total).

    Example:
        >>> variants = clinvar.metadata("all")
        >>> variants = clinvar.metadata("deconfounded-full")
    """
    return variant.metadata("clinvar", PRESETS, preset)
