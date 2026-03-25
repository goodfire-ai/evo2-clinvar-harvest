"""ClinVar variant presets and metadata loader."""

from __future__ import annotations

import polars as pl

import gencode
from utils import CLINVAR_VARIANTS, enrich_variants, finalize_variants, with_cache

from .sampling import _build_deconfounded

_COLUMNS = (
    "chrom", "pos", "ref", "alt", "variant_type", "variant_id",
    "gene_name", "label", "clinical_significance", "stars",
    "consequence", "allele_id", "disease_name", "rs_id",
)

_BINARY_LABELS = {
    "pathogenic": "pathogenic",
    "likely_pathogenic": "pathogenic",
    "benign": "benign",
    "likely_benign": "benign",
}


def _finalize(df: pl.DataFrame) -> pl.DataFrame:
    return finalize_variants(df, _COLUMNS)


def _source() -> pl.DataFrame:
    return pl.read_ipc(CLINVAR_VARIANTS)


def _labeled_source() -> pl.DataFrame:
    return _source().filter(pl.col("label").is_in(_BINARY_LABELS)).with_columns(
        pl.col("label").replace_strict(_BINARY_LABELS)
    )


def _small_gene_snvs(df: pl.DataFrame) -> pl.DataFrame:
    """SNVs in genes <= 100kb with stars >= 1."""
    gene_lengths = gencode.metadata(sequences=False).select("gene_name", "length").unique(subset=["gene_name"])
    return (
        df.filter((pl.col("stars") >= 1) & (pl.col("variant_type") == "snv"))
        .join(gene_lengths, on="gene_name", how="inner")
        .filter(pl.col("length") <= 100_000)
        .drop("length")
    )


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
    "pilot":     lambda: _finalize(_build_deconfounded(_small_gene_snvs(_labeled_source()), n=8_000)),
    "labeled":   lambda: _finalize(_labeled_source().filter(pl.col("stars") >= 1)),
    "unlabeled": lambda: _finalize(_source().filter(
        ~pl.col("label").is_in(["pathogenic", "likely_pathogenic", "benign", "likely_benign"])
        | (pl.col("stars") < 1)
    )),
}


def metadata(preset: str = "labeled") -> pl.DataFrame:
    """Load enriched ClinVar variant metadata.

    Args:
        preset: One of "pilot", "labeled", "unlabeled".

    Returns:
        DataFrame with variant metadata + gene coordinates (20 columns total).
    """
    assert preset in PRESETS, f"Unknown preset '{preset}'. Available: {sorted(PRESETS)}"
    df = with_cache("clinvar", preset, PRESETS[preset])
    return enrich_variants(df, gencode.metadata(sequences=False))
