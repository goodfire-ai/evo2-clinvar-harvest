"""CADD-deconfounded sampling for ClinVar variant datasets."""

from __future__ import annotations

import polars as pl


def _build_deconfounded(
    df: pl.DataFrame,
    n: int = 200_000,
    deconfound: float = 0.5,
) -> pl.DataFrame:
    """Balanceability-weighted deconfounding with natural consequence distribution.

    Allocates a budget across (consequence, label) groups, pushing labels toward
    50/50 while preserving consequence proportions and maximizing gene diversity.

    Args:
        df: Eligible SNVs (stars >= 1, labeled pathogenic/benign).
        n: Target dataset size.
        deconfound: Deconfounding strength (0=raw, 0.5=recommended, 1=strong).
    """
    strength = min(deconfound, 1.0)

    counts = df.group_by("consequence", "label").agg(pl.len().alias("obs_n"))
    cons_stats = counts.group_by("consequence").agg(
        raw_frac=pl.col("obs_n").sum() / len(df),
        _bal=2 * pl.col("obs_n").min() / pl.col("obs_n").sum(),
    )
    quotas = (
        counts
        .with_columns(label_frac=pl.col("obs_n") / pl.col("obs_n").sum().over("consequence"))
        .join(cons_stats, on="consequence")
        .with_columns(_target=(
            pl.col("raw_frac") * pl.col("_bal").pow(deconfound)
            * (pl.col("label_frac") * (1 - strength) + 0.5 * strength)
        ))
        .with_columns(quota=pl.min_horizontal(
            (pl.col("_target") / pl.col("_target").sum() * n).cast(pl.Int64), pl.col("obs_n"),
        ))
        .select("consequence", "label", "quota")
    )

    return (
        df
        .sort(["stars", "allele_id"], descending=[True, False])
        .with_columns(pl.int_range(pl.len()).over("consequence", "label", "gene_name").alias("_gene_rank"))
        .sort(["consequence", "label", "_gene_rank", "stars", "allele_id"], descending=[False, False, False, True, False])
        .with_columns(pl.int_range(pl.len()).over("consequence", "label").alias("_rank"))
        .join(quotas, on=["consequence", "label"])
        .filter(pl.col("_rank") < pl.col("quota"))
        .drop("_gene_rank", "_rank", "quota")
    )
