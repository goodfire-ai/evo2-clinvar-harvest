"""Sampling and subsetting strategies for ClinVar variant datasets.

Pure DataFrame → DataFrame functions. No data loading or schema enforcement —
that stays in main.py. Preset lambdas compose the pipeline.
"""

from __future__ import annotations

import polars as pl

# Consequence → category mapping for stratified sampling
_CONSEQUENCE_TO_CATEGORY = {
    **{c: "coding" for c in (
        "missense_variant", "nonsense", "splice_acceptor_variant", "splice_donor_variant",
        "frameshift_variant", "inframe_deletion", "inframe_insertion", "start_lost", "stop_lost",
    )},
    **{c: "regulatory" for c in (
        "splice_region_variant", "5_prime_UTR_variant", "3_prime_UTR_variant",
    )},
}

# Pilot: small clinically important genes for activation profiling
# 8 genes, 10kb total, 394 variants (stars>=2)
PILOT_GENES = (
    "MT-ND3",   # 346bp - Leigh syndrome
    "MT-ND6",   # 525bp - Leber hereditary optic neuropathy
    "MT-ATP6",  # 681bp - NARP/Leigh syndrome
    "HBA2",     # 835bp - Alpha-thalassemia
    "MT-ND1",   # 956bp - MELAS/LHON
    "INS",      # 1443bp - Neonatal diabetes
    "GP9",      # 1657bp - Bernard-Soulier syndrome
    "HBB",      # 3932bp - Sickle cell/Beta-thalassemia
)

# Compact: 102 well-balanced small genes for pathogenicity prediction
# ~6,500 variants (stars>=2, capped at 75/gene), ~480GB evo2 storage
# Selection criteria: <15kb length, >=10 variants per class, 15-85% pathogenic
COMPACT_GENES = (
    "ACADS", "ACADVL", "ACTA1", "AGA", "AGXT", "AIRE", "AMT", "AQP2", "ARSA",
    "BBS10", "BBS12", "BCS1L", "COMP", "CTSA", "CYP11B1", "CYP11B2", "CYP17A1",
    "CYP21A2", "CYP27B1", "DDX41", "DES", "ECHS1", "EIF2B5", "ELANE", "EMD",
    "F7", "FANCG", "FOXC1", "FOXG1", "G6PC1", "GALK1", "GALT", "GAMT", "GFAP",
    "GJB1", "GJB2", "GLA", "GMPPB", "GNPTG", "GP1BA", "GRHPR", "GRN", "HBB",
    "HMBS", "HRAS", "HSD3B2", "HSPB1", "IL2RG", "INPP5E", "INS", "KCNJ2",
    "KRT14", "KRT5", "LAMB2", "MCOLN1", "MEN1", "MFRP", "MKS1", "MMACHC",
    "MPI", "MPZ", "MUTYH", "NAGLU", "NDUFV1", "NEFL", "NTHL1", "PEX10",
    "PKLR", "PNPO", "PRF1", "PROC", "PYGM", "RAPSN", "RHO", "RIT1",
    "SERPINA1", "SERPINC1", "SGCA", "SGSH", "SKIC2", "SLC34A3", "SLC37A4",
    "SLC4A11", "SLC6A8", "SMPD1", "SOD1", "SOX11", "SOX9", "STAR", "STXBP2",
    "SURF1", "TAFAZZIN", "TCIRG1", "TH", "TNNI3", "TPP1", "TUBA1A", "TUBB4A",
    "TWNK", "TYMP", "VHL", "WAS",
)


def _sample_balanced(df: pl.DataFrame, n: int) -> pl.DataFrame:
    """Sample n variants with 50/50 pathogenic/benign balance.

    Selection is deterministic: prefers higher-quality variants (stars)
    and uses allele_id for stable tie-breaking within each label group.
    """
    return (
        df
        .sort(["stars", "allele_id"], descending=[True, False])
        .group_by("label", maintain_order=True)
        .head(n // 2)
    )


def _sample_stratified(
    df: pl.DataFrame,
    n: int,
    target_coding: float = 0.72,
    target_noncoding: float = 0.20,
    target_regulatory: float = 0.08,
) -> pl.DataFrame:
    """Sample n variants stratified by consequence category with label balance.

    Computes category from consequence (coding/regulatory/noncoding) as a temporary
    column for stratification. Default ratios (72/20/8) approximate the ClinVar
    population distribution.

    Selection is deterministic: prefers higher-quality variants (stars)
    and uses allele_id for stable tie-breaking within each (category, label) stratum.
    """
    return (
        df
        .with_columns(
            pl.col("consequence")
            .replace_strict(_CONSEQUENCE_TO_CATEGORY, default="noncoding")
            .alias("_category")
        )
        .with_columns(
            target_n=pl.when(pl.col("_category") == "coding").then(int(n * target_coding) // 2)
                      .when(pl.col("_category") == "regulatory").then(int(n * target_regulatory) // 2)
                      .otherwise(int(n * target_noncoding) // 2)
        )
        .sort(["stars", "allele_id"], descending=[True, False])
        .with_columns(pl.int_range(pl.len()).over("_category", "label").alias("_rank"))
        .filter(pl.col("_rank") < pl.col("target_n"))
        .drop("_category", "target_n", "_rank")
    )


def _per_gene_limit(df: pl.DataFrame, per_gene: int) -> pl.DataFrame:
    """Cap variants per (gene, label), preferring higher stars."""
    return (
        df
        .sort("stars", descending=True)
        .group_by(["gene_name", "label"], maintain_order=True)
        .head(per_gene)
    )


def _build_comprehensive(df: pl.DataFrame) -> pl.DataFrame:
    """Intra-gene benign matching: match benign count to pathogenic per gene."""
    pathogenic = df.filter(pl.col("label") == "pathogenic")
    benign = df.filter(pl.col("label") == "benign")

    # Count pathogenic per gene, match benign to balance
    counts = pathogenic.group_by("gene_name").agg(pl.len().alias("n"))
    benign_matched = (
        benign
        .join(counts, on="gene_name", how="inner")
        .sort(["stars", "allele_id"], descending=[True, False])
        .with_columns(pl.int_range(pl.len()).over("gene_name").alias("_rank"))
        .filter(pl.col("_rank") < pl.col("n"))
        .drop("n", "_rank")
    )

    return pl.concat([pathogenic, benign_matched])


def _build_label_diverse(df: pl.DataFrame, n: int = 50_000) -> pl.DataFrame:
    """Gene-diverse round-robin sampling.

    Variants are ranked within each gene by stars (higher first), then
    interleaved globally so one variant per gene is selected before a
    second from any gene.

    Args:
        df: Single-label SNVs.
        n: Target number of variants.
    """
    return (
        df
        .sort(["stars", "allele_id"], descending=[True, False])
        .with_columns(
            pl.int_range(pl.len())
            .over("gene_name")
            .alias("_gene_rank")
        )
        .sort(
            ["_gene_rank", "stars", "allele_id"],
            descending=[False, True, False],
        )
        .head(n)
        .drop("_gene_rank")
    )


def _build_deconfounded(
    df: pl.DataFrame,
    n: int = 200_000,
    deconfound: float = 0.5,
) -> pl.DataFrame:
    """Balanceability-weighted deconfounding with natural consequence distribution.

    Input should be eligible SNVs (stars >= 1, genes <= 100kb).

    Allocates a budget of `n` variants across (consequence, label) groups using a
    formula that jointly optimizes two objectives:

    1. **Deconfounding**: Within each consequence type, push labels toward 50/50.
       Controlled by `min(deconfound, 1)` as interpolation strength.
    2. **Natural distribution**: Preserve the raw consequence type proportions,
       but shift budget away from unbalanceable types (e.g., synonymous: 99.5% benign)
       toward balanceable ones (e.g., missense: ~50/50). Controlled by
       `balanceability^deconfound`.

    The per-(consequence, label) quota formula::

        target(c, l) = raw_frac(c) * balanceability(c)^deconfound
                       * lerp(label_frac(c, l), 0.5, min(deconfound, 1))

    where balanceability(c) = 2 * min(path_c, ben_c) / (path_c + ben_c).

    Gene diversity is maximized by round-robin: within each (consequence, label)
    group, variants are ranked so that one variant per gene is picked before a
    second from any gene.

    Args:
        df: Eligible SNVs.
        n: Target dataset size (actual size may be smaller if capped by data).
        deconfound: Deconfounding strength.
            0.0 = raw distribution (no rebalancing).
            0.5 = recommended default (77% consequence accuracy, 2% syn pathogenic).
            1.0 = strong deconfounding (budget fully shifted to balanceable types, 50/50 labels).
            >1.0 = extreme (budget hyper-concentrated on balanced types).
    """
    strength = min(deconfound, 1.0)

    # Per-(consequence, label) counts — all statistics derive from this
    counts = df.group_by("consequence", "label").agg(pl.len().alias("obs_n"))

    # Per-consequence stats: raw fraction + balanceability
    # Each consequence has exactly 2 rows (pathogenic + benign) in counts
    cons_stats = (
        counts
        .group_by("consequence")
        .agg(
            raw_frac=pl.col("obs_n").sum() / len(df),
            _bal=2 * pl.col("obs_n").min() / pl.col("obs_n").sum(),
        )
    )

    # Compute quotas: target ∝ raw_frac * bal^dc * lerp(label_frac, 0.5, strength)
    quotas = (
        counts
        .with_columns(label_frac=pl.col("obs_n") / pl.col("obs_n").sum().over("consequence"))
        .join(cons_stats, on="consequence")
        .with_columns(
            _target=(
                pl.col("raw_frac")
                * pl.col("_bal").pow(deconfound)
                * (pl.col("label_frac") * (1 - strength) + 0.5 * strength)
            ),
        )
        .with_columns(
            quota=pl.min_horizontal(
                (pl.col("_target") / pl.col("_target").sum() * n).cast(pl.Int64),
                pl.col("obs_n"),
            ),
        )
        .select("consequence", "label", "quota")
    )

    # Gene round-robin ranking, then apply quotas
    return (
        df
        .sort(["stars", "allele_id"], descending=[True, False])
        .with_columns(
            pl.int_range(pl.len())
            .over("consequence", "label", "gene_name")
            .alias("_gene_rank")
        )
        .sort(
            ["consequence", "label", "_gene_rank", "stars", "allele_id"],
            descending=[False, False, False, True, False],
        )
        .with_columns(
            pl.int_range(pl.len())
            .over("consequence", "label")
            .alias("_rank")
        )
        .join(quotas, on=["consequence", "label"])
        .filter(pl.col("_rank") < pl.col("quota"))
        .drop("_gene_rank", "_rank", "quota")
    )
