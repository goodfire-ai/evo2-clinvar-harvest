"""Evaluation utilities for poolers and probes.

Gene-level splitting, per-task metric computation, bootstrap confidence
intervals, stratified evaluation by variant consequence, and probe → SAE
feature attribution.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import polars as pl
import torch
from loguru import logger
from torch import Tensor, nn
from tqdm import tqdm

from .utils.metrics import aupr, auroc, f_max, pearson_corr, r2_score

# Legacy bidir window dataset names (4-dataset format).
WINDOW_DATASETS = ("variant_fwd_window", "variant_bwd_window", "ref_fwd_window", "ref_bwd_window")

# ── VEP consequence constants ────────────────────────────────────────────

#: VEP consequence terms indicating a coding variant.
#: Includes both VEP standard ("stop_gained") and ClinVar legacy ("nonsense").
CODING_CONSEQUENCES: frozenset[str] = frozenset({
    "frameshift_variant",
    "inframe_deletion",
    "inframe_insertion",
    "stop_gained",
    "nonsense",
    "stop_lost",
    "start_lost",
    "missense_variant",
    "synonymous_variant",
    "coding_sequence_variant",
    "protein_altering_variant",
    "incomplete_terminal_codon_variant",
})

#: VEP consequence terms indicating splice-site proximity.
SPLICE_CONSEQUENCES: frozenset[str] = frozenset({
    "splice_donor_variant",
    "splice_acceptor_variant",
    "splice_region_variant",
    "splice_donor_region_variant",
    "splice_polypyrimidine_tract_variant",
    "splice_donor_5th_base_variant",
})


def gene_split(
    metadata: pl.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    gene_col: str = "gene_name",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Split metadata by genes — no gene appears in both sets.

    Gene-level splitting prevents data leakage from shared gene effects.

    Deterministic: same (metadata, test_size, seed) → same split.
    To persist, save the test gene list:
        test_df[gene_col].unique().sort().write_ipc(path / "test_genes.feather")

    Args:
        metadata: DataFrame with a gene column.
        test_size: Fraction of unique genes held out for test.
        seed: Random seed for reproducibility.
        gene_col: Column name containing gene identifiers.

    Returns:
        (train_df, test_df) with no gene overlap.

    Example:
        >>> train_df, test_df = gene_split(master, test_size=0.2)
        >>> assert set(train_df["gene_name"]) & set(test_df["gene_name"]) == set()

    """
    genes = metadata.select(gene_col).unique().sort(gene_col)
    n_test = int(len(genes) * test_size)
    shuffled = genes.sample(fraction=1.0, seed=seed, shuffle=True)
    test_genes = set(shuffled[gene_col][:n_test].to_list())

    train_df = metadata.filter(~pl.col(gene_col).is_in(test_genes))
    test_df = metadata.filter(pl.col(gene_col).is_in(test_genes))
    return train_df, test_df


def gene_kfold(
    metadata: pl.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    gene_col: str = "gene_name",
) -> pl.DataFrame:
    """Assign each row a fold (0..n_folds-1) grouped by gene.

    Same shuffle logic as gene_split: unique genes → deterministic shuffle
    → round-robin fold assignment → join back.

    Args:
        metadata: DataFrame with a gene column.
        n_folds: Number of folds.
        seed: Random seed for reproducibility.
        gene_col: Column name containing gene identifiers.

    Returns:
        metadata with an added ``fold`` column (Int32).

    Example:
        >>> folded = gene_kfold(manifest, n_folds=5)
        >>> folded.group_by("fold").len()
    """
    genes = metadata.select(gene_col).unique().sort(gene_col)
    shuffled = genes.sample(fraction=1.0, seed=seed, shuffle=True)
    gene_folds = shuffled.with_columns(
        pl.Series("fold", [i % n_folds for i in range(len(shuffled))], dtype=pl.Int32),
    )
    return metadata.join(gene_folds, on=gene_col)


def compute_per_task_classification_metrics(
    probs: Tensor,
    labels: Tensor,
    task_names: tuple[str, ...],
) -> pl.DataFrame:
    """Compute per-task classification metrics.

    Args:
        probs: Prediction probabilities, shape (N, L)
        labels: Binary labels, shape (N, L)
        task_names: Names for each task/label column

    Returns:
        DataFrame with per-task metrics (auc, aupr, f_max, n_positive, n_samples)

    Raises:
        ValueError: If len(task_names) != number of label columns

    """
    if len(task_names) != labels.shape[-1]:
        raise ValueError(f"task_names length ({len(task_names)}) != labels columns ({labels.shape[-1]})")

    label_sums = labels.sum(0)
    valid = (label_sums > 0) & (label_sums < len(labels))

    # Filter to valid tasks
    valid_probs = probs[:, valid]
    valid_labels = labels[:, valid]
    valid_names = [task_names[i] for i in range(len(task_names)) if valid[i]]

    # Vectorized per-task metrics (single GPU sync)
    metrics = torch.stack([
        auroc(valid_probs, valid_labels, reduction="none"),
        aupr(valid_probs, valid_labels, reduction="none"),
        f_max(valid_probs, valid_labels, reduction="none"),
        valid_labels.sum(0).float(),
    ]).cpu()

    return pl.DataFrame({
        "task": valid_names,
        "auc": metrics[0],
        "aupr": metrics[1],
        "f_max": metrics[2],
        "n_positive": metrics[3].int(),
        "n_samples": [len(labels)] * len(valid_names),
    })


def bootstrap_ci(
    scores: Tensor,
    labels: Tensor,
    metric_fn: Callable[[Tensor, Tensor], Tensor],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a binary classification metric.

    Vectorized: generates all bootstrap indices at once and evaluates
    the metric in a single batched call. Degenerate samples (all-same labels)
    are filtered automatically (NaN from metric function).

    Args:
        scores: Predicted scores, shape (N,).
        labels: Binary labels, shape (N,).
        metric_fn: ``(scores, labels) -> Tensor``. Must support batch dims
                   ``(..., N, 1)``. Compatible with ``auroc`` and ``aupr``.
        n_bootstrap: Number of bootstrap resamples.
        alpha: Two-sided significance level (0.05 → 95% CI).
        seed: RNG seed for reproducibility.

    Returns:
        ``(lower, upper)`` confidence interval bounds.

    Example:
        >>> from src.utils.metrics import auroc
        >>> lo, hi = bootstrap_ci(scores, labels, auroc)
    """
    generator = torch.Generator(device=scores.device).manual_seed(seed)
    n = labels.shape[0]

    # All bootstrap indices at once: [B, N]
    indices = torch.randint(0, n, (n_bootstrap, n), generator=generator, device=scores.device)

    # Gather and add label dim: [B, N, 1]
    boot_scores = scores[indices].unsqueeze(-1)
    boot_labels = labels[indices].unsqueeze(-1)

    # Vectorized metric across bootstrap dimension: [B, N, 1] → [B]
    values = metric_fn(boot_scores, boot_labels)
    if values.dim() > 1:
        values = values.squeeze(-1)

    # Drop degenerate samples (NaN from constant labels)
    values = values[~torch.isnan(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"))

    lo = values.quantile(alpha / 2).item()
    hi = values.quantile(1 - alpha / 2).item()
    return (lo, hi)


# ── Stratified evaluation ────────────────────────────────────────────────


def annotate_strata(
    metadata: pl.DataFrame,
    consequence_col: str = "consequence",
    ref_col: str = "ref",
    alt_col: str = "alt",
) -> pl.DataFrame:
    """Add stratification columns to variant metadata.

    Added columns:
        is_frameshift : bool — consequence contains "frameshift"
        is_coding     : bool — consequence is in CODING_CONSEQUENCES
        is_splice     : bool — consequence is in SPLICE_CONSEQUENCES

    If *ref_col* and *alt_col* are present, also adds:
        indel_len     : int  — signed length change (positive = insertion)
        indel_size    : int  — absolute length change
        size_bin      : str  — "1bp" / "2-5bp" / "6-20bp" / ">20bp"

    Args:
        metadata: Variant metadata with at least a consequence column.
        consequence_col: Column containing VEP consequence terms.
        ref_col: Column containing reference allele.
        alt_col: Column containing alternate allele.

    Returns:
        DataFrame with strata columns added.
    """
    result = metadata.with_columns(
        pl.col(consequence_col).str.contains("(?i)frameshift").alias("is_frameshift"),
        pl.col(consequence_col).is_in(CODING_CONSEQUENCES).alias("is_coding"),
        pl.col(consequence_col).is_in(SPLICE_CONSEQUENCES).alias("is_splice"),
    )

    if ref_col in metadata.columns and alt_col in metadata.columns:
        result = result.with_columns(
            (pl.col(alt_col).str.len_chars().cast(pl.Int32) - pl.col(ref_col).str.len_chars().cast(pl.Int32)).alias("indel_len"),
        ).with_columns(
            pl.col("indel_len").abs().alias("indel_size"),
        ).with_columns(
            pl.when(pl.col("indel_size") <= 1).then(pl.lit("1bp"))
            .when(pl.col("indel_size") <= 5).then(pl.lit("2-5bp"))
            .when(pl.col("indel_size") <= 20).then(pl.lit("6-20bp"))
            .otherwise(pl.lit(">20bp"))
            .alias("size_bin"),
        )

    return result


#: Strata definitions as polars expressions. Requires ``annotate_strata()`` first.
STRATA_DEFS: dict[str, pl.Expr] = {
    "overall": pl.lit(True),
    "frameshift": pl.col("is_frameshift"),
    "inframe_coding": ~pl.col("is_frameshift") & pl.col("is_coding"),
    "noncoding": ~pl.col("is_coding"),
    "splice": pl.col("is_splice"),
    "non_splice": ~pl.col("is_splice"),
}

#: Extended strata for indel-specific analysis. Requires indel columns from ``annotate_strata()``.
INDEL_STRATA_DEFS: dict[str, pl.Expr] = {
    **STRATA_DEFS,
    "insertion": pl.col("indel_len") > 0,
    "deletion": pl.col("indel_len") < 0,
    "size_1bp": pl.col("size_bin") == "1bp",
    "size_2_5bp": pl.col("size_bin") == "2-5bp",
    "size_6_20bp": pl.col("size_bin") == "6-20bp",
    "size_gt20bp": pl.col("size_bin") == ">20bp",
}


def compute_stratified_metrics(
    scores: Tensor,
    labels: Tensor,
    metadata: pl.DataFrame,
    strata: dict[str, pl.Expr] | None = None,
    bootstrap: bool = True,
    n_bootstrap: int = 1000,
    seed: int = 42,
    min_positive: int = 5,
    min_negative: int = 5,
) -> pl.DataFrame:
    """Compute AUROC and AUPRC for each stratum of variants.

    Evaluates pre-computed scores within each stratum subset (no retraining).

    Args:
        scores: Predicted scores (N,), higher = more pathogenic.
        labels: Binary labels (N,), 1 = pathogenic.
        metadata: Variant metadata aligned with scores/labels, with strata
                  columns from ``annotate_strata()``.
        strata: Dict of ``{name: polars_expr}``. Defaults to ``STRATA_DEFS``.
        bootstrap: If True, add 95% bootstrap confidence intervals.
        n_bootstrap: Number of bootstrap resamples.
        seed: RNG seed.
        min_positive: Skip stratum if fewer positives.
        min_negative: Skip stratum if fewer negatives.

    Returns:
        DataFrame with columns: stratum, n_variants, n_pathogenic, n_benign,
        auroc, auprc, and optionally auroc_ci_lo/hi, auprc_ci_lo/hi.

    Example:
        >>> metadata = annotate_strata(clinvar.metadata("labeled"))
        >>> results = compute_stratified_metrics(scores, labels, metadata)
        >>> results.filter(pl.col("stratum") == "frameshift")
    """
    if strata is None:
        strata = STRATA_DEFS

    rows: list[dict] = []
    for name, expr in strata.items():
        mask = metadata.with_columns(expr.alias("_m"))["_m"]
        mask_t = torch.from_numpy(mask.to_numpy())

        s = scores[mask_t]
        y = labels[mask_t]

        n_pos = int(y.sum().item())
        n_neg = len(y) - n_pos

        if n_pos < min_positive or n_neg < min_negative:
            continue

        s_2d = s.unsqueeze(-1)
        y_2d = y.unsqueeze(-1)

        row: dict = {
            "stratum": name,
            "n_variants": len(y),
            "n_pathogenic": n_pos,
            "n_benign": n_neg,
            "auroc": auroc(s_2d, y_2d).item(),
            "auprc": aupr(s_2d, y_2d).item(),
        }

        if bootstrap:
            auroc_lo, auroc_hi = bootstrap_ci(s, y, auroc, n_bootstrap=n_bootstrap, seed=seed)
            auprc_lo, auprc_hi = bootstrap_ci(s, y, aupr, n_bootstrap=n_bootstrap, seed=seed)
            row["auroc_ci_lo"] = auroc_lo
            row["auroc_ci_hi"] = auroc_hi
            row["auprc_ci_lo"] = auprc_lo
            row["auprc_ci_hi"] = auprc_hi

        rows.append(row)

    return pl.DataFrame(rows)


def compute_per_task_regression_metrics(
    preds: Tensor,
    labels: Tensor,
    task_names: tuple[str, ...],
) -> pl.DataFrame:
    """Compute per-task regression metrics.

    Args:
        preds: Predictions, shape (N, L)
        labels: True values, shape (N, L)
        task_names: Names for each task column

    Returns:
        DataFrame with per-task metrics (r2, pearson, mse, std_true, std_pred)

    Raises:
        ValueError: If len(task_names) != number of label columns

    """
    if len(task_names) != labels.shape[-1]:
        raise ValueError(f"task_names length ({len(task_names)}) != labels columns ({labels.shape[-1]})")

    # Per-column metrics with NaN masking (each column may have different NaN pattern)
    rows = []
    for i in range(labels.shape[-1]):
        mask = ~torch.isnan(labels[:, i])
        p, t = preds[mask, i], labels[mask, i]
        rows.append(torch.stack([
            r2_score(p, t),
            pearson_corr(p, t),
            (p - t).pow(2).mean(),
            t.std(),
            p.std(),
        ]))
    metrics = torch.stack(rows).cpu()

    return pl.DataFrame({
        "task": task_names,
        "r2": metrics[:, 0],
        "pearson": metrics[:, 1],
        "mse": metrics[:, 2],
        "std_true": metrics[:, 3],
        "std_pred": metrics[:, 4],
    })


# ── Cross-validation ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProbeResult:
    """Out-of-fold predictions from K-fold gene-holdout cross-validation.

    Every sample gets exactly one out-of-fold score, enabling stratified
    analysis by consequence type, gene family, etc.
    """

    scores: Tensor  # shape (n,), out-of-fold P(pathogenic)
    labels: Tensor  # shape (n,), ground truth
    folds: Tensor   # shape (n,), fold assignment per sample
    ids: tuple[str, ...]
    auroc: float
    per_fold_auroc: tuple[float, ...]


def _to_scores(logits: Tensor) -> Tensor:
    """Convert logits to P(positive) scores. Handles scalar and 2-class logits."""
    if logits.dim() == 1:
        return logits.sigmoid()
    return torch.softmax(logits, dim=-1)[:, 1]


def evaluate_folds(
    probes: list[nn.Module],
    stream: Callable[[set[str]], Iterator[tuple[Tensor, list[str]]]],
    metadata: pl.DataFrame,
    *,
    n_folds: int = 5,
    gene_col: str = "gene_name",
    label_col: str = "pathogenic",
    id_col: str = "variant_id",
    seed: int = 42,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> ProbeResult:
    """Score every sample with its held-out fold probe.

    Streams all data once, routing each sample to the probe that was
    NOT trained on that sample's fold.

    Args:
        probes: One trained probe per fold.
        stream: Callable taking target_ids → Iterator[(Tensor, list[str])].
            Build from ``iter_dataset`` or ``iter_datasets`` + a transform.
        metadata: Variant metadata with gene and label columns.
        n_folds: Number of CV folds.
        gene_col: Column containing gene identifiers.
        label_col: Column containing binary labels.
        id_col: Column containing variant IDs.
        seed: Random seed (must match training).

    Returns:
        ProbeResult with out-of-fold scores, labels, and fold AUROC.
    """
    assert len(probes) == n_folds, f"Expected {n_folds} probes, got {len(probes)}"

    device_obj = torch.device(device)
    folded = gene_kfold(metadata, n_folds=n_folds, seed=seed, gene_col=gene_col)

    id_list = folded[id_col].to_list()
    fold_list = folded["fold"].to_list()
    label_list = folded[label_col].to_list()
    id_to_fold = dict(zip(id_list, fold_list, strict=True))
    id_to_label = dict(zip(id_list, label_list, strict=True))

    for p in probes:
        p.to(dtype).to(device_obj).eval()

    all_ids: list[str] = []
    all_scores: list[Tensor] = []
    all_labels: list[int] = []
    all_folds: list[int] = []

    logger.info("Evaluating out-of-fold predictions...")
    with torch.no_grad():
        for acts, ids in tqdm(stream(set(id_list)), desc="eval"):
            batch_folds = torch.tensor([id_to_fold[sid] for sid in ids])
            for fold_k in batch_folds.unique().tolist():
                mask = batch_folds == fold_k
                scores_batch = _to_scores(probes[fold_k](acts[mask])).cpu()
                sids = [sid for sid, m in zip(ids, mask.tolist(), strict=True) if m]

                all_scores.append(scores_batch)
                all_labels.extend(id_to_label[sid] for sid in sids)
                all_folds.extend([fold_k] * len(sids))
                all_ids.extend(sids)

    scores = torch.cat(all_scores)
    labels_t = torch.tensor(all_labels, dtype=torch.float32)
    folds_t = torch.tensor(all_folds, dtype=torch.int64)

    overall_auroc = auroc(scores.unsqueeze(1), labels_t.unsqueeze(1)).item()
    per_fold = tuple(
        auroc(scores[folds_t == k].unsqueeze(1), labels_t[folds_t == k].unsqueeze(1)).item()
        for k in range(n_folds)
    )
    for k, auc in enumerate(per_fold):
        logger.info(f"Fold {k} AUROC: {auc:.4f} (n={(folds_t == k).sum().item():,})")
    logger.info(f"Overall AUROC: {overall_auroc:.4f}")

    return ProbeResult(
        scores=scores,
        labels=labels_t.to(torch.int64),
        folds=folds_t,
        ids=tuple(all_ids),
        auroc=overall_auroc,
        per_fold_auroc=per_fold,
    )
