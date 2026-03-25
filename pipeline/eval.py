"""Evaluate probe scores: AUROC, AUPRC, Spearman, stratified by consequence.

Reads scores.feather + metadata, computes metrics. No GPU needed.

ClinVar mode (auto-discovers scores, split, and output from --probe):
    python scripts/eval.py --probe /path/to/storage/cov64 --preset labeled

DMS mode (explicit labels):
    python scripts/eval.py \\
        --scores path/to/dms/scores.feather \\
        --labels-csv path/to/variant_manifest.csv \\
        --label-col pathogenic \\
        --continuous-col score \\
        --output path/to/dms_metrics.csv
"""

import argparse
from pathlib import Path

import polars as pl
import torch
from loguru import logger
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

import clinvar

# ── Metrics (eval-only) ──────────────────────────────────────────────────

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

STRATA_DEFS: dict[str, pl.Expr] = {
    "overall": pl.lit(True),
    "frameshift": pl.col("is_frameshift"),
    "inframe_coding": ~pl.col("is_frameshift") & pl.col("is_coding"),
    "noncoding": ~pl.col("is_coding"),
    "splice": pl.col("is_splice"),
    "non_splice": ~pl.col("is_splice"),
}


@torch.inference_mode()
def auroc(probs: torch.Tensor, labels: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """AUROC (Area Under ROC Curve)."""
    n_samples, n_labels = probs.shape[-2], probs.shape[-1]
    batch_shape = probs.shape[:-2]
    sorted_idx = probs.argsort(dim=-2, descending=True)
    y_sorted = labels.gather(-2, sorted_idx).bool()
    n_pos, n_neg = labels.sum(-2), n_samples - labels.sum(-2)
    tps = y_sorted.cumsum(-2).float()
    fps = (~y_sorted).cumsum(-2).float()
    tpr = tps / n_pos.unsqueeze(-2).clamp(min=1)
    fpr = fps / n_neg.unsqueeze(-2).clamp(min=1)
    zeros = torch.zeros(*batch_shape, 1, n_labels, device=probs.device, dtype=fpr.dtype)
    aucs = (torch.diff(fpr, dim=-2, prepend=zeros) * tpr).sum(-2)
    valid = (n_pos > 0) & (n_neg > 0)
    aucs = torch.where(valid, aucs, torch.full_like(aucs, float("nan")))
    return aucs.mean(-1) if reduction == "mean" else aucs


@torch.inference_mode()
def aupr(probs: torch.Tensor, labels: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """AUPR (Area Under Precision-Recall Curve)."""
    n_samples = probs.shape[-2]
    sorted_idx = probs.argsort(dim=-2, descending=True)
    y_sorted = labels.gather(-2, sorted_idx).float()
    n_pos = labels.sum(-2)
    tps = y_sorted.cumsum(-2)
    ranks = torch.arange(1, n_samples + 1, device=probs.device, dtype=torch.float32)
    ranks = ranks.view(*([1] * (probs.dim() - 2)), n_samples, 1)
    aps = ((tps / ranks) * y_sorted).sum(-2) / n_pos.clamp(min=1)
    valid = n_pos > 0
    aps = torch.where(valid, aps, torch.zeros_like(aps))
    return aps.mean(-1) if reduction == "mean" else aps


def bootstrap_ci(scores, labels, metric_fn, n_bootstrap=1000, alpha=0.05, seed=42):
    """Percentile bootstrap confidence interval for a binary metric."""
    generator = torch.Generator(device=scores.device).manual_seed(seed)
    indices = torch.randint(0, labels.shape[0], (n_bootstrap, labels.shape[0]), generator=generator, device=scores.device)
    values = metric_fn(scores[indices].unsqueeze(-1), labels[indices].unsqueeze(-1))
    if values.dim() > 1:
        values = values.squeeze(-1)
    values = values[~torch.isnan(values)]
    if len(values) == 0:
        return (float("nan"), float("nan"))
    return (values.quantile(alpha / 2).item(), values.quantile(1 - alpha / 2).item())


def annotate_strata(metadata: pl.DataFrame) -> pl.DataFrame:
    """Add stratification columns (is_frameshift, is_coding, is_splice)."""
    result = metadata.with_columns(
        pl.col("consequence").str.contains("(?i)frameshift").alias("is_frameshift"),
        pl.col("consequence").is_in(CODING_CONSEQUENCES).alias("is_coding"),
        pl.col("consequence").is_in(SPLICE_CONSEQUENCES).alias("is_splice"),
    )
    if "ref" in metadata.columns and "alt" in metadata.columns:
        result = result.with_columns(
            (pl.col("alt").str.len_chars().cast(pl.Int32) - pl.col("ref").str.len_chars().cast(pl.Int32)).alias("indel_len"),
        ).with_columns(pl.col("indel_len").abs().alias("indel_size")).with_columns(
            pl.when(pl.col("indel_size") <= 1).then(pl.lit("1bp"))
            .when(pl.col("indel_size") <= 5).then(pl.lit("2-5bp"))
            .when(pl.col("indel_size") <= 20).then(pl.lit("6-20bp"))
            .otherwise(pl.lit(">20bp")).alias("size_bin"),
        )
    return result


def compute_stratified_metrics(
    scores, labels, metadata,
    strata=None, bootstrap=True, n_bootstrap=1000, seed=42, min_positive=5, min_negative=5,
) -> pl.DataFrame:
    """Compute AUROC and AUPRC for each stratum of variants."""
    if strata is None:
        strata = STRATA_DEFS
    rows = []
    for name, expr in strata.items():
        mask_t = torch.from_numpy(metadata.with_columns(expr.alias("_m"))["_m"].to_numpy())
        s, y = scores[mask_t], labels[mask_t]
        n_pos, n_neg = int(y.sum().item()), len(y) - int(y.sum().item())
        if n_pos < min_positive or n_neg < min_negative:
            continue
        s_2d, y_2d = s.unsqueeze(-1), y.unsqueeze(-1)
        row = {"stratum": name, "n_variants": len(y), "n_pathogenic": n_pos, "n_benign": n_neg,
               "auroc": auroc(s_2d, y_2d).item(), "auprc": aupr(s_2d, y_2d).item()}
        if bootstrap:
            alo, ahi = bootstrap_ci(s, y, auroc, n_bootstrap=n_bootstrap, seed=seed)
            plo, phi = bootstrap_ci(s, y, aupr, n_bootstrap=n_bootstrap, seed=seed)
            row.update(auroc_ci_lo=alo, auroc_ci_hi=ahi, auprc_ci_lo=plo, auprc_ci_hi=phi)
        rows.append(row)
    return pl.DataFrame(rows)


def eval_clinvar(
    scores_df: pl.DataFrame,
    preset: str,
    split_path: Path | None,
    split_name: str,
) -> pl.DataFrame:
    """Evaluate on ClinVar: stratified AUROC/AUPRC with bootstrap CIs."""
    meta = clinvar.metadata(preset).with_columns(
        (pl.col("label") == "pathogenic").cast(pl.Int32).alias("pathogenic"),
    )

    meta_cols = ["variant_id", "pathogenic", "consequence"]
    if "ref" in meta.columns:
        meta_cols.extend(["ref", "alt"])
    df = scores_df.join(meta.select(meta_cols), on="variant_id")

    if split_path is not None:
        split = pl.read_ipc(str(split_path))
        df = df.join(split.select("variant_id", "split"), on="variant_id")
        if split_name != "all":
            df = df.filter(pl.col("split") == split_name)
            logger.info(f"Filtered to split={split_name}: {df.height:,} variants")

    logger.info(f"Evaluating {df.height:,} variants")
    df = annotate_strata(df)

    scores_t = torch.tensor(df["score"].to_list())
    labels_t = torch.tensor(df["pathogenic"].to_list())

    return compute_stratified_metrics(scores_t, labels_t, df)


def eval_dms(
    scores_df: pl.DataFrame,
    labels_csv: Path,
    label_col: str,
    continuous_col: str | None,
) -> pl.DataFrame:
    """Evaluate on DMS: AUROC + optional Spearman vs continuous score."""
    manifest = pl.read_csv(str(labels_csv))
    df = scores_df.rename({"score": "probe_score"}).join(manifest, on="variant_id", how="inner")

    y = df[label_col].to_numpy()
    s = df["probe_score"].to_numpy()

    auc = roc_auc_score(y, s)
    row: dict = {"n": df.height, "n_pathogenic": int(y.sum()), "auroc": round(auc, 6)}

    if continuous_col and continuous_col in df.columns:
        rho = spearmanr(s, df[continuous_col].to_numpy())[0]
        row["spearman"] = round(rho, 6)

    logger.info(f"AUROC={auc:.4f}" + (f", Spearman={row.get('spearman', 'N/A')}" if continuous_col else ""))
    return pl.DataFrame([row])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=Path, required=True, help="Probe dir or weights.pt (auto-discovers scores + split)")
    parser.add_argument("--scores", type=Path, help="Override scores.feather path (default: {probe}/scores.feather)")
    parser.add_argument("--output", type=Path, help="Output metrics CSV (default: {probe}/metrics.csv)")

    # ClinVar mode
    parser.add_argument("--preset", help="ClinVar preset (labeled, deconfounded)")
    parser.add_argument("--split", type=Path, help="Explicit split feather (overrides --probe)")
    parser.add_argument("--split-name", default="test", help="Which split to evaluate (test, train, all)")

    # DMS mode
    parser.add_argument("--labels-csv", type=Path, help="CSV with variant_id + label columns")
    parser.add_argument("--label-col", default="pathogenic", help="Binary label column name")
    parser.add_argument("--continuous-col", help="Continuous score column for Spearman (e.g. ldl_uptake_score)")

    args = parser.parse_args()

    # Resolve probe dir from --probe (can be dir or weights.pt)
    probe_dir = args.probe.parent if args.probe.suffix == ".pt" else args.probe

    # Auto-discover scores and output from probe dir
    scores_path = args.scores or probe_dir / "scores.feather"
    output_path = args.output or probe_dir / "metrics.csv"

    scores_df = pl.read_ipc(str(scores_path))
    logger.info(f"Loaded {scores_df.height:,} scores from {scores_path}")

    if args.preset:
        # Resolve split: explicit --split > auto-discover from probe dir > None
        split_path = args.split
        if split_path is None:
            candidate = probe_dir / "split.feather"
            if candidate.exists():
                split_path = candidate
                logger.info(f"Auto-discovered split: {split_path}")
            else:
                logger.warning(f"No split.feather found in {probe_dir} — evaluating on all variants")

        metrics = eval_clinvar(scores_df, args.preset, split_path, args.split_name)
    elif args.labels_csv:
        metrics = eval_dms(scores_df, args.labels_csv, args.label_col, args.continuous_col)
    else:
        parser.error("Provide --preset (ClinVar) or --labels-csv (DMS)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.write_csv(str(output_path))
    logger.info(f"Saved {output_path}")
    print(metrics)


if __name__ == "__main__":
    main()
