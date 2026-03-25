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
from utils import annotate_strata, compute_stratified_metrics


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
