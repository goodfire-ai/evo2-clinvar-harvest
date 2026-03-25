"""Train a CovarianceProbe on unified [2, 3, K, d] activation data.

Uses goodfire-core's train_probe with DDP, checkpointing, and EMuon.

Saves artifacts to {activations}/{name}/ (default name: cov64):
    - weights.pt — probe checkpoint
    - checkpoint_epoch_{N}.pt — per-epoch checkpoints (resumable)
    - split.feather — gene-holdout split (variant_id, gene_name, split)
    - config.json — full experiment config

Single GPU:
    uv run python scripts/train_pooler.py \\
        --activations /path/to/storage --preset deconfounded-full

Multi-GPU (torchrun):
    uv run torchrun --nproc-per-node=4 scripts/train_pooler.py \\
        --activations /path/to/storage --preset deconfounded-full
"""

import argparse
import json
import os
from pathlib import Path

import polars as pl
import torch
import torch.distributed as dist
from goodfire_core.configs.probes import (
    CovarianceProbeConfig,
    ProbeDataConfig,
    ProbeTrainingConfig,
    ProbeWorkflowConfig,
)
from goodfire_core.data.interfaces import TensorActivations
from goodfire_core.probes.covariance import SequenceCovarianceProbe
from goodfire_core.probes.trainer import train_probe
from goodfire_core.storage import ActivationDataset, FilesystemStorage
from goodfire_core.training.optimizers import EMuon
from loguru import logger
from torch.nn.parallel import DistributedDataParallel

import clinvar
from utils import gene_split, unified_diff


def _unified_diff(batch: TensorActivations) -> TensorActivations:
    """[B, 2, 3, K, d] -> [B, K, 2*d]: var-ref diff, concat fwd+bwd."""
    return TensorActivations(
        acts=unified_diff(batch.acts).float(),
        labels=batch.labels,
        sequence_ids=batch.sequence_ids,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations", type=Path, required=True)
    parser.add_argument("--preset", default="deconfounded-full")
    parser.add_argument("--name", default="cov64")

    parser.add_argument("--d-model", type=int, default=8192)
    parser.add_argument("--d-hidden", type=int, default=64)
    parser.add_argument("--d-probe", type=int, default=256)

    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    torch.manual_seed(args.seed)

    # ── DDP ───────────────────────────────────────────────────────────────
    distributed = dist.is_available() and "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        rank = 0
        device = torch.device("cuda")

    # ── Data ──────────────────────────────────────────────────────────────
    manifest = clinvar.metadata(args.preset).with_columns(
        (pl.col("label") == "pathogenic").cast(pl.Int32).alias("pathogenic"),
    )
    train_df, test_df = gene_split(manifest, test_size=args.test_size, seed=args.seed)
    train_ids = train_df["variant_id"].to_list()

    # ── Save split (rank 0) ──────────────────────────────────────────────
    out_dir = args.activations / args.name
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        pl.concat([
            train_df.select("variant_id", "gene_name").with_columns(pl.lit("train").alias("split")),
            test_df.select("variant_id", "gene_name").with_columns(pl.lit("test").alias("split")),
        ]).write_ipc(out_dir / "split.feather")
        logger.info(f"Split: {train_df.height:,} train, {test_df.height:,} test")
    if distributed:
        dist.barrier()

    # ── Iterator ──────────────────────────────────────────────────────────
    storage = FilesystemStorage(args.activations)
    world_size = dist.get_world_size() if distributed else 1
    per_gpu_batch = max(1, args.batch_size // world_size)
    dataset = ActivationDataset(storage, "activations", batch_size=per_gpu_batch)
    iterator = dataset.training_iterator(
        device=str(device), n_epochs=args.epochs, sequence_ids=train_ids,
    )
    iterator.add_transform(_unified_diff)

    if rank == 0:
        logger.info(f"Train: {len(train_ids):,} variants, {iterator.steps_per_epoch} steps/epoch")

    # ── Probe ─────────────────────────────────────────────────────────────
    probe = SequenceCovarianceProbe(d_model=args.d_model, d_hidden=args.d_hidden, d_probe=args.d_probe)
    probe = probe.to(device)
    optimizer = EMuon(probe.parameters(), lr=args.lr)

    if distributed:
        probe = DistributedDataParallel(probe, device_ids=[rank])

    # ── Config for train_probe ────────────────────────────────────────────
    workflow_config = ProbeWorkflowConfig(
        experiment_name=f"clinvar/{args.preset}",
        base_dir=str(out_dir),
        checkpoints_dir=str(out_dir),
        model={"name": "evo2_7b", "hook_sites": ["blocks.27"], "d_model": 4096},
        activation_extraction={"batch_size": 1, "dataset_name": "activations"},
        data=ProbeDataConfig(data_sources={}, max_seq_length=1, extra_data={}),
        probe=CovarianceProbeConfig(
            probe_type="covariance", d_model=args.d_model, n_outputs=2,
            d_hidden=args.d_hidden, d_probe=args.d_probe, extra_data={},
        ),
        training=ProbeTrainingConfig(
            n_epochs=args.epochs, batch_size=args.batch_size,
            learning_rate=args.lr, weight_decay=0.0,
            log_every_steps=1, save_every_epochs=1,
        ),
        extra_data={},
    )

    train_probe(
        probe, iterator, workflow_config,
        optimizer=optimizer,
        is_distributed=distributed, rank=rank,
    )

    # ── Save final probe + config (rank 0) ────────────────────────────────
    if rank == 0:
        raw_probe = probe.module if distributed else probe
        raw_probe.save_checkpoint(str(out_dir / "weights.pt"))
        logger.info(f"Saved: {out_dir / 'weights.pt'}")

        (out_dir / "config.json").write_text(json.dumps({
            "preset": args.preset, "seed": args.seed, "test_size": args.test_size,
            "d_model": args.d_model, "d_hidden": args.d_hidden, "d_probe": args.d_probe,
            "lr": args.lr, "epochs": args.epochs, "batch_size": args.batch_size,
            "n_train": train_df.height, "n_test": test_df.height,
            "distributed": distributed,
            "world_size": dist.get_world_size() if distributed else 1,
        }, indent=2))

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
