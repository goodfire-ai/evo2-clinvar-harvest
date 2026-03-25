"""Extract covariance embeddings + scores from a trained CovarianceProbe.

Streams unified activation diffs through the probe, writes embeddings
(goodfire-core chunked format) and scores. Idempotent. Supports SLURM
array parallelism via --shard-id / --n-shards.

Single GPU:
    python pipeline/embed.py \\
        --probe /path/to/storage/cov64 \\
        --activations /path/to/storage \\
        --preset labeled

Parallel (SLURM array):
    EMBED=$(sbatch --parsable --array=0-15 --gres=gpu:1 --wrap \\
        "cd \\${SLURM_SUBMIT_DIR} && uv run python pipeline/embed.py \\
         --probe /path/to/storage/cov64 --activations /path/to/storage \\
         --preset labeled --shard-id \\${SLURM_ARRAY_TASK_ID} \\
         --n-shards \\${SLURM_ARRAY_TASK_COUNT}")
    sbatch --dependency=afterok:${EMBED} pipeline/finalize_embed.sh /path/to/storage/cov64
"""

import argparse
from pathlib import Path

import polars as pl
import torch
from goodfire_core.data.interfaces import TensorActivations
from goodfire_core.probes.covariance import SequenceCovarianceProbe
from goodfire_core.storage import ActivationWriter, FilesystemStorage
from loguru import logger
from tqdm import tqdm

from src.streaming import iter_dataset, unified_diff


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--probe", type=Path, required=True, help="Probe dir or weights.pt path")
    parser.add_argument("--activations", type=Path, required=True, help="Base storage directory")
    parser.add_argument("--batch-size", type=int, default=512)

    # Sharding (for SLURM array parallelism)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)

    id_group = parser.add_mutually_exclusive_group(required=True)
    id_group.add_argument("--preset", help="ClinVar preset (e.g. labeled, deconfounded)")
    id_group.add_argument("--manifest", type=Path, help="CSV manifest with variant_id column")
    id_group.add_argument("--split", type=Path, help="Split feather with variant_id column")

    args = parser.parse_args()

    # ── Load probe ────────────────────────────────────────────────────────
    probe_dir = args.probe.parent if args.probe.suffix == ".pt" else args.probe
    weights_path = args.probe if args.probe.suffix == ".pt" else args.probe / "weights.pt"
    probe = SequenceCovarianceProbe.from_checkpoint(str(weights_path)).cuda().eval()
    logger.info(f"Probe: d_model={probe.d_model}, d_hidden={probe.d_hidden}")

    # ── Resolve target IDs ────────────────────────────────────────────────
    if args.preset:
        from src.datasets import clinvar
        all_ids = clinvar.metadata(args.preset)["variant_id"].to_list()
    elif args.manifest:
        all_ids = pl.read_csv(str(args.manifest))["variant_id"].to_list()
    else:
        all_ids = pl.read_ipc(str(args.split))["variant_id"].to_list()

    # Shard the ID list
    n = len(all_ids)
    n_shards = min(args.n_shards, n)
    if args.shard_id >= n_shards:
        logger.info(f"Shard {args.shard_id} >= n_shards {n_shards}, nothing to do")
        return
    start = args.shard_id * n // n_shards
    end = (args.shard_id + 1) * n // n_shards
    target_ids = set(all_ids[start:end])
    logger.info(f"Shard {args.shard_id}/{n_shards}: {len(target_ids):,} variants [{start}:{end})")

    # ── Output dir ────────────────────────────────────────────────────────
    probe_name = probe_dir.name
    output_dir = args.activations / probe_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_storage = FilesystemStorage(output_dir)
    logger.info(f"Output: {output_dir}")

    # ── Stream and extract ────────────────────────────────────────────────
    storage = FilesystemStorage(args.activations)
    d_hidden = probe.d_hidden

    partition_id = args.shard_id if args.n_shards > 1 else None
    writer = ActivationWriter(
        output_storage, "embeddings",
        d_model=(d_hidden, d_hidden),
        mode="tensor",
        dtype="bfloat16",
        shuffle=False,
        shuffle_buffer_size=1024,
        partition_id=partition_id,
    )

    all_out_ids: list[str] = []
    all_scores: list[float] = []

    with torch.no_grad():
        for acts, ids in tqdm(
            iter_dataset(
                storage, "activations", target_ids, unified_diff,
                batch_size=args.batch_size, dtype=torch.bfloat16, device="cuda",
            ),
            desc=f"embed s{args.shard_id}",
        ):
            acts = acts.float()
            emb = probe.embedding(acts)
            logits = probe(acts)
            probs = torch.softmax(logits, dim=-1)[:, 1].cpu()

            writer.add_activations(TensorActivations(acts=emb.cpu(), sequence_ids=ids))
            all_out_ids.extend(ids)
            all_scores.extend(probs.tolist())

    writer.finalize()

    # Save scores
    scores_df = pl.DataFrame({"variant_id": all_out_ids, "score": all_scores})
    if args.n_shards > 1:
        scores_path = output_dir / f"scores_shard_{args.shard_id}.feather"
    else:
        scores_path = output_dir / "scores.feather"
    scores_df.write_ipc(scores_path)
    logger.info(f"Extracted {len(all_out_ids):,} embeddings [{d_hidden}, {d_hidden}] + scores -> {scores_path}")


if __name__ == "__main__":
    main()
