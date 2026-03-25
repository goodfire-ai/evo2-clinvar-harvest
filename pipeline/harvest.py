#!/usr/bin/env python3
"""Harvest Evo2 bidir activation diffs using fixed genomic windows.

For each variant: read W bp centered on the mutation from the chromosome,
run bidir Evo2, select top-K positions per direction with a fixed window
after the variant, and save 3 activation views at each position.

No gene boundaries, no gene length limits. Handles SNVs and indels natively.
Supports per-shard parallelism and checkpoint/resume on crash.

Outputs (goodfire-core chunked format):
    activations [2, 3, K, d]  fp16  per-direction, 3 views at K positions
    positions   [2, K]        int64 0-based genomic coordinates (-1 = pad)
    mean        [2, 3, d]     fp32  mean(rms_norm(X)) per view per direction
    loss        [2, K, 1]     fp32  cross-entropy loss diff at top-K positions

Activations layout:
    dim 0 (direction): [fwd_positions, bwd_positions]
    dim 1 (view):      [var_same, ref_same, ref_cross]
    dim 2 (position):  K positions, position-ordered, window first
    dim 3 (feature):   d_model

var_same and ref_same are on the selecting strand. ref_cross is the
reference on the opposite strand at the same position.

Compute diffs as acts[:, 0] - acts[:, 1] (var_same - ref_same).

Fwd positions: a fixed window of --window positions immediately downstream
of the variant, plus top-(K-window) most divergent by cosine similarity.
Bwd positions: top-K most divergent upstream of the variant (no window).

Positions are 0-based reference genome coordinates: genome[chrom][position]
gives the nucleotide.

Anchor alignment for indels: ref_anchor = offset + len(ref), var_anchor =
offset + len(alt). Fwd from [anchor, end), bwd from [0, offset).

Usage:
    # From ClinVar preset (no manifest needed)
    python scripts/harvest.py --preset labeled \\
        --storage /path/to/output --shard-id 0 --n-shards 16

    # From CSV manifest
    python scripts/harvest.py --manifest manifest.csv \\
        --storage /path/to/output --shard-id 0 --n-shards 16
"""

import argparse
import gc
import logging
import time
from pathlib import Path

import polars as pl
import torch
import torch.nn.functional as func
from goodfire_core.data.interfaces import TensorActivations
from goodfire_core.storage import ActivationWriter, FilesystemStorage
from tqdm import tqdm

logger = logging.getLogger(__name__)

_DNA_COMPLEMENT = str.maketrans("ATGCatgcNn", "TACGtacgNn")


def reverse_complement(seq: str) -> str:
    """Reverse complement a DNA sequence."""
    return seq.translate(_DNA_COMPLEMENT)[::-1]


# ── Model ────────────────────────────────────────────────────────────────────


class Evo2Bidir:
    """Evo2 bidirectional activation + loss extractor."""

    def __init__(self, model_name: str = "evo2_7b", block: int = 27, device: str = "cuda:0"):
        from evo2 import Evo2

        self.model = Evo2(model_name)
        self.device = device
        self.d_model: int = self.model.model.config.hidden_size
        self.layer = f"blocks.{block}"
        logger.info("Loaded %s (d_model=%d, layer=%s)", model_name, self.d_model, self.layer)

    def _forward(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Single forward pass. Returns (activations [L, d_model], loss [L-1])."""
        input_ids = torch.tensor(
            self.model.tokenizer.tokenize(sequence), dtype=torch.int,
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs, embeddings = self.model(
                input_ids, return_embeddings=True, layer_names=[self.layer],
            )
            loss = func.cross_entropy(
                outputs[0][:, :-1].reshape(-1, outputs[0].shape[-1]),
                input_ids[:, 1:].reshape(-1).long(),
                reduction="none",
            ).cpu()

        acts = embeddings[self.layer].squeeze(0).cpu()  # [L, d_model]
        return acts, loss

    def __call__(self, sequence: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Bidirectional pass. Returns (fwd_acts, bwd_acts, fwd_loss, bwd_loss) in sense order."""
        fwd_acts, fwd_loss = self._forward(sequence)
        bwd_acts, bwd_loss = self._forward(reverse_complement(sequence))
        return fwd_acts, bwd_acts.flip(0), fwd_loss, bwd_loss.flip(0)


# ── Top-K selection ──────────────────────────────────────────────────────────


def select_topk(
    var: torch.Tensor, ref: torch.Tensor,
    k: int, start: int, end: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Top-K most divergent positions in [start, end). Returns (var, ref, pos) all length K."""
    d = var.shape[-1]
    n = max(0, end - start)
    actual_k = min(k, n)

    if actual_k == 0:
        z = torch.zeros(k, d, dtype=torch.float16)
        return z, z.clone(), torch.full((k,), -1, dtype=torch.long)

    cos = func.cosine_similarity(var[start:end].float(), ref[start:end].float(), dim=-1)
    _, idx = cos.topk(actual_k, largest=False)
    idx = idx.sort().values + start

    var_sel, ref_sel = var[idx].half(), ref[idx].half()
    pos = idx.long()

    if actual_k < k:
        pad = k - actual_k
        var_sel = torch.cat([var_sel, torch.zeros(pad, d, dtype=torch.float16)])
        ref_sel = torch.cat([ref_sel, torch.zeros(pad, d, dtype=torch.float16)])
        pos = torch.cat([pos, torch.full((pad,), -1, dtype=torch.long)])

    return var_sel, ref_sel, pos


def select_positions(
    var_same: torch.Tensor, ref_same: torch.Tensor, ref_cross: torch.Tensor,
    k: int, start: int, end: int,
    *,
    window: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select K positions with optional guaranteed window, storing 3 views.

    Selects positions from [start, end) in two parts:
    1. **Window**: positions [start, start+window) are always included
       (the immediate context after the variant).
    2. **Divergent**: top-(K-window) most divergent positions from
       [start+window, end), by lowest cosine similarity between
       var_same and ref_same.

    At each selected position, 3 activation views are stored:
    - var_same: variant on the selecting strand
    - ref_same: reference on the selecting strand
    - ref_cross: reference on the opposite strand

    All positions are sorted in ascending order.

    Args:
        var_same: Variant activations on the selecting strand [L, d].
        ref_same: Reference activations on the selecting strand [L, d].
        ref_cross: Reference activations on the opposite strand [L, d].
        k: Total number of positions to select.
        start: Start of valid range (inclusive).
        end: End of valid range (exclusive).
        window: Positions [start, start+window) always retained (default 0).

    Returns:
        (acts, positions) where:
        - acts: ``[3, K, d]`` fp16 — [var_same, ref_same, ref_cross]
        - positions: ``[K]`` int64 — window-relative indices (-1 = padding)
    """
    d = var_same.shape[-1]
    n = max(0, end - start)

    if n == 0:
        return torch.zeros(3, k, d, dtype=torch.float16), torch.full((k,), -1, dtype=torch.long)

    # Part 1: window (always included)
    actual_window = min(window, n, k)
    window_idx = torch.arange(start, start + actual_window)

    # Part 2: top-K divergent from [start+window, end)
    remaining_k = k - actual_window
    divergent_start = start + actual_window
    divergent_n = max(0, end - divergent_start)
    actual_divergent = min(remaining_k, divergent_n)

    if actual_divergent > 0:
        cos = func.cosine_similarity(
            var_same[divergent_start:end].float(),
            ref_same[divergent_start:end].float(),
            dim=-1,
        )
        _, div_idx = cos.topk(actual_divergent, largest=False)
        div_idx = div_idx.sort().values + divergent_start
        all_idx = torch.cat([window_idx, div_idx])
    else:
        all_idx = window_idx

    all_idx = all_idx.sort().values

    acts = torch.stack([
        var_same[all_idx].half(),
        ref_same[all_idx].half(),
        ref_cross[all_idx].half(),
    ])  # [3, K', d]

    pos = all_idx.long()

    # Pad to K
    actual_total = len(all_idx)
    if actual_total < k:
        pad = k - actual_total
        acts = torch.cat([acts, torch.zeros(3, pad, d, dtype=torch.float16)], dim=1)
        pos = torch.cat([pos, torch.full((pad,), -1, dtype=torch.long)])

    return acts, pos


def loss_at_positions(
    var_loss: torch.Tensor, ref_loss: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Loss diff (var - ref) at top-K positions. Returns [K, 1] fp32."""
    k = positions.shape[0]
    result = torch.zeros(k, 1, dtype=torch.float32)
    valid = (positions >= 0) & (positions < len(var_loss)) & (positions < len(ref_loss))
    if valid.any():
        idx = positions[valid].long()
        result[valid] = (var_loss[idx] - ref_loss[idx]).float().unsqueeze(-1)
    return result


# ── Sequence building ────────────────────────────────────────────────────────


def read_window(
    genome: dict[str, str], chrom: str, pos: int,
    upstream: int, downstream: int,
) -> tuple[str, int]:
    """Read a window of upstream + downstream bp around pos from chromosome.

    Args:
        genome: Chromosome name → sequence mapping.
        chrom: Bare chromosome name (e.g. '1', 'M').
        pos: 0-based genomic position of the variant.
        upstream: Number of bp before the variant.
        downstream: Number of bp after the variant.

    Returns:
        (sequence, offset) where offset is the variant position within the
        window. For chromosomes shorter than the requested window, returns
        the entire chromosome.
    """
    chrom_seq = genome[chrom]
    chrom_len = len(chrom_seq)
    window = min(upstream + downstream, chrom_len)

    start = max(0, pos - upstream)
    end = start + window
    if end > chrom_len:
        end = chrom_len
        start = end - window

    return chrom_seq[start:end], pos - start


def mutate(ref_seq: str, offset: int, ref_allele: str, alt_allele: str) -> str:
    """Splice alt_allele into ref_seq at offset, replacing ref_allele.

    Both ref_allele and alt_allele are in VCF convention (+ strand), matching
    the chromosome window which is also + strand. No strand correction needed.
    """
    actual = ref_seq[offset:offset + len(ref_allele)]
    assert actual == ref_allele, f"ref mismatch at offset {offset}: expected '{ref_allele}', got '{actual}'"
    return ref_seq[:offset] + alt_allele + ref_seq[offset + len(ref_allele):]


# ── Manifest loading ─────────────────────────────────────────────────────────


def load_manifest(args: argparse.Namespace) -> pl.DataFrame:
    """Load variant manifest from --preset or --manifest CSV.

    Returns DataFrame with columns: variant_id, chrom, pos, ref, alt, label.
    Labels are converted to int (1=pathogenic, 0=benign) at the DataFrame level.
    """
    if args.preset:
        import clinvar
        df = clinvar.metadata(args.preset)
        df = df.select(
            "variant_id",
            pl.col("chrom"),
            pl.col("pos").cast(pl.Int64),
            pl.col("ref"),
            pl.col("alt"),
            pl.col("label"),
        )
    else:
        df = pl.read_csv(args.manifest)
        renames = {"ref_corrected": "ref", "alt_corrected": "alt"}
        for old, new in renames.items():
            if old in df.columns and new not in df.columns:
                df = df.rename({old: new})

    # Normalize chrom to bare number (1, 2, ..., X, Y, M) to match gencode convention
    df = df.with_columns(pl.col("chrom").str.replace("^chr", ""))

    # Convert labels to int once (1=pathogenic, 0=benign)
    label_col = df["label"]
    if label_col.dtype == pl.Utf8:
        df = df.with_columns(
            pl.when(pl.col("label") == "pathogenic").then(1).otherwise(0).alias("label")
        )
    else:
        df = df.with_columns(pl.col("label").cast(pl.Int32))

    logger.info("Manifest: %d variants", df.height)
    return df


# ── Checkpoint ───────────────────────────────────────────────────────────────


def load_checkpoint(path: Path) -> set[str]:
    """Load set of completed variant IDs from checkpoint file."""
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--preset", help="ClinVar preset (e.g. labeled, deconfounded)")
    input_group.add_argument("--manifest", type=Path, help="CSV manifest with variant columns")

    # Output
    parser.add_argument("--storage", type=Path, required=True, help="Output directory for activation datasets")

    # Sharding
    parser.add_argument("--shard-id", type=int, required=True)
    parser.add_argument("--n-shards", type=int, default=16)

    # Model
    parser.add_argument("--model-name", default="evo2_7b")
    parser.add_argument("--block", type=int, default=27)
    parser.add_argument("--device", default="cuda:0")

    # Extraction
    parser.add_argument("--upstream", type=int, default=3 * 2**14, help="bp before variant (default 3*2^14 = 49152)")
    parser.add_argument("--downstream", type=int, default=2**14, help="bp after variant (default 2^14 = 16384)")
    parser.add_argument("--topk", type=int, default=256)
    parser.add_argument("--window", type=int, default=64, help="Fixed window after variant (always retained)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s %(levelname)s [s{args.shard_id}] %(message)s")

    topk = args.topk

    # Load and shard manifest — each shard gets a contiguous slice
    manifest = load_manifest(args)
    n = manifest.height
    n_shards = min(args.n_shards, n)  # no more shards than variants
    if args.shard_id >= n_shards:
        logger.info("Shard %d >= n_shards %d (only %d variants), nothing to do", args.shard_id, n_shards, n)
        return
    start = args.shard_id * n // n_shards
    end = (args.shard_id + 1) * n // n_shards
    shard = manifest.slice(start, end - start)
    logger.info("Shard %d/%d: %d variants [%d:%d), up=%d down=%d K=%d",
                args.shard_id, n_shards, shard.height, start, end, args.upstream, args.downstream, topk)

    import gencode
    genome = gencode.chromosomes()
    model = Evo2Bidir(args.model_name, args.block, args.device)

    # Column-wise access
    vids = shard["variant_id"].to_list()

    # Checkpoint — validate against current shard to catch stale files from prior runs
    ckpt_dir = args.storage / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"shard_{args.shard_id}.txt"
    completed = load_checkpoint(ckpt_path)
    if completed:
        shard_vids = set(vids)
        stale = completed - shard_vids
        if stale:
            raise RuntimeError(
                f"Checkpoint {ckpt_path} has {len(stale)} entries not in this shard's "
                f"manifest. Stale checkpoint from a prior run? Delete it to start fresh."
            )
        logger.info("Resuming from %d completed variants", len(completed))

    # Writers — one primary dataset with all activations, plus small metadata datasets.
    # activations: [4, K, d] = [var_fwd, var_bwd, ref_fwd, ref_bwd] at top-K positions.
    # Diff is not stored — compute as acts[:2] - acts[2:] at read time.
    storage = FilesystemStorage(args.storage)
    d = model.d_model
    # shuffle=False: datasets stay aligned by insertion order.
    # Shuffling at read time via training_iterator.
    # Buffer sized for ~1GB chunks (2x default chunk_target_bytes for margin).
    item_bytes = 2 * 3 * topk * d * 2  # largest item: activations [2, 3, K, d] fp16
    buffer_items = max(1, 1_000_000_000 // item_bytes)
    writer_cfg = dict(mode="tensor", partition_id=args.shard_id, shuffle=False,
                      shuffle_buffer_size=buffer_items, overwrite=False, compute_checksum=False)
    writers = {
        "activations": ActivationWriter(storage, "activations", d_model=(2, 3, topk, d), dtype="float16", **writer_cfg),
        "positions": ActivationWriter(storage, "positions", d_model=(2, topk), dtype="int64", **writer_cfg),
        "mean": ActivationWriter(storage, "mean", d_model=(2, 3, d), dtype="float32", **writer_cfg),
        "loss": ActivationWriter(storage, "loss", d_model=(2, topk, 1), dtype="float32", **writer_cfg),
    }
    chroms = shard["chrom"].to_list()
    pos_list = shard["pos"].to_list()
    refs = shard["ref"].to_list()
    alts = shard["alt"].to_list()
    labels = shard["label"].to_list()

    n_ok, n_fail = 0, 0
    t0 = time.time()
    pbar = tqdm(range(shard.height), desc=f"shard {args.shard_id}")

    for i in pbar:
        vid = vids[i]
        if vid in completed:
            continue

        label = int(labels[i])

        try:
            ref_seq, offset = read_window(genome, chroms[i], int(pos_list[i]), args.upstream, args.downstream)
            var_seq = mutate(ref_seq, offset, refs[i], alts[i])
            assert len(ref_seq) > 0, f"empty window for {vid} on {chroms[i]}"

            ref_anchor = offset + len(refs[i])
            var_anchor = offset + len(alts[i])

            try:
                ref_fwd, ref_bwd, ref_fwd_loss, ref_bwd_loss = model(ref_seq)
                var_fwd, var_bwd, var_fwd_loss, var_bwd_loss = model(var_seq)
            except RuntimeError as e:
                if "cuFFT" not in str(e):
                    raise
                logger.info("cuFFT retry: %s", vid)
                gc.collect()
                torch.cuda.empty_cache()
                ref_fwd, ref_bwd, ref_fwd_loss, ref_bwd_loss = model(ref_seq)
                var_fwd, var_bwd, var_fwd_loss, var_bwd_loss = model(var_seq)

            shared_len = min(len(var_fwd), len(ref_fwd))

            # Forward (downstream): window + top-K divergent
            fwd_acts, pf = select_positions(
                var_fwd, ref_fwd, ref_bwd,
                topk, max(ref_anchor, var_anchor), shared_len,
                window=args.window,
            )
            # Backward (upstream): top-K divergent only (no window)
            bwd_acts, pb = select_positions(
                var_bwd, ref_bwd, ref_fwd,
                topk, 0, offset,
            )

            # Stack: [2, 3, K, d] — [direction, view, position, feature]
            acts = torch.stack([fwd_acts, bwd_acts])  # [2, 3, K, d]

            loss_fwd = loss_at_positions(var_fwd_loss, ref_fwd_loss, pf)
            loss_bwd = loss_at_positions(var_bwd_loss, ref_bwd_loss, pb)

            # Mean pool: mean(rms_norm(X)) per view per direction → [2, 3, d]
            # RMS-normalize each position first, then mean — preserves direction,
            # normalizes magnitude per token before averaging.
            norm_shape = (var_fwd.shape[-1],)
            sl = slice(None, shared_len)
            mean_pool = torch.stack([
                # fwd direction: [var_same, ref_same, ref_cross]
                torch.stack([
                    func.rms_norm(var_fwd[sl].float(), norm_shape).mean(0),
                    func.rms_norm(ref_fwd[sl].float(), norm_shape).mean(0),
                    func.rms_norm(ref_bwd[sl].float(), norm_shape).mean(0),
                ]),
                # bwd direction: [var_same, ref_same, ref_cross]
                torch.stack([
                    func.rms_norm(var_bwd[sl].float(), norm_shape).mean(0),
                    func.rms_norm(ref_bwd[sl].float(), norm_shape).mean(0),
                    func.rms_norm(ref_fwd[sl].float(), norm_shape).mean(0),
                ]),
            ])  # [2, 3, d]

            # Convert window-relative positions to 0-based genomic coordinates
            window_start = int(pos_list[i]) - offset
            pf_genomic = pf.clone()
            pb_genomic = pb.clone()
            pf_genomic[pf >= 0] += window_start
            pb_genomic[pb >= 0] += window_start

            label_t = torch.tensor([label])
            writers["activations"].add_activations(TensorActivations(
                acts=acts.unsqueeze(0), labels=label_t, sequence_ids=[vid],
            ))
            writers["positions"].add_activations(TensorActivations(
                acts=torch.stack([pf_genomic.long(), pb_genomic.long()]).unsqueeze(0), labels=label_t, sequence_ids=[vid],
            ))
            writers["mean"].add_activations(TensorActivations(
                acts=mean_pool.unsqueeze(0), labels=label_t, sequence_ids=[vid],
            ))
            writers["loss"].add_activations(TensorActivations(
                acts=torch.stack([loss_fwd, loss_bwd]).unsqueeze(0), labels=label_t, sequence_ids=[vid],
            ))

            del ref_fwd, ref_bwd, var_fwd, var_bwd
            del ref_fwd_loss, ref_bwd_loss, var_fwd_loss, var_bwd_loss
            n_ok += 1
            with ckpt_path.open("a") as f:
                f.write(vid + "\n")

        except (torch.cuda.OutOfMemoryError, KeyboardInterrupt):
            raise
        except Exception as e:
            logger.error("FAILED %s: %s", vid, e, exc_info=True)
            n_fail += 1

        if (n_ok + n_fail) % 50 == 0:
            elapsed = time.time() - t0
            pbar.set_postfix(ok=n_ok, fail=n_fail, s_per_var=f"{elapsed / max(n_ok, 1):.1f}")

    for w in writers.values():
        w.finalize()

    elapsed = time.time() - t0
    logger.info("Done: %d ok, %d fail, %.0fs (%.1f s/variant)", n_ok, n_fail, elapsed, elapsed / max(n_ok, 1))
    if n_fail:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
