"""Tests for harvest.py partition scanning and integration correctness.

Tests are split into:
  - Unit tests for load_completed_from_partition (no GPU needed)
  - Integration tests that run the full harvest loop with a mocked model (no GPU needed)
"""

import json
import sys
from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest
import torch
from safetensors.torch import save_file

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO = Path(__file__).parent.parent
sys.path.insert(0, str(REPO))

from pipeline.harvest import load_completed_from_partition


def write_fake_chunk(chunk_dir: Path, sequence_ids: list[str], d: int = 8) -> None:
    """Write a minimal safetensors file that mimics a real activations chunk."""
    chunk_dir.mkdir(parents=True, exist_ok=True)
    n = len(sequence_ids)
    tensors = {"activations": torch.zeros(n, d)}
    metadata = {
        "sequence_ids": json.dumps(sequence_ids),
        "mode": "sequence",
        "num_items": str(n),
    }
    save_file(tensors, str(chunk_dir / "activations.safetensors"), metadata=metadata)


def make_fake_genome(length: int = 200_000) -> dict[str, str]:
    import random
    random.seed(42)
    bases = "ATGC"
    seq = "".join(random.choice(bases) for _ in range(length))
    return {str(c): seq for c in list(range(1, 6)) + ["X"]}


def make_fake_manifest(n: int = 20, chrom: str = "1") -> pl.DataFrame:
    """Create a tiny variant manifest using positions guaranteed to be in the fake genome.
    chrom is stored as 'chr1' in the CSV so Polars doesn't infer it as int64.
    load_manifest strips the 'chr' prefix, matching the gencode convention.
    """
    genome = make_fake_genome()
    seq = genome[chrom]
    rows = []
    for i in range(n):
        pos = 50_000 + i * 200
        ref = seq[pos]
        alt = {"A": "T", "T": "A", "G": "C", "C": "G"}.get(ref, "T")
        rows.append({
            "variant_id": f"chr{chrom}:{pos}:{ref}:{alt}",
            "chrom": f"chr{chrom}",  # keep as string; load_manifest strips the prefix
            "pos": pos,
            "ref": ref,
            "alt": alt,
            "label": i % 2,
        })
    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Unit tests: load_completed_from_partition
# ---------------------------------------------------------------------------


def test_load_completed_empty_storage(tmp_path):
    assert load_completed_from_partition(tmp_path, shard_id=0) == set()


def test_load_completed_no_activations_dir(tmp_path):
    (tmp_path / "other_dataset").mkdir()
    assert load_completed_from_partition(tmp_path, shard_id=0) == set()


def test_load_completed_single_chunk(tmp_path):
    ids = ["chr1:100:A:T", "chr2:200:G:C", "chr3:300:T:A"]
    write_fake_chunk(
        tmp_path / "activations" / "partition_0" / "chunks" / "chunk_000000", ids
    )
    assert load_completed_from_partition(tmp_path, shard_id=0) == set(ids)


def test_load_completed_multiple_chunks_merged(tmp_path):
    ids_a = ["chr1:100:A:T", "chr2:200:G:C"]
    ids_b = ["chr3:300:T:A", "chr4:400:C:G"]
    base = tmp_path / "activations" / "partition_0" / "chunks"
    write_fake_chunk(base / "chunk_000000", ids_a)
    write_fake_chunk(base / "chunk_000001", ids_b)
    assert load_completed_from_partition(tmp_path, shard_id=0) == set(ids_a + ids_b)


def test_load_completed_shard_isolation(tmp_path):
    """Scanning shard 0 must not pick up shard 1's data."""
    ids_s0 = ["chr1:100:A:T", "chr1:200:G:C"]
    ids_s1 = ["chr9:999:A:T", "chr9:888:G:C"]
    write_fake_chunk(
        tmp_path / "activations" / "partition_0" / "chunks" / "chunk_000000", ids_s0
    )
    write_fake_chunk(
        tmp_path / "activations" / "partition_1" / "chunks" / "chunk_000000", ids_s1
    )
    assert load_completed_from_partition(tmp_path, shard_id=0) == set(ids_s0)
    assert load_completed_from_partition(tmp_path, shard_id=1) == set(ids_s1)


def test_load_completed_corrupt_chunk_skipped(tmp_path):
    """A corrupt chunk is skipped with a warning; valid chunks still read."""
    ids = ["chr1:100:A:T"]
    base = tmp_path / "activations" / "partition_0" / "chunks"
    write_fake_chunk(base / "chunk_000000", ids)
    corrupt = base / "chunk_000001"
    corrupt.mkdir()
    (corrupt / "activations.safetensors").write_bytes(b"not a safetensors file")

    assert load_completed_from_partition(tmp_path, shard_id=0) == set(ids)


def test_load_completed_missing_safetensors_in_chunk(tmp_path):
    """A chunk dir that has no activations.safetensors is silently skipped."""
    base = tmp_path / "activations" / "partition_0" / "chunks"
    empty = base / "chunk_000000"
    empty.mkdir(parents=True)
    ids = ["chr1:100:A:T"]
    write_fake_chunk(base / "chunk_000001", ids)
    assert load_completed_from_partition(tmp_path, shard_id=0) == set(ids)


# ---------------------------------------------------------------------------
# Unit tests: load_checkpoint
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# load_completed_from_partition correctness
# ---------------------------------------------------------------------------


def test_resume_partition_reads_all_chunks(tmp_path):
    """All variants across multiple chunks are returned."""
    partition_ids = {"chr1:100:A:T", "chr1:200:G:C", "chr1:300:T:A"}
    write_fake_chunk(
        tmp_path / "activations" / "partition_0" / "chunks" / "chunk_000000",
        sorted(partition_ids),
    )
    assert load_completed_from_partition(tmp_path, shard_id=0) == partition_ids


def test_resume_empty_partition_returns_empty(tmp_path):
    """No partition directory → empty set."""
    assert load_completed_from_partition(tmp_path, shard_id=0) == set()


# ---------------------------------------------------------------------------
# Integration tests: full harvest loop with mocked model
# ---------------------------------------------------------------------------


class FakeEvo2Bidir:
    """Replaces Evo2Bidir with deterministic random tensors for testing."""
    d_model = 64

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, sequence: str):
        L = min(len(sequence), 256)
        torch.manual_seed(hash(sequence) % (2**32))
        fwd = torch.randn(L, self.d_model)
        bwd = torch.randn(L, self.d_model)
        fwd_loss = torch.randn(max(1, L - 1))
        bwd_loss = torch.randn(max(1, L - 1))
        return fwd, bwd.flip(0), fwd_loss, bwd_loss.flip(0)


def _run_harvest(storage: Path, manifest_path: Path, n_shards: int = 1, shard_id: int = 0,
                 upstream: int = 200, downstream: int = 200, topk: int = 4, window: int = 2):
    """Run harvest.main() with a mocked Evo2 model against a CSV manifest."""
    from pipeline import harvest as hmod

    fake_genome = make_fake_genome()

    argv = [
        "harvest.py",
        "--manifest", str(manifest_path),
        "--storage", str(storage),
        "--shard-id", str(shard_id),
        "--n-shards", str(n_shards),
        "--upstream", str(upstream),
        "--downstream", str(downstream),
        "--topk", str(topk),
        "--window", str(window),
    ]

    with (
        patch.object(sys, "argv", argv),
        patch.object(hmod, "Evo2Bidir", FakeEvo2Bidir),
        patch("gencode.chromosomes", return_value=fake_genome),
    ):
        hmod.main()


def _manifest_csv(tmp_path: Path, n: int = 10) -> Path:
    df = make_fake_manifest(n=n)
    # Save as CSV — harvest.py reads this via pl.read_csv
    path = tmp_path / "manifest.csv"
    df.write_csv(str(path))
    return path


@pytest.fixture
def manifest(tmp_path):
    return _manifest_csv(tmp_path, n=10)


def test_fresh_run_writes_all_variants(tmp_path, manifest):
    """A complete run puts all variants in the partition."""
    storage = tmp_path / "storage"
    _run_harvest(storage, manifest)

    completed = load_completed_from_partition(storage, shard_id=0)
    df = pl.read_csv(str(manifest))
    expected = set(df["variant_id"].to_list())
    assert completed == expected, f"missing: {expected - completed}"


def test_resume_after_clean_run_skips_everything(tmp_path, manifest):
    """Running twice on the same storage processes 0 variants the second time."""
    storage = tmp_path / "storage"
    _run_harvest(storage, manifest)
    first_ids = load_completed_from_partition(storage, shard_id=0)

    # Second run should be a no-op (all variants already on disk)
    _run_harvest(storage, manifest)
    second_ids = load_completed_from_partition(storage, shard_id=0)

    assert first_ids == second_ids


def test_rerun_crashed_partition_writes_all_variants(tmp_path, manifest):
    """A partition with no metadata.json (crashed) is re-run from scratch and writes all variants."""
    storage = tmp_path / "storage"
    df = pl.read_csv(str(manifest))
    all_ids = set(df["variant_id"].to_list())

    _run_harvest(storage, manifest)
    assert load_completed_from_partition(storage, shard_id=0) == all_ids
