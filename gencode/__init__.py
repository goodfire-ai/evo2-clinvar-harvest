"""GENCODE gene coordinates and chromosome sequences."""

from __future__ import annotations

import polars as pl

from utils import CHROMOSOMES, GENCODE_CHROMOSOMES, GENCODE_GENES

PRESETS: dict[str, None] = {}


def metadata(sequences: bool = True) -> pl.DataFrame:
    """Gene coordinates and optionally sequences."""
    columns = None if sequences else ("gene_id", "gene_name", "chrom", "start", "end", "strand", "length", "hgnc_id", "level")
    return pl.read_ipc(GENCODE_GENES, columns=columns).filter(pl.col("chrom").is_in(CHROMOSOMES))


def chromosomes() -> dict[str, str]:
    """Full chromosome sequences keyed by chromosome name (1-22, X, Y, M)."""
    df = pl.read_ipc(GENCODE_CHROMOSOMES)
    return dict(zip(df["chrom"].to_list(), df["sequence"].to_list(), strict=True))
