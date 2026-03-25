"""GENCODE gene coordinates and chromosome sequences."""

from __future__ import annotations

import polars as pl

from .. import paths
from ..utils import CHROMOSOMES

PRESETS: dict[str, None] = {}


def metadata(sequences: bool = True) -> pl.DataFrame:
    """Gene coordinates and optionally sequences.

    Gene IDs are unversioned (ENSG00000139618), chromosomes have no prefix (17).
    Normalization happens at setup time (gencode.setup), not here.

    Args:
        sequences: Include DNA sequences (default True). False skips the large
            sequence column for performance.

    Example:
        >>> genes = gencode.metadata()
        >>> genes = gencode.metadata(sequences=False)
    """
    columns = None if sequences else ("gene_id", "gene_name", "chrom", "start", "end", "strand", "length", "hgnc_id", "level")
    df = pl.read_ipc(paths.GENCODE_GENES, columns=columns)
    return df.filter(pl.col("chrom").is_in(CHROMOSOMES))


def chromosomes() -> dict[str, str]:
    """Full chromosome sequences keyed by chromosome name (1-22, X, Y, M).

    Built once from GRCh38 FASTA during setup, cached as feather.
    Sequences are upper-case, + strand.

    Example:
        >>> genome = gencode.chromosomes()
        >>> genome["17"][7_500_000:7_500_100]  # 100bp window on chr17
    """
    df = pl.read_ipc(paths.GENCODE_CHROMOSOMES)
    return dict(zip(df["chrom"].to_list(), df["sequence"].to_list(), strict=True))
