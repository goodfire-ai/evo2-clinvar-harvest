"""Shared loaders for variant dataset modules.

Encapsulates the metadata/activations/embeddings boilerplate that all variant
dataset modules (clinvar, dms, mavedb, opentargets) share. Each module keeps
its own explicit function signatures and delegates in one line.
"""

from __future__ import annotations

from collections.abc import Callable

import polars as pl
from goodfire_core.storage import ActivationDataset, FilesystemStorage

from src.datasets.paths import activations_dir, embeddings_dir

from . import gencode
from .utils import enrich_variants, mutate_variants, with_cache


def metadata(
    module: str,
    presets: dict[str, Callable[[], pl.DataFrame] | None],
    preset: str,
    unique_on: tuple[str, ...] = ("variant_id",),
) -> pl.DataFrame:
    """Resolve preset, cache, and enrich with GENCODE coordinates."""
    assert preset in presets, f"Unknown preset '{preset}'. Available: {sorted(presets)}"
    builder = presets[preset]
    df = with_cache(module, preset, builder)
    return enrich_variants(df, gencode.metadata(sequences=False), unique_on=unique_on)


def activations(
    module: str,
    df: pl.DataFrame,
    preset: str,
    model: str,
    sequences: bool = False,
    unique_on: tuple[str, ...] = ("variant_id",),
) -> tuple[ActivationDataset, pl.DataFrame]:
    """Load variant activations and enriched metadata.

    Returns:
        (acts, metadata) tuple. Use .training_iterator() for streaming batch access.
    """
    if sequences:
        df = mutate_variants(df, gencode.metadata(), unique_on=unique_on)
    path = activations_dir(module, preset, model)
    acts = ActivationDataset(FilesystemStorage(path.parent), path.name, include_provenance=True)
    return acts, df


def embeddings(
    module: str,
    df: pl.DataFrame,
    preset: str,
    model: str,
    pooler: str,
) -> tuple[ActivationDataset, pl.DataFrame]:
    """Load variant embeddings and enriched metadata.

    Returns:
        (acts, metadata) tuple. Use .training_iterator() for streaming batch access.
    """
    path = embeddings_dir(module, preset, model, pooler)
    return ActivationDataset(FilesystemStorage(path.parent), path.name, include_provenance=True), df
