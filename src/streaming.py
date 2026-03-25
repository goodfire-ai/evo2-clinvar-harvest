"""Streaming iteration over activation datasets with transforms.

Example:
    >>> from src.streaming import iter_dataset, unified_diff
    >>> for acts, ids in iter_dataset(storage, "activations", target_ids, unified_diff):
    ...     train_step(acts, ids)
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch
from goodfire_core.storage import ActivationDataset, FilesystemStorage
from torch import Tensor


def iter_dataset(
    storage: FilesystemStorage,
    dataset_name: str,
    target_ids: set[str],
    transform: Callable[[Tensor], Tensor] | None = None,
    *,
    batch_size: int = 512,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> Iterator[tuple[Tensor, list[str]]]:
    """Stream one dataset with optional per-batch transform.

    Yields (acts, ids) where acts is on device after transform.
    """
    ds = ActivationDataset(storage, dataset_name, batch_size=batch_size, include_provenance=True)
    for batch in ds.training_iterator(
        device=device, n_epochs=1, shuffle=False, drop_last=False, sequence_ids=list(target_ids),
    ):
        x = batch.acts.to(dtype=dtype)
        if transform is not None:
            x = transform(x)
        yield x, batch.sequence_ids


# ── Transforms ────────────────────────────────────────────────────────────
# Activation layout: [B, direction, view, K, d]
#   direction: 0=fwd, 1=bwd
#   view: 0=var_same, 1=ref_same, 2=ref_cross


def unified_diff(x: Tensor) -> Tensor:
    """[B, 2, 3, K, d] -> [B, K, 2*d]: var-ref diff, concat fwd+bwd."""
    diff = x[:, :, 0] - x[:, :, 1]
    return torch.cat([diff[:, 0], diff[:, 1]], dim=-1)


def unified_ref(x: Tensor) -> Tensor:
    """[B, 2, 3, K, d] -> [B, K, 2*d]: reference activations, concat fwd+bwd."""
    ref = x[:, :, 1]
    return torch.cat([ref[:, 0], ref[:, 1]], dim=-1)


def unified_var(x: Tensor) -> Tensor:
    """[B, 2, 3, K, d] -> [B, K, 2*d]: variant activations, concat fwd+bwd."""
    var = x[:, :, 0]
    return torch.cat([var[:, 0], var[:, 1]], dim=-1)
