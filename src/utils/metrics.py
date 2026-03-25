"""Evaluation metrics for multi-label classification and regression.

All functions support arbitrary leading batch dimensions:
    - Input shape: (..., N, L) where N=samples, L=labels/tasks
    - Output shape: (..., L) for per-task, (...) for macro-averaged

Design: per-task functions are the base; macro versions just add .mean(-1).
"""

import torch
from torch import Tensor

# ============================================================================
# Classification metrics (per-task base implementations)
# ============================================================================


@torch.inference_mode()
def auroc(probs: Tensor, labels: Tensor, reduction: str = "mean") -> Tensor:
    """Compute AUROC (Area Under ROC Curve).

    Args:
        probs: Prediction probabilities, shape (..., N, L)
        labels: Binary labels, shape (..., N, L)
        reduction: "none" for per-task (..., L), "mean" for macro-averaged (...)

    Returns:
        AUROC scores. Returns 0.5 for degenerate labels (all pos or all neg).
    """
    n_samples = probs.shape[-2]
    n_labels = probs.shape[-1]
    batch_shape = probs.shape[:-2]

    sorted_idx = probs.argsort(dim=-2, descending=True)
    y_sorted = labels.gather(-2, sorted_idx).bool()

    n_pos = labels.sum(-2)
    n_neg = n_samples - n_pos

    tps = y_sorted.cumsum(-2).float()
    fps = (~y_sorted).cumsum(-2).float()

    tpr = tps / n_pos.unsqueeze(-2).clamp(min=1)
    fpr = fps / n_neg.unsqueeze(-2).clamp(min=1)

    # Trapezoidal integration
    zeros = torch.zeros(*batch_shape, 1, n_labels, device=probs.device, dtype=fpr.dtype)
    fpr_diff = torch.diff(fpr, dim=-2, prepend=zeros)
    aucs = (fpr_diff * tpr).sum(-2)

    # NaN for degenerate batches (all-positive or all-negative) — avoids
    # spiky 0.5 values in logged metrics with imbalanced class distributions.
    valid = (n_pos > 0) & (n_neg > 0)
    aucs = torch.where(valid, aucs, torch.full_like(aucs, float("nan")))

    return aucs.mean(-1) if reduction == "mean" else aucs


@torch.inference_mode()
def aupr(probs: Tensor, labels: Tensor, reduction: str = "mean") -> Tensor:
    """Compute AUPR (Area Under Precision-Recall Curve / Average Precision).

    Args:
        probs: Prediction probabilities, shape (..., N, L)
        labels: Binary labels, shape (..., N, L)
        reduction: "none" for per-task (..., L), "mean" for macro-averaged (...)

    Returns:
        AUPR scores. Returns 0 for labels with no positives.
    """
    n_samples = probs.shape[-2]

    sorted_idx = probs.argsort(dim=-2, descending=True)
    y_sorted = labels.gather(-2, sorted_idx).float()

    n_pos = labels.sum(-2)

    tps = y_sorted.cumsum(-2)
    ranks = torch.arange(1, n_samples + 1, device=probs.device, dtype=torch.float32)
    ranks = ranks.view(*([1] * (probs.dim() - 2)), n_samples, 1)
    precision = tps / ranks
    aps = (precision * y_sorted).sum(-2) / n_pos.clamp(min=1)

    valid = n_pos > 0
    aps = torch.where(valid, aps, torch.zeros_like(aps))

    return aps.mean(-1) if reduction == "mean" else aps


@torch.inference_mode()
def f_max(probs: Tensor, labels: Tensor, reduction: str = "mean", n_thresholds: int = 50) -> Tensor:
    """Compute F_max (best F1 over thresholds).

    Args:
        probs: Prediction probabilities, shape (..., N, L)
        labels: Binary labels, shape (..., N, L)
        reduction: "none" for per-task (..., L), "mean" for micro-averaged (...)
        n_thresholds: Number of thresholds to search

    Returns:
        F_max scores.
    """
    thresholds = torch.linspace(0.01, 0.99, n_thresholds, device=probs.device)
    labels_bool = labels.bool()

    # Expand thresholds for broadcasting: (T, 1, ..., 1, 1, 1)
    thresh_shape = (n_thresholds,) + (1,) * probs.dim()
    preds = probs.unsqueeze(0) > thresholds.view(*thresh_shape)

    if reduction == "mean":
        # Micro-averaged: sum over both samples and labels
        tp = (preds & labels_bool).sum(dim=(-2, -1)).float()
        fp = (preds & ~labels_bool).sum(dim=(-2, -1)).float()
        fn = (~preds & labels_bool).sum(dim=(-2, -1)).float()
    else:
        # Per-task: sum over samples only
        tp = (preds & labels_bool).sum(dim=-2).float()
        fp = (preds & ~labels_bool).sum(dim=-2).float()
        fn = (~preds & labels_bool).sum(dim=-2).float()

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    return f1.max(dim=0).values


# ============================================================================
# Regression metrics
# ============================================================================


def r2_score(preds: Tensor, targets: Tensor) -> Tensor:
    """Per-task R² score.

    Args:
        preds: Predictions, shape (N,), (..., N), or (..., N, L)
        targets: True values, same shape as preds

    Returns:
        R² per task. Shape: scalar for (N,), (...) for (..., N), (..., L) for (..., N, L).
        Returns 0 for constant-target columns.
    """
    squeezed = preds.dim() == 1
    if squeezed:
        preds, targets = preds.unsqueeze(-1), targets.unsqueeze(-1)

    ss_res = ((targets - preds) ** 2).sum(-2)
    ss_tot = ((targets - targets.mean(-2, keepdim=True)) ** 2).sum(-2)
    valid = ss_tot > 1e-10
    result = torch.where(valid, 1 - ss_res / ss_tot.clamp(min=1e-10), torch.zeros_like(ss_tot))
    return result.squeeze(-1) if squeezed else result


def pearson_corr(preds: Tensor, targets: Tensor) -> Tensor:
    """Per-task Pearson correlation.

    Args:
        preds: Predictions, shape (N,), (..., N), or (..., N, L)
        targets: True values, same shape as preds

    Returns:
        Pearson r per task. Shape: scalar for (N,), (...) for (..., N), (..., L) for (..., N, L).
        Returns 0 for constant columns.
    """
    squeezed = preds.dim() == 1
    if squeezed:
        preds, targets = preds.unsqueeze(-1), targets.unsqueeze(-1)

    preds_c = preds - preds.mean(-2, keepdim=True)
    targets_c = targets - targets.mean(-2, keepdim=True)
    cov = (preds_c * targets_c).sum(-2)
    denom = preds_c.norm(dim=-2) * targets_c.norm(dim=-2)
    valid = denom > 1e-10
    result = torch.where(valid, cov / denom.clamp(min=1e-10), torch.zeros_like(cov))
    return result.squeeze(-1) if squeezed else result


