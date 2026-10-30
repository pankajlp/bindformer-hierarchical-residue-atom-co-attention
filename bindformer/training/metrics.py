from __future__ import annotations

import math

import torch


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.sqrt(torch.mean((pred - target) ** 2)).item())


def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean((pred - target).abs()).item())


def pearson_r(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = pred.detach().flatten().double()
    target = target.detach().flatten().double()
    if pred.numel() < 2:
        return float("nan")
    pred = pred - pred.mean()
    target = target - target.mean()
    denom = torch.sqrt((pred.square().sum() * target.square().sum()).clamp_min(1e-12))
    return float((pred * target).sum().div(denom).item())


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Average ranks, so ties get the mean of the ranks they span."""

    x = x.detach().flatten().double()
    sorted_x, order = torch.sort(x)
    ranks = torch.empty_like(x)
    _, inverse, counts = torch.unique_consecutive(sorted_x, return_inverse=True, return_counts=True)
    ends = torch.cumsum(counts, 0).double()
    mean_rank = ends - (counts.double() + 1) / 2
    ranks[order] = mean_rank[inverse]
    return ranks


def spearman_rho(pred: torch.Tensor, target: torch.Tensor) -> float:
    return pearson_r(_rank(pred), _rank(target))


def casf_sd(pred: torch.Tensor, target: torch.Tensor) -> float:
    """CASF scoring-power SD: residual std of the linear fit target ~ a * pred + b."""

    x = pred.detach().flatten().double()
    y = target.detach().flatten().double()
    n = x.numel()
    if n < 3:
        return float("nan")
    xm, ym = x.mean(), y.mean()
    var_x = (x - xm).square().sum()
    a = ((x - xm) * (y - ym)).sum() / var_x.clamp_min(1e-12)
    b = ym - a * xm
    resid = y - (a * x + b)
    return float(torch.sqrt(resid.square().sum() / (n - 1)).item())


def regression_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "rmse": rmse(pred, target),
        "mae": mae(pred, target),
        "pearson_r": pearson_r(pred, target),
        "spearman_rho": spearman_rho(pred, target),
        "sd": casf_sd(pred, target),
    }


def top_fraction_success(scores: torch.Tensor, rmsd: torch.Tensor, threshold: float = 2.0, fraction: float = 0.01) -> float:
    if scores.numel() == 0:
        return math.nan
    k = max(1, int(scores.numel() * fraction))
    top_idx = torch.topk(scores, k=k, largest=True).indices
    return float((rmsd[top_idx] <= threshold).float().mean().item())
