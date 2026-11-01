from __future__ import annotations

import torch
from torch import nn


class RadialBasisBias(nn.Module):
    """Gaussian radial-basis expansion of pairwise distances -> additive per-head attention bias.

    The binary distance mask in co-attention says only whether two things are within
    the cutoff; this lets attention see *how far apart* they are. Distances are
    expanded on `num_kernels` Gaussians evenly spaced over `[0, cutoff]` and mixed
    into one logit per head.

    `proj` is zero-initialized, so a fresh model emits an all-zero bias and is
    numerically identical to one without it. Any gain therefore comes from training,
    not from a different initialization.
    """

    def __init__(self, num_heads: int, num_kernels: int = 16, cutoff: float = 20.0, learn_centers: bool = False):
        super().__init__()
        if num_kernels < 1:
            raise ValueError("num_kernels must be >= 1")
        self.cutoff = cutoff
        centers = torch.linspace(0.0, cutoff, num_kernels)
        width = cutoff / max(1, num_kernels - 1) if num_kernels > 1 else cutoff
        if learn_centers:
            self.centers = nn.Parameter(centers)
            self.log_width = nn.Parameter(torch.full((num_kernels,), float(torch.log(torch.tensor(width)))))
        else:
            self.register_buffer("centers", centers)
            self.register_buffer("log_width", torch.full((num_kernels,), float(torch.log(torch.tensor(width)))))
        self.proj = nn.Linear(num_kernels, num_heads)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def expand(self, distances: torch.Tensor) -> torch.Tensor:
        """`[B, N, M]` distances -> `[B, N, M, K]` radial-basis features."""

        d = distances.clamp(min=0.0, max=self.cutoff).unsqueeze(-1)
        return torch.exp(-(((d - self.centers) / self.log_width.exp().clamp_min(1e-4)) ** 2))

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """`[B, N, M]` distances -> `[B, H, N, M]` additive attention bias."""

        return self.proj(self.expand(distances)).permute(0, 3, 1, 2)
