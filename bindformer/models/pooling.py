from __future__ import annotations

import torch
from torch import nn


class AttentionPool(nn.Module):
    """Learned pooling over a masked token sequence.

    A scorer assigns one logit per token and the pooled vector is the softmax-weighted
    sum. `score` is zero-initialized, so every token starts with an equal weight and
    the layer is *exactly* mean pooling at initialization — uniform attention is the
    mean. Any difference from the masked-mean baseline is therefore learned.
    """

    def __init__(self, hidden_dim: int, temperature: float = 1.0):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        self.temperature = temperature

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(x).squeeze(-1) / self.temperature
        if mask is not None:
            logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min / 2)
        weights = torch.softmax(logits, dim=-1)
        if mask is not None:
            # A row with no valid tokens would otherwise come back as uniform nonsense.
            weights = torch.where(mask, weights, torch.zeros_like(weights))
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            weights = torch.where(mask.any(dim=-1, keepdim=True), weights, torch.zeros_like(weights))
        return torch.einsum("bn,bnd->bd", weights, x)
