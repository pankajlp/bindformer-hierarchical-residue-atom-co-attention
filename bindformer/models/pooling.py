from __future__ import annotations

import torch
from torch import nn


class AttentionPool(nn.Module):
    """Learned pooling over a masked token sequence.

    A scorer assigns one logit per token and the pooled vector is the softmax-weighted
    sum. The scorer's last layer is zero-initialized, so every token starts with an
    equal weight and the layer is *exactly* mean pooling at initialization — uniform
    attention is the mean. Any difference from the masked-mean baseline is learned.

    A single linear scorer trained at the model's own learning rate never left uniform
    (attention entropy stayed at 0.999 of uniform across a full run), which made the
    null result it produced uninformative. `scorer_hidden` gives the scorer a hidden
    layer and `temperature` below 1 sharpens the softmax at the same logit scale; the
    scorer also wants a larger learning rate than the rest of the model, which
    `training.pool_lr_mult` supplies.
    """

    def __init__(self, hidden_dim: int, scorer_hidden: int = 0, temperature: float = 1.0):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if scorer_hidden > 0:
            self.score = nn.Sequential(
                nn.Linear(hidden_dim, scorer_hidden), nn.GELU(), nn.Linear(scorer_hidden, 1)
            )
            last = self.score[-1]
        else:
            self.score = nn.Linear(hidden_dim, 1)
            last = self.score
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
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
