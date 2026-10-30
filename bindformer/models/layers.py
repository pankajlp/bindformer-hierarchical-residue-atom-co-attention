from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BiasedTransformerLayer(nn.Module):
    """Pre-norm transformer encoder layer whose self-attention accepts an additive `[B, H, N, N]` bias.

    Same train and eval code path (unlike `nn.TransformerEncoderLayer`'s fast path), so
    padding and per-head biases behave identically in both modes.
    """

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float, ffn_mult: int = 4):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_mult, hidden_dim),
        )
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor | None = None) -> torch.Tensor:
        bsz, n, dim = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(bsz, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        if attn_bias is not None:
            attn_bias = attn_bias.to(q.dtype)
        attn = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_bias, dropout_p=self.dropout if self.training else 0.0
        )
        x = x + self.resid_dropout(self.out(attn.transpose(1, 2).reshape(bsz, n, dim)))
        return x + self.resid_dropout(self.ffn(self.norm2(x)))


class BiasedTransformer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        self.num_heads = num_heads
        self.layers = nn.ModuleList([BiasedTransformerLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None, bias: torch.Tensor | None = None) -> torch.Tensor:
        """`mask`: `[B, N]` True for real tokens. `bias`: optional `[B, H, N, N]` additive attention bias."""

        bsz, n, _ = x.shape
        attn_bias = bias
        if mask is not None:
            # Large finite negative (not -inf) keeps fully padded query rows NaN-free.
            key_mask = torch.zeros(bsz, 1, 1, n, dtype=x.dtype, device=x.device)
            key_mask = key_mask.masked_fill(~mask[:, None, None, :], torch.finfo(x.dtype).min / 2)
            attn_bias = key_mask if attn_bias is None else attn_bias + key_mask
        for layer in self.layers:
            x = layer(x, attn_bias)
        return self.final_norm(x)
