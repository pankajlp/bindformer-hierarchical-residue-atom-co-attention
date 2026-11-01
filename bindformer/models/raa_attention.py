from __future__ import annotations

import math

import torch
from torch import nn

from bindformer.models.rbf import RadialBasisBias


class ResidueAtomAttentionLayer(nn.Module):
    """Bidirectional sparse co-attention between residues and atoms."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = nn.Dropout(dropout)
        self.qp = nn.Linear(hidden_dim, hidden_dim)
        self.kp = nn.Linear(hidden_dim, hidden_dim)
        self.vp = nn.Linear(hidden_dim, hidden_dim)
        self.ql = nn.Linear(hidden_dim, hidden_dim)
        self.kl = nn.Linear(hidden_dim, hidden_dim)
        self.vl = nn.Linear(hidden_dim, hidden_dim)
        self.protein_out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))
        self.ligand_out = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout), nn.LayerNorm(hidden_dim))

    def forward(
        self,
        protein: torch.Tensor,
        ligand: torch.Tensor,
        attention_mask: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`bias`: optional `[B, H, residues, atoms]` additive bias; transposed for the atom->residue pass."""

        p_context, p_attn = self._cross_attention(
            self.qp(protein), self.kl(ligand), self.vl(ligand), attention_mask, bias
        )
        l_context, _ = self._cross_attention(
            self.ql(ligand),
            self.kp(protein),
            self.vp(protein),
            attention_mask.transpose(1, 2),
            None if bias is None else bias.transpose(-1, -2),
        )
        protein = self.protein_out(protein + p_context)
        ligand = self.ligand_out(ligand + l_context)
        return protein, ligand, p_attn

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        bsz, length, _ = x.shape
        return x.view(bsz, length, self.num_heads, self.head_dim).transpose(1, 2)

    def _cross_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        allowed_mask: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        qh, kh, vh = self._shape(q), self._shape(k), self._shape(v)
        logits = torch.matmul(qh, kh.transpose(-1, -2)) / math.sqrt(self.head_dim)
        if bias is not None:
            logits = logits + bias.to(logits.dtype)
        mask = allowed_mask.unsqueeze(1)
        has_any = mask.any(dim=-1, keepdim=True)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(has_any, weights, torch.zeros_like(weights))
        weights = self.dropout(weights)
        context = torch.matmul(weights, vh).transpose(1, 2).contiguous().view(q.shape)
        return context, weights.mean(dim=1)


class ResidueAtomAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_layers: int,
        dropout: float,
        rbf_kernels: int = 0,
        rbf_cutoff: float = 10.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [ResidueAtomAttentionLayer(hidden_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.rbf = RadialBasisBias(num_heads, rbf_kernels, rbf_cutoff) if rbf_kernels > 0 else None

    def forward(
        self,
        protein: torch.Tensor,
        ligand: torch.Tensor,
        attention_mask: torch.Tensor,
        distances: torch.Tensor | None = None,
    ):
        bias = None
        if self.rbf is not None and distances is not None:
            bias = self.rbf(distances)
        attentions = []
        for layer in self.layers:
            protein, ligand, attn = layer(protein, ligand, attention_mask, bias)
            attentions.append(attn)
        return protein, ligand, attentions

