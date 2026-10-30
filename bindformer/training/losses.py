from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class BindFormerLoss(nn.Module):
    """Affinity MSE plus an optional residue-atom contact BCE.

    `contact_pos_weight` up-weights the rare positive contacts (roughly 1-5% of pocket pairs).
    """

    def __init__(self, affinity_weight: float = 1.0, contact_weight: float = 0.2, contact_pos_weight: float = 10.0):
        super().__init__()
        self.affinity_weight = affinity_weight
        self.contact_weight = contact_weight
        self.contact_pos_weight = contact_pos_weight

    def forward(self, outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        affinity_loss = F.mse_loss(outputs["affinity"].float(), batch["affinity"])
        losses = {"loss": self.affinity_weight * affinity_loss, "affinity_loss": affinity_loss.detach()}

        if "contacts" in batch and self.contact_weight > 0:
            contact_mask = batch["protein_mask"].unsqueeze(-1) & batch["ligand_mask"].unsqueeze(1)
            pos_weight = torch.tensor(self.contact_pos_weight, device=batch["contacts"].device)
            contact_loss = F.binary_cross_entropy_with_logits(
                outputs["contact_logits"][contact_mask].float(),
                batch["contacts"][contact_mask],
                pos_weight=pos_weight,
            )
            losses["loss"] = losses["loss"] + self.contact_weight * contact_loss
            losses["contact_loss"] = contact_loss.detach()

        return losses
