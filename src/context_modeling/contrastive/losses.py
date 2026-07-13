"""Contrastive Losses — InfoNCE and related contrastive learning objectives."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE loss for contrastive pretraining (CLIP-style).

    Args:
        temperature: Temperature for softmax (learnable if not None).

    Usage:
        loss_fn = InfoNCELoss(temperature=0.07)
        loss = loss_fn(ecg_embeddings, text_embeddings)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = nn.Parameter(torch.tensor(temperature))

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute symmetric InfoNCE loss.

        Args:
            z1, z2: L2-normalized embeddings (B, D).

        Returns:
            Scalar loss.
        """
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)

        # Similarity matrix
        logits = (z1 @ z2.T) / self.temperature  # (B, B)

        # Labels: diagonal is positive
        labels = torch.arange(logits.shape[0], device=logits.device)

        loss1 = F.cross_entropy(logits, labels)
        loss2 = F.cross_entropy(logits.T, labels)

        return (loss1 + loss2) / 2.0


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss — leverages label information.

    Positive pairs are samples with the same label.

    Args:
        temperature: Temperature parameter.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Supervised contrastive loss.

        Args:
            features: (B, D) L2-normalized features.
            labels: (B,) class indices or (B, C) multi-hot.

        Returns:
            Scalar loss.
        """
        features = F.normalize(features, dim=-1)
        sim = features @ features.T / self.temperature  # (B, B)

        # Positive mask: same label
        if labels.dim() == 1:
            pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        else:
            # Multi-label: share at least one label
            pos_mask = (labels @ labels.T) > 0

        # Remove self
        pos_mask = pos_mask.fill_diagonal_(False)

        # Compute loss
        exp_sim = torch.exp(sim)
        pos_sum = (exp_sim * pos_mask.float()).sum(dim=1)
        neg_sum = (exp_sim * (~pos_mask).float()).sum(dim=1)

        loss = -torch.log(pos_sum / (pos_sum + neg_sum + 1e-8))
        return loss[pos_sum > 0].mean()
