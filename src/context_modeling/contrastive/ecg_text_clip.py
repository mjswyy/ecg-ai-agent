"""ECG-Text CLIP — Contrastive pretraining for ECG and clinical text.

Aligns ECG and text representations in a shared embedding space using
contrastive learning (CLIP-style).

Usage:
    model = ECGTextCLIP(ecg_encoder, text_encoder, proj_dim=256)
    ecg_emb, text_emb = model(ecg_signals, clinical_texts)
    loss = InfoNCELoss()(ecg_emb, text_emb)
"""

import torch
import torch.nn as nn

from .losses import InfoNCELoss


class ECGTextCLIP(nn.Module):
    """ECG-Text CLIP: Joint embedding for ECG signals and clinical text.

    Args:
        ecg_encoder: ECG backbone (from ecg_models.backbone).
        text_encoder: Text encoder (TextEncoder).
        proj_dim: Shared projection dimension.
        temperature: Initial temperature for InfoNCE.
    """

    def __init__(
        self,
        ecg_encoder: nn.Module,
        text_encoder: nn.Module,
        proj_dim: int = 256,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.text_encoder = text_encoder

        # Projection heads
        self.ecg_proj = nn.Sequential(
            nn.Linear(ecg_encoder.feature_dim, proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim * 2, proj_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_encoder.output_dim, proj_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim * 2, proj_dim),
        )

        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / temperature)))

        self.loss_fn = InfoNCELoss(temperature=temperature)

    def encode_ecg(self, ecg: torch.Tensor) -> torch.Tensor:
        """Encode ECG to normalized embedding.

        Args:
            ecg: (B, 12, L) signal.

        Returns:
            (B, proj_dim) normalized embedding.
        """
        features = self.ecg_encoder(ecg)
        return nn.functional.normalize(self.ecg_proj(features), dim=-1)

    def encode_text(self, texts) -> torch.Tensor:
        """Encode text to normalized embedding.

        Args:
            texts: List of strings or (B, ...) tokens.

        Returns:
            (B, proj_dim) normalized embedding.
        """
        if isinstance(texts, (list, tuple)):
            features = self.text_encoder(texts)
        else:
            features = self.text_encoder(texts)
        return nn.functional.normalize(self.text_proj(features), dim=-1)

    def forward(
        self,
        ecg: torch.Tensor,
        texts,
    ) -> tuple:
        """Forward pass: encode both modalities.

        Args:
            ecg: (B, 12, L) ECG signals.
            texts: List of clinical text strings.

        Returns:
            (ecg_emb, text_emb, loss) tuple.
        """
        ecg_emb = self.encode_ecg(ecg)
        text_emb = self.encode_text(texts)
        loss = self.loss_fn(ecg_emb, text_emb)
        return ecg_emb, text_emb, loss
