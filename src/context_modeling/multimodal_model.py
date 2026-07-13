"""Multi-modal Model — Unified ECG + Text + Metadata classification model.

Combines ECG backbone, text encoder, metadata encoder, and fusion module
into a single end-to-end model for diagnosis classification.

Usage:
    model = MultimodalModel(ecg_encoder, text_encoder, fusion, num_classes=27)
    logits = model(ecg, texts, ages, sexes)
"""

import torch
import torch.nn as nn


class MultimodalModel(nn.Module):
    """Multi-modal ECG diagnosis model.

    Args:
        ecg_encoder: ECG backbone or SimpleECGProjector.
        text_encoder: Text encoder (TextEncoder).
        metadata_encoder: Metadata encoder (MetadataEncoder).
        fusion: Fusion module (CrossAttentionFusion/GatedFusion/LateFusion).
        num_classes: Number of output classes (27).
        dropout: Classification head dropout.
    """

    def __init__(
        self,
        ecg_encoder: nn.Module,
        text_encoder: nn.Module,
        metadata_encoder: nn.Module,
        fusion: nn.Module,
        num_classes: int = 27,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.ecg_encoder = ecg_encoder
        self.text_encoder = text_encoder
        self.metadata_encoder = metadata_encoder
        self.fusion = fusion

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion.output_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(
        self,
        ecg: torch.Tensor,
        texts=None,
        ages: torch.Tensor = None,
        sexes: torch.Tensor = None,
        sources=None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            ecg: (B, 12, L) ECG signals.
            texts: List of clinical text strings (or None).
            ages: (B,) normalized ages (-1.0 = unknown).
            sexes: (B,) sex indices (0=Female, 1=Male, 2=Unknown).
            sources: List of data source names.

        Returns:
            (B, num_classes) logits.
        """
        # ECG features
        ecg_feat = self.ecg_encoder(ecg)  # (B, ecg_dim) or (B, seq, ecg_dim)

        # Text features
        if texts is not None:
            text_feat = self.text_encoder(texts)  # (B, text_dim)
        else:
            text_feat = torch.zeros(
                ecg_feat.shape[0], self.text_encoder.output_dim,
                device=ecg.device,
            )

        # Metadata features
        meta_feat = self.metadata_encoder(
            age=ages, sex=sexes, source=sources,
        )  # (B, meta_dim)

        # Fusion
        fused = self.fusion(ecg_feat, text_feat, meta_feat)

        # Classify
        return self.classifier(fused)
