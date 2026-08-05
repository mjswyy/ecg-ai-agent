"""ECGFounder Fine-tuning — 在 30K 数据上微调 10M+ 预训练模型."""
import sys
from pathlib import Path

sys.path.insert(0, r'C:\Users\llyun\Desktop\ecg资料\GitHub上的一些项目\ECGFounder')
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import logging
import numpy as np
import torch
import torch.nn as nn
from net1d import Net1D

from src.data_pipeline.dataset import ECGDataModule
from src.data_pipeline.augmentor import ECGAugmentor
from src.data_pipeline.label_extractor import LabelExtractor
from src.ecg_models.trainer import ECGTrainer
from src.utils.device_utils import detect_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ECGFounderClassifier(nn.Module):
    """ECGFounder backbone + 27-class head for fine-tuning."""

    def __init__(self, pretrained_path, n_classes=27, dropout=0.3):
        super().__init__()
        # Build original 150-class model
        self.backbone = Net1D(
            in_channels=12, base_filters=64, ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16, stride=2, groups_width=16,
            n_classes=150, use_bn=False, use_do=False, verbose=False,
        )

        # Load pretrained weights
        ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
        self.backbone.load_state_dict(ckpt["state_dict"], strict=True)
        logger.info(f"Loaded ECGFounder from epoch {ckpt.get('epoch', '?')}")

        # Replace head: 150 → 27 classes with learnable BN + Dropout
        feat_dim = ckpt["state_dict"]["dense.weight"].shape[1]  # 1024
        self.backbone.dense = nn.Identity()  # Remove old head

        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # Pad 4096 → 5000 (model trained on 5000-sample inputs)
        if x.shape[-1] < 5000:
            pad = 5000 - x.shape[-1]
            x = nn.functional.pad(x, (pad // 2, pad - pad // 2))
        elif x.shape[-1] > 5000:
            x = x[..., :5000]

        feat = self.backbone(x)  # returns (B, 1024) after swapping dense to Identity
        return self.head(feat)

    def predict(self, x):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained", default="checkpoints/ECGFounder/12_lead_ECGFounder.pth")
    parser.add_argument("--data-dir", default="/cache/data/processed")
    parser.add_argument("--device", default=detect_device())
    parser.add_argument("--output-dir", default="/cache/output/ecgfounder_ft")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="只训练分类头（线性探针模式）")
    args = parser.parse_args()

    device = args.device
    logger.info(f"Device: {device} | Fine-tuning ECGFounder")

    # Build model
    model = ECGFounderClassifier(args.pretrained, dropout=args.dropout)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total: {total_params:,} params | Trainable: {trainable_params:,}")

    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Frozen backbone. Trainable: {trainable_params:,}")

    # Data
    le = LabelExtractor()
    augmentor = ECGAugmentor(random_seed=42, apply_prob=0.8)
    dm = ECGDataModule(args.data_dir, batch_size=args.batch_size,
                       num_workers=args.num_workers,
                       augmentor=augmentor, label_extractor=le)
    dm.setup()

    # Trainer
    trainer = ECGTrainer(model, device=device, output_dir=args.output_dir,
                         use_amp=not args.no_amp)

    history = trainer.train_multilabel(
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        epochs=args.epochs,
        loss_fn=nn.BCEWithLogitsLoss(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        early_stopping_patience=10,
    )

    # Test
    logger.info("=" * 60)
    logger.info("Test Evaluation")
    logger.info("=" * 60)
    metrics = trainer.evaluate(dm.test_dataloader())
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
