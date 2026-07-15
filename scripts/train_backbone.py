#!/usr/bin/env python3
"""Train ECG Backbone — Contrastive pretraining + multi-label fine-tuning.

Usage:
    # Train xResNet1D-101 baseline
    python scripts/train_backbone.py --backbone xresnet1d_101 --epochs 50

    # Train ECG Transformer (innovation backbone)
    python scripts/train_backbone.py --backbone ecg_transformer --epochs 50

    # Contrastive pretraining only
    python scripts/train_backbone.py --backbone inception_time --pretrain_only
"""

import argparse
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.dataset import (
    ECGDataset, ECGContrastiveDataset, ECGDataModule,
)
from src.data_pipeline.augmentor import ECGAugmentor
from src.data_pipeline.label_extractor import LabelExtractor
from src.ecg_models.backbone.xresnet1d import xresnet1d_101
from src.ecg_models.backbone.inception_time import inception_time
from src.ecg_models.backbone.resnet1d import resnet1d_34
from src.ecg_models.backbone.transformer_encoder import ecg_transformer
from src.ecg_models.classifiers.arrhythmia_classifier import (
    ArrhythmiaClassifier, AsymmetricLoss,
)
from src.ecg_models.trainer import ECGTrainer
from src.utils.device_utils import detect_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKBONES = {
    "xresnet1d_101": xresnet1d_101,
    "inception_time": inception_time,
    "resnet1d_34": resnet1d_34,
    "ecg_transformer": ecg_transformer,
}


def main():
    parser = argparse.ArgumentParser(description="Train ECG Backbone")
    parser.add_argument("--backbone", default="inception_time",
                        choices=list(BACKBONES.keys()))
    parser.add_argument("--data-dir", default="/cache/data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="L2 regularization strength (对InceptionTime推荐1e-3)")
    parser.add_argument("--dropout", type=float, default=None,
                        help="Override backbone dropout rate (InceptionTime默认0.1, 过拟合时可升到0.3-0.5)")
    parser.add_argument("--label-smoothing", type=float, default=0.0,
                        help="Label smoothing factor (0=关闭, 推荐0.05-0.1)")
    parser.add_argument("--augment-prob", type=float, default=1.0,
                        help="每种数据增强的独立应用概率 (默认1.0, 减小可降低噪声)")
    parser.add_argument("--grad-noise", type=float, default=0.0,
                        help="梯度噪声标准差 (0=关闭, 推荐0.001-0.01)")
    parser.add_argument("--device", default=detect_device())
    parser.add_argument("--output-dir", default="/cache/output")
    parser.add_argument("--pretrain-only", action="store_true")
    parser.add_argument("--pretrain-epochs", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--quick-test", action="store_true",
                        help="Quick overfitting test on 100 samples")
    parser.add_argument("--patience", type=int, default=10,
                        help="Early stopping patience (0 = disable)")
    args = parser.parse_args()

    device = args.device
    logger.info(f"Device: {device} | Backbone: {args.backbone}")

    # Build model
    backbone_fn = BACKBONES[args.backbone]
    backbone_kwargs = {"in_channels": 12}
    if args.dropout is not None:
        backbone_kwargs["dropout"] = args.dropout
    backbone = backbone_fn(**backbone_kwargs)
    model = ArrhythmiaClassifier(backbone, num_classes=27)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters"
                f" | dropout={backbone_kwargs.get('dropout', 'default')}"
                f" | weight_decay={args.weight_decay}"
                f" | label_smoothing={args.label_smoothing}"
                f" | augment_prob={args.augment_prob}"
                f" | grad_noise={args.grad_noise}")

    # Data
    label_extractor = LabelExtractor()
    augmentor = ECGAugmentor(random_seed=42, apply_prob=args.augment_prob)

    if args.quick_test:
        logger.info("Quick test mode: small epochs for smoke test")
        args.pretrain_epochs = min(args.pretrain_epochs, 3)
        args.epochs = min(args.epochs, 2)

    # === Step 1: Contrastive Pretraining (optional, opt-in) ===
    if args.pretrain_only:
        logger.info("=== Contrastive Pretraining ===")
        contrastive_dataset = ECGContrastiveDataset(
            data_dir=args.data_dir,
            augmentor=ECGAugmentor(segment_shuffle=True, random_seed=42),
            target_length=4096,
        )

        if args.quick_test:
            contrastive_dataset.file_list = contrastive_dataset.file_list[:200]

        contrastive_loader = torch.utils.data.DataLoader(
            contrastive_dataset,
            batch_size=min(args.batch_size, 64),
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )

        trainer = ECGTrainer(
            model.backbone,  # Pretrain backbone only
            device=device,
            output_dir=args.output_dir,
            use_amp=not args.no_amp,
        )
        trainer.train_contrastive(
            contrastive_loader,
            epochs=args.pretrain_epochs,
            lr=args.lr,
        )

        if args.pretrain_only:
            logger.info("Pretraining complete. Exiting.")
            return

    # === Step 2: Multi-label Fine-tuning ===
    logger.info("=== Multi-label Fine-tuning ===")

    dm = ECGDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augmentor=augmentor,
        label_extractor=label_extractor,
    )
    dm.setup()

    trainer = ECGTrainer(
        model,
        device=device,
        output_dir=args.output_dir,
        use_amp=not args.no_amp,
    )

    loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=0.0)

    history = trainer.train_multilabel(
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        epochs=args.epochs,
        loss_fn=loss_fn,
        lr=args.lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        grad_noise=args.grad_noise,
        early_stopping_patience=args.patience,
    )

    # === Step 3: Evaluate on test set ===
    logger.info("=== Test Evaluation ===")
    metrics = trainer.evaluate(dm.test_dataloader())
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
