#!/usr/bin/env python3
# ⚠️ DEPRECATED（2026-08-24，检查报告第六步）: 旧 NPU SimCLR 预训练（0.841 历史对照，见论文 §4.1 † 注记；代码已按检查报告 1.4/1.5 修复）
"""SimCLR Pretraining + Fine-tuning for ECG backbone.

Phase 1: Self-supervised contrastive pretraining on train split only (~30K ECGs, no labels; 泄漏红线)
Phase 2: Multi-label fine-tuning on 30K labeled ECGs
Phase 3: Test evaluation

Usage:
    python scripts/train_simclr.py --backbone xresnet1d_101 --device auto
"""

import argparse
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.data_pipeline.dataset import ECGContrastiveDataset, ECGDataModule
from src.data_pipeline.augmentor import ECGAugmentor
from src.data_pipeline.label_extractor import LabelExtractor
from src.ecg_models.backbone.xresnet1d import xresnet1d_101
from src.ecg_models.backbone.transformer_encoder import ecg_transformer
from src.ecg_models.classifiers.arrhythmia_classifier import ArrhythmiaClassifier
from src.ecg_models.trainer import ECGTrainer
from src.utils.device_utils import detect_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BACKBONES = {
    "xresnet1d_101": xresnet1d_101,
    "ecg_transformer": ecg_transformer,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="xresnet1d_101", choices=list(BACKBONES))
    parser.add_argument("--data-dir", default="/cache/data/processed")
    parser.add_argument("--device", default=detect_device())
    parser.add_argument("--output-dir", default="/cache/output")
    # SimCLR
    parser.add_argument("--simclr-epochs", type=int, default=100)
    parser.add_argument("--simclr-lr", type=float, default=1e-4)
    parser.add_argument("--simclr-batch", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.07)
    # Fine-tuning
    parser.add_argument("--ft-epochs", type=int, default=50)
    parser.add_argument("--ft-lr", type=float, default=5e-5)
    parser.add_argument("--ft-batch", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    args = parser.parse_args()

    # 检查报告 1.9 修复：补齐随机种子（torch/numpy/python），保证可复现
    import random
    import numpy as np
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = args.device
    logger.info(f"Device: {device} | Backbone: {args.backbone} | seed={seed}")

    # Output dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pretrain_path = out_dir / "simclr_pretrained.pt"

    # ============================================================
    # Phase 1: SimCLR Contrastive Pretraining
    # ============================================================

    if not args.skip_pretrain:
        logger.info("=" * 60)
        logger.info("Phase 1: SimCLR Contrastive Pretraining")
        # 4E/4F 审查修复：🟡-6 —— 实际仅用 train 划分（约 30K，泄漏红线），修正日志
        logger.info(f"  train-split unlabeled ECGs (~30K, 泄漏红线) | {args.simclr_epochs} epochs | lr={args.simclr_lr}")
        logger.info("=" * 60)

        # Build backbone (no classification head, pretrain backbone only)
        backbone_fn = BACKBONES[args.backbone]
        backbone = backbone_fn(in_channels=12)

        # Contrastive dataset: two augmented views per sample
        contrastive_ds = ECGContrastiveDataset(
            data_dir=args.data_dir,
            augmentor=ECGAugmentor(segment_shuffle=True, random_seed=42),
            target_length=4096,
        )
        logger.info(f"Contrastive samples: {len(contrastive_ds)}")

        contrastive_loader = torch.utils.data.DataLoader(
            contrastive_ds,
            batch_size=min(args.simclr_batch, 64),
            shuffle=True, num_workers=args.num_workers,
            pin_memory=True, drop_last=True,
        )

        pretrain_trainer = ECGTrainer(
            backbone, device=device,
            output_dir=str(out_dir / "simclr"),
            use_amp=not args.no_amp,
        )
        pretrain_trainer.train_contrastive(
            contrastive_loader,
            epochs=args.simclr_epochs,
            lr=args.simclr_lr,
            temperature=args.temperature,
        )

        # Save pretrained backbone
        torch.save(backbone.state_dict(), pretrain_path)
        logger.info(f"Pretrained backbone saved: {pretrain_path}")
    else:
        logger.info("Skipping pretraining, loading from checkpoint")
        pretrain_path = out_dir / "simclr_pretrained.pt"

    # ============================================================
    # Phase 2: Multi-label Fine-tuning
    # ============================================================

    logger.info("=" * 60)
    logger.info("Phase 2: Multi-label Fine-tuning")
    logger.info(f"  30K labeled ECGs | {args.ft_epochs} epochs | lr={args.ft_lr}")
    logger.info("=" * 60)

    # Build model with pretrained backbone
    backbone_fn = BACKBONES[args.backbone]
    backbone = backbone_fn(in_channels=12, dropout=args.dropout)
    if pretrain_path.exists():
        # 3F-YELLOW-8 修复：与全仓 safe_torch_load 口径一致
        # （旧版裸 torch.load 在 torch≥2.6 默认 weights_only=True 下行为不同）
        from src.utils.safe_load import safe_torch_load
        backbone.load_state_dict(safe_torch_load(pretrain_path, map_location="cpu"))
        logger.info(f"Loaded pretrained backbone from {pretrain_path}")
    else:
        # 第三轮审查 3F-R3-06：--skip-pretrain 但预训练权重缺失时必须显式失败
        # （旧版静默以随机初始化继续微调，用户误以为加载了预训练权重）
        logger.error(f"预训练权重不存在: {pretrain_path}（无法按 --skip-pretrain 继续）")
        sys.exit(1)

    model = ArrhythmiaClassifier(backbone, num_classes=27)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Labeled data
    label_extractor = LabelExtractor()
    augmentor = ECGAugmentor(random_seed=42, apply_prob=0.8)
    dm = ECGDataModule(args.data_dir, batch_size=args.ft_batch,
                       num_workers=args.num_workers,
                       augmentor=augmentor, label_extractor=label_extractor)
    dm.setup()

    trainer = ECGTrainer(
        model, device=device, output_dir=str(out_dir),
        use_amp=not args.no_amp,
    )

    history = trainer.train_multilabel(
        train_loader=dm.train_dataloader(),
        val_loader=dm.val_dataloader(),
        epochs=args.ft_epochs,
        loss_fn=torch.nn.BCEWithLogitsLoss(),
        lr=args.ft_lr,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
    )

    # ============================================================
    # Phase 3: Test Evaluation
    # ============================================================

    logger.info("=" * 60)
    logger.info("Phase 3: Test Evaluation")
    logger.info("=" * 60)

    metrics = trainer.evaluate(dm.test_dataloader(),
                               thresholds=trainer.val_thresholds)  # 2E-O7：验证集阈值
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
