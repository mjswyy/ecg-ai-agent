"""Evaluate a trained model checkpoint on the test set."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    import torch
    from src.ecg_models.backbone.xresnet1d import xresnet1d_101
    from src.ecg_models.backbone.inception_time import inception_time
    from src.ecg_models.backbone.transformer_encoder import ecg_transformer
    from src.ecg_models.classifiers.arrhythmia_classifier import ArrhythmiaClassifier
    from src.ecg_models.trainer import ECGTrainer
    from src.data_pipeline.dataset import ECGDataModule
    from src.data_pipeline.label_extractor import LabelExtractor

    BACKBONES = {
        "xresnet1d_101": xresnet1d_101,
        "inception_time": inception_time,
        "ecg_transformer": ecg_transformer,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/xresnet1d_101/best_model.pt")
    parser.add_argument("--backbone", default="xresnet1d_101", choices=list(BACKBONES))
    parser.add_argument("--data-dir", default="data/physionet2020/processed")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    backbone_fn = BACKBONES[args.backbone]
    backbone = backbone_fn(in_channels=12, dropout=args.dropout)
    model = ArrhythmiaClassifier(backbone, num_classes=27)

    trainer = ECGTrainer(model, device=args.device)
    trainer.load_checkpoint(args.checkpoint)

    # num_workers=0 for Windows multiprocessing compatibility
    dm = ECGDataModule(
        args.data_dir, batch_size=args.batch_size,
        num_workers=0, label_extractor=LabelExtractor()
    )
    dm.setup()
    metrics = trainer.evaluate(dm.test_dataloader())

    print("\n" + "=" * 50)
    print("Test Set Evaluation")
    print("=" * 50)
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("=" * 50)


if __name__ == '__main__':
    main()
