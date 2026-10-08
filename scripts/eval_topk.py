# ⚠️ DEPRECATED（2026-08-24，检查报告第六步）: 旧 Top-K 评估脚本
"""Top-K Accuracy — how often the top-K predictions hit at least one ground truth label."""
from src.utils.safe_load import safe_torch_load
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from src.data_pipeline.dataset import ECGDataModule
from src.data_pipeline.label_extractor import LabelExtractor
from src.ecg_models.backbone.xresnet1d import xresnet1d_101
from src.ecg_models.backbone.inception_time import inception_time
from src.ecg_models.backbone.transformer_encoder import ecg_transformer
from src.ecg_models.classifiers.arrhythmia_classifier import ArrhythmiaClassifier


def load_model(backbone_fn, checkpoint, device, dropout=0.3):
    bb = backbone_fn(in_channels=12, dropout=dropout)
    model = ArrhythmiaClassifier(bb, num_classes=27)
    ckpt = safe_torch_load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def topk_hit_rate(probs, labels, k):
    """样本中至少有一个真实标签出现在top-K预测中的比例。"""
    hits = 0
    total = len(probs)
    topk_indices = np.argsort(-probs, axis=1)[:, :k]
    for i in range(total):
        true_active = set(np.where(labels[i] > 0)[0])
        if not true_active:
            total -= 1
            continue
        if true_active & set(topk_indices[i]):
            hits += 1
    return hits / total if total > 0 else 0.0


def main():
    device = "cpu"
    data_dir = str(Path(__file__).parent.parent / "data/physionet2020/processed")

    # Models to test
    models = {}
    try:
        models["xResNet1D-101"] = load_model(
            xresnet1d_101,
            "checkpoints/xresnet1d_101/best_model.pt", device)
    except Exception as e:
        print(f"xResNet: SKIP ({e})")

    try:
        models["ECG Transformer"] = load_model(
            ecg_transformer,
            "checkpoints/ecg_transformer_fp32/best_model.pt", device)
    except Exception as e:
        print(f"ECG Transformer: SKIP ({e})")

    try:
        models["InceptionTime"] = load_model(
            inception_time,
            "checkpoints/inception_time/best_model.pt", device,
            dropout=0.1)
    except Exception as e:
        print(f"InceptionTime: SKIP ({e})")

    try:
        models["SimCLR xResNet"] = load_model(
            xresnet1d_101,
            "checkpoints/simclr_xresnet/best_model.pt", device)
    except Exception as e:
        print(f"SimCLR xResNet: SKIP ({e})")

    print(f"\nLoaded {len(models)} models: {list(models.keys())}\n")

    # Data
    dm = ECGDataModule(data_dir, batch_size=128, num_workers=0,
                       label_extractor=LabelExtractor())
    dm.setup()
    loader = dm.test_dataloader()

    # Collect predictions
    all_labels = []
    all_probs = {name: [] for name in models}
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            all_labels.append(labels.numpy())
            for name, model in models.items():
                logits = model(signals)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs[name].append(probs)

    labels = np.concatenate(all_labels, axis=0)
    for name in models:
        all_probs[name] = np.concatenate(all_probs[name], axis=0)

    # Ensemble: average probabilities
    if len(models) >= 2:
        ensemble_probs = np.mean([all_probs[n] for n in models], axis=0)
        models["集成 Ensemble"] = None  # placeholder
        all_probs["集成 Ensemble"] = ensemble_probs

    # Results
    print(f"Test samples: {len(labels)}")
    print(f"{'Model':<25} {'Top-1':>8} {'Top-3':>8} {'Top-5':>8} {'AUC':>8}")
    print("-" * 60)

    for name, probs in all_probs.items():
        top1 = topk_hit_rate(probs, labels, 1)
        top3 = topk_hit_rate(probs, labels, 3)
        top5 = topk_hit_rate(probs, labels, 5)
        aucs = []
        for c in range(27):
            if 0 < labels[:, c].sum() < len(labels):
                aucs.append(roc_auc_score(labels[:, c], probs[:, c]))
        auc = float(np.mean(aucs)) if aucs else 0.0
        print(f"{name:<25} {top1:7.1%} {top3:7.1%} {top5:7.1%} {auc:8.4f}")


if __name__ == "__main__":
    main()
