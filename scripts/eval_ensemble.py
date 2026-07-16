"""Model Ensemble Evaluation — average probabilities from multiple backbones."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score


def main():
    import torch
    from src.ecg_models.backbone.xresnet1d import xresnet1d_101
    from src.ecg_models.backbone.transformer_encoder import ecg_transformer
    from src.ecg_models.classifiers.arrhythmia_classifier import ArrhythmiaClassifier
    from src.data_pipeline.dataset import ECGDataModule
    from src.data_pipeline.label_extractor import LabelExtractor

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/physionet2020/processed")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    # Define ensemble members
    members = [
        {
            "name": "xResNet1D-101",
            "backbone_fn": xresnet1d_101,
            "checkpoint": "checkpoints/xresnet1d_101/best_model.pt",
            "dropout": 0.3,
        },
        {
            "name": "ECG Transformer (FP32)",
            "backbone_fn": ecg_transformer,
            "checkpoint": "checkpoints/ecg_transformer_fp32/best_model.pt",
            "dropout": 0.3,
        },
    ]

    models = []
    for m in members:
        backbone = m["backbone_fn"](in_channels=12, dropout=m["dropout"])
        model = ArrhythmiaClassifier(backbone, num_classes=27)
        ckpt = torch.load(m["checkpoint"], map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(args.device)
        model.eval()
        models.append(model)
        print(f"Loaded: {m['name']} (epoch {ckpt.get('epoch', '?')})")

    # Data
    dm = ECGDataModule(args.data_dir, batch_size=args.batch_size, num_workers=0,
                       label_extractor=LabelExtractor())
    dm.setup()
    loader = dm.test_dataloader()

    all_probs, all_labels = [], []
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(args.device)
            # Average probabilities from all models
            probs_list = []
            for model in models:
                logits = model(signals)
                probs = torch.sigmoid(logits).cpu().numpy()
                probs_list.append(probs)
            avg_probs = np.mean(probs_list, axis=0)
            all_probs.append(avg_probs)
            all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    num_classes = labels.shape[1]

    # Macro AUC
    aucs = []
    for c in range(num_classes):
        if 0 < labels[:, c].sum() < len(labels):
            aucs.append(roc_auc_score(labels[:, c], probs[:, c]))
    macro_auc = float(np.mean(aucs)) if aucs else 0.0

    # Macro F1 (optimal threshold per class)
    from sklearn.metrics import precision_recall_curve
    f1s = []
    for c in range(num_classes):
        if labels[:, c].sum() == 0:
            continue
        prec, rec, thresh = precision_recall_curve(labels[:, c], probs[:, c])
        f1_curve = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-8)
        best_idx = int(np.argmax(f1_curve))
        best_thresh = float(thresh[best_idx]) if len(thresh) > best_idx else 0.5
        preds_c = (probs[:, c] >= best_thresh).astype(np.float32)
        f1s.append(f1_score(labels[:, c], preds_c, zero_division=0))
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0

    # mAP
    mAP = float(average_precision_score(labels, probs, average="macro"))

    print("\n" + "=" * 55)
    print("ENSEMBLE Evaluation (Probability Averaging)")
    print("=" * 55)
    for m in members:
        print(f"  {m['name']}")
    print("=" * 55)
    print(f"  macro_auc: {macro_auc:.4f}")
    print(f"  macro_f1:  {macro_f1:.4f}")
    print(f"  mAP:       {mAP:.4f}")


if __name__ == '__main__':
    main()
