# ⚠️ DEPRECATED（2026-08-24，检查报告第六步）: 旧综合评估；其产物 comprehensive_eval.json 仍是成绩单历史数字（0.791/0.808/0.822）的来源，保留文件与产物
"""Comprehensive Model Evaluation — Bootstrap CI + Youden Thresholds + Top-K + McNemar."""
from src.utils.safe_load import safe_torch_load
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve
from sklearn.metrics import average_precision_score

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


def bootstrap_auc_ci(labels, probs, n_bootstrap=1000, alpha=0.05):
    """Bootstrap 95% confidence interval for macro AUC.

    2B 🟠-4 修复：固定种子（RandomState(42)），CI 可复现。
    """
    rng = np.random.RandomState(42)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(len(labels), len(labels), replace=True)
        scores.append(macro_auc(labels[idx], probs[idx]))
    lo = np.percentile(scores, 100 * alpha / 2)
    hi = np.percentile(scores, 100 * (1 - alpha / 2))
    mean = np.mean(scores)
    return mean, lo, hi


def youden_thresholds(labels, probs):
    """逐类 Youden's J 统计量最优阈值 (maximize TPR - FPR)."""
    num_classes = labels.shape[1]
    thresholds = []
    for c in range(num_classes):
        if labels[:, c].sum() == 0:
            thresholds.append(0.5)
            continue
        fpr, tpr, thresh = roc_curve(labels[:, c], probs[:, c])
        j = tpr - fpr
        best_idx = int(np.argmax(j))
        thresholds.append(float(thresh[best_idx]))
    return thresholds


def topk_hit_rate(probs, labels, k):
    """样本中至少有一个真实标签出现在 Top-K 预测中的比例."""
    hits, total = 0, len(probs)
    topk = np.argsort(-probs, axis=1)[:, :k]
    for i in range(total):
        true_active = set(np.where(labels[i] > 0)[0])
        if not true_active:
            total -= 1
            continue
        if true_active & set(topk[i]):
            hits += 1
    return hits / total if total > 0 else 0.0


def macro_auc(labels, probs):
    """Macro-averaged AUC."""
    aucs = []
    for c in range(labels.shape[1]):
        if 0 < labels[:, c].sum() < len(labels):
            aucs.append(roc_auc_score(labels[:, c], probs[:, c]))
    return float(np.mean(aucs)) if aucs else 0.0


def macro_f1_optimal(labels, probs, thresholds):
    """Macro F1 using per-class Youden optimal thresholds."""
    f1s = []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() > 0:
            preds = (probs[:, c] >= thresholds[c]).astype(np.float32)
            f1s.append(f1_score(labels[:, c], preds, zero_division=0))
    return float(np.mean(f1s)) if f1s else 0.0


# ============================================================
# Main
# ============================================================

device = "cpu"
data_dir = str(Path(__file__).parent.parent / "data/physionet2020/processed")
label_extractor = LabelExtractor()

# Load models
models = {}

try:
    models["InceptionTime"] = load_model(
        inception_time, "checkpoints/inception_time/best_model.pt", device, dropout=0.1)
except Exception as e:
    print(f"InceptionTime: SKIP ({e})")

try:
    models["xResNet1D-101"] = load_model(
        xresnet1d_101, "checkpoints/xresnet1d_101/best_model.pt", device)
except Exception as e:
    print(f"xResNet1D-101: SKIP ({e})")

try:
    models["ECG Transformer"] = load_model(
        ecg_transformer, "checkpoints/ecg_transformer_fp32/best_model.pt", device)
except Exception as e:
    print(f"ECG Transformer: SKIP ({e})")

print(f"\nLoaded {len(models)} models: {list(models.keys())}\n")

# Data
dm = ECGDataModule(data_dir, batch_size=128, num_workers=0, label_extractor=label_extractor)
dm.setup()
loader = dm.test_dataloader()

# 2B 🔴-2 修复：阈值必须在验证集上确定（旧版在测试集现算 Youden → F1 乐观高估）
val_loader = dm.val_dataloader()

def collect_probs(model, loader):
    ps, ys = [], []
    with torch.no_grad():
        for signals, labels in loader:
            signals = signals.to(device)
            logits = model(signals)
            ps.append(torch.sigmoid(logits).cpu().numpy())
            ys.append(labels.numpy())
    return np.concatenate(ys, axis=0), np.concatenate(ps, axis=0)

val_probs_all, val_labels_all = {}, {}
for name, model in models.items():
    yl, pl = collect_probs(model, val_loader)
    val_labels_all[name], val_probs_all[name] = yl, pl

# Collect predictions
labels_test = []
all_probs = {name: [] for name in models}

with torch.no_grad():
    for signals, labels in loader:
        signals = signals.to(device)
        labels_test.append(labels.numpy())
        for name, model in models.items():
            logits = model(signals)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs[name].append(probs)

labels_test = np.concatenate(labels_test, axis=0)
for name in models:
    all_probs[name] = np.concatenate(all_probs[name], axis=0)

# Ensemble
if len(models) >= 2:
    ensemble_probs = np.mean(list(all_probs.values()), axis=0)
    models["集成 Ensemble"] = None
    all_probs["集成 Ensemble"] = ensemble_probs
    # 集成在验证集上的概率（用于验证集定阈值）
    val_probs_all["集成 Ensemble"] = np.mean(list(val_probs_all.values()), axis=0)

# ============================================================
# Full Comparison Table
# ============================================================

print(f"{'='*105}")
print(f"Test Set: {len(labels_test)} samples | 27 classes | 1000 bootstrap samples")
print(f"{'='*105}")
print(f"{'Model':<25} {'macro_auc':>10} {'95% CI':>18} {'macro_f1':>10} {'mAP':>8} {'Top-1':>7} {'Top-5':>7}")
print(f"{'-'*105}")

results = {}
for name, probs in all_probs.items():
    # Macro AUC + Bootstrap CI
    auc_mean, auc_lo, auc_hi = bootstrap_auc_ci(labels_test, probs)

    # 2B 🔴-2 修复：Youden 最优阈值在验证集上确定，测试集只评估
    thresholds = youden_thresholds(val_labels_all[name], val_probs_all[name])

    # Macro F1 with optimal thresholds
    f1_opt = macro_f1_optimal(labels_test, probs, thresholds)

    # mAP
    mAP = float(average_precision_score(labels_test, probs, average="macro"))

    # Top-K
    top1 = topk_hit_rate(probs, labels_test, 1)
    top5 = topk_hit_rate(probs, labels_test, 5)

    results[name] = {
        "macro_auc": auc_mean, "auc_ci": (auc_lo, auc_hi),
        "macro_f1": f1_opt, "mAP": mAP,
        "top1": top1, "top5": top5,
        "thresholds": thresholds,
    }

    ci = f"[{auc_lo:.4f}, {auc_hi:.4f}]"
    print(f"{name:<25} {auc_mean:10.4f} {ci:>18} {f1_opt:10.4f} {mAP:8.4f} {top1:6.1%} {top5:6.1%}")

print(f"{'='*105}")

# ============================================================
# Per-Class Analysis (for the best model)
# ============================================================
best_name = "集成 Ensemble" if "集成 Ensemble" in results else list(results.keys())[-1]
best_probs = all_probs[best_name]
best_thresh = results[best_name]["thresholds"]

print(f"\n{'='*80}")
print(f"Per-Class Breakdown: {best_name}")
print(f"{'='*80}")
print(f"{'#':>3} {'Class':<35} {'Samples':>8} {'AUC':>8} {'Thresh':>7} {'F1':>7}")
print(f"{'-'*80}")

for c in range(27):
    n_pos = int(labels_test[:, c].sum())
    if n_pos == 0:
        continue
    auc_c = roc_auc_score(labels_test[:, c], best_probs[:, c])
    thresh_c = best_thresh[c]
    preds_c = (best_probs[:, c] >= thresh_c).astype(np.float32)
    f1_c = f1_score(labels_test[:, c], preds_c, zero_division=0)
    class_name = label_extractor.class_names[c] if c < len(label_extractor.class_names) else f"C{c}"
    print(f"{c:3d} {class_name:<35} {n_pos:8d} {auc_c:8.4f} {thresh_c:7.3f} {f1_c:7.4f}")

# Save results
out_path = Path(__file__).parent.parent / "outputs" / "comprehensive_eval.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump({k: {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist())
                   for kk, vv in v.items()} for k, v in results.items()}, f, indent=2)
print(f"\nResults saved to: {out_path}")
