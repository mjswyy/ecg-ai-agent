"""ECGFounder Benchmark: linear probe vs our models."""
import sys
from pathlib import Path

# Import ECGFounder source
sys.path.insert(0, r'C:\Users\llyun\Desktop\ecg资料\GitHub上的一些项目\ECGFounder')
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

from src.data_pipeline.dataset import ECGDataModule
from src.data_pipeline.label_extractor import LabelExtractor
from net1d import Net1D


# ============================================================
# 1. Load ECGFounder with original architecture
# ============================================================

def load_ecgfounder(ckpt_path):
    model = Net1D(
        in_channels=12, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16, stride=2, groups_width=16,
        n_classes=150, use_bn=False, use_do=False, verbose=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    feature_dim = ckpt["state_dict"]["dense.weight"].shape[1]
    return model, feature_dim


# ============================================================
# 2. Feature extraction
# ============================================================

@torch.no_grad()
def extract_features(model, loader, device):
    """Extract 1024-dim features by hooking before the dense layer."""
    feats_list, labs_list = [], []

    # Register hook to capture pre-dense features
    features = []
    def hook_fn(module, input, output):
        features.append(input[0].detach().cpu().numpy())

    handle = model.dense.register_forward_hook(hook_fn)

    for signals, labels in loader:
        # Pad from 4096 to 5000 (model was trained on 5000-sample inputs)
        if signals.shape[-1] < 5000:
            pad = 5000 - signals.shape[-1]
            signals = torch.nn.functional.pad(signals, (pad // 2, pad - pad // 2))
        elif signals.shape[-1] > 5000:
            signals = signals[..., :5000]
        features.clear()
        _ = model(signals.to(device))
        feats_list.append(features[0])
        labs_list.append(labels.numpy())

    handle.remove()
    return np.concatenate(feats_list, axis=0), np.concatenate(labs_list, axis=0)


# ============================================================
# 3. Main
# ============================================================

device = "cpu"
data_dir = str(Path(__file__).parent.parent / "data/physionet2020/processed")

print("Loading ECGFounder...")
model, feature_dim = load_ecgfounder("checkpoints/ECGFounder/12_lead_ECGFounder.pth")
model.to(device)
print(f"Feature dim: {feature_dim}")

# Load data
le = LabelExtractor()
dm = ECGDataModule(data_dir, batch_size=32, num_workers=0, label_extractor=le)
dm.setup()

print("Extracting features (this may take a while on CPU)...")
train_feat, train_labels = extract_features(model, dm.train_dataloader(), device)
val_feat, val_labels = extract_features(model, dm.val_dataloader(), device)
test_feat, test_labels = extract_features(model, dm.test_dataloader(), device)

print(f"Train: {train_feat.shape}, Val: {val_feat.shape}, Test: {test_feat.shape}")

# ============================================================
# Linear Probe: Logistic Regression on frozen features
# ============================================================

print("\n=== Linear Probe (ECGFounder features + Logistic Regression) ===")
clfs = []
test_probs = np.zeros((len(test_feat), 27), dtype=np.float32)
for c in range(27):
    y_train_c = train_labels[:, c]
    if y_train_c.sum() == 0:
        continue
    clf = LogisticRegression(C=1.0, max_iter=500, solver='lbfgs')
    clf.fit(train_feat, y_train_c)
    clfs.append(clf)
    test_probs[:, c] = clf.predict_proba(test_feat)[:, 1]

aucs = []
for c in range(27):
    if test_labels[:, c].sum() > 0:
        aucs.append(roc_auc_score(test_labels[:, c], test_probs[:, c]))
macro_auc_lp = float(np.mean(aucs)) if aucs else 0.0

# Top-K
def topk(probs, labels, k):
    hits, total = 0, len(probs)
    topk_idx = np.argsort(-probs, axis=1)[:, :k]
    for i in range(total):
        true = set(np.where(labels[i] > 0)[0])
        if not true: total -= 1; continue
        if true & set(topk_idx[i]): hits += 1
    return hits / total if total > 0 else 0.0

mAP_lp = float(average_precision_score(test_labels, test_probs, average="macro"))
top1_lp = topk(test_probs, test_labels, 1)
top5_lp = topk(test_probs, test_labels, 5)

print(f"  macro_auc: {macro_auc_lp:.4f}")
print(f"  mAP:       {mAP_lp:.4f}")
print(f"  Top-1:     {top1_lp:.1%}")
print(f"  Top-5:     {top5_lp:.1%}")

# Comparison
print("\n=== Comparison ===")
print(f"{'Model':<30} {'macro_auc':>10} {'Top-1':>8} {'Top-5':>8}")
print(f"{'-'*60}")
print(f"{'Ours: xResNet (from scratch)':<30} {'0.791':>10} {'68.7%':>8} {'92.7%':>8}")
print(f"{'Ours: Transformer (from scratch)':<30} {'0.808':>10} {'69.3%':>8} {'92.8%':>8}")
print(f"{'Ours: Ensemble':<30} {'0.822':>10} {'70.7%':>8} {'93.1%':>8}")
print(f"{'ECGFounder + Linear Probe':<30} {macro_auc_lp:10.4f} {top1_lp:7.1%} {top5_lp:7.1%}")
print(f"\nNote: ECGFounder trained on 10M+ proprietary ECGs.")
print(f"Ours trained on 30K public ECGs (300x less data).")
