#!/usr/bin/env python3
"""M4.2 — 27 类逐类 AUC 表 + 混淆对分析（误差模式报告）。

用法: python scripts/eval_per_class.py
"""

from src.utils.safe_load import safe_torch_load
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.metrics.classification import youden_thresholds
from src.data_pipeline.label_extractor import LabelExtractor
from sklearn.metrics import roc_auc_score

FEAT_DIR = Path("data/physionet2020/ecgfounder_features")
HEAD = Path("outputs/ecgfounder_mlp/mlp_head.pt")
VAL_FEAT = Path("data/physionet2020/ecgfounder_features/val_features.npy")
VAL_LABEL = Path("data/physionet2020/ecgfounder_features/val_labels.npy")


def load_head():
    import torch.nn as nn
    head = nn.Sequential(
        nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
        nn.Linear(512, 27),
    )
    sd = safe_torch_load(HEAD, map_location="cpu")
    if any(k.startswith("net.") for k in sd):
        sd = {k[4:]: v for k, v in sd.items()}
    head.load_state_dict(sd)
    head.eval()
    return head


def main():
    le = LabelExtractor(num_classes=27)
    x = np.load(FEAT_DIR / "test_features.npy")
    y = np.load(FEAT_DIR / "test_labels.npy")

    head = load_head()
    with torch.no_grad():
        probs = torch.sigmoid(head(torch.from_numpy(x))).numpy()

    # 验证集阈值（与主评测一致）
    xv = np.load(VAL_FEAT)
    yv = np.load(VAL_LABEL)
    with torch.no_grad():
        pv = torch.sigmoid(head(torch.from_numpy(xv))).numpy()
    ths = youden_thresholds(yv, pv)
    pred = (probs >= ths).astype(int)

    # ---- 逐类 AUC + 支持度 ----
    rows = []
    for c in range(27):
        n_pos = int(y[:, c].sum())
        auc = roc_auc_score(y[:, c], probs[:, c]) if 0 < n_pos < len(y) else None
        rows.append({
            "class": le.class_names[c],
            "n_positive": n_pos,
            "auc": round(auc, 4) if auc is not None else None,
        })
    rows_sorted = sorted(rows, key=lambda r: -(r["auc"] or -1))
    print(f"{'类别':35s} {'阳性数':>6s} {'AUC':>8s}")
    for r in rows_sorted:
        print(f"{r['class']:35s} {r['n_positive']:6d} {str(r['auc']):>8s}")

    # ---- 混淆对：对每类，找其假阳性样本中最常被同时预测的"竞争类" ----
    print("\n=== Top 混淆对（假阳性竞争类）===")
    confusion = Counter()
    for c in range(27):
        if y[:, c].sum() == 0:
            continue
        # 该类为真、但被预测为其它类的样本（漏报去向）
        missed = (y[:, c] == 1) & (pred[:, c] == 0)
        if missed.sum() == 0:
            continue
        sub = pred[missed]
        for c2 in range(27):
            if c2 != c and sub[:, c2].sum() >= 2:
                confusion[(le.class_names[c], le.class_names[c2])] += int(sub[:, c2].sum())
    for (a, b), cnt in confusion.most_common(12):
        print(f"  {a} → 误判为 {b}: {cnt} 例")

    out = Path("outputs/external_eval")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_class_auc.json", "w", encoding="utf-8") as f:
        json.dump({"classes": rows_sorted,
                   "top_confusion_pairs": [(f"{a}->{b}", cnt)
                                           for (a, b), cnt in confusion.most_common(15)]},
                  f, ensure_ascii=False, indent=1)
    print(f"\n保存 → {out / 'per_class_auc.json'}")


if __name__ == "__main__":
    main()
