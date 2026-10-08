#!/usr/bin/env python3
"""M4.1 — 跨数据源分解验证：ECGFounder-MLP 在各数据源测试子集上的 macro AUC。

意义: 6 个数据源来自不同国家/机构/设备（中国 CPSC、美国 Georgia、德国 PTB/PTB-XL、
俄罗斯 INCART），per-source AUC 揭示域内 vs 域外泛化差异。
（注: 模型在含这些源的混合训练集上训练，本分析为"域分解"而非严格外部验证；
严格外部验证见 eval_mimic_domain.py。）

用法: python scripts/eval_per_source.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
# 论文终态核对（2026-08-30）：safe_torch_load 的 import 必须在 sys.path.insert
# 之后（旧版位于其前，`python scripts/eval_per_source.py` 直接 ModuleNotFoundError）
from src.utils.safe_load import safe_torch_load
from src.evaluation.metrics.classification import macro_auc_with_ci, youden_thresholds, evaluate
from src.ecg_models.backbone.ecgfounder_net1d import Net1D  # noqa: F401 (仅环境检查)

FEAT_DIR = Path("data/physionet2020/ecgfounder_features")
HEAD = Path("outputs/ecgfounder_mlp/mlp_head.pt")


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
    x = np.load(FEAT_DIR / "test_features.npy")
    y = np.load(FEAT_DIR / "test_labels.npy")
    ids = json.load(open(FEAT_DIR / "test_ids.json", encoding="utf-8"))

    head = load_head()
    with torch.no_grad():
        probs = torch.sigmoid(head(torch.from_numpy(x))).numpy()

    # 逐源分组（长前缀优先匹配，区分 cpsc_2018 与 cpsc_2018_extra）
    SOURCES = ["cpsc_2018_extra", "cpsc_2018", "st_petersburg_incart",
               "georgia", "ptb-xl", "ptb"]
    by_source = {}
    unmatched = []
    for i, eid in enumerate(ids):
        for src in SOURCES:
            if eid.startswith(src + "_"):
                by_source.setdefault(src, []).append(i)
                break
        else:
            unmatched.append(eid)
    # 2E-Y10 修复：未匹配 id 显式告警（旧版静默丢弃）
    if unmatched:
        print(f"⚠️ 未匹配数据源前缀的 id: {len(unmatched)} 条（已排除）: "
              f"{unmatched[:5]} ...")

    def _n_active_classes(ys):
        return int(sum(0 < ys[:, c].sum() < len(ys) for c in range(ys.shape[1])))

    results = {}
    print(f"{'数据源':22s} {'n':>6s} {'类数':>4s} {'macro_auc':>10s} {'95% CI':>18s}")
    print("-" * 64)
    for src in sorted(by_source):
        idx = np.array(by_source[src])
        ys, ps = y[idx], probs[idx]
        # 2E-Y10 修复：bootstrap 与主评测一致（1000）
        auc, lo, hi = macro_auc_with_ci(ys, ps, n_bootstrap=1000)
        n_cls = _n_active_classes(ys)
        # 3D 审查（💡-6）：小样本源（n<50）bootstrap CI 退化（n=2 → [0.5,0.5]、
        # n=7 → [1.0,1.0]），显式标注"仅参考"，避免与其它源并列误读
        small = len(idx) < 50
        note = "  ⚠️ 样本过少，仅参考" if small else ""
        results[src] = {"n": int(len(idx)), "n_active_classes": n_cls,
                        "macro_auc": round(auc, 4),
                        "ci": [round(lo, 4), round(hi, 4)],
                        "note": ("样本过少，仅参考" if small else None)}
        print(f"{src:22s} {len(idx):6d} {n_cls:4d} {auc:10.4f} [{lo:.4f}, {hi:.4f}]{note}")

    # 总体
    auc, lo, hi = macro_auc_with_ci(y, probs)
    results["overall"] = {"n": len(y), "n_active_classes": _n_active_classes(y),
                          "macro_auc": round(auc, 4),
                          "ci": [round(lo, 4), round(hi, 4)],
                          "note": None}
    print("-" * 64)
    print(f"{'overall':22s} {len(y):6d} {_n_active_classes(y):4d} "
          f"{auc:10.4f} [{lo:.4f}, {hi:.4f}]")
    print("（注：各源参与宏平均的类数不同，macro 不可直接跨源并列——2E-Y10）")

    out = Path("outputs/external_eval")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "per_source_auc.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(f"\n保存 → {out / 'per_source_auc.json'}")


if __name__ == "__main__":
    main()
