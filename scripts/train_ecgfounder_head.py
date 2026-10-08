#!/usr/bin/env python3
"""
ECGFounder 冻结特征 + MLP 分类头训练与评估（里程碑 M0.2 / 评估协议 M0.4）

输入: extract_ecgfounder_features.py 产出的特征 npy
流程:
    1. 训练 2 层 MLP（1024→512→27），BCE + 标签平滑 0.1，AdamW
    2. 验证集上: 早停(val macro AUC) + 逐类 Youden 最优阈值
    3. 测试集评估: macro AUC + Bootstrap 95% CI + macro F1@Youden + mAP + Top-1/3/5
    4. 同时给出逐类 LogisticRegression 线性探针（下界参考）

用法:
    python scripts/train_ecgfounder_head.py
    python scripts/train_ecgfounder_head.py --epochs 30 --lr 1e-3
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils.device_utils import detect_device

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================
# 评估协议（M0.4 固化版）
# 注（检查报告 1.9 / 口径统一）：以下指标函数与
# src/evaluation/metrics/classification.py 存在并行实现。为不扰动已验证的
# 论文数字（患者级划分 0.9456 及 CI 等由本文件本地实现产出），保留本地版本不动；
# 新代码请直接 import canonical 版本。
# ============================================================

def macro_auc_with_ci(labels, probs, n_bootstrap=1000, seed=42):
    """逐类 AUC 宏平均 + Bootstrap 95% CI。"""
    from sklearn.metrics import roc_auc_score
    rng = np.random.RandomState(seed)
    aucs = []
    for c in range(labels.shape[1]):
        y, p = labels[:, c], probs[:, c]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        aucs.append(roc_auc_score(y, p))
    macro = float(np.mean(aucs)) if aucs else 0.0

    boot = []
    n = len(labels)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, n)
        vals = []
        for c in range(labels.shape[1]):
            y, p = labels[idx, c], probs[idx, c]
            if y.sum() == 0 or y.sum() == len(y):
                continue
            vals.append(roc_auc_score(y, p))
        if vals:
            boot.append(float(np.mean(vals)))
    lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (macro, macro)
    return macro, float(lo), float(hi)


def youden_thresholds(labels, probs):
    """逐类 Youden J 最优阈值。

    5F 审查（🟡-2）：sklearn 1.9 的 roc_curve 恒返回 t[0]=inf，AUC≤0.5 的
    类 argmax 退化取 inf 阈值（测试集恒判负）——非有限阈值兜底 0.5。
    """
    from sklearn.metrics import roc_curve
    ths = np.full(labels.shape[1], 0.5, dtype=np.float32)
    for c in range(labels.shape[1]):
        y, p = labels[:, c], probs[:, c]
        if y.sum() == 0 or y.sum() == len(y):
            continue
        fpr, tpr, t = roc_curve(y, p)
        j = tpr - fpr
        best = float(t[np.argmax(j)])
        ths[c] = best if np.isfinite(best) else 0.5
    return ths


def topk_hits(probs, labels, k):
    hits, total = 0, 0
    topk_idx = np.argsort(-probs, axis=1)[:, :k]
    for i in range(len(probs)):
        true = set(np.where(labels[i] > 0)[0])
        if not true:
            continue
        total += 1
        if true & set(topk_idx[i]):
            hits += 1
    return hits / total if total else 0.0


def evaluate(labels, probs, thresholds=None):
    from sklearn.metrics import f1_score, average_precision_score
    macro_auc, lo, hi = macro_auc_with_ci(labels, probs)
    ths = thresholds if thresholds is not None else youden_thresholds(labels, probs)
    pred = (probs >= ths).astype(int)
    f1s = []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() == 0:
            continue
        f1s.append(f1_score(labels[:, c], pred[:, c], zero_division=0))
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    # 4E/4F 审查修复：🟡-4 —— 与 trainer.py:512-518 口径一致：仅对 0<sum<len 的类
    # 计算 AP 后取平均；旧版 average="macro" 把无正样本类按 0 计入，系统性拉低 mAP。
    aps = [average_precision_score(labels[:, c], probs[:, c])
           for c in range(labels.shape[1])
           if 0 < labels[:, c].sum() < len(labels)]
    mAP = float(np.mean(aps)) if aps else 0.0
    return {
        "macro_auc": round(macro_auc, 4),
        "auc_ci": [round(lo, 4), round(hi, 4)],
        "macro_f1_youden": round(macro_f1, 4),
        "mAP": round(mAP, 4),
        "top1": round(topk_hits(probs, labels, 1), 4),
        "top3": round(topk_hits(probs, labels, 3), 4),
        "top5": round(topk_hits(probs, labels, 5), 4),
        "thresholds": [round(float(t), 4) for t in ths],
    }


# ============================================================
# 模型与训练
# ============================================================

class MLPHead(nn.Module):
    def __init__(self, in_dim, n_classes=27, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(x_train, y_train, x_val, y_val, epochs, lr, batch_size, device, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)

    model = MLPHead(x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(x_train)
    best_val, best_state, patience_counter = 0.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = x_train[idx].to(device), y_train[idx].to(device)
            opt.zero_grad()
            logits = model(xb)
            # 标签平滑 0.1
            # 第三轮审查 3F-R3-13：均匀标签平滑 ε=0.1（正类 0.95/负类 0.05）
            y_smooth = yb * 0.9 + 0.05
            loss = loss_fn(logits, y_smooth)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(x_val.to(device))).cpu().numpy()
        from sklearn.metrics import roc_auc_score
        aucs = [roc_auc_score(y_val[:, c].numpy(), vp[:, c])
                for c in range(y_val.shape[1])
                if y_val[:, c].sum() > 0 and y_val[:, c].sum() < len(y_val)]
        val_auc = float(np.mean(aucs)) if aucs else 0.0
        if val_auc > best_val + 1e-4:
            best_val, patience_counter = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= 10:
                logger.info(f"早停 @ epoch {epoch} (best val_auc={best_val:.4f})")
                break
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"epoch {epoch}: val_auc={val_auc:.4f} (best {best_val:.4f})")

    # 第三轮审查 3F-R3-05：best_state 恒为 None 时兜底（验证集所有类 AUC 均 0
    # 的退化场景下旧版 load_state_dict(None) 崩溃）
    if best_state is None:
        logger.warning("验证集全程无改善（val_auc 恒 0），使用最后一轮权重")
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", default="data/physionet2020/ecgfounder_features")
    parser.add_argument("--output-dir", default="outputs/ecgfounder_mlp")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--with-linear-probe", action="store_true",
                        help="额外跑逐类 LogisticRegression 线性探针下界")
    args = parser.parse_args()

    fdir = Path(args.feature_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train = np.load(fdir / "train_features.npy")
    y_train = np.load(fdir / "train_labels.npy")
    x_val = np.load(fdir / "val_features.npy")
    y_val = np.load(fdir / "val_labels.npy")
    x_test = np.load(fdir / "test_features.npy")
    y_test = np.load(fdir / "test_labels.npy")
    logger.info(f"特征载入: train {x_train.shape}, val {x_val.shape}, test {x_test.shape}")

    # 4E/4F 审查修复：💡-2 —— 默认 --device auto，经 detect_device() 自动探测
    # GPU/NPU（旧版默认 cpu 且仅当显式传 cuda 且 CUDA 可用才用 GPU，从不自动加速）。
    device = torch.device(detect_device(args.device))
    logger.info(f"训练设备: {device}")

    model = train_mlp(x_train, y_train, x_val, y_val,
                      args.epochs, args.lr, args.batch_size, device)
    torch.save(model.state_dict(), out_dir / "mlp_head.pt")

    model.eval()
    with torch.no_grad():
        test_probs = torch.sigmoid(
            model(torch.from_numpy(x_test).to(device))).cpu().numpy()

    # 验证集上定阈值
    with torch.no_grad():
        val_probs = torch.sigmoid(
            model(torch.from_numpy(x_val).to(device))).cpu().numpy()
    ths = youden_thresholds(y_val, val_probs)

    metrics = evaluate(y_test, test_probs, thresholds=ths)
    logger.info("=== ECGFounder + MLP Head (test) ===")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    result = {"model": "ECGFounder frozen + MLP head", "metrics": metrics}

    if args.with_linear_probe:
        from sklearn.linear_model import LogisticRegression
        logger.info("训练逐类线性探针（下界参考）...")
        test_lp = np.zeros_like(test_probs)
        val_lp = np.zeros_like(val_probs)
        for c in range(27):
            if y_train[:, c].sum() == 0:
                continue
            clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs")
            clf.fit(x_train, y_train[:, c])
            test_lp[:, c] = clf.predict_proba(x_test)[:, 1]
            val_lp[:, c] = clf.predict_proba(x_val)[:, 1]
        lp_ths = youden_thresholds(y_val, val_lp)
        lp_metrics = evaluate(y_test, test_lp, thresholds=lp_ths)
        logger.info("=== ECGFounder + Linear Probe (test) ===")
        for k, v in lp_metrics.items():
            logger.info(f"  {k}: {v}")
        result["linear_probe"] = lp_metrics

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    logger.info(f"结果保存: {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
