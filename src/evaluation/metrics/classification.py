"""评估协议（M0.4 固化版）— 分类指标统一入口。

包含: macro AUC + Bootstrap 95% CI、逐类 Youden 最优阈值、
macro F1@Youden、mAP、Top-K 命中率。
所有模型对比必须走本模块，保证口径一致。
"""

from typing import Optional, Tuple

import numpy as np


def macro_auc_with_ci(labels, probs, n_bootstrap=1000, seed=42):
    """逐类 AUC 宏平均 + Bootstrap 95% CI。

    Args:
        labels: (N, C) 多标签。
        probs:  (N, C) 预测概率。
    Returns:
        (macro_auc, ci_low, ci_high)
    """
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


def youden_thresholds(labels, probs) -> np.ndarray:
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


def topk_hits(probs, labels, k) -> float:
    """Top-K 命中率（每条记录至少一个正类出现在前 K 预测中）。"""
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


def evaluate(labels, probs, thresholds: Optional[np.ndarray] = None) -> dict:
    """完整评估。

    2E-O7 修复：thresholds 必须由验证集确定并显式传入——缺省时在评估集上
    现算 Youden 阈值属样本内拟合（乐观估计）。调用方应统一从验证集阈值加载
    （与 eval_per_class.py / ECGFounderClassifier 口径一致）；未传时告警并
    用 0.5 默认值，不再现算。
    """
    import logging
    from sklearn.metrics import f1_score, average_precision_score

    # 2E-O7 修复（回归补回）：macro AUC 与 CI 计算（此前误删）
    macro_auc, lo, hi = macro_auc_with_ci(labels, probs)

    if thresholds is None:
        logging.getLogger(__name__).warning(
            "evaluate() 未传 thresholds：用 0.5 默认值（禁止评估集现算 Youden，"
            "见 2E-O7）。请从验证集阈值加载。")
        ths = np.full(labels.shape[1], 0.5, dtype=np.float32)
    else:
        ths = thresholds
    pred = (probs >= ths).astype(int)

    f1s = []
    for c in range(labels.shape[1]):
        if labels[:, c].sum() == 0:
            continue
        f1s.append(f1_score(labels[:, c], pred[:, c], zero_division=0))
    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    # 4H-ORANGE-1 修复：mAP 只对 0<正类<全部 的类取平均——旧版
    # average="macro" 把全正/全负类计 AP=0 系统性拉低（与 trainer.py
    # 3C-TRAIN-3 口径一致；实测空类把 macro AP 从 1.0 拉到 0.6667）
    aps = []
    for c in range(labels.shape[1]):
        s = float(labels[:, c].sum())
        if 0 < s < len(labels):
            aps.append(average_precision_score(labels[:, c], probs[:, c]))
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


def retrieval_recall(query_emb, key_emb, query_ids, key_ids, ks=(1, 5, 10)):
    """跨模态检索 recall@K（query_ids 与 key_ids 均为字符串列表）。

    Args:
        query_emb: (Nq, D) 查询嵌入（如文本）。
        key_emb:   (Nk, D) 键嵌入（如 ECG），D 相同。
        query_ids: 查询样本 id 列表；key_ids: 键样本 id 列表。
    Returns:
        {"r1": float, "r5": float, "r10": float}
    """
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(query_emb, key_emb)  # (Nq, Nk)
    key_id2idx = {k: i for i, k in enumerate(key_ids)}
    hits = {k: 0 for k in ks}
    total = 0
    for i, qid in enumerate(query_ids):
        if qid not in key_id2idx:
            continue
        total += 1
        order = np.argsort(-sim[i])
        rank = int(np.where(order == key_id2idx[qid])[0][0]) + 1
        for k in ks:
            if rank <= k:
                hits[k] += 1
    return {f"r{k}": round(hits[k] / total, 4) if total else 0.0 for k in ks}
