#!/usr/bin/env python3
"""M4.3 — MIMIC-IV-ECG 零样本域分析（真实外部验证）。

对 MIMIC-IV-ECG 样本（D 盘，只读）跑 ECGFounder **150 类原始头**（零样本、无任何
MIMIC 训练），与 CinC2020 测试集样本的 150 类输出做域对比:

    [1] 任务分布对比（top 任务频率的域间差异）
    [2] 置信度与熵对比（域偏移下的校准差异）
    [3] 信号质量 vs 置信度（平导联/噪声鲁棒性）——MIMIC 含真实临床噪声

说明: MIMIC-IV-ECG 无现成诊断标签（仅报告文本），故本分析是"域行为分析"而非
有监督精度；有监督外部验证留待标签来源（如 mimic-iv-ecg-ext-icd-labels）核实后补充。

用法: python scripts/eval_mimic_domain.py --n-samples 500 --batch-size 32
"""

from src.utils.safe_load import safe_torch_load
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wfdb

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from src.ecg_models.backbone.ecgfounder_net1d import Net1D
import preprocess_data_5k as p5

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MIMIC_ROOT = Path(r"D:\ecg\mimic_iv_ecg\extracted\data")
CKPT = "checkpoints/ECGFounder/12_lead_ECGFounder.pth"
TASKS = [l.strip() for l in open("data/physionet2020/ecgfounder_tasks.txt",
                                encoding="utf-8") if l.strip()]
CINC_TEST_MANIFEST = Path("data/physionet2020/processed_5k/test_manifest.json")


def load_ecgfounder_150():
    """带原始 150 类头的 ECGFounder（零样本用）。"""
    model = Net1D(
        in_channels=12, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16, stride=2, groups_width=16,
        n_classes=150, use_bn=False, use_do=False, verbose=False,
    )
    ckpt = safe_torch_load(CKPT, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


@torch.no_grad()
def predict_batch(model, signals):
    """signals: list of (12,5000) → sigmoid (N,150)。"""
    x = torch.from_numpy(np.stack(signals))
    logits = model(x)
    return torch.sigmoid(logits).numpy()


def domain_stats(probs):
    """置信度/熵/任务分布统计。"""
    maxp = probs.max(axis=1)
    # 熵（对 150 类 softmax 归一）
    pn = probs / probs.sum(axis=1, keepdims=True)
    entropy = -(pn * np.log(pn + 1e-12)).sum(axis=1)
    top_tasks = probs.argmax(axis=1)
    n_pos = (probs >= 0.5).sum(axis=1)
    return {
        "mean_max_prob": round(float(maxp.mean()), 4),
        "mean_entropy": round(float(entropy.mean()), 4),
        "mean_n_positive_tasks": round(float(n_pos.mean()), 3),
        "top5_tasks": [(TASKS[i], int((top_tasks == i).sum()))
                       for i in np.argsort(-np.bincount(top_tasks,
                                                        minlength=150))[:5]],
    }


def load_mimic_samples(n):
    rng = np.random.RandomState(42)
    df = pd.read_csv(MIMIC_ROOT / "record_list.csv")
    idx = rng.choice(len(df), size=min(n, len(df)), replace=False)
    signals, qualities = [], []
    for i in idx:
        row = df.iloc[i]
        try:
            rec = wfdb.rdrecord(str(MIMIC_ROOT / row["path"]))
            sig = rec.p_signal.T.astype(np.float32)   # (12, 5000)
            if sig.shape[0] != 12:
                continue
            # ECGFounder 协议预处理（500Hz 原生，无需重采样）
            sig, _ = p5.reorder_leads(sig, rec.sig_name)
            filt = p5.filter_bandpass(sig, 500.0)
            seg, pl, pr = p5.segment_to_5000(filt)
            # 4H-ORANGE-2 修复：平导联统计在 z-score 之前（mV 域）计算——
            # z-score 归一化把非零导联 std 拉到 ~1，之后 std<0.05 恒不成立
            # （旧版 mean_flat_leads=0.002/n_all_flat=0 属归一化伪影，无信息量）
            flat = (filt.std(axis=1) < 0.02).sum()
            norm = p5.official_zscore(seg, pl, pr)
            signals.append(norm)
            qualities.append({"record": row["file_name"],
                              "flat_leads": int(flat),
                              "global_std": round(float(filt.std()), 3)})
        except Exception as e:
            logger.warning(f"读取失败 {row['file_name']}: {str(e)[:60]}")
    return np.array(signals), qualities


def load_cinc_samples(n):
    with open(CINC_TEST_MANIFEST, encoding="utf-8") as f:
        manifest = json.load(f)["files"]
    rng = np.random.RandomState(42)
    idx = rng.choice(len(manifest), size=min(n, len(manifest)), replace=False)
    data_dir = Path("data/physionet2020/processed_5k")
    signals, qualities = [], []
    for i in idx:
        p = data_dir / manifest[i]["signal_file"]
        if not p.exists():
            continue
        sig = np.load(p)
        # 4H-ORANGE-2：CinC 侧为已 z-score 的 processed_5k 产物，幅度信息已
        # 归一化，平导联统计无信息量——仅作记录不作质量结论（MIMIC 侧在
        # z-score 前统计，两者不可直接并列比较）
        flat = (sig.std(axis=1) < 0.05).sum()
        signals.append(sig)
        qualities.append({"record": manifest[i]["record_id"],
                          "flat_leads": int(flat),
                          "global_std": round(float(sig.std()), 3)})
    return np.array(signals), qualities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    logger.info("加载 ECGFounder（150 类原始头，零样本）...")
    model = load_ecgfounder_150()

    logger.info(f"加载 MIMIC-IV-ECG 样本（目标 {args.n_samples}）...")
    mimic_sig, mimic_q = load_mimic_samples(args.n_samples)
    logger.info(f"MIMIC 有效样本: {len(mimic_sig)}")

    logger.info("加载 CinC2020 测试集对照样本...")
    cinc_sig, cinc_q = load_cinc_samples(args.n_samples)
    logger.info(f"CinC 有效样本: {len(cinc_sig)}")

    logger.info("推理 MIMIC...")
    mimic_probs = []
    for i in range(0, len(mimic_sig), args.batch_size):
        mimic_probs.append(predict_batch(model, mimic_sig[i:i + args.batch_size]))
    mimic_probs = np.concatenate(mimic_probs)

    logger.info("推理 CinC...")
    cinc_probs = []
    for i in range(0, len(cinc_sig), args.batch_size):
        cinc_probs.append(predict_batch(model, cinc_sig[i:i + args.batch_size]))
    cinc_probs = np.concatenate(cinc_probs)

    results = {
        "mimic": {"n": len(mimic_sig), **domain_stats(mimic_probs),
                  "signal_quality": {
                      "mean_flat_leads": round(float(np.mean([q["flat_leads"] for q in mimic_q])), 3),
                      "n_all_flat": int(sum(1 for q in mimic_q if q["flat_leads"] >= 6)),
                  }},
        "cinc": {"n": len(cinc_sig), **domain_stats(cinc_probs),
                 "signal_quality": {
                     "mean_flat_leads": round(float(np.mean([q["flat_leads"] for q in cinc_q])), 3),
                     "n_all_flat": int(sum(1 for q in cinc_q if q["flat_leads"] >= 6)),
                 }},
    }

    # 信号质量 vs 置信度（MIMIC）
    flat_idx = np.array([q["flat_leads"] for q in mimic_q])
    clean = flat_idx <= 2
    if clean.sum() > 0 and (~clean).sum() > 0:
        results["mimic"]["quality_vs_confidence"] = {
            "max_prob_clean": round(float(mimic_probs[clean].max(axis=1).mean()), 4),
            "max_prob_degraded": round(float(mimic_probs[~clean].max(axis=1).mean()), 4),
        }
    else:
        # 4H-ORANGE-2 修复：静默不产出 → 显式告警（旧版无任何提示）
        logger.warning(
            f"quality_vs_confidence 未产出：clean={int(clean.sum())} / "
            f"degraded={int((~clean).sum())}（需要两侧均非空才有意义）")

    out = Path("outputs/external_eval")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "mimic_domain.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    logger.info("=" * 60)
    for dom, stats in results.items():
        logger.info(f"[{dom}] n={stats['n']} maxp={stats['mean_max_prob']} "
                    f"entropy={stats['mean_entropy']} "
                    f"n_pos={stats['mean_n_positive_tasks']}")
        logger.info(f"  top tasks: {stats['top5_tasks']}")
    logger.info(f"保存 → {out / 'mimic_domain.json'}")


if __name__ == "__main__":
    main()
