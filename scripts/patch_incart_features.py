#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补丁 ECGFounder 特征缓存中的 INCART 行（检查报告 1.6 修复执行第二步）。

reprocess_incart_5k.py 重算信号后，train/val/test 三个批量特征文件中的
74 行 INCART 特征仍基于旧（损坏）信号。本脚本只重算这些行的特征并原位
替换（其余行保持逐位不变），原文件备份为 .bak。

用法: python scripts/patch_incart_features.py
"""

import json
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from extract_ecgfounder_features import load_ecgfounder  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FEAT_DIR = Path("data/physionet2020/ecgfounder_features")
DATA_DIR = Path("data/physionet2020/processed_5k")
CKPT = "checkpoints/ECGFounder/12_lead_ECGFounder.pth"


@torch.no_grad()
def compute_features(model, signals):
    """对 (n,12,5000) 信号批量前向，取 dense 层输入特征 (n,1024)。"""
    captured = []

    def hook_fn(module, inp, out):
        captured.append(inp[0].cpu().numpy())

    handle = model.dense.register_forward_hook(hook_fn)
    try:
        out = []
        for i in range(0, len(signals), 16):
            x = torch.from_numpy(signals[i:i + 16])
            model(x)
            out.append(np.concatenate(captured, axis=0))
            captured.clear()
        return np.concatenate(out, axis=0)
    finally:
        handle.remove()


def main():
    logger.info("加载 ECGFounder...")
    model, feat_dim = load_ecgfounder(CKPT)
    logger.info(f"特征维度: {feat_dim}")

    summary = {}
    for split in ("train", "val", "test"):
        ids_path = FEAT_DIR / f"{split}_ids.json"
        feat_path = FEAT_DIR / f"{split}_features.npy"
        ids = json.load(open(ids_path, encoding="utf-8"))
        feats = np.load(feat_path)
        idx = [i for i, eid in enumerate(ids)
               if eid.startswith("st_petersburg_incart_")]
        logger.info(f"{split}: {len(idx)} 条 INCART 行 / 共 {len(ids)} 行")

        if not idx:
            summary[split] = {"n_patched": 0}
            continue

        bak_path = Path(str(feat_path) + ".bak")
        # 5C 审查（🟡-4）：备份幂等——只在 .bak 不存在时首次备份（旧版无条件
        # 覆盖，重跑会丢失原始备份）
        if not bak_path.exists():
            shutil.copy2(feat_path, bak_path)
        signals = []
        for i in idx:
            p = DATA_DIR / f"{ids[i]}.npy"
            x = np.nan_to_num(np.load(p).astype(np.float32),
                              nan=0.0, posinf=0.0, neginf=0.0)
            if x.shape[0] != 12 and x.shape[1] == 12:
                x = x.T
            signals.append(x)
        new_feats = compute_features(model, np.stack(signals))
        assert new_feats.shape == (len(idx), feats.shape[1]), \
            (new_feats.shape, feats.shape)

        feats_new = feats.copy()
        feats_new[idx] = new_feats
        np.save(feat_path, feats_new.astype(np.float32))

        # 校验：非 INCART 行逐位不变
        mask = np.ones(len(ids), dtype=bool)
        mask[idx] = False
        unchanged = bool(np.array_equal(feats[mask], feats_new[mask]))
        summary[split] = {
            "n_patched": len(idx),
            "non_incart_rows_unchanged": unchanged,
            "ids_unchanged": True,
        }
        logger.info(f"{split}: 已写回，非 INCART 行不变 = {unchanged}")

    with open("outputs/incart_fix/feature_patch_report.json", "w",
              encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    logger.info("完成 → outputs/incart_fix/feature_patch_report.json")


if __name__ == "__main__":
    main()
