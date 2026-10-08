#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重处理 INCART（st_petersburg_incart）记录 — 检查报告 1.6 修复执行。
（5C 审查 🟡-3：现役 INCART 共 33 条——relabel 官方 27 类丢弃了 41 条无评分类
记录；旧 docstring"74 条"为 relabel 前口径。）

旧 processed_5k 的 INCART 信号由钳制比率 63/33 重采样而来（时间压缩 ~1.9%，
尾部 93 样本为 0，z-score 被污染）。本脚本用修复后的 resample_to_500
（真实比率 500/257）重算全部 INCART 记录并覆盖 npy（原文件备份为 .bak），
不触碰 manifest（记录集合/标签不变）。

用法: python scripts/reprocess_incart_5k.py [--raw-dir ...] [--output-dir ...]
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from preprocess_data_5k import (  # noqa: E402
    filter_bandpass, official_zscore, reorder_leads,
    resample_to_500, segment_to_5000, _default_raw_dir,
)
from src.data_pipeline.loader import ECGLoader  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 4C 审查修复：🟡-7 —— 删除硬编码 Desktop\ECG 绝对路径，改用与
# preprocess_data_5k.py 一致的 _default_raw_dir() 自动探测（换机不再失效）。
DEFAULT_RAW = _default_raw_dir()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default=DEFAULT_RAW)
    parser.add_argument("--output-dir", default="data/physionet2020/processed_5k")
    parser.add_argument("--report-dir", default="outputs/incart_fix")
    args = parser.parse_args()

    # 4C 审查修复：🟡-7 —— 探测失败时给出清晰报错
    if not args.raw_dir:
        sys.exit("未探测到原始数据目录——请用 --raw-dir 显式指定（4C 审查修复：🟡-7）")
    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.output_dir)
    rep_dir = Path(args.report_dir)
    rep_dir.mkdir(parents=True, exist_ok=True)

    # 1) 收集三划分中的 INCART 记录
    targets = {}  # (source, record_id) -> split
    for split in ("train", "val", "test"):
        with open(out_dir / f"{split}_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)["files"]
        for m in manifest:
            if m["source"] == "st_petersburg_incart":
                targets[(m["source"], m["record_id"])] = split
    logger.info(f"待重处理 INCART 记录: {len(targets)} 条 "
                f"(train {sum(1 for v in targets.values() if v=='train')}, "
                f"val {sum(1 for v in targets.values() if v=='val')}, "
                f"test {sum(1 for v in targets.values() if v=='test')})")

    # 2) 定位原始 .hea（在 st_petersburg_incart/g1 子目录下）
    hea_map = {}
    for hea in (raw_dir / "st_petersburg_incart").rglob("*.hea"):
        hea_map[hea.stem] = hea
    missing = [rid for _, rid in targets if rid not in hea_map]
    if missing:
        logger.error(f"找不到原始文件的记录: {missing}")
        sys.exit(1)

    loader = ECGLoader(raw_dir)
    report = []
    t0 = time.time()
    n_done = 0
    for (source, rid), split in sorted(targets.items()):
        out_path = out_dir / f"{source}_{rid}.npy"
        try:
            sample = loader.load_record(str(hea_map[rid]))
            if sample is None:
                raise RuntimeError("load_record 返回 None")
            old = np.load(out_path)
            # 4C 审查修复：💡-11 —— 备份幂等：仅当 .bak 不存在时备份，避免二次
            # 重跑覆盖掉真正的"原始损坏版"（与 relabel_official_27.py 写法一致）。
            bak_path = Path(str(out_path) + ".bak")
            if not bak_path.exists():
                shutil.copy2(out_path, bak_path)

            signal, did_reorder = reorder_leads(sample.signal, sample.lead_names)
            filtered = filter_bandpass(signal, sample.fs)
            resampled = resample_to_500(filtered, sample.fs)
            seg, pad_l, pad_r = segment_to_5000(resampled)
            normalized = official_zscore(seg, pad_l, pad_r)
            np.save(out_path, normalized)

            old_tail0 = int((old[:, -100:] == 0).sum())
            new_tail0 = int((normalized[:, -100:] == 0).sum())
            report.append({
                "record_id": rid, "split": split, "fs": float(sample.fs),
                "raw_len": int(sample.signal.shape[1]),
                "resampled_len": int(resampled.shape[1]),
                "seg_len": int(seg.shape[1]), "pad": [int(pad_l), int(pad_r)],
                "old_tail100_zeros": old_tail0, "new_tail100_zeros": new_tail0,
                "old_std": round(float(old.std()), 5),
                "new_std": round(float(normalized.std()), 5),
                "reordered": did_reorder,
            })
            n_done += 1
            logger.info(f"[{n_done}/{len(targets)}] {rid} ({split}) ok, "
                        f"tail0: {old_tail0}->{new_tail0}")
        except Exception as e:
            logger.error(f"失败 {rid}: {e}")

    dt = time.time() - t0
    summary = {
        "n_targets": len(targets), "n_done": n_done, "elapsed_s": round(dt, 1),
        "records": report,
        "any_new_tail_zero": any(r["new_tail100_zeros"] > 0 for r in report),
        "old_tail_zero_total": sum(r["old_tail100_zeros"] for r in report),
    }
    with open(rep_dir / "reprocess_report.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    logger.info(f"完成 {n_done}/{len(targets)}，耗时 {dt:.0f}s → "
                f"{rep_dir / 'reprocess_report.json'}")


if __name__ == "__main__":
    main()
