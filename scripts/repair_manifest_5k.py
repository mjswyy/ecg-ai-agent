#!/usr/bin/env python3
# ⚠️ DEPRECATED（2026-08-24，检查报告第六步）: 一次性 manifest 修复脚本（仅存档）
"""修复 processed_5k 的 manifest 缺条目问题（M0.1 收尾）。

背景: --skip-existing 旧版逻辑跳过了冒烟测试的 10 个已有文件，导致新 manifest
比旧划分少 11 条。本脚本从旧 manifest 找出缺失条目:
    - npy 文件存在 → 直接补进对应 split 的 manifest（labels 用旧值）
    - npy 文件不存在 → 单独重处理该记录后补录
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
# 4C 审查修复：🟡-7 —— 允许 import 同目录 preprocess_data_5k 的探测函数
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from preprocess_data_5k import _default_raw_dir  # noqa: E402

OLD_DIR = Path("data/physionet2020/processed")
NEW_DIR = Path("data/physionet2020/processed_5k")
# 4C 审查修复：🟡-7 —— 删除硬编码 Desktop\ECG 绝对路径，改用与
# preprocess_data_5k.py 一致的 _default_raw_dir() 自动探测。
_raw = _default_raw_dir()
if not _raw:
    sys.exit("未探测到原始数据目录——请设置 --raw-dir 或修正探测逻辑（4C 审查修复：🟡-7）")
RAW_DIR = Path(_raw)


def load_manifest(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    old = {s: load_manifest(OLD_DIR / f"{s}_manifest.json")["files"]
           for s in ("train", "val", "test")}
    new = {s: load_manifest(NEW_DIR / f"{s}_manifest.json")["files"]
           for s in ("train", "val", "test")}

    present = set()
    for s in ("train", "val", "test"):
        for m in new[s]:
            present.add((m["source"], m["record_id"]))

    missing = []
    for s in ("train", "val", "test"):
        for m in old[s]:
            if (m["source"], m["record_id"]) not in present:
                missing.append((s, m))

    print(f"缺失条目: {len(missing)}")
    for s, m in missing:
        print(f"  [{s}] {m['source']}/{m['record_id']}  npy存在={ (NEW_DIR / m['signal_file']).exists() }")

    # 1) 文件已存在的直接补录
    for s, m in missing:
        if (NEW_DIR / m["signal_file"]).exists():
            entry = {
                "record_id": m["record_id"],
                "source": m["source"],
                "signal_file": m["signal_file"],
                "fs_original": m.get("fs_original", 500.0),
                "fs_target": 500.0,
                "duration_original": m.get("duration_original"),
                "signal_shape": [12, 5000],
                "age": m.get("age"),
                "sex": m.get("sex"),
                "dx_codes": m.get("dx_codes", []),
                "labels": m.get("labels", [0] * 27),
                "has_labels": m.get("has_labels", len(m.get("dx_codes", [])) > 0),
            }
            new[s].append(entry)
            print(f"补录: [{s}] {m['source']}/{m['record_id']}")

    # 2) 文件不存在的重新处理
    need_reproc = [(s, m) for s, m in missing if not (NEW_DIR / m["signal_file"]).exists()]
    if need_reproc:
        print(f"\n需要重新处理 {len(need_reproc)} 条:")
        sys.path.insert(0, str(Path(__file__).parent))
        from src.data_pipeline.loader import ECGLoader
        import preprocess_data_5k as p5

        loader = ECGLoader(RAW_DIR)
        for s, m in need_reproc:
            key = f"{m['source']}/{m['record_id']}"
            sample = loader.load_record(key)
            if sample is None:
                print(f"  !! 无法加载 {key}，保持缺失")
                continue
            signal, _ = p5.reorder_leads(sample.signal, sample.lead_names)
            filtered = p5.filter_bandpass(signal, sample.fs)
            resampled = p5.resample_to_500(filtered, sample.fs)
            seg, pl, pr = p5.segment_to_5000(resampled)
            norm = p5.official_zscore(seg, pl, pr)
            np.save(NEW_DIR / m["signal_file"], norm)
            entry = {
                "record_id": m["record_id"], "source": m["source"],
                "signal_file": m["signal_file"],
                "fs_original": float(sample.fs), "fs_target": 500.0,
                "duration_original": sample.duration,
                "signal_shape": list(norm.shape),
                "age": sample.age if sample.age is not None else None,
                "sex": sample.sex, "dx_codes": sample.dx_codes,
                "labels": m.get("labels", [0] * 27),
                "has_labels": len(sample.dx_codes) > 0,
            }
            new[s].append(entry)
            print(f"  重处理完成: [{s}] {key}")

    # 3) 保存
    for s in ("train", "val", "test"):
        with open(NEW_DIR / f"{s}_manifest.json", "w", encoding="utf-8") as f:
            json.dump({"files": new[s], "count": len(new[s])}, f, indent=2)
        print(f"保存 {s}: {len(new[s])} 条")

    print("\n=== 与旧划分核对 ===")
    for s in ("train", "val", "test"):
        old_ids = {(m["source"], m["record_id"]) for m in old[s]}
        new_ids = {(m["source"], m["record_id"]) for m in new[s]}
        print(f"  {s}: 旧 {len(old_ids)} / 新 {len(new_ids)} / 差异 {len(old_ids ^ new_ids)}")


if __name__ == "__main__":
    main()
