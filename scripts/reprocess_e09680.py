# ⚠️ DEPRECATED（2026-08-24，检查报告第六步）: 一次性重处理脚本（同上，仅存档）
# -*- coding: utf-8 -*-
"""恢复并重处理 georgia/E09680（v4：解压到工作区临时目录，绕过外部目录只读限制）"""
import zipfile
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import wfdb

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
from src.data_pipeline.loader import ECGLoader  # noqa: E402
import preprocess_data_5k as p5  # noqa: E402

# 4C 审查修复：🟡-7 —— 删除硬编码 Desktop\ECG\压缩包\...zip 绝对路径，改用自动探测。
def _default_zip():
    stem = "classification-of-12-lead-ecgs-the-physionetcomputing-" \
           "in-cardiology-challenge-2020-1.0.2"
    candidates = [
        Path.home() / "Desktop" / "ECG" / "压缩包" / f"{stem}.zip",
        Path.home() / "Desktop" / "ecg资料" / "压缩包" / f"{stem}.zip",
        Path.home() / "Desktop" / "ECG" / f"{stem}.zip",
        Path.home() / "Desktop" / "ecg资料" / f"{stem}.zip",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


ZIP = _default_zip()
if not ZIP:
    sys.exit("未探测到原始 zip 压缩包——请修正探测逻辑（4C 审查修复：🟡-7）")
TMP = Path("data/_recovered_e09680")
NEW_DIR = Path("data/physionet2020/processed_5k")

# ---- 1. 解压相关条目到工作区临时目录 ----
TMP.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(ZIP) as z:
    names = [n for n in z.namelist() if "09680" in n and not n.endswith("/")]
    print(f"zip 条目: {len(names)}")
    for n in names:
        dest = TMP / Path(n).name
        with z.open(n) as src, open(dest, "wb") as dst:
            dst.write(src.read())
        print(f"  解压 {n} → {dest} ({dest.stat().st_size} bytes)")

# ---- 2. 用 wfdb 读取并处理 ----
record = wfdb.rdrecord(str(TMP / "E09680"))
signal = record.p_signal.T.astype(np.float32)
print(f"信号: {signal.shape}, fs={record.fs}, 导联={record.sig_name}")

signal, did_reorder = p5.reorder_leads(signal, record.sig_name)
filtered = p5.filter_bandpass(signal, record.fs)
resampled = p5.resample_to_500(filtered, record.fs)
seg, pl, pr = p5.segment_to_5000(resampled)
norm = p5.official_zscore(seg, pl, pr)
print(f"处理后: {norm.shape}, mean={norm.mean():.4f}, std={norm.std():.4f}")

np.save(NEW_DIR / "georgia_E09680.npy", norm)

# ---- 3. 从旧 train manifest 取标签并补录 ----
with open("data/physionet2020/processed/train_manifest.json", encoding="utf-8") as f:
    old_train = json.load(f)["files"]
entry_old = next(m for m in old_train
                 if m["source"] == "georgia" and m["record_id"] == "E09680")

entry = {
    "record_id": "E09680", "source": "georgia",
    "signal_file": "georgia_E09680.npy",
    "fs_original": float(record.fs), "fs_target": 500.0,
    "duration_original": signal.shape[1] / record.fs,
    "signal_shape": list(norm.shape),
    "age": entry_old.get("age"), "sex": entry_old.get("sex"),
    "dx_codes": entry_old.get("dx_codes", []),
    "labels": entry_old.get("labels", [0] * 27),
    "has_labels": entry_old.get("has_labels", False),
}

with open(NEW_DIR / "train_manifest.json", encoding="utf-8") as f:
    train_new = json.load(f)
train_new["files"].append(entry)
train_new["count"] = len(train_new["files"])
with open(NEW_DIR / "train_manifest.json", "w", encoding="utf-8") as f:
    json.dump(train_new, f, indent=2)
print(f"已补录 train manifest → {train_new['count']} 条")

# ---- 4. 清理临时目录 ----
shutil.rmtree(TMP, ignore_errors=True)
print("临时目录已清理。")
