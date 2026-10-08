# -*- coding: utf-8 -*-
"""M0.1 最终校验（v3：对照现役权威口径）。

4C 审查修复：🟡-4 —— 旧版硬编码 npy=43101 并与旧 processed/ 逐条比对
labels+split，而现役 processed_5k 已由 relabel_official_27.py 重贴官方 27 类
标签、并由 patient_level_split.py 改为患者级划分（30232/3763/3754 = 37749），
旧口径必判 FAIL（误导使用者以为数据损坏）。现改为对照权威口径：
    - npy 数 = 37749；manifest 30232/3763/3754
    - 信号形状 (12, 5000)
    - manifest 行序 == ecgfounder_features ids 行序 == features/labels 行数
    - 与 record_patient.json 患者级泄漏 = 0
    - labels 与 LabelExtractor 在线编码一致
"""
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

NEW = Path("data/physionet2020/processed_5k")
FEAT = Path("data/physionet2020/ecgfounder_features")
REC_PAT_PATH = Path("outputs/patient_level/record_patient.json")

EXPECT_COUNTS = {"train": 30232, "val": 3763, "test": 3754}
EXPECT_TOTAL = 37749

ok = True


def fail(msg):
    global ok
    ok = False
    print(f"  ❌ {msg}", flush=True)


print("=== 1. npy 文件总数 ===", flush=True)
n_npy = sum(1 for _ in NEW.glob("*.npy"))
print(f"  processed_5k: {n_npy} 个 npy (期望 {EXPECT_TOTAL})", flush=True)
if n_npy != EXPECT_TOTAL:
    fail(f"npy 数 {n_npy} != {EXPECT_TOTAL}")

print("=== 2. manifest 数量与引用完整性 ===", flush=True)
random.seed(42)
manifest = {}
for s in ("train", "val", "test"):
    with open(NEW / f"{s}_manifest.json", encoding="utf-8") as f:
        data = json.load(f)["files"]
    manifest[s] = data
    if len(data) != EXPECT_COUNTS[s]:
        fail(f"{s} manifest {len(data)} != {EXPECT_COUNTS[s]}")
    missing = [m["signal_file"] for m in data if not (NEW / m["signal_file"]).exists()]
    sample = random.sample(data, min(60, len(data)))
    bad_shape = []
    for m in sample:
        x = np.load(NEW / m["signal_file"])
        if x.shape != (12, 5000):
            bad_shape.append((m["signal_file"], x.shape))
    print(f"  {s}: {len(data)} 条, 文件缺失 {len(missing)}, "
          f"抽样形状异常 {len(bad_shape)}", flush=True)
    if missing:
        fail(f"{s} 文件缺失 {len(missing)}")
    if bad_shape:
        fail(f"{s} 形状异常 {bad_shape[:3]}")

print("=== 3. 特征缓存与 manifest 逐行对齐 ===", flush=True)
for s in ("train", "val", "test"):
    ids = json.load(open(FEAT / f"{s}_ids.json", encoding="utf-8"))
    feats = np.load(FEAT / f"{s}_features.npy")
    labels = np.load(FEAT / f"{s}_labels.npy")
    manifest_order = [f"{m['source']}_{m['record_id']}" for m in manifest[s]]
    if ids != manifest_order:
        fail(f"{s} ids 行序 != manifest 行序")
    if feats.shape[0] != len(ids) or labels.shape[0] != len(ids):
        fail(f"{s} features/labels 行数 != ids 长度")
    print(f"  {s}: ids {len(ids)} 行 == features {feats.shape[0]} 行 == "
          f"labels {labels.shape[0]} 行, 顺序一致={ids == manifest_order}", flush=True)

print("=== 4. 患者级泄漏（对照 record_patient.json）===", flush=True)
rec_pat = json.load(open(REC_PAT_PATH, encoding="utf-8"))
pat_sets = {}
for s in ("train", "val", "test"):
    pat_sets[s] = {rec_pat[f"{m['source']}_{m['record_id']}"] for m in manifest[s]}
leak = {
    "train_test": len(pat_sets["train"] & pat_sets["test"]),
    "train_val": len(pat_sets["train"] & pat_sets["val"]),
    "val_test": len(pat_sets["val"] & pat_sets["test"]),
}
print(f"  泄漏: {leak}", flush=True)
if any(v != 0 for v in leak.values()):
    fail(f"患者级泄漏 != 0: {leak}")

print("=== 5. labels 与 LabelExtractor 在线编码一致 ===", flush=True)
from src.data_pipeline.label_extractor import LabelExtractor  # noqa: E402
_le = LabelExtractor(num_classes=27)
lab_mismatch = 0
for s in ("train", "val", "test"):
    for m in manifest[s]:
        want = _le.encode(m.get("dx_codes") or []).tolist()
        if m.get("labels") != want:
            lab_mismatch += 1
print(f"  labels 与在线编码不一致: {lab_mismatch}", flush=True)
if lab_mismatch:
    fail(f"{lab_mismatch} 条 labels 与 LabelExtractor 在线编码不一致")

print("=== 5b. 内容级校验：恒定非零填充段（旧 z-score bug 残留）===", flush=True)
# 5C 审查（🟡-1）：旧版只查形状，漏掉 9 条"恒定非零填充/缺 1 样本零填充"
# 的旧管线产物。修复版 official_zscore 填充区显式置零、信号区含噪声——故
# 检查"所有导联同时恒定且非零、长度≥50 样本"的段（正常 10s 信号不存在）
def _const_nonzero_runs(x, min_len=50):
    d = np.abs(np.diff(x, axis=1)).sum(axis=0)
    same = d < 1e-9
    runs = []
    start = None
    for i in range(len(same)):
        if same[i] and start is None:
            start = i
        elif not same[i] and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(same) - start >= min_len:
        runs.append((start, len(same)))
    out = []
    for a, b in runs:
        if np.abs(x[:, a:b + 1].mean()) > 1e-6:
            out.append((a, b))
    return out

bad_content = []
bad_short_pad = []
n_checked = 0
for s in ("train", "val", "test"):
    for m in manifest[s]:
        x = np.load(NEW / m["signal_file"])
        if _const_nonzero_runs(x):
            bad_content.append(m["signal_file"])
        # 复检A（🟡-3）：短记录（duration<10s）必须有 ≥1 个全零填充列——
        # 修复版 segment_to_5000 会 pad 零列；"缺 1 样本零填充"型 stale 产物
        # 信号占满 5000 无 pad，恒定非零段检测（≥50 样本）覆盖不到
        dur = m.get("duration_original")
        if dur is not None and dur < 10.0:
            n_zero_cols = int((np.abs(x).sum(axis=0) < 1e-9).sum())
            if n_zero_cols == 0:
                bad_short_pad.append(m["signal_file"])
        n_checked += 1
        if n_checked >= 8000:  # 全量 37,749 条较慢，抽样覆盖全部短记录
            break
    if n_checked >= 8000:
        break
print(f"  抽查 {n_checked} 条，恒定非零填充残留: {len(bad_content)}, "
      f"短记录缺 pad 残留: {len(bad_short_pad)}", flush=True)
if bad_content:
    fail(f"恒定非零填充残留 {bad_content[:5]}（旧 z-score bug 产物，需 fix_short_records.py 重建）")
if bad_short_pad:
    fail(f"短记录缺零填充残留 {bad_short_pad[:5]}（旧管线产物，需 fix_short_records.py 重建）")

print("=== 6. 信号数值健康度（各数据源抽 1 条）===", flush=True)
seen = set()
for m in manifest["train"]:
    if m["source"] in seen:
        continue
    seen.add(m["source"])
    x = np.load(NEW / m["signal_file"])
    print(f"  {m['source']:20s} {m['record_id']:12s} shape={x.shape} "
          f"mean={x.mean():.4f} std={x.std():.4f} nan={np.isnan(x).sum()}", flush=True)

print(f"\n{'M0.1 校验全部通过 OK' if ok else '存在异常 FAIL'}", flush=True)
sys.exit(0 if ok else 1)
