# -*- coding: utf-8 -*-
"""审查补充：E09680 补录、填充记录分布、遗留权重盘点"""
import json
import sys
from pathlib import Path

import numpy as np

print("=== 1. E09680 补录核对 ===", flush=True)
with open("data/physionet2020/processed_5k/train_manifest.json", encoding="utf-8") as f:
    train = json.load(f)["files"]
e = [m for m in train if m["source"] == "georgia" and m["record_id"] == "E09680"]
if e:
    x = np.load(Path("data/physionet2020/processed_5k") / e[0]["signal_file"])
    print(f"  在 train: 是; labels={e[0]['labels'][:5]}...; shape={x.shape}; std={x.std():.4f}", flush=True)
    old = json.load(open("data/physionet2020/processed/train_manifest.json", encoding="utf-8"))["files"]
    oe = [m for m in old if m["source"] == "georgia" and m["record_id"] == "E09680"][0]
    print(f"  标签与旧版一致: {oe['labels'] == e[0]['labels']}", flush=True)
else:
    print("  !! E09680 不在 train manifest 中", flush=True)

print("\n=== 2. 零填充记录（<9.5s）分布 ===", flush=True)
from collections import Counter
padded = []
for s in ("train", "val", "test"):
    with open(f"data/physionet2020/processed_5k/{s}_manifest.json", encoding="utf-8") as f:
        for m in json.load(f)["files"]:
            d = m.get("duration_original")
            if isinstance(d, (int, float)) and d < 9.5:
                padded.append((s, m["source"], m["record_id"], round(d, 1)))
print(f"  共 {len(padded)} 条", flush=True)
for row in padded[:8]:
    print(f"    {row}", flush=True)
print(f"  来源分布: {dict(Counter(p[1] for p in padded))}", flush=True)

print("\n=== 3. 遗留权重盘点（区分公平/泄漏版）===", flush=True)
for name in ["multimodal", "simclr_xresnet", "xresnet1d_101", "ecg_transformer",
             "ecg_transformer_fp32", "ECGFounder"]:
    p = Path("checkpoints") / name
    files = list(p.rglob("*")) if p.exists() else []
    size = sum(f.stat().st_size for f in files if f.is_file()) / 1024 / 1024
    print(f"  checkpoints/{name}: {'存在' if p.exists() else '不存在'} ({size:.1f}MB)", flush=True)
