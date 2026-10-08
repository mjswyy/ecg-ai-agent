#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""官方 27 评分类体系重贴标签（第二次审查 R1 方案 (a) 执行脚本）。

1. 备份并按官方 dx_mapping_scored.csv 27 类重贴三个划分 manifest 的 labels；
   丢弃"无任何评分类标签"的记录（官方基准不对其评分）。
2. 同步重写 ECGFounder 特征缓存（features/labels/ids 三件套按行过滤）。
3. 重新生成持久化映射 data/physionet2020/labels/snomed_to_class.json。
4. 输出 relabel_report.json（新旧计数、逐类分布、丢弃样例）。

用法: python scripts/relabel_official_27.py
"""

import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_pipeline.label_extractor import LabelExtractor

DATA = Path("data/physionet2020/processed_5k")
FEAT = Path("data/physionet2020/ecgfounder_features")
REP = Path("outputs/relabel_official27")
REP.mkdir(parents=True, exist_ok=True)

le = LabelExtractor(num_classes=27)
report = {"classes": le.class_names, "splits": {}}

for split in ("train", "val", "test"):
    mp = DATA / f"{split}_manifest.json"
    bak = Path(str(mp) + ".bak_old27")
    if not bak.exists():
        shutil.copy2(mp, bak)

    with open(mp, encoding="utf-8") as f:
        manifest = json.load(f)["files"]

    kept, dropped = [], []
    for m in manifest:
        new_labels = le.encode(m.get("dx_codes", []))
        if new_labels.sum() == 0:
            dropped.append(m)
        else:
            m["labels"] = new_labels.tolist()
            kept.append(m)

    with open(mp, "w", encoding="utf-8") as f:
        json.dump({"files": kept, "count": len(kept)}, f, indent=2)

    report["splits"][split] = {
        "old_count": len(manifest), "new_count": len(kept),
        "dropped": len(dropped),
        "dropped_dx_samples": [
            (m["source"], m["record_id"], m.get("dx_codes", [])[:6])
            for m in dropped[:5]],
    }

    # ---- 特征缓存按行过滤 + 重写 labels ----
    ids = json.load(open(FEAT / f"{split}_ids.json", encoding="utf-8"))
    feats = np.load(FEAT / f"{split}_features.npy")
    labels = np.load(FEAT / f"{split}_labels.npy")
    keep_ids = {f"{m['source']}_{m['record_id']}" for m in kept}
    new_labels_by_id = {
        f"{m['source']}_{m['record_id']}": m["labels"] for m in kept}
    sel = [i for i, eid in enumerate(ids) if eid in keep_ids]
    assert len(sel) == len(kept), (split, len(sel), len(kept))
    assert all(ids[i] in new_labels_by_id for i in sel)
    np.save(FEAT / f"{split}_features.npy", feats[sel])
    np.save(FEAT / f"{split}_labels.npy",
            np.asarray([new_labels_by_id[ids[i]] for i in sel],
                       dtype=np.float32))
    json.dump([ids[i] for i in sel],
              open(FEAT / f"{split}_ids.json", "w", encoding="utf-8"))
    report["splits"][split]["features_rows_kept"] = len(sel)
    print(f"{split}: {len(manifest)} -> {len(kept)} (丢弃 {len(dropped)}), "
          f"特征行 {len(sel)}")

# ---- 持久化映射 ----
labels_dir = Path("data/physionet2020/labels")
labels_dir.mkdir(parents=True, exist_ok=True)
le.save_mapping(labels_dir / "snomed_to_class.json")

# ---- 新体系逐类分布（test）----
with open(DATA / "test_manifest.json", encoding="utf-8") as f:
    test = json.load(f)["files"]
dist = Counter()
for m in test:
    for i, v in enumerate(m["labels"]):
        if v > 0:
            dist[le.class_names[i]] += 1
report["test_class_distribution"] = dict(dist.most_common())
print("test 逐类正样本分布:")
for k, v in dist.most_common():
    print(f"  {k}: {v}")

json.dump(report, open(REP / "relabel_report.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"报告 → {REP / 'relabel_report.json'}")
