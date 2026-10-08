#!/usr/bin/env python3
"""患者级重划分（第二次检查报告 2A 🔴-2 修复执行）。

把记录级划分改为患者级划分：
    - 同一 patient_id 的所有记录必须落在同一划分（train/val/test）
    - PTB-XL 用官方 ptbxl_database.csv 的 patient_id；其他源按记录归患者
    - 患者级 80/10/10 随机分配（seed=42，患者级无跨划分泄漏）
产出：
    - data/physionet2020/processed_5k/{train,val,test}_manifest.json（患者级）
    - data/physionet2020/ecgfounder_features/{split}_{features,labels,ids}（重排）
    - outputs/patient_level/patient_split_report.json（统计+泄漏验证）
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DATA = Path(r"data/physionet2020/processed_5k")
FEAT = Path(r"data/physionet2020/ecgfounder_features")
OUT = Path(r"outputs/patient_level")
OUT.mkdir(parents=True, exist_ok=True)

rng = random.Random(42)

# ---- 1. 读取 ----
# 2026-08-29 修复（顺序确定性）：划分结果依赖输入记录顺序（患者分组后的
# shuffle 顺序），旧版直接读当前 manifest 导致"重跑 = 换顺序 = 换划分"。
# 现固定为权威输入顺序：.bak_old27（relabel 前 manifest，保序过滤）——
# 该顺序与 2026-08-28 首次执行完全一致，保证重跑幂等（输出恒为权威划分）。
rec_pat = json.load(open(OUT / "record_patient.json", encoding="utf-8"))

from src.data_pipeline.label_extractor import LabelExtractor  # noqa: E402
_le = LabelExtractor(num_classes=27)

records = []
bak_used = False
for split in ("train", "val", "test"):
    bak = Path(f"{DATA}/{split}_manifest.json.bak_old27")
    if bak.exists():
        bak_used = True
        m = json.load(open(bak, encoding="utf-8"))
        for rec in m["files"]:
            dx = rec.get("dx_codes") or []
            if dx and _le.encode(dx).sum() > 0:
                records.append(rec)
    else:
        # 4C 审查修复：🟡-8 —— 无 .bak_old27 时回退分支统一与 bak 分支相同的
        # 过滤逻辑（只保留有评分类标签的记录），否则会得到 43101 条并触发下方
        # 37749 断言。注意：无 bak 时无法保证输入顺序与权威首跑一致，结果可能
        # 不可复现，仅作为兜底；正常流程应先恢复 .bak_old27。
        m = json.load(open(DATA / f"{split}_manifest.json", encoding="utf-8"))
        for rec in m["files"]:
            dx = rec.get("dx_codes") or []
            if dx and _le.encode(dx).sum() > 0:
                records.append(rec)
print(f"输入顺序: {'bak_old27 权威序' if bak_used else '当前 manifest 序（无 bak，结果可能不可复现）'}")
print(f"总记录: {len(records)}")
assert len(records) == 37749, len(records)

# ---- 2. 患者分组 ----
pat_recs = defaultdict(list)
for rec in records:
    key = f"{rec['source']}_{rec['record_id']}"
    pat_recs[rec_pat[key]].append(rec)

pats = list(pat_recs.keys())
rng.shuffle(pats)
n = len(pats)
n_tr = int(n * 0.8)
n_va = int(n * 0.1)
assign = {}
for i, p in enumerate(pats):
    assign[p] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"

# ---- 3. 新 manifest ----
new_splits = {"train": [], "val": [], "test": []}
for p, recs in pat_recs.items():
    new_splits[assign[p]].extend(recs)

# 2026-08-29 修复：权威划分锚点断言（30232/3763/3754）——
# 防止任何输入顺序/算法变化静默产生不同划分（论文数字与评测均基于此划分）
sizes = {k: len(v) for k, v in new_splits.items()}
print("患者级划分:", sizes)
if sizes != {"train": 30232, "val": 3763, "test": 3754}:
    raise SystemExit(
        f"患者级划分 {sizes} 与权威划分 (30232/3763/3754) 不一致——"
        "请检查输入顺序或 record_patient.json，禁止覆盖权威划分")

for split, files in new_splits.items():
    # 2026-08-29 修复：从 .bak_old27 读取的记录携带 relabel 前旧版 labels
    # （码-名错配）——写回前必须用当前 LabelExtractor 重新编码，否则覆盖正确标签
    for rec in files:
        rec["labels"] = _le.encode(rec.get("dx_codes") or []).tolist()
    # 保持与 relabel 一致的确定性排序
    files.sort(key=lambda r: (r["source"], r["record_id"]))
    out = DATA / f"{split}_manifest.json"
    json.dump({"files": files, "count": len(files)}, open(out, "w", encoding="utf-8"),
              indent=2)
    print(f"{split}: {len(files)} 条")

# ---- 4. 特征缓存重排 ----
# 先加载全部旧划分文件，id -> (旧文件, 行号) 全局索引，
# 再按新划分组装（患者级重划分会跨旧划分移动记录）
old_feats, old_labels, old_ids = {}, {}, {}
for split in ("train", "val", "test"):
    old_ids[split] = json.load(open(FEAT / f"{split}_ids.json", encoding="utf-8"))
    old_feats[split] = np.load(FEAT / f"{split}_features.npy")
    old_labels[split] = np.load(FEAT / f"{split}_labels.npy")

id_loc = {}
for split in ("train", "val", "test"):
    for i, eid in enumerate(old_ids[split]):
        id_loc[eid] = (split, i)

for split, files in new_splits.items():
    want = [f"{r['source']}_{r['record_id']}" for r in files]
    rows = [id_loc[eid] for eid in want]  # (旧文件, 旧行号)
    new_feats = np.stack([old_feats[s][i] for s, i in rows])
    new_labels = np.stack([old_labels[s][i] for s, i in rows])
    np.save(FEAT / f"{split}_features.npy", new_feats)
    np.save(FEAT / f"{split}_labels.npy", new_labels)
    json.dump(want, open(FEAT / f"{split}_ids.json", "w", encoding="utf-8"))
    print(f"特征重排 {split}: {new_feats.shape} / 行对应 {len(want)}")

# ---- 5. 泄漏验证 + 分布统计 ----
train_pats = {rec_pat[f"{r['source']}_{r['record_id']}"] for r in new_splits["train"]}
val_pats = {rec_pat[f"{r['source']}_{r['record_id']}"] for r in new_splits["val"]}
test_pats = {rec_pat[f"{r['source']}_{r['record_id']}"] for r in new_splits["test"]}
leak = {
    "train_test": len(train_pats & test_pats),
    "train_val": len(train_pats & val_pats),
    "val_test": len(val_pats & test_pats),
}
print(f"患者泄漏验证（应为 0）: {leak}")
assert all(v == 0 for v in leak.values())

# 类别分布（test）
dist = Counter()
for m in new_splits["test"]:
    for i, v in enumerate(m["labels"]):
        if v > 0:
            dist[i] += 1
from src.data_pipeline.label_extractor import LabelExtractor  # noqa: E402
le = LabelExtractor(num_classes=27)
dist_named = {le.class_names[i]: c for i, c in dist.most_common()}
print("新 test 逐类分布（前 10）:", dict(list(dist_named.items())[:10]))

report = {
    "n_records": len(records),
    "n_patients": len(pats),
    "split_patients": {"train": len(train_pats), "val": len(val_pats), "test": len(test_pats)},
    "split_records": {k: len(v) for k, v in new_splits.items()},
    "leak_verification": leak,
    "test_class_distribution": dist_named,
}
json.dump(report, open(OUT / "patient_split_report.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
print(f"报告 → {OUT / 'patient_split_report.json'}")
