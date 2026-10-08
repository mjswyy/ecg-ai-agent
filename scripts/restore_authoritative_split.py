#!/usr/bin/env python3
"""写回权威患者级划分：manifest + 特征缓存重排 + 泄漏验证。

⚠️ 4C 审查修复：🟡-5 —— 输入 `_authoritative_ids.json` 来源说明：
    该文件是历史备份产物（2026-08-28 首次患者级划分时手工留存的权威
    train/val/test id 快照），全仓库无代码生产者——patient_level_split.py
    产出的是 patient_split_report.json，patient_split_prep.py 产出的是
    record_patient.json，均不写 _authoritative_ids.json。本脚本仅用于从该
    快照**恢复**权威划分；换机/仓库重灌后该文件不存在会 FileNotFoundError，
    此时应改用 scripts/patient_level_split.py 重新生成患者级划分，而非运行本脚本。
"""
import json
from pathlib import Path

import numpy as np

DATA = Path("data/physionet2020/processed_5k")
FEAT = Path("data/physionet2020/ecgfounder_features")

auth = json.load(open("outputs/patient_level/_authoritative_ids.json", encoding="utf-8"))
rec_pat = json.load(open("outputs/patient_level/record_patient.json", encoding="utf-8"))

# 当前 manifest 记录（含 labels 等完整字段）→ id 映射
cur = {}
for s in ("train", "val", "test"):
    for x in json.load(open(DATA / f"{s}_manifest.json", encoding="utf-8"))["files"]:
        cur[f"{x['source']}_{x['record_id']}"] = x

# 1) 写回 manifest（权威划分）
# 4C 审查修复：🟠-2 —— manifest 与特征缓存必须按同一顺序写回。旧版此处对
# manifest 按 (source, record_id) 排序，而第 2 步特征按 _authoritative_ids.json
# 原始（未排序）序组装，导致运行后 manifest 行 ↔ 特征行错位（下游严格逐行对齐，
# 泄漏/evalset 校验均查不出该错位）。现统一按 _authoritative_ids.json 原顺序，
# 不再排序，并在第 2 步加"行对齐"断言兜底。
manifest_order = {}
for split, ids in auth.items():
    files = [cur[i] for i in ids]  # 保持 _authoritative_ids.json 原顺序，不排序
    manifest_order[split] = [f"{r['source']}_{r['record_id']}" for r in files]
    json.dump({"files": files, "count": len(files)},
              open(DATA / f"{split}_manifest.json", "w", encoding="utf-8"), indent=2)
    print(f"{split}: {len(files)} 条 manifest 已恢复")

# 2) 特征缓存重排（按权威 id 从当前特征组装）
old_feats, old_labels, old_ids = {}, {}, {}
for split in ("train", "val", "test"):
    old_ids[split] = json.load(open(FEAT / f"{split}_ids.json", encoding="utf-8"))
    old_feats[split] = np.load(FEAT / f"{split}_features.npy")
    old_labels[split] = np.load(FEAT / f"{split}_labels.npy")

id_loc = {}
for split in ("train", "val", "test"):
    for i, eid in enumerate(old_ids[split]):
        id_loc[eid] = (split, i)

for split, ids in auth.items():
    rows = [id_loc[eid] for eid in ids]
    new_feats = np.stack([old_feats[s][i] for s, i in rows])
    new_labels = np.stack([old_labels[s][i] for s, i in rows])
    np.save(FEAT / f"{split}_features.npy", new_feats)
    np.save(FEAT / f"{split}_labels.npy", new_labels)
    json.dump(ids, open(FEAT / f"{split}_ids.json", "w", encoding="utf-8"))
    # 4C 审查修复：🟠-2 —— 行对齐断言：manifest 行序 == ids 行序 == features/labels 行数
    assert manifest_order[split] == ids, (
        f"{split} manifest 行序与特征 ids 行序不一致（4C 审查修复：🟠-2）")
    assert new_feats.shape[0] == len(ids) == new_labels.shape[0], (
        f"{split} features/labels 行数与 ids 长度不一致（4C 审查修复：🟠-2）")
    print(f"特征重排 {split}: {new_feats.shape}")

# 3) 泄漏验证
train_pats = {rec_pat[i] for i in auth["train"]}
val_pats = {rec_pat[i] for i in auth["val"]}
test_pats = {rec_pat[i] for i in auth["test"]}
leak = {"train_test": len(train_pats & test_pats),
        "train_val": len(train_pats & val_pats),
        "val_test": len(val_pats & test_pats)}
print("泄漏验证:", leak)
assert all(v == 0 for v in leak.values())

# 4) 与 evalset 一致性（evalset 基于权威 test 构建）
evalset = json.load(open("outputs/agent_evalset/evalset.json", encoding="utf-8"))
test_ids = set(auth["test"])
ev_in_test = all(f"{it['source']}_{it['record_id']}" in test_ids
                 for it in evalset["items"])
print(f"evalset {evalset['count']} 条全部属于权威 test: {ev_in_test}")
assert ev_in_test

print("\n✅ 权威划分 + 特征缓存 + evalset 一致性全部确认")
