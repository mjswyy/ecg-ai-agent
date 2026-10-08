#!/usr/bin/env python3
"""患者级重划分 — 数据准备（第一阶段，只读统计+映射构建）。

复现第二次检查报告 2A 🔴-2：PTB-XL 患者跨 train/test 泄漏规模，
并构建 patient_id 映射，为患者级重划分做准备。

方案（检查报告 4.2）:
    - PTB-XL: record_id 'HR00004' → ecg_id 4 → ptbxl_database.csv 的 patient_id
    - 其他源: 无患者信息，按记录归患者（record_id 即 patient_id）
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r".")
PROC = Path(r"data/physionet2020/processed_5k")

# ---- 1. PTB-XL patient 映射 ----
eid2pid = {}
with open(r"data/physionet2020/ptbxl_database.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        eid2pid[int(row["ecg_id"])] = row["patient_id"]

def record_patient(rec):
    """record -> patient_id（ptb-xl 用官方 patient_id；其他源按记录）。"""
    src = rec["source"]
    rid = rec["record_id"]
    if src == "ptb-xl":
        eid = int(rid[2:])  # 'HR00004' -> 4
        return f"ptbxl_p{eid2pid.get(eid, eid)}"
    return f"{src}_{rid}"

# ---- 2. 统计当前划分的患者泄漏 ----
splits = {}
for split in ("train", "val", "test"):
    m = json.load(open(PROC / f"{split}_manifest.json", encoding="utf-8"))
    splits[split] = m["files"]

pat_split = defaultdict(set)   # patient -> set(splits)
rec_pat = {}
for split, files in splits.items():
    for rec in files:
        p = record_patient(rec)
        rec_pat[(rec["source"], rec["record_id"])] = p
        pat_split[p].add(split)

cross = {p: s for p, s in pat_split.items() if len(s) > 1}
n_rec_cross = sum(
    1 for rec in rec_pat if len(pat_split[rec_pat[rec]]) > 1)

# 与检查报告数字对照：train∩test 547 名患者
train_pats = {rec_pat[(r["source"], r["record_id"])] for r in splits["train"]}
test_pats = {rec_pat[(r["source"], r["record_id"])] for r in splits["test"]}
val_pats = {rec_pat[(r["source"], r["record_id"])] for r in splits["val"]}
tr_te = len(train_pats & test_pats)
tr_va = len(train_pats & val_pats)
va_te = len(val_pats & test_pats)

# 跨划分记录数（患者泄漏影响到的记录）
def n_records(pats_set):
    return sum(1 for p in rec_pat.values() if p in pats_set)

print("=== 当前划分（记录级）患者泄漏统计 ===")
print(f"train 患者: {len(train_pats)}, val: {len(val_pats)}, test: {len(test_pats)}")
print(f"train∩test 患者: {tr_te}（跨划分记录 {n_records(train_pats & test_pats)} 条）")
print(f"train∩val 患者: {tr_va}（跨划分记录 {n_records(train_pats & val_pats)} 条）")
print(f"val∩test 患者: {va_te}（跨划分记录 {n_records(val_pats & test_pats)} 条）")
print(f"涉及任意跨划分的记录总数: {n_rec_cross} 条")

# 保存 patient 映射（供重划分脚本使用）
out_map = {f"{s}_{r}": p for (s, r), p in rec_pat.items()}
Path(r"outputs/patient_level").mkdir(parents=True, exist_ok=True)
with open(r"outputs/patient_level/record_patient.json", "w", encoding="utf-8") as f:
    json.dump(out_map, f, ensure_ascii=False, indent=1)
print(f"\n映射已保存: outputs/patient_level/record_patient.json ({len(out_map)} 条)")
