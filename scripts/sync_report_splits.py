#!/usr/bin/env python3
"""第三轮审查 3F-R3-02：report_splits.json 同步为患者级划分。
report_ids 中仍存在的记录更新为当前 manifest split（患者级）；
已被 relabel 剔除的 185 条（无评分类标签）保留旧值。

⚠️ 4C 审查修复：🟡-6 数据现状说明（不改逻辑）——嵌入缓存与现役报告集脱节：
    report_ids.json 仅 15110 条，而 ptbxl_reports.json 现为 21566 条；本脚本只
    同步 report_ids 在册记录的 split，覆盖不了 6641 条无嵌入记录（部分 PTB-XL
    无嵌入），属数据级现状，需重跑 extract_text_embeddings.py 才能补齐。
"""
import json
from pathlib import Path

SPLITS = Path("data/physionet2020/text_embeddings/report_splits.json")
IDS = Path("data/physionet2020/text_embeddings/report_ids.json")
DATA = Path("data/physionet2020/processed_5k")

old = json.load(open(SPLITS, encoding="utf-8"))
ids = json.load(open(IDS, encoding="utf-8"))

manifest = {}
for s in ("train", "val", "test"):
    for x in json.load(open(DATA / f"{s}_manifest.json", encoding="utf-8"))["files"]:
        if x["source"] == "ptb-xl":
            manifest[x["record_id"]] = s

n_upd = 0
n_keep = 0
for rid in ids:
    if rid in manifest:
        old[rid] = manifest[rid]
        n_upd += 1
    else:
        n_keep += 1

json.dump(old, open(SPLITS, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(f"更新 {n_upd} 条为患者级 split, 保留 {n_keep} 条（relabel 剔除记录）")
# 验证
bad = 0
for rid in ids:
    if rid in manifest and old.get(rid) != manifest[rid]:
        bad += 1
print(f"验证: 不一致 {bad} 条")
