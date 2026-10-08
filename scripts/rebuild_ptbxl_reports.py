#!/usr/bin/env python3
"""验证映射 + 从官方 ptbxl_database.csv 重建 ptbxl_reports.json（2E-Y5 彻底修复）。

第三轮审查 3F-R3-01 修复：重建时保留 split 字段（从旧备份回填），
并仅输出 processed_5k 实际使用的记录（与 manifest 对齐），
不再把整份 CSV 21799 条全量写入——旧版丢弃 split 使
train_ecg_text_clip 的"ECG split == report split"防泄漏双保险失效。

⚠️ 4C 审查修复：🟠-3 数据缺口说明（数据级，无法代码修复）——
    本地 data/physionet2020/ptbxl_database.csv 为 21799 行（min ecg_id=1,
    max ecg_id=21837），存在 38 个 ecg_id 空洞（官方应为 21837 行，本仓库
    版本被截断 38 行），导致 38 条 PTB-XL 记录（train 36 / test 2，其中 2 条
    在权威 test 划分）无报告文本。rebuild 对这 38 条只能跳过，最终
    ptbxl_reports.json 为 21566 条（manifest ptb-xl 21604 - 38 缺口）。
    该缺口直接影响报告臂评测完整性，需补全 CSV 才能修复——代码层面无法
    凭空生成缺失报告文本。
"""
import csv
import json
from pathlib import Path

CSV = Path(r"data/physionet2020/ptbxl_database.csv")
OUT = Path(r"data/physionet2020/processed_5k/ptbxl_reports.json")
BAK = Path(r"data/physionet2020/processed_5k/ptbxl_reports.json.mojibake_bak")
DATA = Path(r"data/physionet2020/processed_5k")

by_eid = {}
with open(CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        by_eid[int(row["ecg_id"])] = row["report"]

# 1) 映射验证：旧报告 HR01278 与官方 ecg_id=1278
# 4C 审查修复：💡-13 —— .mojibake_bak 由 fix_ptbxl_mojibake.py 生成，全新
# checkout 无该文件时旧版直接 FileNotFoundError；现降级为 {}（split 兜底走
# manifest_split），仅告警，消除脚本间隐式执行顺序依赖。
if BAK.exists():
    old = json.load(open(BAK, encoding="utf-8"))
else:
    old = {}
    print(f"⚠️ 未找到 {BAK}（fix_ptbxl_mojibake.py 未运行），split 仅按 manifest 回填")
for rid in ["HR01278", "HR01809", "HR02598", "HR00004"]:
    eid = int(rid[2:])
    print(f"{rid} (ecg_id={eid}):")
    o = old.get(rid)
    print(f"   旧: {o['report'][:90]!r}" if o else "   旧: (无备份)")
    print(f"   官: {by_eid.get(eid, 'MISSING')[:90]!r}")

# 2) split 确定：优先当前患者级 manifest（权威划分），旧备份仅兜底
# （旧备份为记录级划分，直接回填会与 train_ecg_text_clip 的红线检查冲突）
def old_split(rid):
    o = old.get(rid)
    if isinstance(o, dict) and o.get("split"):
        return o["split"]
    return None


manifest_split = {}
for split in ("train", "val", "test"):
    mp = DATA / f"{split}_manifest.json"
    if not mp.exists():
        continue
    m = json.load(open(mp, encoding="utf-8"))
    for item in m["files"]:
        if item["source"] == "ptb-xl":
            manifest_split[item["record_id"]] = split

# 3) 重建：仅输出 processed_5k manifest 实际使用的 ptb-xl 记录（带 split）
used_ids = set()
for split in ("train", "val", "test"):
    mp = DATA / f"{split}_manifest.json"
    if not mp.exists():
        continue
    for item in json.load(open(mp, encoding="utf-8"))["files"]:
        if item["source"] == "ptb-xl":
            used_ids.add(item["record_id"])

reports = {}
n_kept = 0
n_split = 0
missing = []  # 4C 复查（β）：缺失清单落盘，下游可程序化感知（旧版仅 print）
for rid in sorted(used_ids):
    eid = int(rid[2:])
    rep = by_eid.get(eid)
    if rep is None:
        # 4C 审查修复：🟠-3 —— 此处跳过即 docstring 所述 38 条 CSV 空洞记录
        #（数据级缺口，train 36 / test 2，无法代码修复）
        print(f"⚠️ 官方 CSV 缺记录 {rid}，跳过")
        missing.append(rid)
        continue
    sp = manifest_split.get(rid) or old_split(rid) or "unknown"
    if sp != "unknown":
        n_split += 1
    reports[rid] = {"report": rep, "split": sp}
    n_kept += 1

OUT.write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
if missing:
    miss_out = DATA / "ptbxl_reports_missing.json"
    miss_out.write_text(
        json.dumps({"n": len(missing), "record_ids": missing}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(f"⚠️ {len(missing)} 条记录无官方报告文本 → {miss_out}（需补全 CSV）")
print(f"\n重建完成: {n_kept} 条（仅 processed_5k 使用记录，含 split 回填 {n_split} 条）→ {OUT}")
