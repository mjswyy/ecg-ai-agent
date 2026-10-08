# -*- coding: utf-8 -*-
"""构造情境评测的 40 条分层样本（覆盖严重疾病 + 正常窦律 + 常见异常）"""
import json
import random
from pathlib import Path

OUT_DIR = Path("outputs/context_eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

with open("outputs/agent_evalset/evalset.json", encoding="utf-8") as f:
    evalset = json.load(f)["items"]

GROUPS = {
    # 官方 27 评分类体系（R1 修复）
    "serious": {"Atrial Fibrillation", "Atrial Flutter",
                "Complete Right Bundle Branch Block", "Bradycardia",
                "Prolonged QT Interval"},
    "blocks": {"Left Bundle Branch Block", "Right Bundle Branch Block",
               "First Degree AV Block", "Incomplete Right Bundle Branch Block",
               "Left Anterior Fascicular Block",
               "Nonspecific Intraventricular Conduction Disorder"},
    "normal": {"Sinus Rhythm"},
    "other": set(),
}

random.seed(42)
by_group = {g: [] for g in GROUPS}
for it in evalset:
    dx = set(it["dx_names"])
    for g, s in GROUPS.items():
        if g != "other" and dx & s:
            by_group[g].append(it)
            break
    else:
        by_group["other"].append(it)

picked = []
for g, pool in by_group.items():
    n = {"serious": 15, "blocks": 10, "normal": 8, "other": 7}[g]
    picked.extend(random.sample(pool, min(n, len(pool))))

random.shuffle(picked)
picked = picked[:40]

with open(OUT_DIR / "context_subset.json", "w", encoding="utf-8") as f:
    json.dump({"items": picked, "count": len(picked)}, f, ensure_ascii=False, indent=1)

from collections import Counter
c = Counter()
for it in picked:
    for d in it["dx_names"]:
        c[d] += 1
print(f"选取 {len(picked)} 条，标签分布:")
for k, v in c.most_common():
    print(f"  {k}: {v}")
