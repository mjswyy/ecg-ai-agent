# -*- coding: utf-8 -*-
"""检查 findings 空集情况（解释一致性指标的 3 条'不一致'）"""
import json

with open("outputs/context_eval/records.jsonl", encoding="utf-8") as f:
    recs = [json.loads(l) for l in f if l.strip()]

empty_counts = {}
for r in recs:
    n_empty = 0
    sets = {}
    for cid, c in r["contexts"].items():
        if not isinstance(c, dict):
            continue
        s = {f.get("name") for f in c.get("findings", []) if isinstance(f, dict)}
        sets[cid] = s
        if not s:
            n_empty += 1
    key = (n_empty, len(sets))
    empty_counts[key] = empty_counts.get(key, 0) + 1
    if n_empty > 0:
        print(f"{r['record_id']}: {n_empty}/{len(sets)} 个情境空 findings | "
              f"分类器阳性={r['classifier_positives']} | 各集合={ {k: sorted(v) for k, v in sets.items()} }")

print("\n空集分布 (n_empty, n_contexts):", empty_counts)
