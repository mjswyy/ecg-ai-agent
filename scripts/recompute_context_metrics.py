# -*- coding: utf-8 -*-
"""从已存记录重算情境评测指标（修正空集一致性定义后）"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.eval_agent_context import ContextEvaluator, URGENCY_MAP, SERIOUS_LABELS

with open("outputs/context_eval/records.jsonl", encoding="utf-8") as f:
    recs = [json.loads(l) for l in f if l.strip()]

# 用修正后的 aggregate（实例化一个不带 LLM 的评估器，只调用 aggregate 静态逻辑）
dummy = ContextEvaluator.__new__(ContextEvaluator)
metrics = dummy.aggregate(recs)

print("=== 修正后的情境敏感性基准（36 条 × 3 情境）===")
for k, v in metrics.items():
    print(f"  {k}: {v}")

with open("outputs/context_eval/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print("\n已更新 outputs/context_eval/metrics.json")
