# -*- coding: utf-8 -*-
"""v3 后置重算（检查报告 1.1 收尾）：
  1) 注入不应污染发现层 Top-5 —— 对注入记录本地重跑分类器还原真实 top-5；
  2) 注入捕获口径改为"报告显式标记快/慢诊断冲突" + 未注入对照组（判别力）。
用已存 LLM 输出重算，避免整轮重跑；口径与 eval_agent_llm.aggregate_metrics 一致。

复检C（🟡-2）登记：本脚本仅处理 main 臂（records_main/metrics_main）——
report/kb 臂的注入仅存在于 main（三臂同策略注入仅主分支执行），其余两臂
无需重算注入口径；如需处理其他臂，参照 record_id 对齐逻辑扩展。"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_agent_llm import aggregate_metrics
from src.agent.tools.ecgfounder_classifier import ECGFounderClassifier

OUT = ROOT / "outputs/agent_eval"
evalset = json.load(open(ROOT / "outputs/agent_evalset/evalset.json",
                         encoding="utf-8"))
items = evalset["items"]
records = [json.loads(l) for l in
           open(OUT / "records_main.jsonl", encoding="utf-8") if l.strip()]
assert len(items) == len(records), (len(items), len(records))

old_metrics = json.load(open(OUT / "metrics_main.json", encoding="utf-8"))

data_dir = ROOT / "data/physionet2020/processed_5k"
clf = ECGFounderClassifier()
# 5G 审查（🟡-1）：按 record_id 对齐（与 recompute_hallucinations 口径一致）——
# 旧版按位置 zip(items, records)，顺序变化即用错信号重判 top-5
ev_by_id = {it["record_id"]: it for it in items}
patched = []
for i, r in enumerate(records):
    item = ev_by_id.get(r.get("record_id"))
    if item is None:
        print(f"⚠️ 记录 #{i} ({r.get('record_id', '?')}) 不在评估集，跳过")
        continue
    if r.get("injected"):
        signal = np.load(data_dir / item["signal_file"])
        topk = clf.predict(signal).get("top_k", [])
        names = [d["name"] for d in topk[:5]]
        r["classifier_top5"] = names
        patched.append((i, item["record_id"], names[0] if names else None))
    r["top5_hit"] = bool(set(item["dx_names"]) & set(r["classifier_top5"]))

ok = [r for r in records if "planned_tools" in r]
# 2E-O5 修复：透传旧 metrics 中的 judge 结果（旧版硬编码 judge_scores=None，
# 会把原跑 --judge 的判官分数静默抹掉）
metrics = aggregate_metrics(ok, judge_scores=old_metrics.get("judge"),
                            use_report=False, use_kb=False)

json.dump(metrics, open(OUT / "metrics_main.json", "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
with open(OUT / "records_main.jsonl", "w", encoding="utf-8") as f:
    for r in records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

lines = [f"patched injected records ({len(patched)}):"]
for i, rid, top1 in patched:
    lines.append(f"  #{i} {rid} true_top1={top1}")
lines.append("")
lines.append("=== new metrics_main.json ===")
lines.append(json.dumps(metrics, indent=2, ensure_ascii=False))
lines.append("")
lines.append("=== old metrics_main.json (before recompute) ===")
lines.append(json.dumps(old_metrics, indent=2, ensure_ascii=False))
open(OUT / "_recompute_v3_report.txt", "w", encoding="utf-8").write("\n".join(lines))
print("done")
