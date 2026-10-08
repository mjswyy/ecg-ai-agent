#!/usr/bin/env python3
"""Rubric 重打分 — 复用三臂已缓存报告，统一三维评分（不重跑诊断流水线）。

输入: outputs/agent_eval/records_{main,report_arm,kb_arm}.jsonl
输出: outputs/agent_eval/judge_rubric_results.json
     {arm: {correctness/completeness/grounding/total 的 mean+dist}}

用法:
    python scripts/rejudge_with_rubric.py            # 全部三臂
    python scripts/rejudge_with_rubric.py --limit 20 # 试点
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.agent.llm.llm_interface import LLMInterface
from src.agent.llm.rubric_judge import RubricJudge, RubricCache, aggregate

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ARMS = {
    "main": "records_main.jsonl",
    "report_arm": "records_report_arm.jsonl",
    "kb_arm": "records_kb_arm.jsonl",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--arms", nargs="+", default=["main", "report_arm", "kb_arm"])
    parser.add_argument("--model", default="deepseek-chat")
    args = parser.parse_args()

    llm = LLMInterface(backend="deepseek", model=args.model)
    if not llm.is_available:
        logger.error("无 API key（.env）")
        sys.exit(1)

    out_dir = Path("outputs/agent_eval")
    cache = RubricCache(out_dir / "judge_rubric_cache.jsonl")
    judge = RubricJudge(llm, cache)

    results = {}
    for arm in args.arms:
        rec_path = out_dir / ARMS[arm]
        if not rec_path.exists():
            logger.warning(f"跳过 {arm}: 缺 {rec_path}")
            continue
        records = [json.loads(l) for l in rec_path.read_text(encoding="utf-8").splitlines()
                   if l.strip()]
        records = [r for r in records if r.get("report")]
        if args.limit:
            records = records[:args.limit]

        scores = []
        for i, r in enumerate(records):
            s = judge.score(r.get("gt_dx_names", []), r["report"])
            scores.append(s)
            if (i + 1) % 20 == 0:
                logger.info(f"[{arm}] {i+1}/{len(records)}")

        agg = aggregate(scores)
        results[arm] = agg
        logger.info(f"[{arm}] 完成: {agg['n']} 条")

    with open(out_dir / "judge_rubric_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    for arm, agg in results.items():
        logger.info(f"{arm:12s} n={agg['n']} "
                    f"correctness={agg['correctness']['mean']} "
                    f"completeness={agg['completeness']['mean']} "
                    f"grounding={agg['grounding']['mean']} "
                    f"total={agg['total']['mean']}")
    logger.info(f"保存 → {out_dir / 'judge_rubric_results.json'}")


if __name__ == "__main__":
    main()
