#!/usr/bin/env python3
"""M3 — 情境敏感性评测（双层输出架构）。

设计（临床诚实原则）:
    - 发现层（findings）: 由工具/分类器锚定，**情境无关**——同一心电图在任何情境下必须一致
    - 风险层（risk_assessment）+ 建议层（recommendations）: **情境条件化**——随患者情境合理变化

评测（每条记录 × 3 种情境卡: 急诊胸痛 / 常规体检 / 房颤随访）:
    [1] 发现层一致性: 三种情境下 findings 诊断集合必须完全相同（防情境污染发现层）
    [2] 发现层与真实标签重合率: findings vs ground truth（发现层质量）
    [3] 紧急度单调性: urgency(急诊) ≥ urgency(体检) 必须恒成立（共识硬规则，违例应=0）
    [4] 情境敏感性: 异常记录上三种情境的建议/风险是否发生变化（应该变化，否则情境无作用）
    [5] 严重疾病急诊紧急度: 房颤/室速/心梗/ST抬高/完全阻滞记录在急诊情境应 ≥ urgent
    （2E-Y6 修复：docstring 旧版声称第 [6] 维"幻觉"与 --judge 参数，实际未实现——已删除该声明）

注意: 情境卡中的 urgency 提示字段**不注入** LLM（防泄漏答案），只注入 setting/complaint/history。

用法:
    python scripts/eval_agent_context.py --limit 40
"""

import argparse
import hashlib
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent.llm.llm_interface import LLMInterface
from src.agent.llm.prompt_templates import SYSTEM_PROMPT
from src.agent.tools.ecgfounder_classifier import ECGFounderClassifier
from src.ecg_models.feature_extraction.r_peak_detector import RPeakDetector
from src.ecg_models.feature_extraction.hrv_analyzer import HRVAnalyzer
from src.ecg_models.feature_extraction.qt_analyzer import QTAnalyzer

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

URGENCY_MAP = {"routine": 0, "urgent": 1, "emergent": 2}
# 官方 27 评分类体系下的"严重疾病"清单（R1 修复：旧清单含已移除的 VT/MI/STE 等类）
SERIOUS_LABELS = {"Atrial Fibrillation", "Atrial Flutter",
                  "Complete Right Bundle Branch Block", "Bradycardia",
                  "Prolonged QT Interval"}


class ContextCache:
    """JSONL 缓存（key=sha256(messages)）。"""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                    self.cache[e["key"]] = e["value"]
                except json.JSONDecodeError:
                    pass

    def get(self, messages, t, model=""):
        key = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
            + f"|t={t}|model={model}".encode()).hexdigest()
        return self.cache.get(key)

    def put(self, messages, t, value, model=""):
        key = hashlib.sha256(
            json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
            + f"|t={t}|model={model}".encode()).hexdigest()
        self.cache[key] = value
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}, ensure_ascii=False) + "\n")


class ContextEvaluator:
    def __init__(self, llm, cache, data_dir="data/physionet2020/processed_5k"):
        self.llm = llm
        self.cache = cache
        self.data_dir = Path(data_dir)
        logger.info("加载工具（ECGFounder 分类器 + 特征提取）...")
        self.classifier = ECGFounderClassifier()
        self.rpeak = RPeakDetector(method="pan_tompkins")
        self.hrv = HRVAnalyzer()
        self.qt = QTAnalyzer()

    def ask(self, messages, json_mode=True):
        # 2E-O2 修复：key 用真实温度+模型（旧版写死 t=0.0）
        # 4G-RED-2 修复：mock 模式不读不写缓存——旧版把占位文本写入共享
        # context_cache.jsonl，先 mock 后真实运行同 key 命中占位回复 →
        # 全情境 parse_error、指标静默归零（与 eval_agent_llm.ask 对齐）
        if getattr(self.llm, "mock_mode", False):
            return self.llm.chat(
                messages,
                **({"response_format": {"type": "json_object"}} if json_mode else {}))
        model = getattr(self.llm, "model", "")
        t = getattr(self.llm, "temperature", 0.0)
        hit = self.cache.get(messages, t, model)
        if hit is not None:
            return hit
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        raw = self.llm.chat(messages, **kwargs)
        # 5G 审查（🟡-3）：空响应自动重试一次，且空结果不写缓存（与
        # eval_agent_llm.ask 的 2026-08-29 修复对齐——旧版空串写缓存后
        # 重跑永远命中 parse_error）
        if not raw or not raw.strip():
            logger.warning("LLM 返回空响应，重试一次")
            raw = self.llm.chat(messages, **kwargs)
        if not raw or not raw.strip():
            logger.error("LLM 两次返回空响应，返回空串（不写缓存）")
            return ""
        self.cache.put(messages, t, raw, model)
        return raw

    def run_tools(self, signal):
        out = {}
        r = None  # 2E-Y7：先初始化
        try:
            lead_ii = signal[1]
            r = self.rpeak.detect(lead_ii, fs=500)
            out["r_peaks"] = {"heart_rate": round(float(r.get("heart_rate", 0)), 1),
                              "rhythm": r.get("rhythm", "unknown"),
                              "insufficient": bool(r.get("insufficient"))}
        except Exception as e:
            out["r_peaks"] = {"error": str(e)[:80]}
        try:
            q = self.qt.analyze(lead_ii, (r or {}).get("r_peaks", []), fs=500)
            # R3：QT 测量失败（None）显式透传
            out["qt"] = {"qtc_bazett": (round(float(q["qtc_bazett"]), 1)
                                        if q.get("qtc_bazett") is not None else None),
                         "error": q.get("error")}
        except Exception as e:
            out["qt"] = {"error": str(e)[:80]}
        try:
            out["classifier"] = self.classifier.predict(signal)
        except Exception as e:
            out["classifier"] = {"error": str(e)[:80]}
        return out

    def eval_one(self, item, contexts):
        signal = np.load(self.data_dir / item["signal_file"])
        tools_out = self.run_tools(signal)
        positives = tools_out.get("classifier", {}).get("positives", [])
        top5 = [d["name"] for d in tools_out.get("classifier", {}).get("top_k", [])[:5]]
        pos_names = [d["name"] for d in positives]

        feature_summary = json.dumps({
            "r_peaks": tools_out.get("r_peaks"),
            "qt": tools_out.get("qt"),
            "classifier_positives": pos_names,
            "classifier_top5": tools_out.get("classifier", {}).get("top_k", [])[:5],
        }, ensure_ascii=False, default=str)

        rec = {"record_id": item["record_id"], "gt_dx_names": item["dx_names"],
               "classifier_positives": pos_names, "contexts": {}}

        for card in contexts:
            # 只注入临床情境字段，不注入 urgency 提示（防泄漏）
            patient_info = (f"{item.get('age') or '未知'}岁 {item.get('sex') or '未知'}，"
                            f"{card['setting']}，主诉: {card['chief_complaint']}，"
                            f"病史: {card['history']}")
            prompt = f"""你是心内科 AI 助手。请基于以下信息做双层分析：

【患者情境】(仅用于风险与建议层，不得影响发现层)
{patient_info}

【确定性工具输出】(发现层的唯一依据，数值不可编造)
{feature_summary}

要求：
1. findings(发现层): 列出波形诊断结论。该层与情境无关——同一份心电图在任何情境下必须给出完全相同的 findings。只能从 classifier_positives 中选取，每条附一句工具证据。
2. risk_assessment(风险层): 结合患者情境评估风险等级(low/moderate/high)并说明理由。
3. recommendations(建议层): 给出紧急度(routine/urgent/emergent)与具体建议，必须结合情境。

输出 JSON:
{{"findings": [{{"name": 诊断名, "evidence": 工具证据}}],
  "risk_assessment": {{"level": "low|moderate|high", "rationale": "理由"}},
  "recommendations": {{"urgency": "routine|urgent|emergent", "actions": ["建议1", "建议2"]}}}}"""
            raw = self.ask([{"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt}], json_mode=True)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"parse_error": raw[:120]}
            rec["contexts"][card["id"]] = parsed

        return rec

    def run(self, evalset, limit=None, context_ids=None):
        items = evalset["items"]
        if limit:
            items = items[:limit]
        all_recs = []
        for i, item in enumerate(items):
            logger.info(f"[{i+1}/{len(items)}] {item['source']}/{item['record_id']}")
            cards = item["context_cards"]
            if context_ids:
                cards = [c for c in cards if c["id"] in context_ids]
            try:
                all_recs.append(self.eval_one(item, cards))
            except Exception as e:
                logger.error(f"失败 {item['record_id']}: {e}")

        metrics = self.aggregate(all_recs)
        return metrics, all_recs

    # ---------------- 指标 ----------------
    def aggregate(self, recs):
        ctx_ids = ["ed_chest_pain", "routine_checkup", "af_followup"]

        def get_findings(ctx):
            out = ctx.get("findings")
            if isinstance(out, list):
                return {f.get("name", "") for f in out if isinstance(f, dict)}
            return set()

        def get_urgency(ctx):
            try:
                u = str(ctx["recommendations"]["urgency"]).lower().strip()
                return u if u in URGENCY_MAP else None
            except Exception:
                return None

        def get_risk(ctx):
            try:
                r = str(ctx["risk_assessment"]["level"]).lower().strip()
                return r if r in ("low", "moderate", "high") else None
            except Exception:
                return None

        n = len(recs)
        findings_consistent = 0
        urgency_violations = 0
        sensitivity = 0
        abnormal_ed_urgent = 0
        n_abnormal = 0
        findings_gt_overlap = 0
        gt_total = 0
        risk_change = 0
        parse_fail = 0

        for r in recs:
            ctxs = {c: r["contexts"].get(c) for c in ctx_ids}
            # 检查报告 1.9 修复：仅解析成功（dict）的上下文参与指标；
            # 单个情境 parse_error 的 truthy dict 不再被当作空集参与比较
            # 第三轮审查 3D-Y4：parse_error dict 不得通过过滤
            # （旧版 isinstance(v, dict) 把 {"parse_error":...} 当有效上下文参与比较）
            valid = {c: v for c, v in ctxs.items()
                     if isinstance(v, dict) and "parse_error" not in v}
            if len(valid) < 2:
                parse_fail += 1
                continue
            f_sets = {c: get_findings(valid[c]) for c in valid}
            # [1] 发现层一致性（与情境无关；空集=分类器无阳性，视为一致）
            nonempty = [s for s in f_sets.values() if s]
            if nonempty:
                if all(s == nonempty[0] for s in f_sets.values()):
                    findings_consistent += 1
            else:
                findings_consistent += 1  # 全空 = 一致（分类器无阳性，属阈值现象）
            # [2] 发现层与真实标签重合（第二次审查 R1 修复：按"记录×情境"逐情境
            # 平均，分子分母对齐——旧版分子按情境累加、分母每记录只加一次，
            # 产出的 1.5484/1.8214 > 1 为伪指标；正确口径 = 各情境 overlap 分数的均值 ∈ [0,1]）
            # 第三轮审查 3D-Y5：GT 非空但三情境 findings 全空的记录计入分母（overlap=0）
            # 4G-ORANGE-3 修复：分母含全部 valid 情境——部分情境 findings 为空集时
            # 同样计入（overlap=0；旧版只遍历 nonempty，重合率被系统性高估）
            gt = set(r["gt_dx_names"])
            if gt:
                for c in valid:
                    s = f_sets.get(c, set())
                    findings_gt_overlap += len(s & gt) / len(gt)
                    gt_total += 1
            # [3]/[4]/[5] 紧急度
            u = {c: get_urgency(valid[c]) for c in valid}
            if "ed_chest_pain" in u and "routine_checkup" in u and \
                    u["ed_chest_pain"] is not None and u["routine_checkup"] is not None:
                if URGENCY_MAP[u["ed_chest_pain"]] < URGENCY_MAP[u["routine_checkup"]]:
                    urgency_violations += 1
            # 检查报告 1.9 修复：None（解析失败）不再被当作独立取值虚增敏感性
            u_vals = [v for v in u.values() if v is not None]
            if len(set(u_vals)) > 1 and len(u_vals) >= 2:
                sensitivity += 1
            # 5G 审查（🟡-2）：abnormal_ed_urgent 分子分母对齐——分母只计
            # "ed 情境 urgency 成功解析"的严重记录；旧版 ed 解析失败但其余
            # 情境成功的严重记录进分母不进分子，率被系统性低估
            if gt & SERIOUS_LABELS:
                ed_u = u.get("ed_chest_pain")
                if ed_u is not None:
                    n_abnormal += 1
                    if ed_u in ("urgent", "emergent"):
                        abnormal_ed_urgent += 1
            # 风险层变化
            rk = {c: get_risk(valid[c]) for c in valid}
            rk_vals = [v for v in rk.values() if v is not None]
            if len(set(rk_vals)) > 1 and len(rk_vals) >= 2:
                risk_change += 1

        return {
            "n_records": n,
            "parse_failures": parse_fail,
            "findings_consistency_rate": round(findings_consistent / max(n - parse_fail, 1), 4),
            "findings_gt_overlap": round(findings_gt_overlap / max(gt_total, 1), 4),
            "urgency_monotonicity_violations": urgency_violations,
            "urgency_sensitivity_rate": round(sensitivity / max(n - parse_fail, 1), 4),
            "abnormal_ed_urgent_rate": round(abnormal_ed_urgent / max(n_abnormal, 1), 4)
            if n_abnormal else None,
            "risk_change_rate": round(risk_change / max(n - parse_fail, 1), 4),
            "n_abnormal": n_abnormal,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalset", default="outputs/agent_evalset/evalset.json")
    parser.add_argument("--output", default="outputs/context_eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--mock", action="store_true",
                        help="mock 模式（管道测试，不调用 API）")
    args = parser.parse_args()

    with open(args.evalset, encoding="utf-8") as f:
        evalset = json.load(f)

    if args.mock:
        llm = LLMInterface(backend="deepseek", model=args.model, mock=True)
        logger.info("Mock 模式（管道测试，不调用 API）")
    else:
        llm = LLMInterface(backend="deepseek", model=args.model)
        if not llm.is_available:
            logger.error("无 API key（.env）")
            sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = ContextCache(out_dir / "context_cache.jsonl")

    evaluator = ContextEvaluator(llm, cache)
    metrics, recs = evaluator.run(evalset, limit=args.limit)

    # mock 产物隔离（防止占位结果覆盖真实评测产物）
    suffix = "_mock" if args.mock else ""
    with open(out_dir / f"metrics{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / f"records{suffix}.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("=" * 60)
    logger.info("情境敏感性基准结果:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"产物: {out_dir}")


if __name__ == "__main__":
    main()
