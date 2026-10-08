"""Rubric 化 LLM 判官 — ECG 诊断报告三维质量评分（正确性/完整性/依据性）。

替代旧的无细则 1-5 分判官（双峰分布、不可比）。
三维（每维 1-5）:
    correctness 正确性: 主要诊断与真实标签的符合度；是否误报严重异常
    completeness 完整性: 是否系统覆盖心率/节律、间期、电轴、形态；是否遗漏工具已检出的异常
    grounding   依据性: 结论是否引用工具输出/知识库依据；有无未锚定的凭空声明
    total       综合分: 1-5

用法:
    judge = RubricJudge(llm, cache)
    scores = judge.score(gt_dx_names, report)  # {"correctness":..., "completeness":..., "grounding":..., "total":..., "comments":...}
"""

import hashlib
import json
import logging
import threading

logger = logging.getLogger(__name__)

JUDGE_SYSTEM = ("You are a senior cardiology quality-control expert evaluating AI-generated "
                "ECG diagnostic reports against ground-truth labels.")

JUDGE_PROMPT = """按以下三个维度给 AI 生成的 ECG 诊断报告打分（每维 1-5 分，1=很差 5=优秀）：

1. correctness(正确性)：主要诊断与真实标签(ground truth)的符合程度；是否误报严重异常、是否漏报关键异常
2. completeness(完整性)：是否系统覆盖心率/节律、间期(PR/QRS/QT)、电轴、形态(ST-T/肥大等)；是否遗漏工具已检出的异常
3. grounding(依据性)：结论是否明确引用工具输出或知识库依据；是否存在未锚定的凭空声明或编造数值

真实标签(ground truth): {gt}

AI 报告:
{report}

输出 JSON: {{"correctness": 1-5整数, "completeness": 1-5整数, "grounding": 1-5整数, "total": 1-5整数, "comments": "简短理由(中文)"}}"""


class RubricJudge:
    def __init__(self, llm, cache=None):
        self.llm = llm
        self.cache = cache

    def _key(self, gt, report, model="", temperature=0.0):
        # 2E-O3 修复：key 加模型名/温度/json_mode（旧版只含 gt|report，
        # 跨模型重打分会命中旧模型缓存）；与主评测 ResponseCache 口径一致。
        # 第三轮审查 AGENT-LLM-O2 修复：key 用完整 report 全文哈希
        # （旧版 report[:1500] 截断导致长报告尾部差异碰撞命中错误评分）
        payload = (json.dumps(gt, ensure_ascii=False, sort_keys=True) + "|" + report
                   + f"|model={model}|t={temperature}|json=1")
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def score(self, gt_dx_names, report):
        trunc = report[:1500]
        model = getattr(self.llm, "model", "")
        temperature = getattr(self.llm, "temperature", 0.0)
        # 第三轮审查 AGENT-LLM-O2：key 用完整报告（prompt 仍用 1500 截断）
        key = self._key(gt_dx_names, report, model, temperature)
        if self.cache is not None:
            hit = self.cache.get(key, 0.0)
            if hit is not None:
                return hit

        prompt = JUDGE_PROMPT.format(gt=json.dumps(gt_dx_names, ensure_ascii=False),
                                     report=trunc)
        raw = self.llm.chat(
            [{"role": "system", "content": JUDGE_SYSTEM},
             {"role": "user", "content": prompt}],
            response_format={"type": "json_object"})
        try:
            scores = json.loads(raw)
            # 第三轮审查 AGENT-LLM-O1：非对象 JSON（数组/null/数字/字符串）
            # 同样视为判官故障返回 None（旧版抛 AttributeError 中断评测）
            if not isinstance(scores, dict):
                logger.warning(f"判官输出非对象 JSON: {raw[:100]}")
                return None
            # 4A/4J 审查修复：🟠-3 形状正确但字段缺失/被篡改的 dict 不再静默补 1
            # （旧版 scores.get(field, 1) 把缺失维度补成 1，畸形输出被当成有效
            # 评分且 judge_failures 不计数）。任一必需字段缺失 → 判官故障返回 None。
            for field in ("correctness", "completeness", "grounding", "total"):
                if field not in scores:
                    logger.warning(f"判官输出缺少必需字段 {field}: {raw[:100]}")
                    return None
            # 数值校验（第三轮审查 AGENT-LLM-T1：round 四舍五入而非 int 截断）
            for field in ("correctness", "completeness", "grounding", "total"):
                scores[field] = min(5, max(1, int(round(float(scores[field])))))
        except (json.JSONDecodeError, ValueError, TypeError, AttributeError):
            # 检查报告 1.9 修复：判官故障返回 None 由调用方跳过并单独统计，
            # 旧版全给 1 分会把"LLM 故障"伪装成"报告质量极差"
            logger.warning(f"判官输出解析失败: {raw[:100]}")
            return None

        if self.cache is not None:
            self.cache.put(key, 0.0, scores)
        return scores


class RubricCache:
    """独立的判官响应缓存（JSONL，key=sha256(gt|report)）。"""

    def __init__(self, path):
        import pathlib
        self.path = pathlib.Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # 4A/4J 审查修复：🟡-6 JSONL 追加加线程锁（旧版无锁，并行评测时
        # 行级交错/污染）
        self._lock = threading.Lock()
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

    def get(self, key, _t):
        return self.cache.get(key)

    def put(self, key, _t, value):
        # 4A/4J 审查修复：🟡-6 加锁保护追加写（避免并行线程行级交错）
        with self._lock:
            self.cache[key] = value
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "value": value},
                                   ensure_ascii=False) + "\n")


def aggregate(scores_list):
    """汇总一组三维评分 → 每维均值/分布 + 总分均值（跳过判官失败项）。"""
    import numpy as np
    from collections import Counter
    valid = [s for s in scores_list if isinstance(s, dict)]
    out = {
        "n": len(valid),
        "judge_failures": len(scores_list) - len(valid),
    }
    if not valid:
        return out
    for field in ("correctness", "completeness", "grounding", "total"):
        # 4A/4J 审查修复：🟡-6 缺字段 dict 用 .get 防御并跳过（旧版 s[field]
        # 直接下标，遇缺字段评分 dict 抛 KeyError 崩溃而非计入失败）
        vals = [s.get(field) for s in valid if s.get(field) is not None]
        if not vals:
            continue
        out[field] = {
            "mean": round(float(np.mean(vals)), 3),
            "dist": dict(sorted(Counter(vals).items())),
        }
    return out
