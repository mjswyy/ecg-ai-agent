#!/usr/bin/env python3
"""M2.4 — Agent 五维评测 harness（真 LLM 驱动 + 响应缓存）。

流程（每条记录）:
    1. 确定性工具执行（R峰/HRV/QT/ECGFounder 分类）→ 真实工具输出
    2. LLM Planner: 特征摘要 + 情境卡 + 工具清单 → JSON 计划
    3. 按计划取工具结果（确定性，不再重算）
    4. Reflector: 规则检查（心率/QTc 生理范围；3D 审查修正 docstring——
        实际无跨工具一致性规则）+ 可选 LLM 复核
    5. LLM Reasoner: 医学推理 + 最终报告（声明必须锚定工具输出）

五维指标:
    [1] 工具选择: 计划 vs 黄金计划 → 精确率/召回率/完全匹配数
    [2] 自反思纠错: 最多 20% 记录尝试注入与实测心率矛盾的 top-1（实际注入率取决于
        心率分布，当前评测集 7/130≈5.4%，第三轮审查 3D-T5 修正）→ 报告是否显式标记冲突
        （三臂同策略——2E-R3 修复；+未注入对照组判别力）
    [3] 幻觉率: LLM 报告中的数值声明 vs 工具输出（不匹配即幻觉；工具缺失仍声明数值
        也计幻觉——2E-O6 修复）。覆盖 心率/HR/QTc/QT/SDNN/PR间期/QRS时限/电轴
        （3D 审查 💡-1 扩展：PR/QRS/电轴工具无测量，声明数值即幻觉；
        范围/百分比/参考值表述仍排除——O6b/O6c）
    [4] 发现层 Top-5: 分类器 top-5 命中 GT（注入前快照口径，不受注入污染）
    [5] 推理链质量: LLM-as-judge 1-5 分（可选 --judge，失败计 judge_failures——2E-O4）

成本控制: temperature 用 LLMInterface 实例值（默认 0.3，全链路同温）；响应 JSONL 缓存
（key 含消息哈希+真实温度+json_mode+模型名——2E-O2；mock 模式不写缓存）; --limit 抽样。
用法:
    python scripts/eval_agent_llm.py --limit 20 --judge
    python scripts/eval_agent_llm.py --mock --limit 5   # 无 key 管道测试
"""

import argparse
import hashlib
import json
import logging
import re
import sys
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

TOOLS_DESC = """- extract_r_peaks: R 峰检测 → 心率、节律规律性
- compute_hrv: 心率变异性 (SDNN/RMSSD/pNN50，依赖 R 峰)
- measure_qt_interval: QT/QTc 间期测量（依赖 R 峰）
- classify_arrhythmia: 27 类心律失常/传导/形态多标签分类（ECGFounder 基础模型, test macro_auc 0.9456，官方评分类）
（注：4G-YELLOW-5 修复——read_report 从工具清单移除：医生报告在 report_arm 中直接注入
报告生成阶段，并非可调用工具；旧版广告该"幻影工具"使 LLM 计划含 read_report 被计
FP，工具选择 precision 被系统性低估。golden 计划亦从不包含它。）"""


class ResponseCache:
    """JSONL 响应缓存，key = sha256(role序列+内容+temperature)。"""

    def __init__(self, path: Path):
        self.path = path
        self.cache = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                    self.cache[e["key"]] = e["value"]
                except json.JSONDecodeError:
                    pass

    def _key(self, messages, temperature, json_mode=False, model=""):
        payload = (json.dumps(messages, ensure_ascii=False, sort_keys=True)
                   + f"|t={temperature}|json={json_mode}|model={model}")
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, messages, temperature, json_mode=False, model=""):
        return self.cache.get(self._key(messages, temperature, json_mode, model))

    def put(self, messages, temperature, value, json_mode=False, model=""):
        self.cache[self._key(messages, temperature, json_mode, model)] = value
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": self._key(messages, temperature, json_mode, model),
                                "value": value}, ensure_ascii=False) + "\n")


class AgentEvaluator:
    def __init__(self, llm, cache, data_dir="data/physionet2020/processed_5k",
                 inject_error_frac=0.2, use_report=False, use_kb=False,
                 judge=False, seed=42):
        self.llm = llm
        self.cache = cache
        self.data_dir = Path(data_dir)
        self.inject_error_frac = inject_error_frac
        self.use_report = use_report
        self.use_kb = use_kb
        self.judge = judge
        self.rng = np.random.RandomState(seed)

        logger.info("加载工具（ECGFounder 分类器 + 特征提取）...")
        self.classifier = ECGFounderClassifier()
        self.rpeak = RPeakDetector(method="pan_tompkins")
        self.hrv = HRVAnalyzer()
        self.qt = QTAnalyzer()

        self.kb = None
        if use_kb:
            try:
                from src.knowledge.kb_loader import KnowledgeBase
                self.kb = KnowledgeBase()
                logger.info(f"知识库已加载: {len(self.kb.classes)} 类")
            except Exception as e:
                logger.warning(f"知识库不可用（先跑 compile_knowledge.py）: {e}")

    def kb_excerpts(self, positives, max_classes=3, max_chars=1800):
        """为分类器阳性诊断检索知识条目，拼成上下文片段。

        2026-08-28 用户需求升级：不只注入阳性诊断的资料——同时注入
        总论方法论 + 全部 27 类压缩目录，让 LLM 能对照所有类别的
        判定标准验证/质疑 ECGFounder 的判断（鉴别与漏报核查）。
        """
        if self.kb is None:
            return ""
        parts = []
        gen = self.kb.general_excerpt()
        if gen:
            parts.append(gen)
        for d in positives[:max_classes]:
            try:
                parts.append(self.kb.query(d["name"], max_chars=max_chars))
            except Exception as e:
                logger.warning(f"知识检索失败 {d['name']}: {e}")
        parts.append(self.kb.format_catalog())
        if parts:
            return ("\n\n--- 医学参考知识库（分级可信来源）---\n"
                    + "\n\n".join(parts))
        return ""

    # ---------- LLM 调用（带缓存） ----------
    def ask(self, messages, temperature=None, json_mode=False):
        """带缓存的 LLM 调用。

        2E-O2 修复：缓存 key 的温度分量用 LLMInterface 实例的真实温度
        （self.llm.temperature，chat() 恒用它采样）——旧版 key 写死 t=0.0
        而真实请求恒为 0.3，key 与采样温度不符；LLMInterface.chat() 不接受
        温度参数，故不向其透传。mock 模式不读不写缓存（防止占位回复污染）。
        3D 审查（💡-2）：temperature 形参仅用于缓存 key 分量；调用点未传温度时
        取实例值，与真实采样温度一致。若未来传入不同值，实际采样温度仍为实例
        温度——保留形参仅为缓存 key 语义完整，请勿依赖其改变采样行为。
        """
        model = getattr(self.llm, "model", "")
        temperature = (temperature if temperature is not None
                       else getattr(self.llm, "temperature", 0.0))
        if getattr(self.llm, "mock_mode", False):
            return self.llm.chat(messages,
                                 **({"response_format": {"type": "json_object"}}
                                    if json_mode else {}))
        hit = self.cache.get(messages, temperature, json_mode, model)
        if hit is not None:
            return hit
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        raw = self.llm.chat(messages, **kwargs)
        # 2026-08-29 修复：LLM 空响应自动重试一次，且空结果不写缓存
        # （实测 deepseek-v4-flash 对部分长 prompt 返回空 content，
        # 旧版把空串写入缓存导致重跑永远命中空结果——main 臂 60/130 条空报告）
        if not raw or not raw.strip():
            logger.warning("LLM 返回空响应，重试一次")
            raw = self.llm.chat(messages, **kwargs)
        if not raw or not raw.strip():
            logger.error("LLM 两次返回空响应，返回空串（不写缓存）")
            return ""
        self.cache.put(messages, temperature, raw, json_mode, model)
        return raw

    # ---------- 确定性工具 ----------
    def run_tools(self, signal):
        out = {}
        r = None  # 2E-Y7 修复：先初始化，防止 rpeak.detect 抛错后 NameError
        try:
            lead_ii = signal[1]
            r = self.rpeak.detect(lead_ii, fs=500)
            out["r_peaks"] = {
                "heart_rate": round(float(r.get("heart_rate", 0)), 1),
                "rhythm": r.get("rhythm", "unknown"),
                "n_beats": int(r.get("num_beats", 0) or 0),
                # R3 修复（2D-R1）：失败标记透传，下游/报告不得合成节律结论
                "insufficient": bool(r.get("insufficient")),
            }
        except Exception as e:
            out["r_peaks"] = {"error": str(e)[:80]}
        try:
            rr = r.get("rr_intervals", np.array([0.8])) if r else np.array([0.8])
            h = self.hrv.analyze(rr)
            # 第三轮审查 3C-HRV-2：HRV 数据不足（None）显式透传，不再 round(None) 崩溃
            out["hrv"] = {"sdnn": (round(float(h["sdnn"]), 1)
                                   if h.get("sdnn") is not None else None),
                          "rmssd": (round(float(h["rmssd"]), 1)
                                    if h.get("rmssd") is not None else None),
                          "insufficient": bool(h.get("insufficient"))}
        except Exception as e:
            out["hrv"] = {"error": str(e)[:80]}
        try:
            q = self.qt.analyze(lead_ii, (r or {}).get("r_peaks", []), fs=500)
            # R3 修复（2D-R3）：QT 测量失败（数值为 None）显式透传，不再 round(None) 崩溃
            out["qt"] = {
                "qt_ms": (round(float(q["qt_ms"]), 1) if q.get("qt_ms") is not None else None),
                "qtc_bazett": (round(float(q["qtc_bazett"]), 1)
                               if q.get("qtc_bazett") is not None else None),
                "error": q.get("error"),
            }
        except Exception as e:
            out["qt"] = {"error": str(e)[:80]}
        try:
            out["classifier"] = self.classifier.predict(signal)
        except Exception as e:
            out["classifier"] = {"error": str(e)[:80]}
        return out

    # ---------- 规则反射器 ----------
    def rule_reflection(self, tools_out):
        alerts = []
        hr = tools_out.get("r_peaks", {}).get("heart_rate")
        if hr is not None and not (20 <= hr <= 300):
            alerts.append(f"心率 {hr} 超出生理范围")
        qtc = tools_out.get("qt", {}).get("qtc_bazett")
        if qtc is not None and not (200 <= qtc <= 700):
            alerts.append(f"QTc {qtc} 超出生理范围")
        return alerts

    def _inject_dx(self, item, new_name):
        """注入矛盾诊断：改写 top_k 条目全部字段（3D 审查 💡-4）。

        旧版只改 name/prob/positive，snomed 仍指原类 → 注入条目内部不一致。
        """
        item["name"] = new_name
        item["prob"] = 0.91
        item["positive"] = True
        try:
            le = self.classifier.le
            idx = le.class_names.index(new_name)
            item["snomed"] = le.idx_to_snomed.get(idx)
        except (ValueError, AttributeError):
            item["snomed"] = None

    # ---------- 单条评估 ----------
    def eval_one(self, item):
        rec = {}
        signal = np.load(self.data_dir / item["signal_file"])
        tools_out = self.run_tools(signal)

        # 未篡改的 top-5（发现层 Top-K 指标必须在注入前计算，防止注入污染该指标）
        orig_top5_names = [d["name"] for d in
                           tools_out.get("classifier", {}).get("top_k", [])[:5]]

        # 注入可检测矛盾（最多 20% 记录，三臂同策略——2E-R3 修复：
        # 旧版仅主分支注入导致 report/kb 臂在干净输入上评测、跨臂不可比；
        # 三臂同 seed 同记录序 → 注入记录集合一致）。
        # 修复（检查报告 1.1）：旧版泄露"注入错误"字样进告警 + top1/top2 对调大多不可检测 → 恒 1.0 伪指标。
        # 新版：告警不含任何注入提示；捕获判定 = 报告显式标记快/慢诊断与心率冲突（见 aggregate_metrics），
        #       Top-5 指标用注入前的 orig_top5_names，避免注入篡改污染发现层指标。
        injected = False
        injection_desc = None
        injected_name = None
        if (self.rng.rand() < self.inject_error_frac
                and "top_k" in tools_out.get("classifier", {})
                and tools_out["classifier"].get("top_k")):
            hr = tools_out.get("r_peaks", {}).get("heart_rate")
            tk = tools_out["classifier"]["top_k"]
            top1 = tk[0].get("name", "")
            if hr is not None and hr > 0:  # R3：hr=0 为测量失败，不作注入对象
                # 4G-YELLOW-8 修复：排除表必须使用 27 类实际存在的类名——
                # 旧版引用不存在的 "Ventricular Tachycardia"（死排除），且漏
                # "Bradycardia"（hr<60 且 top1=Bradycardia 的真实缓搏记录被
                # 错误注入窦速，注入组混入假阳性）
                if hr < 60 and top1 not in ("Sinus Bradycardia", "Bradycardia"):
                    self._inject_dx(tk[0], "Sinus Tachycardia")
                    injected = True
                    injected_name = "Sinus Tachycardia"
                    injection_desc = f"实测心率{hr:.0f}<60 却报窦速"
                elif hr >= 100 and top1 not in ("Sinus Tachycardia",):
                    self._inject_dx(tk[0], "Sinus Bradycardia")
                    injected = True
                    injected_name = "Sinus Bradycardia"
                    injection_desc = f"实测心率{hr:.0f}≥100 却报窦缓"

        if injected:
            # 3D 审查（💡-4）：positives 与注入后的 top_k 保持一致
            # （旧版只改 top_k[0]，positives 仍含原 top1 → 注入条目内部不一致；
            #  kb_excerpts 也基于 positives 检索，必须用注入后的类）
            pos_list = tools_out.get("classifier", {}).get("positives", [])
            tools_out["classifier"]["positives"] = (
                [d for d in pos_list if d.get("name") != top1]
                + [{"name": injected_name, "snomed": tk[0].get("snomed"),
                    "prob": 0.91, "positive": True}])

        # 情境卡
        card = item["context_cards"][0]  # 主情境：急诊胸痛
        patient_info = (f"{item.get('age') or '未知'}岁 {item.get('sex') or '未知'}，"
                        f"{card['setting']}，主诉: {card['chief_complaint']}，"
                        f"病史: {card['history']}")

        feature_summary = json.dumps({
            "r_peaks": tools_out.get("r_peaks"),
            "hrv": tools_out.get("hrv"),
            "qt": tools_out.get("qt"),
            "classifier_top5": tools_out.get("classifier", {}).get("top_k", [])[:5],
        }, ensure_ascii=False, default=str)

        # ---- Planner（规划阶段不给预测量，保证工具选择指标有意义）----
        planning_prompt = f"""你是心内科 AI 助手，规划该心电图的诊断步骤。

可用工具:
{TOOLS_DESC}

患者: {patient_info}

ECG 信号: 12 导联, 500Hz, 时长 10 秒。

请以 JSON 输出诊断计划: {{"plan": [{{"action": 工具名, "reason": 理由}}], "notes": "..."}}
只调用必要工具，输出合法 JSON。"""
        plan_raw = self.ask(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": planning_prompt}],
            json_mode=True)
        plan = []
        try:
            data = json.loads(plan_raw)
            plan = data.get("plan", [])
        except json.JSONDecodeError:
            rec["plan_parse_error"] = plan_raw[:100]

        planned_tools = [s.get("action") for s in plan if isinstance(s, dict)]
        # 3D 审查（🟠-2）：规划 prompt 三臂完全一致，但 LLM 采样温度 0.3 有非确定性。
        # 串行运行（共享 llm_cache）时规划请求 key 相同 → 缓存命中 → planned_tools
        # 跨臂一致；并行运行时各进程缓存互不可见 → 可能产生集合差异（实测 7/130 条），
        # 属 LLM 非确定性而非干预差异。reflection 同理（实测三臂完全一致）。

        # ---- 规则反射（仅生理范围检查，不含任何注入提示）----
        alerts = self.rule_reflection(tools_out)

        # ---- LLM 复核（反射器） ----
        reflection_prompt = f"""请审查以下 ECG 分析结果是否自洽:
患者: {patient_info}
工具输出: {feature_summary}
规则告警: {alerts if alerts else '无'}

回复 JSON: {{"verdict": "continue|revise", "issue": "..."}}"""
        refl_raw = self.ask(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": reflection_prompt}],
            json_mode=True)
        verdict = "unknown"
        try:
            verdict = json.loads(refl_raw).get("verdict", "unknown")
        except json.JSONDecodeError:
            pass

        # ---- 报告（Reasoner）— 强制系统化结构（完整性修复）----
        kb_text = ""
        if self.use_kb:
            positives = tools_out.get("classifier", {}).get("positives", [])
            kb_text = self.kb_excerpts(positives)
        report_prompt = f"""基于以下确定性工具输出生成结构化诊断报告。

报告必须严格按以下结构逐节输出，每一节必须引用工具输出中的具体数值（严禁编造数值；工具未提供的项目如实写"未测量"）：

1. 心率与节律 (Rate & Rhythm)：引用 r_peaks 的心率与节律判断
2. 间期 (Intervals)：引用 qt 工具的 QT/QTc 数值；PR/QRS 如未测量则注明
3. 电轴 (Axis)：工具未提供电轴数据时写"未测量"，不要猜测
4. 形态 (Morphology)：P波/QRS/ST-T 异常，引用分类器发现
5. 主要诊断 (Primary Diagnosis)：列出分类器阳性诊断与概率，逐条给出支持证据
6. 鉴别诊断 (Differential)：与主要诊断易混淆的疾病
7. 建议 (Recommendations)：结合患者情境给出

患者: {patient_info}
工具输出: {feature_summary}
{('医生原始报告: ' + item['report_text']) if self.use_report and item['report_text'] else ''}
{kb_text}

如有知识库资料，第 5/6 节的诊断标准与鉴别应与之保持一致并注明依据。"""
        report = self.ask(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": report_prompt}])
        # 2026-08-29：报告为空视为该记录失败标记（ask 已内部重试；仍空则如实记录，
        # 供下游区分"正常无异常"与"报告生成失败"）
        if not report or not report.strip():
            rec["report_error"] = "LLM 空响应"

        rec.update({
            # 4G-YELLOW-10 修复：记录带 record_id，供后置重算按 id 对齐
            # （旧版 recompute 按 evalset[i] 位置索引，顺序变化即用错信号重判）
            "record_id": item["record_id"],
            "planned_tools": planned_tools,
            "golden_tools": item["golden_tools"],
            "rule_alerts": alerts,
            "reflection_verdict": verdict,
            "injected": injected,
            "injection_desc": injection_desc,
            "report": report,
            "gt_dx_names": item["dx_names"],
            "classifier_top5": orig_top5_names,
            "kb_used": bool(kb_text),
        })

        # ---- 幻觉检查：明确声明模式 vs 工具输出（跳过规范性范围表述）----
        hr = tools_out.get("r_peaks", {}).get("heart_rate")
        qtc = tools_out.get("qt", {}).get("qtc_bazett")
        qt_ms = tools_out.get("qt", {}).get("qt_ms")
        sdnn = tools_out.get("hrv", {}).get("sdnn")

        claim_patterns = [
            # 三臂重跑复查（2026-08-30）：\s* → [ \t]*——\s 含换行，旧版
            # "确认真实心率\n10. 建议…"跨行匹配编号"10"误报幻觉（E09971 实测）
            (r"心率[为是]?[ \t]*[:：]?[ \t]*(\d{2,3})", hr, "心率"),
            (r"(?:HR|hr)[ \t]*[:：=]?[ \t]*(\d{2,3})", hr, "心率"),
            (r"QTc[为是]?[ \t]*[:：]?[ \t]*(\d{2,3})", qtc, "QTc"),
            (r"QT间期[为是]?[ \t]*[:：]?[ \t]*(\d{2,3})", qt_ms, "QT"),
            (r"SDNN[为是]?[ \t]*[:：]?[ \t]*(\d{2,3})", sdnn, "SDNN"),
            # 3D 审查（💡-1）扩展：PR间期/QRS时限/电轴——工具无这些测量，
            # 任何具体数值声明（非"未测量"否定句）即幻觉；tool_val=None 走
            # "工具未提供测量值"分支。head/tail 排除规则（O6b/O6c）继续生效。
            # 5G 审查（🟠-1）：QRS 修饰词需覆盖"时限/宽度/宽"整词（旧版单字符
            # [时限宽]? 漏检 "QRS时限 120ms"）、电轴需覆盖"左偏/右偏"插入语
            # （旧版漏检 "电轴左偏 -30°"——实测 main 臂 2 例编造数值漏检）
            (r"PR间期[为是]?[ \t]*[:：]?[ \t]*(\d{2,3})", None, "PR间期"),
            (r"QRS(?:波群)?(?:时限|宽度|宽|间期)?[为是]?[ \t]*[:：]?[ \t]*[-+]?[ \t]*(\d{2,3})",
             None, "QRS时限"),
            (r"电轴(?:左偏|右偏|正常)?[为是]?[ \t]*[:：]?[ \t]*[-+]?[ \t]*(\d{1,3})",
             None, "电轴"),
        ]
        hallucinations = []
        for pattern, tool_val, label in claim_patterns:
            for m in re.finditer(pattern, report):
                claimed = int(m.group(1))
                # 2E-O6 修复：工具值缺失（测量失败）但报告声明了对应数值
                # → 更强的幻觉信号（编造数据），计入分子而非静默跳过
                # 2E-O6b 修复：排除规范性范围/区间表述（"心率 60–100 bpm"、
                # "SVT 通常心率 150–250 次/分"）与百分比表述（"预测最大心率 80%"）
                # ——数字后紧跟范围连接符或 % 即跳过
                tail = report[m.end():m.end() + 12]
                if re.match(r"^\s*(?:[-–—~～至到/]\s*\d|%)", tail):
                    continue
                # 2E-O6c 修复：排除知识库/参考值表述（"男性平均QTc 422 ms"、
                # "正常 QTc < 450 ms"）——数字前 10 字符含"平均/均值/正常/参考/
                # 阈值"等修饰词即视为参考统计值而非本患者声明
                head = report[max(0, m.start() - 10):m.start()]
                # 三臂重跑复查（2026-08-30）：补疾病一般知识表述修饰词——
                # "房扑常表现为心率150 bpm左右" 是鉴别知识而非患者声称，
                # 旧词表漏"常表现/通常/一般/约/左右/常见"（E03180 实测误报）
                if re.search(r"平均|均值|中位|正常|参考|阈值|标准|典型|常表现|通常|一般|左右|约|常见|mean|avg|normal|threshold", head):
                    continue
                if tool_val is None:
                    hallucinations.append(f"{label}声称{claimed}(工具未提供测量值)")
                    continue
                if abs(claimed - round(tool_val)) > 5:
                    hallucinations.append(f"{label}声称{claimed}(实际{tool_val})")
        rec["hallucinated_claims"] = hallucinations
        # 工具值入库（供后置重算幻觉口径时使用，无需重跑 LLM）
        rec["tool_values"] = {"hr": hr, "qtc": qtc, "qt_ms": qt_ms, "sdnn": sdnn}

        # ---- Top-K 命中（发现层 = 分类器） ----
        gt_set = set(item["dx_names"])
        top5_set = set(rec["classifier_top5"])
        rec["top5_hit"] = bool(gt_set & top5_set)

        return rec

    def run(self, evalset, limit=None):
        items = evalset["items"]
        if limit:
            items = items[:limit]
        records = []
        for i, item in enumerate(items):
            logger.info(f"[{i+1}/{len(items)}] {item['source']}/{item['record_id']}")
            try:
                records.append(self.eval_one(item))
            except Exception as e:
                logger.error(f"评估失败 {item['record_id']}: {e}")
                records.append({"error": str(e)[:120]})

        # ---- 汇总指标（口径见 aggregate_metrics，与后置重算脚本共用） ----
        ok_records = [r for r in records if "planned_tools" in r]

        # [5] LLM-as-judge（rubric 三维：正确性/完整性/依据性）
        # 2E-O4/Y8 修复：直接调用 RubricJudge.score()（删除死代码 rj），
        # 判官解析失败 → None 跳过并计入 judge_failures（旧版失败记 1 分
        # 会把"LLM 故障"伪装成"报告质量极差"）。
        judge_scores = None
        if self.judge and ok_records:
            from src.agent.llm.rubric_judge import RubricJudge, aggregate as rub_agg
            rj = RubricJudge(self.llm)
            judge_failures = 0
            scores = []
            for r in ok_records:
                s = rj.score(r["gt_dx_names"], r["report"][:1500])
                if s is None:
                    judge_failures += 1
                    continue
                scores.append(s)
            agg = rub_agg(scores) if scores else {"n": 0, "total": {"mean": None},
                                                  "correctness": {"mean": None},
                                                  "completeness": {"mean": None},
                                                  "grounding": {"mean": None}}
            judge_scores = {"n": agg["n"], "judge_failures": judge_failures,
                            "total_mean": agg["total"]["mean"],
                            "correctness_mean": agg["correctness"]["mean"],
                            "completeness_mean": agg["completeness"]["mean"],
                            "grounding_mean": agg["grounding"]["mean"]}

        metrics = aggregate_metrics(ok_records, judge_scores=judge_scores,
                                    use_report=self.use_report, use_kb=self.use_kb)
        return metrics, records


CONFLICT_MARK = re.compile(r"矛盾|不一致|不符|存疑|冲突")  # 4G-YELLOW-9：补"冲突"
RATE_WORDS = re.compile(r"心动过速|心动过缓|窦速|窦缓|Sinus Tachycardia|Sinus Bradycardia")


def _report_flags_conflict(text):
    """报告是否显式标记"快/慢节律诊断"冲突（同一句内同时出现矛盾标记与快慢诊断词）。"""
    for s in re.split(r"[。；;\n]+", text or ""):
        if CONFLICT_MARK.search(s) and RATE_WORDS.search(s):
            return True
    return False


def aggregate_metrics(ok_records, judge_scores=None,
                      use_report=False, use_kb=False):
    """五维指标汇总（单一口径，eval harness 与后置重算脚本共用）。

    注入纠错口径（检查报告 1.1 修复）：
      - 捕获 = 报告显式标记快/慢诊断与心率冲突（_report_flags_conflict）；
      - 对照组 = 未注入记录的同口径标记率（判别力），
        以及反射器 verdict 对照组（verdict 基率高，仅作参考不作主指标）。
    """
    # [1] 工具选择：精确率/召回率 + 计划完全匹配数
    tp = fn = fp = 0
    exact = 0
    for r in ok_records:
        planned = set(r["planned_tools"])
        golden = set(r["golden_tools"])
        tp += len(planned & golden)
        fn += len(golden - planned)
        fp += len(planned - golden)
        exact += int(planned == golden)
    tool_precision = tp / (tp + fp) if tp + fp else 0.0
    tool_recall = tp / (tp + fn) if tp + fn else 0.0

    # [2] 注入纠错
    injected = [r for r in ok_records if r.get("injected")]
    noninj = [r for r in ok_records if not r.get("injected")]
    caught_report = [r for r in injected
                     if _report_flags_conflict(r.get("report", ""))]
    control_flags = [r for r in noninj
                     if _report_flags_conflict(r.get("report", ""))]
    caught_verdict = [r for r in injected
                      if r.get("reflection_verdict") == "revise"]
    control_revise = [r for r in noninj
                      if r.get("reflection_verdict") == "revise"]
    injection = {
        "n_injected": len(injected),
        "catch_rate_report": round(len(caught_report) / len(injected), 4)
        if injected else None,
        "conflict_flag_rate_control": round(len(control_flags) / len(noninj), 4)
        if noninj else None,
        "catch_rate_verdict": round(len(caught_verdict) / len(injected), 4)
        if injected else None,
        "revise_rate_control": round(len(control_revise) / len(noninj), 4)
        if noninj else None,
    }

    # [3] 幻觉率（明确声明模式）
    n_hallu = sum(len(r.get("hallucinated_claims", [])) for r in ok_records)
    # 2E-O6b：分母与分子同口径——排除规范性范围/百分比表述
    # 4G-RED-1 修复：分母模式必须与 eval_one 的 claim_patterns 完全一致
    # （旧版漏 PR间期/QRS时限/电轴 3 类 → 分子 8 类分母 5 类，幻觉率可 >100%）
    n_claims = 0
    for r in ok_records:
        rep = r.get("report", "")
        for m in re.finditer(r"心率[为是]?[ \t]*[:：]?[ \t]*\d{2,3}|(?:HR|hr)[ \t]*[:：=]?[ \t]*\d{2,3}|"
                             r"QTc[为是]?[ \t]*[:：]?[ \t]*\d{2,3}|QT间期[为是]?[ \t]*[:：]?[ \t]*\d{2,3}|"
                             r"SDNN[为是]?[ \t]*[:：]?[ \t]*\d{2,3}|"
                             r"PR间期[为是]?[ \t]*[:：]?[ \t]*\d{2,3}|"
                             # 复检C（🟠-1）：分母 QRS/电轴模式必须与分子
                             # claim_patterns 逐字一致（5G-🟠-1 只同步了分子与
                             # recompute，漏了分母——实测"QRS时限 120ms"分子命中
                             # 分母不命中，幻觉率可重开 4G-RED-1 越界）
                             # 三臂重跑复查（2026-08-30）：\s*→[ \t]* 防跨行
                             r"QRS(?:波群)?(?:时限|宽度|宽|间期)?[为是]?[ \t]*[:：]?[ \t]*[-+]?[ \t]*\d{2,3}|"
                             r"电轴(?:左偏|右偏|正常)?[为是]?[ \t]*[:：]?[ \t]*[-+]?[ \t]*\d{1,3}", rep):
            tail = rep[m.end():m.end() + 12]
            if re.match(r"^\s*(?:[-–—~～至到/]\s*\d|%)", tail):
                continue
            head = rep[max(0, m.start() - 10):m.start()]
            # 三臂重跑复查（2026-08-30）：分母排除词与分子同步（疾病一般知识表述）
            if re.search(r"平均|均值|中位|正常|参考|阈值|标准|典型|常表现|通常|一般|左右|约|常见|mean|avg|normal|threshold", head):
                continue
            n_claims += 1
    hallucination_rate = n_hallu / n_claims if n_claims else None

    # [4] Top-5 命中（classifier_top5 为注入前快照）
    # 2E-Y9 注明口径：分母为全部记录（GT 为空/分类器失败计 miss），
    # 与 src/evaluation/metrics/classification.topk_hits（跳过无正类记录）不同，
    # 二者不可直接并列。
    n_hit = sum(1 for r in ok_records if r.get("top5_hit"))
    top5_acc = n_hit / len(ok_records) if ok_records else 0.0

    return {
        "n_records": len(ok_records),
        "tool_selection": {"precision": round(tool_precision, 4),
                           "recall": round(tool_recall, 4),
                           "exact_match": exact},
        "injection": injection,
        "hallucination": {"n_claims": n_claims,
                          "n_hallucinated": n_hallu,
                          "rate": round(hallucination_rate, 4)
                          if hallucination_rate is not None else None},
        "top5_accuracy": round(top5_acc, 4),
        "judge": judge_scores,
        "use_report_arm": use_report,
        "use_kb_arm": use_kb,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evalset", default="outputs/agent_evalset/evalset.json")
    parser.add_argument("--output", default="outputs/agent_eval")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--report-arm", action="store_true",
                        help="开启'报告辅助'分支（与主分支对照用）")
    parser.add_argument("--use-kb", action="store_true",
                        help="开启'知识库'分支：报告生成时注入分级可信医学参考资料")
    parser.add_argument("--model", default="deepseek-chat")
    args = parser.parse_args()

    with open(args.evalset, encoding="utf-8") as f:
        evalset = json.load(f)

    if args.mock:
        # 检查报告 1.9 修复：mock 需显式开启（旧版手动置 _client=None 依赖静默降级）
        llm = LLMInterface(backend="deepseek", model=args.model, mock=True)
        logger.info("Mock 模式（管道测试，不调用 API）")
    else:
        llm = LLMInterface(backend="deepseek", model=args.model)
        if not llm.is_available:
            logger.error("无 API key。请创建 ecg-ai-agent/.env 写入 DEEPSEEK_API_KEY=sk-xxx")
            sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = ResponseCache(out_dir / "llm_cache.jsonl")

    evaluator = AgentEvaluator(llm, cache, use_report=args.report_arm,
                               use_kb=args.use_kb, judge=args.judge)
    metrics, records = evaluator.run(evalset, limit=args.limit)

    suffix = ("kb_arm" if args.use_kb else
              "report_arm" if args.report_arm else "main")
    # mock 模式产物隔离（防止占位结果覆盖真实评测产物）
    if args.mock:
        suffix += "_mock"
    with open(out_dir / f"metrics_{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    with open(out_dir / f"records_{suffix}.jsonl", "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("=" * 50)
    logger.info(f"五维评测结果 ({suffix}):")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")
    logger.info(f"产物: {out_dir}")


if __name__ == "__main__":
    main()
