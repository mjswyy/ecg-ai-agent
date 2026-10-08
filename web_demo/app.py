#!/usr/bin/env python3
"""ECG AI Agent — 情境条件化双层诊断演示（全新设计，M5）。

流水线:
    输入(ECG + 患者情境)
      → 预处理(ECGFounder 官方协议) + 确定性工具(R峰/HRV/QT)
      → ECGFounder 发现层(0.9456, 27 类 + Youden 阈值，患者级划分)
      → 分级可信知识库检索(27 类全覆盖)
      → DeepSeek deepseek-chat 双层报告(发现层情境无关 / 风险建议层情境条件化)

另有一个只读页签「参考答案（真值）」：选中测试集记录时展示 PhysioNet 官方
标注（dx_codes + 27 维 labels）并与分类器阳性逐类对照。该页签**只写界面**，
不进入发现层、不进入 LLM prompt（发现层内容必须完全来自分类器阳性）。

运行:
    cd ecg-ai-agent
    python web_demo/app.py        # 浏览器打开 http://127.0.0.1:7860
    （需 .env 中的 DEEPSEEK_API_KEY；无 key 时自动降级为"无 LLM 模式"）
"""

import html
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import rcParams  # 2F 🟠-11 修复后不再使用 pyplot 状态机

rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from src.agent.llm.llm_interface import LLMInterface  # noqa: E402
from src.agent.llm.prompt_templates import SYSTEM_PROMPT  # noqa: E402
from src.agent.tools.ecgfounder_classifier import ECGFounderClassifier  # noqa: E402
from src.ecg_models.feature_extraction.r_peak_detector import RPeakDetector  # noqa: E402
from src.ecg_models.feature_extraction.hrv_analyzer import HRVAnalyzer  # noqa: E402
from src.ecg_models.feature_extraction.qt_analyzer import QTAnalyzer  # noqa: E402
from src.knowledge.kb_loader import KnowledgeBase  # noqa: E402
import scripts.preprocess_data_5k as p5  # noqa: E402

DATA_DIR = ROOT / "data" / "physionet2020" / "processed_5k"

SCENARIOS = ["急诊科", "心内科门诊", "体检中心", "术后随访", "其他"]

# 清单缓存：test_manifest.json ≈3 MB，原版在 load_signal / 下拉框 / 年龄预填
# 三处各自 json.load 一次（每次请求重复解析）。清单在 demo 运行期不变，
# 故按 split 缓存；换数据需重启。
_MANIFEST_CACHE: dict = {}


def load_manifest(split: str = "test"):
    """读取 processed_5k 的 {split}_manifest.json（带进程内缓存）。"""
    if split not in _MANIFEST_CACHE:
        with open(DATA_DIR / f"{split}_manifest.json", encoding="utf-8") as f:
            _MANIFEST_CACHE[split] = json.load(f)["files"]
    return _MANIFEST_CACHE[split]


# ============================================================
# 管线
# ============================================================
class DemoPipeline:
    def __init__(self):
        self.classifier = ECGFounderClassifier(
            thresholds_path=str(ROOT / "outputs" / "ecgfounder_mlp" / "metrics.json"))
        self.rpeak = RPeakDetector(method="pan_tompkins")
        self.hrv = HRVAnalyzer()
        self.qt = QTAnalyzer()
        try:
            self.kb = KnowledgeBase()
        except FileNotFoundError:
            self.kb = None
        try:
            self.llm = LLMInterface(backend="deepseek", model="deepseek-chat")
            self.llm_ok = self.llm.is_available
        except Exception:
            self.llm = None
            self.llm_ok = False

    # ---------- 输入加载 ----------
    def load_signal(self, record_id=None, files=None):
        """返回 (signal(12,5000), source_info) 或错误信息。"""
        if record_id:
            item = next((m for m in load_manifest("test")
                         if m["record_id"] == record_id), None)
            if item is None:
                return None, f"记录 {record_id} 不存在"
            sig = np.load(DATA_DIR / item["signal_file"])
            return sig, f"测试集: {item['source']}/{item['record_id']}（已预处理）"
        if files:
            # 2F 🟠-8 修复：扩展名大小写不敏感、整段异常兜底、复制而非移动上传文件
            try:
                hea = next((f for f in files if Path(f).suffix.lower() == ".hea"), None)
                mat = next((f for f in files if Path(f).suffix.lower() == ".mat"), None)
                if not hea or not mat:
                    return None, "请同时上传 .hea 与 .mat 两个文件"
                import shutil
                import wfdb
                with tempfile.TemporaryDirectory() as td:
                    # 5J 审查（🟠-1）：保留原文件名复制——gr.Files 保留上传文件
                    # 原名，.hea 内嵌信号文件名与之匹配即可被 wfdb 解析。
                    # 复检D（🟠-1）：勿改写 .hea——WFDB 记录行 [1] 是 n_signals、
                    # 信号行 [1] 是 format（文件名在 [0]），旧版改写 [1] 反而
                    # 触发 HeaderSyntaxError 使所有上传损坏
                    hea_dst = Path(td) / Path(hea).name
                    shutil.copy2(hea, hea_dst)
                    shutil.copy2(mat, Path(td) / Path(mat).name)
                    rec = wfdb.rdrecord(str(hea_dst.with_suffix("")))
                    sig = rec.p_signal.T.astype(np.float32)
                # ECGFounder 协议预处理（原生 fs）
                sig, _ = p5.reorder_leads(sig, rec.sig_name)
                filt = p5.filter_bandpass(sig, rec.fs)
                res = p5.resample_to_500(filt, rec.fs)
                seg, pl, pr = p5.segment_to_5000(res)
                sig = p5.official_zscore(seg, pl, pr)
                return sig, f"上传文件: {rec.record_name} (fs={rec.fs})"
            except Exception as e:
                return None, f"上传文件解析失败: {str(e)[:100]}（请确认 .hea/.mat 为有效 WFDB 记录）"
        return None, "请选择测试集记录或上传 WFDB 文件"

    # ---------- 工具链 ----------
    def run_tools(self, signal):
        out = {}
        lead_ii = signal[1]
        # 检查报告 1.9 修复：R 峰与分类器调用纳入异常兜底，坏输入不炸页面
        try:
            r = self.rpeak.detect(lead_ii, fs=500)
        except Exception as e:
            r = {"error": str(e)[:80]}
        out["r_peaks"] = r
        # 4J-RED-1 修复：sufficient 判断取键值而非键成员——RPeakDetector.detect()
        # 恒返回含 insufficient 键的字典，旧版 `"insufficient" in r` 恒真，
        # 导致 HRV 永远走"测量失败"分支（else 死代码，演示 HRV 工具完全失效）
        if "error" in r or r.get("insufficient") or not r.get("rr_intervals", []).size:
            # 第三轮审查 3G 🟠-2：R 峰失败时不伪造输入（旧版喂 [0.8] 给 HRV，
            # 空结果被显示为 "HRV SDNN: 0 ms"）
            out["hrv"] = {"error": "R 峰检测失败/不足，无法计算 HRV",
                          "sdnn": None, "rmssd": None}
        else:
            try:
                out["hrv"] = self.hrv.analyze(r.get("rr_intervals", np.array([0.8])))
            except Exception as e:
                out["hrv"] = {"error": str(e)[:60]}
        try:
            out["qt"] = self.qt.analyze(lead_ii, r.get("r_peaks", []), fs=500)
        except Exception as e:
            out["qt"] = {"error": str(e)[:60]}
        try:
            out["classifier"] = self.classifier.predict(signal)
        except Exception as e:
            out["classifier"] = {"error": str(e)[:80], "top_k": [], "positives": []}
        return out

    # ---------- LLM 双层报告 ----------
    def llm_report(self, signal, tools_out, context):
        if not self.llm_ok:
            return None
        pos_names = [d["name"] for d in tools_out["classifier"].get("positives", [])]
        feature_summary = json.dumps({
            "r_peaks": {"heart_rate": tools_out["r_peaks"].get("heart_rate"),
                        "rhythm": tools_out["r_peaks"].get("rhythm")},
            "qt": {"qtc_bazett": tools_out["qt"].get("qtc_bazett")},
            "hrv": {"sdnn": tools_out["hrv"].get("sdnn")},
            "classifier_positives": pos_names,
        }, ensure_ascii=False, default=str)

        kb_text = ""
        if self.kb:
            parts = []
            gen = self.kb.general_excerpt(max_chars=1000)
            if gen:
                parts.append(gen)
            parts += [self.kb.query(d["name"], max_chars=1500)
                      for d in tools_out["classifier"].get("positives", [])[:3]]
            parts.append(self.kb.format_catalog(max_chars=240))
            if parts:
                kb_text = "\n\n--- 医学参考知识库（分级可信）---\n" + "\n\n".join(parts)

        patient_info = (f"{context['age'] or '未知'}岁 {context['sex']}，{context['setting']}，"
                        f"主诉: {context['complaint'] or '未提供'}，病史: {context['history'] or '未提供'}")
        prompt = f"""你是心内科 AI 助手，做双层分析：

【患者情境】(仅用于风险与建议层)
{patient_info}

【确定性工具输出】(发现层唯一依据，不可编造)
{feature_summary}
{kb_text}

要求：
1. findings: 波形诊断结论，情境无关，只能从 classifier_positives 选取，逐条附工具证据
2. risk_assessment: 结合情境评估风险(low/moderate/high)并说明理由
3. recommendations: 紧急度(routine/urgent/emergent)与具体建议，结合情境
输出 JSON:
{{"findings": [{{"name": ..., "evidence": ...}}],
  "risk_assessment": {{"level": ..., "rationale": ...}},
  "recommendations": {{"urgency": ..., "actions": [...]}}}}"""
        try:
            raw = self.llm.chat(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": prompt}],
                response_format={"type": "json_object"})
        except Exception as e:
            # 检查报告 1.9 修复：API 异常显式暴露（旧版静默降级为占位文本）
            return {"error": f"模型调用失败: {str(e)[:120]}"}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"parse_error": raw[:200]}

    # ---------- 绘图 ----------
    def plot_waveform(self, signal, r_peaks):
        # 2F 🟠-11 修复：改用 matplotlib.figure.Figure 面向对象 API
        # （旧版 plt.subplots 依赖 pyplot 全局状态机，Gradio 并发请求会互相污染 figure）
        from matplotlib.figure import Figure
        leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        fig = Figure(figsize=(11, 9), dpi=110)
        axes = []
        for i in range(12):
            ax = fig.add_subplot(6, 2, i + 1, sharex=axes[0] if i else None)
            axes.append(ax)
        fig.subplots_adjust(hspace=0.45)
        t = np.arange(signal.shape[1]) / 500.0
        for i, (ax, name) in enumerate(zip(axes, leads)):
            ax.plot(t, signal[i], lw=0.7, color="black")
            if name == "II" and len(r_peaks):
                ax.plot(t[np.array(r_peaks)], signal[i][np.array(r_peaks)],
                        "o", ms=3, color="red", alpha=0.7)
            ax.set_ylabel(name, fontsize=8)
            ax.set_xlim(0, 10)
            ax.set_yticks([])
            ax.grid(True, ls=":", alpha=0.35)
        axes[-2].set_xlabel("时间 (s)", fontsize=9)
        axes[-1].set_xlabel("时间 (s)", fontsize=9)
        fig.suptitle("12 导联心电图（红点 = Ⅱ 导联 R 峰）", fontsize=12)
        return fig

    # ---------- 参考答案（真值，只写界面） ----------
    def ground_truth_md(self, record_id, cls_out=None):
        """测试集记录的 PhysioNet 官方真值 + 与分类器阳性的逐类对照。

        审计口径（论文 §4.4）：发现层内容必须完全来自分类器阳性（570/570）。
        本方法返回值只写入「参考答案（真值）」页签，**禁止**拼进 llm_report
        的 prompt 或 findings 文本——真值一旦进入推理路径，该不变式即失效。
        """
        head = "### 参考答案（真值标签）\n\n"
        if not record_id:
            return (head + "当前输入是**上传的 WFDB 文件**——真实使用场景下系统不持有真值，"
                    "这正是本演示的盲测视角。\n\n"
                    "如需核对答案，请在左侧下拉框选择一条**测试集记录**。")
        try:
            item = next((m for m in load_manifest("test")
                         if m["record_id"] == record_id), None)
        except Exception as e:
            return head + f"⚠️ 读取测试集清单失败: {html.escape(str(e)[:100], quote=False)}"
        if item is None:
            return head + f"⚠️ 记录 {html.escape(str(record_id), quote=False)} 不在测试集清单中"

        le = getattr(self.classifier, "le", None)
        labels = list(item.get("labels") or [])
        codes = list(item.get("dx_codes") or [])

        def _name(i):
            if le is not None and i < len(le.class_names):
                return le.class_names[i]
            return f"C{i}"

        gt_idx = [i for i, v in enumerate(labels) if v == 1]
        gt_names = [_name(i) for i in gt_idx]
        unmapped = [c for c in codes
                    if (le.get_class_name(c) if le is not None else None) is None]

        esc = lambda v: html.escape(str(v), quote=False)  # noqa: E731
        src = esc(item.get("source"))
        rid = esc(record_id)
        L = [head.rstrip("\n"), ""]
        L.append("> 来源: PhysioNet 2020 官方标注（`processed_5k/test_manifest.json`）。"
                 "**真值只在此页签显示**：不进入发现层、不进入 LLM prompt"
                 "（发现层内容必须完全来自分类器阳性）。")
        L.append("")
        L.append(f"**记录**: `{src}/{rid}`")
        L.append("")
        L.append(f"#### ① 官方真值（27 评分类，{len(gt_idx)} 类）")
        if gt_idx:
            L += [f"- **{esc(_name(i))}**" for i in gt_idx]
        else:
            L.append("- （本记录无 27 类中的阳性标注）")
        if unmapped:
            L.append("")
            L.append(f"#### ② 非评分类码（{len(unmapped)} 个，不参与 27 类指标）")
            L.append("- " + "、".join(f"`{esc(c)}`" for c in unmapped))

        # ---- 分类器概率 / 阈值对齐（任一缺失则跳过对照，不猜测） ----
        probs = thresholds = None
        if isinstance(cls_out, dict) and cls_out.get("probs") is not None:
            try:
                probs = np.asarray(cls_out["probs"], dtype=float).ravel()
                thresholds = np.asarray(self.classifier.thresholds, dtype=float).ravel()
                if probs.size != thresholds.size:
                    probs = thresholds = None
            except Exception:
                probs = thresholds = None
        positives = cls_out.get("positives", []) if isinstance(cls_out, dict) else []

        L.append("")
        L.append("#### ③ 与分类器对照（逐类 Youden 阈值，来自 `outputs/ecgfounder_mlp/metrics.json`）")
        if probs is None:
            if cls_out is None:
                # 仅选中记录（未点"开始分析"）：先给真值，对照待分析后补上
                L.append("- 点击「开始分析」后，将在此显示分类器对该记录的逐类对照"
                         "（预测概率 / 该类阈值 / 命中或漏报）。")
                return "\n".join(L)
            err = cls_out.get("error") if isinstance(cls_out, dict) else None
            why = f"（{esc(str(err)[:80])}）" if err else ""
            L.append(f"- ⚠️ 本次分类器未产出概率{why}，仅显示真值，无法逐类对照。")
            return "\n".join(L)

        hit, miss = [], []
        L.append("")
        L.append("| 真值类 | 预测概率 | 该类阈值 | 判定 |")
        L.append("|---|---|---|---|")
        for i in gt_idx:
            if i >= probs.size:
                L.append(f"| {esc(_name(i))} | — | — | ⚠️ 概率越界 |")
                continue
            p, th = float(probs[i]), float(thresholds[i])
            if p >= th:
                hit.append(i)
                verdict = "✅ 命中（判为阳性）"
            else:
                miss.append(i)
                verdict = "❌ 漏报（未达阈值）"
            L.append(f"| {esc(_name(i))} | {p:.1%} | {th:.2f} | {verdict} |")
        L.append("")
        line = f"- **命中**: {len(hit)}/{len(gt_idx)} 类"
        if miss:
            line += "；**漏报**: " + "、".join(f"{esc(_name(i))}" for i in miss)
        else:
            line += "（真值类全部被识别）"
        L.append(line)
        extra = sorted((d for d in positives
                        if isinstance(d, dict) and d.get("name")
                        and d["name"] not in set(gt_names)),
                       key=lambda d: -float(d.get("prob") or 0.0))
        if extra:
            shown = extra[:10]
            txt = "、".join(f"{esc(d['name'])} {float(d.get('prob') or 0):.0%}"
                            for d in shown)
            more = f"（共 {len(extra)} 类）" if len(extra) > len(shown) else ""
            L.append(f"- **额外阳性**（分类器判阳性、该记录真值未标注）: {txt}{more}")
        else:
            L.append("- **额外阳性**: 无")
        L.append("")
        L.append("> ⚠️ 解读注意：PhysioNet 2020 标注**非穷尽**——真值未列出的阳性不等于一定"
                 "误报（可能确实存在但官方未标注）；「漏报」也可能源于标注口径差异。"
                 "另注：官方规定 RBBB/CRBBB、PAC/SVPB、PVC/VPB 三对「计为同一诊断」，"
                 "本系统按官方评分类保留为独立类，故这三对之间常出现「真值标 A、判为 B」"
                 "的显示差异，并非模型错误。"
                 "本页签仅供演示核对，论文口径一律以 test manifest 全量统计为准。")
        return "\n".join(L)

    # ---------- 总入口 ----------
    def run(self, record_id, files, age, sex, setting, complaint, history):
        context = {"age": age, "sex": sex, "setting": setting,
                   "complaint": complaint, "history": history}
        signal, info = self.load_signal(record_id, files)
        if signal is None:
            # 参考答案独立于信号加载：记录存在即可核对真值
            return (None, info, "", "", "❌ " + info,
                    self.ground_truth_md(record_id, None))

        tools_out = self.run_tools(signal)
        # 5K 新增：真值页签（只写界面，不进 llm_report / findings）
        gt_md = self.ground_truth_md(record_id, tools_out.get("classifier"))
        fig = self.plot_waveform(signal, tools_out["r_peaks"].get("r_peaks", []))

        # 发现层 markdown
        r = tools_out["r_peaks"]
        # R3 修复（第二次审查 2D-R1/R3）：测量失败显式呈现，
        # 禁止把 0 bpm / QTc 0 ms 当真实测量值展示
        hr_text = (f"**{r.get('heart_rate')} bpm**（节律: {r.get('rhythm')}）"
                   if r.get("heart_rate") and r.get("rhythm") != "insufficient_data"
                   else "**测量失败（R 峰不足，无法可靠计算心率/节律）**")
        qt_val = tools_out["qt"].get("qtc_bazett")
        qt_text = (f"**{qt_val} ms**" if qt_val is not None
                   else "**测量失败（QT 波形数据不足）**")
        # 第三轮审查 3G 🟠-2/🟡-6：HRV 失败显式呈现（旧版把伪造输入/空结果
        # 显示为 "HRV SDNN: 0 ms"）
        hrv_out = tools_out["hrv"]
        sdnn_val = hrv_out.get("sdnn") if isinstance(hrv_out, dict) else None
        hrv_text = (f"**{sdnn_val} ms**" if sdnn_val is not None
                    else "**测量失败（RR 间期不足）**")
        findings_lines = [
            # 5J 审查（🟡-3）：info（上传 record_name 等）HTML 转义
            f"**输入**: {__import__('html').escape(str(info), quote=False)}",
            "",
            "### 确定性工具输出",
            f"- 心率: {hr_text}",
            f"- QTc(Bazett): {qt_text}",
            f"- HRV SDNN: {hrv_text}",
            "",
            "### ECGFounder 发现层（27 类, test macro AUC 0.9456）",
        ]
        # 4J-ORANGE-3 修复：分类器失败显式呈现——旧版分类器出错/空结果时
        # 发现层静默为空，与"测量失败显式呈现"约定不符（agent 主路径有兜底）
        cls_out = tools_out["classifier"]
        top_k = cls_out.get("top_k", []) if isinstance(cls_out, dict) else []
        if isinstance(cls_out, dict) and cls_out.get("error"):
            findings_lines.append(f"- ⚠️ **诊断分类失败**: {cls_out['error']}"
                                  "（信号质量异常或分类器不可用，以上内容不应作为排除异常的依据）")
        elif not top_k:
            findings_lines.append("- ⚠️ **诊断分类失败**: 分类器未返回任何诊断结果"
                                  "（请重试或检查信号质量）")
        else:
            # 4J-YELLOW-8 修复：发现层只显示分类器阳性（Youden 阈值以上），
            # 不再把非阳性 top-k 以"[观察]"混入发现层（与"发现层只能显示阳性"约定一致）
            # 5J 审查（🟠-2）：阳性集合与 LLM 白名单/知识库一致——旧版只扫 top_k
            # 的 positive（最多 top-5 内的阳性），而 LLM 白名单/知识库用全量
            # positives（27 类逐类阈值判定，排名第 6+ 的阳性被漏显示）
            positives = cls_out.get("positives", [])
            pos_rows = [f"- [阳性] {d['name']}: {d['prob']:.0%}（Youden 阈值以上）"
                        for d in positives if isinstance(d, dict) and d.get("positive")]
            if pos_rows:
                findings_lines.extend(pos_rows)
            else:
                findings_lines.append("- 无 Youden 阈值以上阳性诊断（分类器未检出高置信度异常）")
        findings = "\n".join(findings_lines)

        # 知识库片段（2F 🟠-9 + 2026-08-28：总论 + 阳性详情 + 全 27 类目录）
        kb_md = ""
        if self.kb:
            kb_parts = ["### 知识库依据（分级可信来源）"]
            gen = self.kb.general_excerpt(max_chars=800)
            if gen:
                kb_parts.append(gen)
            for d in tools_out["classifier"].get("positives", [])[:3]:
                kb_parts.append(self.kb.query(d["name"], max_chars=900))
            kb_parts.append(self.kb.format_catalog(max_chars=200))
            kb_md = "\n\n".join(kb_parts)

        # LLM 双层报告
        llm_out = self.llm_report(signal, tools_out, context)
        if llm_out is None:
            report_md = ("⚠️ 未检测到 DeepSeek API key（`.env` 未配置）——当前为**无 LLM 模式**："
                         "发现层与知识库正常，风险/建议层不可用。配置后自动启用。")
            return fig, findings, kb_md, report_md, "无 LLM 模式（工具链正常）", gt_md

        if "parse_error" in llm_out:
            # 5J 审查（🟡-3）：LLM 原始输出/异常消息/上传信息同样 HTML 转义
            # （成功分支已用 esc()，此处口径不一致，注入风险）
            import html as _html2
            report_md = (f"⚠️ LLM 输出解析失败: "
                         f"{_html2.escape(str(llm_out['parse_error']), quote=False)}")
            return fig, findings, kb_md, report_md, "LLM 解析失败", gt_md

        if "error" in llm_out:
            import html as _html2
            report_md = (f"⚠️ LLM 调用失败（模型未连接或 API 错误）: "
                         f"{_html2.escape(str(llm_out['error']), quote=False)}")
            return fig, findings, kb_md, report_md, "LLM 调用失败", gt_md

        # 2F 🟠-4 修复：LLM 输出形状校验 + findings 白名单（仅允许分类器阳性诊断，
        # 越界项丢弃并注明；类型不对时降级为占位而不是炸页面）
        f = llm_out.get("findings", [])
        risk = llm_out.get("risk_assessment", {})
        recs = llm_out.get("recommendations", {})
        if not isinstance(f, list):
            f = []
        if not isinstance(risk, dict):
            risk = {}
        if not isinstance(recs, dict):
            recs = {}
        pos_names = {d["name"] for d in tools_out["classifier"].get("positives", [])}
        kept, dropped = [], 0
        for x in f:
            if not isinstance(x, dict):
                continue
            if x.get("name") in pos_names:
                kept.append(x)
            else:
                dropped += 1
        if dropped:
            f = kept + [{"name": "（发现层白名单过滤）",
                         "evidence": f"LLM 返回 {dropped} 个非分类器阳性诊断，已丢弃"}]
        # 第三轮审查 3G 💡-1：LLM 自由文本 HTML 转义后进 Markdown
        # （旧版原文直拼，<script>/<img onerror> 构成注入风险）
        import html as _html

        def esc(v):
            return _html.escape(str(v or ""), quote=False)

        lines = [
            "## 发现层（情境无关）",
            *(f"- **{esc(x.get('name'))}**: {esc(x.get('evidence'))}" for x in f),
            "",
            f"## 风险层（情境条件化）: **{esc(risk.get('level', 'N/A'))}**",
            f"> {esc(risk.get('rationale'))}",
            "",
            f"## 建议层（情境条件化）: 紧急度 **{esc(recs.get('urgency', 'N/A'))}**",
            *(f"- {esc(a)}" for a in recs.get("actions", []) if isinstance(a, str)),
        ]
        return (fig, findings, kb_md, "\n".join(lines),
                "分析完成（LLM 双层报告已生成）", gt_md)


pipeline = DemoPipeline()


def load_test_records():
    # 检查报告 1.8 修复：显示 "source/record_id"，值传纯 record_id（load_signal 按纯 id 匹配）
    # 第三轮审查 3G 🟡-2：条数动态生成（旧版硬编码 6,472 与患者级划分后的 3,754 不符）
    return [(f"{m['source']}/{m['record_id']}", m["record_id"])
            for m in load_manifest("test")]


def test_records_label():
    try:
        n = len(load_test_records())
    except Exception:
        n = 0
    return f"从测试集选择（{n:,} 条）"


with gr.Blocks(title="ECG AI Agent — 情境条件化双层诊断") as demo:
    gr.Markdown("""
    # 🫀 ECG AI Agent — 情境条件化双层诊断系统
    **流水线**: ECGFounder 发现层(0.9456) → 特征工具(R峰/HRV/QT) → 分级可信知识库(27 类) → deepseek-chat 双层报告
    同一份心电图在不同临床情境下：**发现不变，风险与建议随情境调整**。
    页签「参考答案（真值）」仅在选中**测试集记录**时显示官方标注，用于核对；
    真值不参与发现层与 LLM 报告（上传文件时为真实使用场景，无真值）。
    """)

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### ① 输入")
            record_dd = gr.Dropdown(choices=load_test_records(),
                                    label=test_records_label(), value=None)
            # 2F 🟡-16：限制上传类型与大小
            files_in = gr.Files(label="或上传 WFDB 文件（.hea + .mat）",
                                file_count="multiple",
                                file_types=[".hea", ".mat"])
            gr.Markdown("### ② 患者情境（影响风险/建议层，不影响发现层）")
            with gr.Row():
                # 2F 🟡-13：年龄合法范围
                age = gr.Number(label="年龄", value=65, precision=0,
                                minimum=0, maximum=120)
                sex = gr.Radio(["Male", "Female", "Unknown"], label="性别", value="Male")
                setting = gr.Dropdown(SCENARIOS, label="就诊场景", value="急诊科")
            complaint = gr.Textbox(label="主诉", value="胸痛 3 天，伴心悸", lines=2)
            history = gr.Textbox(label="病史", value="高血压 10 年，吸烟 30 年", lines=2)
            btn = gr.Button("开始分析", variant="primary")

        with gr.Column(scale=3):
            gr.Markdown("### ③ 输出")
            status = gr.Markdown("")
            with gr.Tabs():
                with gr.Tab("波形与发现"):
                    plot_out = gr.Plot(label="12 导联心电图")
                    findings_out = gr.Markdown(label="工具链 + 发现层")
                with gr.Tab("双层诊断报告"):
                    report_out = gr.Markdown(label="风险与建议（情境条件化）")
                with gr.Tab("知识库依据"):
                    kb_out = gr.Markdown(label="分级可信医学参考资料")
                with gr.Tab("参考答案（真值）"):
                    gt_out = gr.Markdown(label="PhysioNet 官方标注 + 分类器对照")

    btn.click(pipeline.run,
              inputs=[record_dd, files_in, age, sex, setting, complaint, history],
              outputs=[plot_out, findings_out, kb_out, report_out, status, gt_out])

    # 2F 🟡-14：选中测试集记录时联动预填 manifest 的 age/sex
    # 5K：同时预填「参考答案（真值）」页签（选中即可核对答案，无需先跑分析）
    def _fill_from_record(record_id):
        gt_now = pipeline.ground_truth_md(record_id, None)
        if not record_id:
            return gr.update(), gr.update(), gt_now
        item = next((m for m in load_manifest("test")
                     if m["record_id"] == record_id), None)
        if item is None:
            return gr.update(), gr.update(), gt_now
        a = item.get("age")
        s = item.get("sex")
        return (gr.update(value=int(a) if a is not None else None),
                gr.update(value=s if s in ("Male", "Female") else "Unknown"),
                gt_now)

    record_dd.change(_fill_from_record, inputs=record_dd,
                     outputs=[age, sex, gt_out])


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, theme=gr.themes.Soft())
