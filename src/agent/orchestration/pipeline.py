"""
Agent 管道 — 端到端 ECG 诊断工作流 / End-to-End ECG Diagnosis Pipeline.

编排完整流程: 加载 ECG → 预处理 → Agent 诊断 → 格式化报告。
提供命令行诊断和交互式多轮对话两种模式。

Orchestrates: Load ECG → Preprocess → Agent diagnosis → Format report.
Supports both single-run and interactive multi-turn modes.

使用示例 / Usage:
    pipeline = AgentPipeline(agent, preprocessor, label_extractor)
    result = pipeline.run("ecg.npy", {"age": 65, "sex": "Male"})
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class AgentPipeline:
    """端到端 ECG 诊断管道 / End-to-end ECG diagnosis pipeline.

    参数 / Args:
        agent:           ECGAIAgent 实例
        preprocessor:    ECGPreprocessor 实例 (可选)
        label_extractor: LabelExtractor 实例 (用于标签解码)
        model:           训练好的 ArrhythmiaClassifier (可选, 用于直接推理)
    """

    def __init__(self, agent, preprocessor=None, label_extractor=None, model=None):
        self.agent, self.preprocessor, self.label_extractor, self.model = agent, preprocessor, label_extractor, model

    def run(self, ecg_input: Union[str, Path, np.ndarray], patient_info: Optional[Dict] = None,
            query: str = "分析这份心电图并给出诊断。") -> Dict:
        """运行完整诊断管道 / Run complete diagnostic pipeline.

        参数 / Args:
            ecg_input:    .npy 文件路径 或 (12, L) ECG 数组
            patient_info: {age, sex, symptoms, history}
            query:        诊断查询文本

        返回 / Returns:
            {diagnosis, confidence, heart_rate, rhythm, intervals, reasoning_chain, recommendations, report}
        """
        ecg_signal = self._load_ecg(ecg_input)                                   # 1. 加载
        if self.preprocessor: ecg_signal = self.preprocessor(ecg_signal, 500.0)  # 2. 预处理
        result = self.agent.diagnose(ecg_signal, patient_info, query)            # 3. Agent诊断
        return {                                                                  # 4. 格式化
            "diagnosis": result.diagnosis, "confidence": result.confidence,
            "heart_rate": result.heart_rate, "rhythm": result.rhythm,
            "intervals": result.intervals,
            "reasoning_chain": [{"step": s.action, "reason": s.reason, "result": s.result, "error": s.error}
                                for s in result.reasoning_chain],
            "recommendations": result.recommendations,
            "report": self._format_report(result),
        }

    def _load_ecg(self, ecg_input) -> np.ndarray:
        """加载 ECG / Load ECG from file or array."""
        if isinstance(ecg_input, np.ndarray): return ecg_input.astype(np.float32)
        path = Path(ecg_input)
        if path.suffix == ".npy": return np.load(path).astype(np.float32)
        if path.suffix == ".hea":
            from src.data_pipeline.loader import ECGLoader
            sample = ECGLoader(path.parent).load_record(path.stem)
            if sample: return sample.signal
        raise ValueError(f"不支持的 ECG 输入: {ecg_input}")

    def _format_report(self, result) -> str:
        """格式化诊断结果为 Markdown 报告 / Format diagnosis as structured markdown report."""
        lines = ["# ECG 诊断报告", "", "## ECG 发现",
                 f"- 心率: {result.heart_rate or 'N/A'} bpm",
                 f"- 节律: {result.rhythm or 'N/A'}"]
        if result.intervals:
            lines.append(f"- QTc (Bazett): {result.intervals.get('qtc_bazett', 'N/A')} ms")
        lines.extend(["", "## 主要诊断"])
        for d in result.diagnosis[:5]:
            lines.append(f"- **{d['name']}** (置信度: {d['confidence']:.1%})")
        lines.extend(["", "## 建议"])
        for r in result.recommendations: lines.append(f"- {r}")
        lines.extend(["", "---", "*由 ECG AI Agent 生成。仅供临床参考。*"])
        return "\n".join(lines)

    def run_interactive(self):
        """交互式诊断会话 / Interactive diagnostic session."""
        print("=" * 60 + "\nECG AI Agent — 交互式诊断会话\n" + "=" * 60)
        result = self.run(input("\nECG 文件路径 (.npy): ").strip())
        print(result["report"])
        return result
