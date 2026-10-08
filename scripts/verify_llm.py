#!/usr/bin/env python3
"""M2.1 — DeepSeek LLM 全链路冒烟验证。

验证四项:
    1. 连通性: DEEPSEEK_API_KEY 是否就绪（.env 或环境变量）
    2. Planner: 输出 JSON 计划且可解析
    3. Reasoner: 结构化医学推理输出
    4. Reflector: continue / revise / complete 三态输出

用法:
    python scripts/verify_llm.py            # 真 API
    python scripts/verify_llm.py --mock     # 无 key 时用 Mock 测试管道
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.agent.llm.llm_interface import LLMInterface
from src.agent.llm.prompt_templates import (
    SYSTEM_PROMPT, PLANNING_PROMPT, REASONING_PROMPT, REFLECTION_PROMPT,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

TOOLS_DESC = """- extract_r_peaks: R 峰检测，输出心率与节律
- compute_hrv: 心率变异性（依赖 R 峰）
- measure_qt_interval: QT/QTc 间期（依赖 R 峰）
- classify_arrhythmia: 27 类心律失常/传导/形态多标签分类（ECGFounder 基础模型）
- plot_waveform: 标注波形绘图"""


def test_planner(llm):
    prompt = f"""你是心内科 AI 助手，规划该心电图的诊断步骤。

可用工具:
{TOOLS_DESC}

患者: 65 岁男性，急诊，胸痛 3 天，既往高血压
ECG: (12, 5000) @ 500Hz

请以 JSON 输出诊断计划: {{"plan": [{{"action": 工具名, "reason": 理由}}]}}
只调用必要工具，输出合法 JSON。"""
    raw = llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}],
                   response_format={"type": "json_object"})
    try:
        data = json.loads(raw)
        # 兼容两种格式: {"plan": [...]} 或顶层数组 [...]
        if isinstance(data, list):
            steps = data
        elif isinstance(data, dict):
            steps = data.get("plan", [])
        else:
            steps = []
        ok = isinstance(steps, list) and len(steps) > 0 and all(
            isinstance(s, dict) and "action" in s for s in steps)
        return ok, steps
    except json.JSONDecodeError:
        return False, raw[:200]


def test_reasoner(llm):
    prompt = REASONING_PROMPT.format(
        patient_info="65 岁男性，急诊胸痛",
        findings="心率 110bpm，心律绝对不齐；分类器: 房颤 92%；QTc 412ms",
    )
    raw = llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}])
    return len(raw) > 50, raw[:150]


def test_reflector(llm):
    prompt = REFLECTION_PROMPT.format(
        step_name="classify_arrhythmia",
        result="房颤 92%，但 R 峰检测显示心率 210bpm",
        previous_findings="心率 210bpm（R 峰检测）",
    )
    raw = llm.chat([{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}])
    verdict = "continue" if "continue" in raw.lower() and "revise" not in raw.lower() \
        else "revise" if "revise" in raw.lower() else "unknown"
    return verdict, raw[:150]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--model", default="deepseek-chat")
    args = parser.parse_args()

    if args.mock:
        llm = LLMInterface(backend="deepseek", model=args.model, mock=True)
        logger.info("Mock 模式（不调用真实 API）")
    else:
        llm = LLMInterface(backend="deepseek", model=args.model)
        if not llm.is_available:
            logger.error("无 API key。请在 ecg-ai-agent/.env 写入 DEEPSEEK_API_KEY=sk-xxx "
                         "或设置环境变量后重试；也可先 --mock 验证管道。")
            sys.exit(1)
        logger.info(f"已连接: backend=deepseek, model={args.model}")

    ok_p, steps = test_planner(llm)
    logger.info(f"[Planner] 解析成功={ok_p}")
    if ok_p:
        logger.info(f"  计划步骤: {[s.get('action') for s in steps]}")

    ok_r, head = test_reasoner(llm)
    logger.info(f"[Reasoner] 输出长度正常={ok_r} | {head}...")

    verdict, head = test_reflector(llm)
    logger.info(f"[Reflector] 判定={verdict} | {head}...")

    all_ok = ok_p and ok_r and verdict in ("continue", "revise")
    logger.info("=" * 50)
    logger.info(f"冒烟结果: {'全部通过 ✅' if all_ok else '存在异常 ❌'}")


if __name__ == "__main__":
    main()
