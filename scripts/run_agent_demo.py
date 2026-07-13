#!/usr/bin/env python3
"""ECG AI Agent Demo — Interactive diagnostic session.

Usage:
    # With real LLM (set DEEPSEEK_API_KEY env var)
    python scripts/run_agent_demo.py --ecg data/physionet2020/processed/cpsc_2018_A0001.npy

    # Mock mode (no API key needed, for testing)
    python scripts/run_agent_demo.py --mock

    # Interactive multi-turn
    python scripts/run_agent_demo.py --interactive
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="ECG AI Agent Demo")
    parser.add_argument("--ecg", type=str, help="Path to .npy ECG file")
    parser.add_argument("--mock", action="store_true", help="Use mock LLM (no API key needed)")
    parser.add_argument("--interactive", action="store_true", help="Interactive multi-turn mode")
    parser.add_argument("--backend", default="deepseek", help="LLM backend")
    parser.add_argument("--model", default="deepseek-v4-pro", help="LLM model")
    args = parser.parse_args()

    # 1. Setup LLM
    from src.agent.llm.llm_interface import LLMInterface

    if args.mock:
        logger.info("Mock mode: using placeholder LLM responses")
        llm = LLMInterface(backend="deepseek", model="mock")
        llm._client = None  # Force mock mode
    else:
        llm = LLMInterface(backend=args.backend, model=args.model)

    # 2. Setup Tool Registry
    from src.agent.tools.registry import ToolRegistry
    from src.agent.tools.ecg_tools import register_ecg_tools
    from src.agent.tools.diagnosis_tools import register_diagnosis_tools

    registry = ToolRegistry()
    register_ecg_tools(registry)

    # Optional: Load trained model for classifier tool
    model = None
    detector = None
    try:
        from src.ecg_models.backbone.inception_time import inception_time
        from src.ecg_models.classifiers.arrhythmia_classifier import ArrhythmiaClassifier
        backbone = inception_time(in_channels=12)
        model = ArrhythmiaClassifier(backbone, num_classes=27)
        logger.info("Model loaded (untrained — train first for real predictions)")
    except Exception as e:
        logger.warning(f"Model not loaded: {e}")

    from src.data_pipeline.label_extractor import LabelExtractor
    label_extractor = LabelExtractor()
    register_diagnosis_tools(registry, model=model, detector=detector, label_extractor=label_extractor)

    # 3. Create Agent
    from src.agent.core.agent import ECGAIAgent
    agent = ECGAIAgent(llm, registry, verbose=True)

    # 4. Load/prepare ECG
    if args.ecg:
        ecg = np.load(args.ecg).astype(np.float32)
    else:
        logger.info("No ECG file provided; using synthetic signal for demo")
        t = np.linspace(0, 10, 5000)
        ecg = 0.5 * np.sin(2 * np.pi * 1.2 * t)
        for i in range(0, 5000, 417):
            if i + 5 < 5000:
                ecg[i:i+5] += 2.0
        ecg = np.tile(ecg, (12, 1)).astype(np.float32)

    # 5. Run diagnosis
    patient_info = {
        "age": 65,
        "sex": "Male",
        "symptoms": "chest pain for 3 days, occasional palpitations",
    }

    print("\n" + "=" * 60)
    print("ECG AI Agent — Diagnostic Analysis")
    print("=" * 60)
    print(f"ECG shape: {ecg.shape}")
    print(f"Patient: {patient_info['age']}yo {patient_info['sex']}, {patient_info['symptoms']}")
    print("=" * 60 + "\n")

    result = agent.diagnose(ecg, patient_info)
    print(format_result(result))


def format_result(result) -> str:
    """Format DiagnosisResult for display."""
    lines = [
        "\n" + "=" * 60,
        "DIAGNOSTIC REPORT",
        "=" * 60,
        f"\nHeart Rate: {result.heart_rate or 'N/A'} bpm",
        f"Rhythm: {result.rhythm or 'N/A'}",
    ]
    if result.intervals:
        lines.append(f"QTc: {result.intervals.get('qtc_bazett', 'N/A')} ms")

    lines.append(f"\nConfidence: {result.confidence:.1%}")
    lines.append("\nDiagnoses:")
    for d in result.diagnosis[:5]:
        lines.append(f"  - {d['name']} ({d['confidence']:.1%})")

    lines.append("\nReasoning Chain:")
    for step in result.reasoning_chain:
        status = "OK" if not step["error"] else f"ERR: {step['error']}"
        lines.append(f"  [{status}] {step['step']}: {step['reason']}")

    lines.append("\nRecommendations:")
    for r in result.recommendations:
        lines.append(f"  - {r}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


if __name__ == "__main__":
    main()
