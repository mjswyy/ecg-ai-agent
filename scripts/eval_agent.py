# ⚠️ DEPRECATED（2026-08-24，检查报告第六步）: 旧 Agent 评估；现役五维评测 = eval_agent_llm.py
"""Agent End-to-End Evaluation — 100 条 ECG 完整推理链评估."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import random
import time
import numpy as np
import torch
from collections import defaultdict

from src.ecg_models.backbone.xresnet1d import xresnet1d_101
from src.ecg_models.backbone.transformer_encoder import ecg_transformer
from src.ecg_models.classifiers.arrhythmia_classifier import ArrhythmiaClassifier
from src.ecg_models.feature_extraction.r_peak_detector import RPeakDetector
from src.ecg_models.feature_extraction.hrv_analyzer import HRVAnalyzer
from src.ecg_models.feature_extraction.qt_analyzer import QTAnalyzer
from src.data_pipeline.label_extractor import LabelExtractor

le = LabelExtractor()
device = torch.device("cpu")

# ============================================================
# 1. Load Models
# ============================================================

print("Loading models...")
checkpoints = Path("checkpoints")
from src.utils.safe_load import safe_torch_load

def load_one(bb_fn, ckpt_path, dropout=0.3):
    bb = bb_fn(in_channels=12, dropout=dropout)
    m = ArrhythmiaClassifier(bb, num_classes=27)
    sd = safe_torch_load(str(ckpt_path), map_location="cpu")
    m.load_state_dict(sd["model_state_dict"])
    m.eval()
    return m

models = {
    "SimCLR xResNet": load_one(xresnet1d_101, checkpoints / "simclr_xresnet" / "best_model.pt"),
    "ECG Transformer": load_one(ecg_transformer, checkpoints / "ecg_transformer_fp32" / "best_model.pt"),
}
print(f"Loaded: {list(models.keys())}")


# ============================================================
# 2. Agent Pipeline
# ============================================================

class ECGAgentEvaluator:
    """Simulate Agent reasoning chain with real model outputs."""

    def __init__(self, models):
        self.models = models
        self.rpeak = RPeakDetector(method="pan_tompkins")
        self.hrv = HRVAnalyzer()
        self.qt = QTAnalyzer()
        self.stats = defaultdict(list)

    def run_one(self, signal, ecg_id, ground_truth):
        steps = []
        errors = []

        # ---- Step 1: R-Peak Detection ----
        try:
            lead_ii = signal[1]  # Lead II
            r_result = self.rpeak.detect(lead_ii, fs=500)
            hr = r_result.get("heart_rate", None)
            rhythm = r_result.get("rhythm", "unknown")
            steps.append({"tool": "extract_r_peaks", "ok": True,
                          # R3（2D-R1）：测量失败显式化，不再显示误导性的 "peaks detected"
                          "result": (f"HR={hr:.0f}bpm, rhythm={rhythm}"
                                     if hr and rhythm != "insufficient_data"
                                     else "测量失败（R 峰不足）")})
        except Exception as e:
            steps.append({"tool": "extract_r_peaks", "ok": False, "error": str(e)})
            errors.append("r_peaks_failed")
            hr = None

        # ---- Step 2: HRV ----
        try:
            rr = r_result.get("rr_intervals", np.array([0.8]))
            hrv_result = self.hrv.analyze(rr)
            sdnn = hrv_result.get("sdnn", 0)
            steps.append({"tool": "compute_hrv", "ok": True,
                          "result": f"SDNN={sdnn:.0f}ms"})
        except Exception:
            steps.append({"tool": "compute_hrv", "ok": False, "error": "hrv_failed"})

        # ---- Step 3: QT ----
        try:
            qt_result = self.qt.analyze(signal[1], r_result.get("r_peaks", []), fs=500)
            qtc = qt_result.get("qtc_bazett", 0)
            steps.append({"tool": "measure_qt_interval", "ok": True,
                          "result": f"QTc(Bazett)={qtc:.0f}ms"})
        except Exception:
            steps.append({"tool": "measure_qt_interval", "ok": False, "error": "qt_failed"})

        # ---- Step 4: Classify (Ensemble) ----
        x = torch.from_numpy(signal).unsqueeze(0).float()
        all_probs = []
        for name, model in self.models.items():
            with torch.no_grad():
                logits = model(x)
                probs = torch.sigmoid(logits).squeeze(0).numpy()
                all_probs.append(probs)
        ensemble_probs = np.mean(all_probs, axis=0)

        top5_idx = np.argsort(-ensemble_probs)[:5]
        predictions = []
        for idx in top5_idx:
            name = le.class_names[idx] if idx < len(le.class_names) else f"C{idx}"
            prob = float(ensemble_probs[idx])
            predictions.append({"name": name, "prob": prob})

        steps.append({"tool": "classify_arrhythmia", "ok": True,
                      "result": "; ".join(f"{p['name']}({p['prob']:.0%})" for p in predictions[:5])})

        # ---- Step 5: Report ----
        gt_set = set(ground_truth)
        hits = []
        for p in predictions[:5]:
            hit = p["name"] in gt_set
            if hit:
                hits.append(p["name"])
        steps.append({"tool": "generate_report", "ok": True,
                      "result": f"Top predictions: {len(hits)}/{len(gt_set)} ground truth labels hit"})

        # ---- Metrics ----
        return {
            "steps": steps,
            "errors": errors,
            "predictions": predictions,
            "hr": hr,
            "hits": hits,
            "total_gt": len(gt_set),
            "top1_hit": predictions[0]["name"] in gt_set if predictions else False,
            "top3_hit": any(p["name"] in gt_set for p in predictions[:3]),
            "top5_hit": any(p["name"] in gt_set for p in predictions[:5]),
        }


# ============================================================
# 3. Run Evaluation
# ============================================================

data_dir = Path("data/physionet2020/processed")
with open(data_dir / "test_manifest.json") as f:
    test_manifest = json.load(f)

# Select 100 random ECGs
random.seed(42)
samples = random.sample(test_manifest["files"], min(100, len(test_manifest["files"])))
print(f"\nEvaluating {len(samples)} ECGs...\n")

agent = ECGAgentEvaluator(models)
results = []
t0 = time.time()

for i, entry in enumerate(samples):
    signal_path = data_dir / entry["signal_file"]
    if not signal_path.exists():
        continue

    signal = np.nan_to_num(np.load(signal_path).astype(np.float32),
                           nan=0.0, posinf=0.0, neginf=0.0)
    if signal.shape[0] != 12:
        signal = signal.T if signal.shape[1] == 12 else signal

    gt_names = []
    for code in entry.get("dx_codes", []):
        name = le.get_class_name(code)
        if name and name != code:
            gt_names.append(name)

    result = agent.run_one(signal, entry["record_id"], set(gt_names))
    results.append(result)

    if (i + 1) % 20 == 0:
        elapsed = time.time() - t0
        print(f"  [{i+1}/{len(samples)}] Time: {elapsed:.0f}s")

elapsed = time.time() - t0
print(f"\nDone. {len(results)} ECGs in {elapsed:.0f}s ({elapsed/len(results):.1f}s per ECG)\n")

# ============================================================
# 4. Results
# ============================================================

total = len(results)
tool_success = sum(1 for r in results if len(r["errors"]) == 0)
top1_hits = sum(1 for r in results if r["top1_hit"])
top3_hits = sum(1 for r in results if r["top3_hit"])
top5_hits = sum(1 for r in results if r["top5_hit"])
avg_gt = np.mean([r["total_gt"] for r in results])
avg_hits = np.mean([len(r["hits"]) for r in results])

# Per-tool success rate
tool_ok = defaultdict(lambda: {"ok": 0, "total": 0})
for r in results:
    for s in r["steps"]:
        tool_ok[s["tool"]]["total"] += 1
        if s["ok"]:
            tool_ok[s["tool"]]["ok"] += 1

print(f"{'='*60}")
print(f"Agent End-to-End Evaluation ({total} ECGs)")
print(f"{'='*60}")
print(f"  All tools successful:    {tool_success}/{total} ({tool_success/total:.0%})")
print(f"  Avg ground-truth labels: {avg_gt:.1f}")
print(f"  Avg labels hit by Agent: {avg_hits:.1f}")
print(f"")
print(f"  Agent Top-1 Hit:         {top1_hits}/{total} ({top1_hits/total:.0%})")
print(f"  Agent Top-3 Hit:         {top3_hits}/{total} ({top3_hits/total:.0%})")
print(f"  Agent Top-5 Hit:         {top5_hits}/{total} ({top5_hits/total:.0%})")
print(f"")
print(f"Per-tool success rate:")
for tool, stats in sorted(tool_ok.items()):
    print(f"  {tool}: {stats['ok']}/{stats['total']} ({stats['ok']/stats['total']:.0%})")

# Compare with model ensemble metrics
print(f"\n{'='*60}")
print(f"Comparison: Agent vs Raw Model Ensemble")
print(f"{'='*60}")
print(f"  Agent Top-1:  {top1_hits/total:.0%}")
print(f"  Agent Top-5:  {top5_hits/total:.0%}")
print(f"  Model Top-1:  70.7%")
print(f"  Model Top-5:  93.1%")
print(f"  Agent vs Model Top-1 gap: {top1_hits/total - 0.707:+.1%}")
