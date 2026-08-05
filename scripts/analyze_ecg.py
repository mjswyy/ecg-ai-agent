"""
ECG 综合分析工具 — 基于 ECGFounder 150 类预训练模型。

直接使用 ECGFounder 的 150 类全标签体系（PTB-XL 标准），
比我们训练的 27 类模型覆盖更全面的心律失常诊断。

用法:
    python scripts/analyze_ecg.py --ecg data/physionet2020/processed/xxx.npy
    python scripts/analyze_ecg.py --ecg xxx.hea
    python scripts/analyze_ecg.py --index 42  # 测试集第42条
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import numpy as np
import torch
from src.ecg_models.backbone.ecgfounder_net1d import Net1D


def load_model(ckpt_path):
    model = Net1D(
        in_channels=12, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16, stride=2, groups_width=16,
        n_classes=150, use_bn=False, use_do=False, verbose=False,
    )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def load_task_names():
    path = Path(__file__).parent.parent / "data/physionet2020/ecgfounder_tasks.txt"
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def analyze(ecg_signal, model, task_names, fs=500, top_k=20, threshold=0.3):
    """返回结构化分析结果."""
    # Pad 4096 → 5000
    x = torch.from_numpy(ecg_signal).float().unsqueeze(0)
    if x.shape[-1] < 5000:
        pad = 5000 - x.shape[-1]
        x = torch.nn.functional.pad(x, (pad // 2, pad - pad // 2))
    elif x.shape[-1] > 5000:
        x = x[..., :5000]

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).squeeze(0).numpy()

    # 排序取 top-K
    indices = np.argsort(-probs)

    findings = []
    for rank, idx in enumerate(indices[:top_k]):
        p = float(probs[idx])
        name = task_names[idx] if idx < len(task_names) else f"Task {idx}"
        cat = _classify_category(name)
        findings.append({
            "rank": rank + 1,
            "name": name,
            "probability": p,
            "category": cat,
            "level": "HIGH" if p >= 0.7 else "MED" if p >= 0.3 else "LOW",
        })

    # 按类别分组
    categories = {}
    for f in findings:
        cat = f["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(f)

    return {
        "findings": findings,
        "categories": categories,
        "summary": _generate_summary(findings),
    }


def _classify_category(name):
    n = name.lower()
    if any(w in n for w in ["sinus", "bradycardia", "tachycardia", "atrial", "ventricular",
                             "fibrillation", "flutter", "rhythm", "pace", "pvc", "pac",
                             "tach", "ectopic", "arrest"]):
        return "Rhythm"
    if any(w in n for w in ["block", "bundle branch", "rbbb", "lbbb", "av block",
                             "conduction", "fascicular", "wpw", "preexcitation"]):
        return "Conduction"
    if any(w in n for w in ["infarct", "ischemia", "st ", "t wave", "q wave", "elevation",
                             "depression", "inversion", "hypertrophy", "strain",
                             "repolarization", "qt", "voltage"]):
        return "Morphology"
    if any(w in n for w in ["normal", "abnormal", "borderline"]):
        return "General"
    return "Other"


def _generate_summary(findings):
    high = [f for f in findings if f["probability"] >= 0.7]
    medium = [f for f in findings if 0.3 <= f["probability"] < 0.7]
    lines = []
    if high:
        lines.append(f"[HIGH] {len(high)} findings:")
        for f in high:
            lines.append(f"  - {f['name']} ({f['probability']:.0%})")
    if medium:
        lines.append(f"\n[MEDIUM] {len(medium)} findings:")
        for f in medium:
            lines.append(f"  - {f['name']} ({f['probability']:.0%})")
    if not high and not medium:
        lines.append("No significant abnormalities detected.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="ECG 综合分析 (ECGFounder 150-class)")
    parser.add_argument("--ecg", type=str, help="ECG .npy 或 .hea 文件路径")
    parser.add_argument("--index", type=int, help="测试集索引（0-6471）")
    parser.add_argument("--ckpt", default="checkpoints/ECGFounder/12_lead_ECGFounder.pth")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式")
    args = parser.parse_args()

    print("Loading ECGFounder...")
    model = load_model(args.ckpt)
    task_names = load_task_names()
    print(f"Model loaded. 150 tasks, ~15M params\n")

    # Get ECG signal
    if args.index is not None:
        data_dir = Path("data/physionet2020/processed")
        with open(data_dir / "test_manifest.json") as f:
            manifest = json.load(f)
        entry = manifest["files"][args.index]
        ecg = np.nan_to_num(np.load(data_dir / entry["signal_file"]).astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
        print(f"Test ECG #{args.index}: {entry['record_id']} ({entry['source']})")
        if entry.get("dx_codes"):
            from src.data_pipeline.label_extractor import LabelExtractor
            le = LabelExtractor()
            names = [le.get_class_name(c) or c for c in entry['dx_codes'][:5]]
            print(f"Ground truth: {', '.join(names)}")
    elif args.ecg:
        ecg_path = Path(args.ecg)
        if ecg_path.suffix == ".npy":
            ecg = np.nan_to_num(np.load(ecg_path).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        elif ecg_path.suffix == ".hea":
            ecg = np.nan_to_num(
                np.loadtxt(ecg_path.parent / ecg_path.stem, skiprows=1).astype(np.float32),
                nan=0.0, posinf=0.0, neginf=0.0)
        else:
            print(f"Unsupported format: {ecg_path.suffix}")
            sys.exit(1)
        print(f"ECG: {ecg_path.name} | shape: {ecg.shape}")
    else:
        print("Specify --ecg or --index")
        sys.exit(1)

    # Analyze
    result = analyze(ecg, model, task_names, top_k=args.top_k, threshold=args.threshold)

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"\n{'='*60}")
        print("ECGFounder 150-class Analysis Report")
        print(f"{'='*60}")
        print(result["summary"])
        print(f"\n--- Details ---")
        for cat, finds in result["categories"].items():
            print(f"\n[{cat}]")
            for f in finds[:8]:
                marker = "***" if f["probability"] >= 0.7 else "** " if f["probability"] >= 0.3 else " * "
                print(f"  {marker} {f['name']:<45s}  {f['probability']:.1%}")


if __name__ == "__main__":
    main()
