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
# 复检3 修复（🟡）：safe_torch_load 的 import 必须在 sys.path.insert 之后
# （旧版位于其前，`python scripts/analyze_ecg.py` 直接 ModuleNotFoundError，
# 仅 -m 模式可用）
from src.utils.safe_load import safe_torch_load
from src.ecg_models.backbone.ecgfounder_net1d import Net1D


def load_model(ckpt_path):
    model = Net1D(
        in_channels=12, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16, stride=2, groups_width=16,
        n_classes=150, use_bn=False, use_do=False, verbose=False,
    )
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    return model


def load_task_names():
    path = Path(__file__).parent.parent / "data/physionet2020/ecgfounder_tasks.txt"
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def analyze(ecg_signal, model, task_names, fs=500, top_k=20, threshold=0.3):
    """返回结构化分析结果."""
    # 第三轮审查 3F-R3-10：threshold 参数真正用于 level 判定
    # （旧版接收后从未使用，level 硬编码 0.7/0.3）
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
            "level": "HIGH" if p >= 0.7 else "MED" if p >= threshold else "LOW",
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
        "summary": _generate_summary(findings, threshold=threshold),
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


def _generate_summary(findings, threshold=0.3):
    # 4A/4J 审查修复：🟡-10 summary 分级使用传入的 threshold 参数作为 MED 下界
    # （旧版硬编码 0.3，--threshold 0.5 时 "level" 与 "summary 的 MEDIUM 段"
    # 口径不一致）。HIGH 上界仍固定 0.7，与 analyze() 的 level 判定一致。
    high = [f for f in findings if f["probability"] >= 0.7]
    medium = [f for f in findings if threshold <= f["probability"] < 0.7]
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
    parser.add_argument("--index", type=int, help="测试集索引（0-3753，患者级划分）")
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
        # 第三轮审查 3F-R3-10：读取现役 processed_5k（患者级划分，test=3754）
        # （旧版读 processed/ 且 help 硬编码 0-6471 与磁盘不符）
        data_dir = Path("data/physionet2020/processed_5k")
        with open(data_dir / "test_manifest.json", encoding="utf-8") as f:
            manifest = json.load(f)
        n_test = len(manifest["files"])
        if not (0 <= args.index < n_test):
            parser.error(f"--index 超出范围（0-{n_test-1}）")
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
            # 检查报告 1.9 修复：旧版用 np.loadtxt 读 WFDB 二进制 → 静默垃圾诊断。
            # 改用 wfdb 正确解析（含 format 16/212 解码），并按 .hea 导联重排。
            # 4A/4J 审查修复：🟠-4 补齐与 web_demo 上传路径一致的完整预处理链
            # （旧版只 reorder+nan_to_num，非 500Hz/非 5000 样本输入时基被错误
            # 拉伸后直接喂给 500Hz 训练的 Net1D，产出失真诊断且无提示）。
            try:
                import wfdb
                # 5J 审查（💡-8）：显式 scripts 包导入（旧版裸 import 仅 -m 模式
                # 可用，且 ImportError 被误报为"需要安装 wfdb"）
                import scripts.preprocess_data_5k as p5
            except ImportError:
                print("需要安装 wfdb（或确认在项目根目录运行）")
                sys.exit(1)
            rec = wfdb.rdrecord(str(ecg_path.with_suffix("")))
            ecg = rec.p_signal.T.astype(np.float32)
            # ECGFounder 协议预处理（原生 fs）：reorder → filter → resample → segment → zscore
            ecg, _ = p5.reorder_leads(ecg, list(rec.sig_name))
            ecg = p5.filter_bandpass(ecg, rec.fs)
            ecg = p5.resample_to_500(ecg, rec.fs)
            ecg, pl, pr = p5.segment_to_5000(ecg)
            ecg = p5.official_zscore(ecg, pl, pr)
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
        # 4A 定向复查（γ）：Details 段分档随 --threshold 贯通（旧版第二档
        # 硬编码 0.3，与 summary 的 MED 下界脱节）；HIGH 档保持 0.7 与 level 一致
        for cat, finds in result["categories"].items():
            print(f"\n[{cat}]")
            for f in finds[:8]:
                marker = ("***" if f["probability"] >= 0.7
                          else "** " if f["probability"] >= args.threshold else " * ")
                print(f"  {marker} {f['name']:<45s}  {f['probability']:.1%}")


if __name__ == "__main__":
    main()
