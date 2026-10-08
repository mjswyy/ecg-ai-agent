#!/usr/bin/env python3
"""M2.3 — Agent 评估集与黄金计划构建。

从 test 划分分层抽样（27 类每类 ≥3 条，总目标 ~150-200 条），为每条生成:
    - golden_tools: 黄金工具计划（由标签规则推导，可解释、可复现）
    - context_cards: 3 种情境卡（急诊胸痛 / 常规体检 / 房颤随访）
    - report_text:   ptb-xl 记录附带医生报告（供"报告辅助"对照分支）

规则（医学常识映射；4G-YELLOW-7 修正：旧 docstring 的 plot_waveform 规则
已废弃——plot_waveform 不在 Agent 工具清单，27 类亦无 ST/心梗/缺血类）:
    - 任何标签 → classify_arrhythmia
    - 节律类（房颤/房扑/窦缓/窦速/窦律不齐/心动过缓/PAC/SVPB/PVC/VPB）→ extract_r_peaks, compute_hrv
    - QT 延长 → measure_qt_interval
    - 传导阻滞（LBBB/RBBB/CRBBB/IRBBB/IAVB/LAFB/NSIVCB）→ extract_r_peaks
    - 形态/轴/间期/起搏类只触发 classify_arrhythmia

用法:
    python scripts/build_agent_evalset.py --n-per-class 5 --max-total 200
"""

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_pipeline.label_extractor import LabelExtractor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RHYTHM_CLASSES = {"Atrial Fibrillation", "Atrial Flutter", "Sinus Bradycardia",
                  "Sinus Tachycardia", "Sinus Arrhythmia", "Bradycardia",
                  "Premature Atrial Contraction", "Supraventricular Premature Beats",
                  "Premature Ventricular Contractions", "Ventricular Premature Beats"}
# 官方 27 评分类体系（R1 修复）
CONDUCTION_CLASSES = {"Left Bundle Branch Block", "Right Bundle Branch Block",
                      "Complete Right Bundle Branch Block",
                      "Incomplete Right Bundle Branch Block",
                      "First Degree AV Block", "Left Anterior Fascicular Block",
                      "Nonspecific Intraventricular Conduction Disorder"}
QT_CLASSES = {"Prolonged QT Interval"}
# 注意: 形态/轴/间期/起搏类在 golden 规则中不触发额外工具
# （plot_waveform 不在 Agent 工具清单）；形态异常只需 classify_arrhythmia
MORPH_CLASSES = {"Q Wave Abnormal", "T Wave Abnormal", "T Wave Inversion",
                 "Low QRS Voltages", "Left Axis Deviation", "Right Axis Deviation",
                 "Prolonged PR Interval", "Pacing Rhythm"}

SCENARIOS = [
    {
        "id": "ed_chest_pain",
        "setting": "急诊科",
        "chief_complaint": "胸痛 3 天，伴心悸",
        "history": "高血压 10 年，吸烟 30 年",
        "urgency": "高",
    },
    {
        "id": "routine_checkup",
        "setting": "体检中心",
        "chief_complaint": "无症状，常规体检",
        "history": "无特殊病史",
        "urgency": "低",
    },
    {
        "id": "af_followup",
        "setting": "心内科门诊",
        "chief_complaint": "房颤术后随访，偶感心慌",
        "history": "房颤史 5 年，口服抗凝药",
        "urgency": "中",
    },
]


def golden_tools(dx_names):
    """黄金工具计划（最小必需集，与 Agent 工具清单严格对齐）。"""
    tools = ["classify_arrhythmia"]
    if dx_names & RHYTHM_CLASSES:
        tools += ["extract_r_peaks", "compute_hrv"]
    if dx_names & QT_CLASSES:
        tools.append("measure_qt_interval")
    if dx_names & CONDUCTION_CLASSES:
        if "extract_r_peaks" not in tools:
            tools.append("extract_r_peaks")
    # 去重保序
    return list(dict.fromkeys(tools))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/physionet2020/processed_5k")
    parser.add_argument("--output", default="outputs/agent_evalset/evalset.json")
    parser.add_argument("--n-per-class", type=int, default=5)
    parser.add_argument("--max-total", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    le = LabelExtractor(num_classes=27)
    data_dir = Path(args.data_dir)

    with open(data_dir / "test_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)["files"]

    # 按类建立索引
    by_class = {i: [] for i in range(27)}
    for m in manifest:
        labels = m.get("labels", [])
        if len(labels) != 27:
            continue
        for i, v in enumerate(labels):
            if v > 0:
                by_class[i].append(m)
    by_class = {i: v for i, v in by_class.items() if v}

    # 分层抽样
    picked = {}
    for i, items in by_class.items():
        for m in random.sample(items, min(args.n_per_class, len(items))):
            picked[m["record_id"]] = m
        logger.info(f"类 {le.class_names[i]:35s} 样本 {len(items):4d} → 抽 {min(args.n_per_class, len(items))}")

    # 若超总量，裁剪到 max_total（保持类覆盖）——4G-YELLOW-6 修复：
    # 旧版全量 random.shuffle 会破坏"27 类全覆盖"；4E 定向复查（ε）指出
    # "唯一代表"方案只保护 count==1 的类——现改 round-robin：先按类轮流
    # 各保 1 条（类覆盖兜底），再从剩余记录随机补足额度
    picked_list = list(picked.values())
    if len(picked_list) > args.max_total:
        # 每条记录对哪些类有正标签（多标签记录可同时兜底多个类）
        def _pos_classes(m):
            labels = m.get("labels", [])
            return {i for i, v in enumerate(labels) if v > 0} \
                if len(labels) == 27 else set()

        by_class = {i: [m for m in picked_list if i in _pos_classes(m)]
                    for i in range(27)}
        keep, kept_ids = [], set()
        # round-robin：每轮为每个仍有额度且仍有候选的类保 1 条
        progressed = True
        while progressed and len(keep) < args.max_total:
            progressed = False
            for i in range(27):
                if len(keep) >= args.max_total:
                    break
                cands = [m for m in by_class.get(i, [])
                         if m["record_id"] not in kept_ids]
                if not cands:
                    continue
                m = cands[0]
                keep.append(m)
                kept_ids.add(m["record_id"])
                progressed = True
        rest = [m for m in picked_list if m["record_id"] not in kept_ids]
        random.shuffle(rest)
        keep += rest[:args.max_total - len(keep)]
        picked_list = keep
    logger.info(f"评估集: {len(picked_list)} 条 (目标 ≤{args.max_total})")

    # 报告文本（仅 ptb-xl）
    reports = None
    rp = Path("data/physionet2020/processed_5k/ptbxl_reports.json")
    if rp.exists():
        reports = json.load(open(rp, encoding="utf-8"))

    evalset = []
    for m in picked_list:
        dx_names = set()
        for code in m.get("dx_codes", []):
            n = le.get_class_name(code)
            if n:
                dx_names.add(n)
        item = {
            "record_id": m["record_id"],
            "source": m["source"],
            "signal_file": m["signal_file"],
            "dx_codes": m.get("dx_codes", []),
            "dx_names": sorted(dx_names),
            "labels": m.get("labels", []),
            "age": m.get("age"),
            "sex": m.get("sex"),
            "golden_tools": golden_tools(dx_names),
            "context_cards": SCENARIOS,
            "report_text": (reports or {}).get(m["record_id"], {}).get("report")
            if m["source"] == "ptb-xl" else None,
        }
        evalset.append(item)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"items": evalset, "count": len(evalset)}, f, indent=1, ensure_ascii=False)

    # 统计
    gt_counter = Counter()
    for it in evalset:
        for t in it["golden_tools"]:
            gt_counter[t] += 1
    logger.info(f"黄金工具分布: {dict(gt_counter)}")
    n_with_report = sum(1 for it in evalset if it["report_text"])
    logger.info(f"含医生报告(报告辅助分支可用): {n_with_report} 条")
    logger.info(f"保存 → {out}")


if __name__ == "__main__":
    main()
