#!/usr/bin/env python3
"""2E-O6b 修复：重算三臂记录的幻觉声明（排除规范性范围表述误报）。

已生成 records 的 hallucinated_claims 含"心率 60–100 bpm"式误报；
本脚本用确定性工具重算工具值 + 修正口径重算声明，更新 records 并重算 metrics。
不调用 LLM API（省时）。
"""
import json
import logging
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ecg_models.feature_extraction.r_peak_detector import RPeakDetector
from src.ecg_models.feature_extraction.hrv_analyzer import HRVAnalyzer
from src.ecg_models.feature_extraction.qt_analyzer import QTAnalyzer

# 4E 定向复查（ε）：4G-YELLOW-10 修复引用了 logger 但脚本无 logger 定义
# （旧记录缺 record_id 时触发 NameError 崩溃，跳过兜底完全失效）
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUT = Path(r"outputs/agent_eval")
evalset = json.load(open(r"outputs/agent_evalset/evalset.json", encoding="utf-8"))["items"]
data_dir = Path(r"data/physionet2020/processed_5k")

rpk = RPeakDetector(method="pan_tompkins")
hrv = HRVAnalyzer()
qt = QTAnalyzer()

CLAIM_PATTERNS = [
    # 三臂重跑复查（2026-08-30）：\s* → [ \t]* 防跨行误匹配（"心率\n10." 编号）
    (r"心率[为是]?[ \t]*[:：]?[ \t]*(\d{2,3})", "hr", "心率"),
    (r"(?:HR|hr)[ \t]*[:：=]?[ \t]*(\d{2,3})", "hr", "心率"),
    (r"QTc[为是]?[ \t]*[:：]?[ \t]*(\d{2,3})", "qtc", "QTc"),
    (r"QT间期[为是]?[ \t]*[:：]?[ \t]*(\d{2,3})", "qt_ms", "QT"),
    (r"SDNN[为是]?[ \t]*[:：]?[ \t]*(\d{2,3})", "sdnn", "SDNN"),
    # 4G-ORANGE-4 修复：与 eval_agent_llm.eval_one 的 8 类 claim_patterns 对齐
    # （旧版只 5 类，重算会静默抹掉 PR/QRS/电轴幻觉，口径与在线评测脱节；
    #  工具无这些测量 → key=None 走"工具未提供测量值"分支）
    (r"PR间期[为是]?[ \t]*[:：]?[ \t]*(\d{2,3})", None, "PR间期"),
    # 5G 审查（🟠-1）同步：QRS 修饰词覆盖"时限/宽度/宽"整词、电轴覆盖
    # "左偏/右偏"插入语（与 eval_agent_llm.eval_one 的 claim_patterns 一致）
    (r"QRS(?:波群)?(?:时限|宽度|宽|间期)?[为是]?[ \t]*[:：]?[ \t]*[-+]?[ \t]*(\d{2,3})",
     None, "QRS时限"),
    (r"电轴(?:左偏|右偏|正常)?[为是]?[ \t]*[:：]?[ \t]*[-+]?[ \t]*(\d{1,3})",
     None, "电轴"),
]

def scan_claims(report, tv):
    hallucinations = []
    for pattern, key, label in CLAIM_PATTERNS:
        for m in re.finditer(pattern, report):
            claimed = int(m.group(1))
            tail = report[m.end():m.end() + 12]
            if re.match(r"^\s*(?:[-–—~～至到/]\s*\d|%)", tail):
                continue  # 规范性范围/百分比表述
            head = report[max(0, m.start() - 10):m.start()]
            # 三臂重跑复查（2026-08-30）：补疾病一般知识表述修饰词
            # （"房扑常表现为心率150 bpm左右" 是鉴别知识而非患者声称）
            if re.search(r"平均|均值|中位|正常|参考|阈值|标准|典型|常表现|通常|一般|左右|约|常见|mean|avg|normal|threshold", head):
                continue  # 知识库/参考统计值表述
            tool_val = tv.get(key)
            if tool_val is None:
                hallucinations.append(f"{label}声称{claimed}(工具未提供测量值)")
                continue
            if abs(claimed - round(tool_val)) > 5:
                hallucinations.append(f"{label}声称{claimed}(实际{tool_val})")
    return hallucinations

def tool_values(item):
    sig = np.load(data_dir / item["signal_file"])
    tv = {"hr": None, "qtc": None, "qt_ms": None, "sdnn": None}
    try:
        r = rpk.detect(sig[1], fs=500)
        tv["hr"] = r.get("heart_rate")
        rr = r.get("rr_intervals", np.array([0.8]))
        try:
            h = hrv.analyze(rr)
            tv["sdnn"] = h.get("sdnn")
        except Exception:
            pass
        try:
            q = qt.analyze(sig[1], r.get("r_peaks", []), fs=500)
            tv["qt_ms"] = q.get("qt_ms")
            tv["qtc"] = q.get("qtc_bazett")
        except Exception:
            pass
    except Exception:
        pass
    return tv

for arm, fn in [("main", "records_main.jsonl"),
                ("report_arm", "records_report_arm.jsonl"),
                ("kb_arm", "records_kb_arm.jsonl")]:
    p = OUT / fn
    if not p.exists():
        print(f"跳过 {arm}: 缺 {fn}")
        continue
    lines = p.read_text(encoding="utf-8").splitlines()
    records = [json.loads(l) for l in lines if l.strip()]
    # 4G-YELLOW-10 修复：按 record_id 对齐评估集（旧版按 evalset[i] 位置索引，
    # 顺序变化即用错信号重判幻觉）
    ev_by_id = {it.get("record_id"): it for it in evalset}
    n_fixed = 0
    for r in records:
        if "report" not in r:
            continue
        item = ev_by_id.get(r.get("record_id"))
        if item is None:
            logger.warning(f"记录 {r.get('record_id', '?')} 不在评估集，跳过重算")
            continue
        tv = tool_values(item)
        new_claims = scan_claims(r.get("report", ""), tv)
        if new_claims != r.get("hallucinated_claims"):
            n_fixed += 1
        r["hallucinated_claims"] = new_claims
        r["tool_values"] = tv
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{arm}: {len(records)} 条，声明修正 {n_fixed} 条")

# 重算 metrics（与 eval_agent_llm.aggregate_metrics 同口径）
from eval_agent_llm import aggregate_metrics  # noqa: E402
for arm in ("main", "report_arm", "kb_arm"):
    rp = OUT / f"records_{arm}.jsonl"
    if not rp.exists():
        continue
    records = [json.loads(l) for l in rp.read_text(encoding="utf-8").splitlines()
               if l.strip()]
    ok = [r for r in records if "planned_tools" in r]
    old = json.load(open(OUT / f"metrics_{arm}.json", encoding="utf-8"))
    m = aggregate_metrics(ok, judge_scores=old.get("judge"),
                          use_report=(arm == "report_arm"),
                          use_kb=(arm == "kb_arm"))
    json.dump(m, open(OUT / f"metrics_{arm}.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    h = m["hallucination"]
    print(f"{arm}: 幻觉 {h['n_hallucinated']}/{h['n_claims']} = {h['rate']}")
print("完成")
