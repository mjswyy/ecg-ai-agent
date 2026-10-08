#!/usr/bin/env python3
"""论文四图生成：成绩单曲线 / 五维雷达 / 情境矩阵 / 知识库流水线。

输出: paper/figures/{scoreboard.png, radar.png, context_matrix.png, kb_pipeline.png}
"""

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# 中文字体（与项目 demo 一致）
rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

FIG_DIR = Path(__file__).parent.parent / "paper" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

C_BLUE = "#2E5E9E"
C_ORANGE = "#E07B39"
C_GREEN = "#3F8F5B"
C_GRAY = "#8a8f98"
C_RED = "#C0504D"


def fig_scoreboard():
    # ECGFounder+MLP 分数从产物读取（检查报告 1.6 修复后数字会变，不再硬编码）
    with open("outputs/ecgfounder_mlp/metrics.json", encoding="utf-8") as f:
        mlp = json.load(f)["metrics"]["macro_auc"]
    # 第三轮审查 3G 🟡-5 / 3D 🟡-3：历史行（无磁盘产物）统一加"旧标签体系/历史"标注，
    # 不再与现役官方评分类柱并列为"同口径"
    models = [
        ("InceptionTime", 0.590, C_GRAY, "旧标签体系·历史"),
        # 4H-ORANGE-3 修复：0.791/0.808/0.822 来自 comprehensive_eval.json
        # （旧标签体系时代产物），标注"从零训练"会与患者级 MLP 柱误读为同口径
        ("xResNet1D-101", 0.791, C_GRAY, "旧标签体系·历史"),
        ("ECG Transformer", 0.808, C_GRAY, "旧标签体系·历史"),
        ("2模型集成", 0.822, C_GRAY, "旧标签体系·历史"),
        ("SimCLR xResNet", 0.841, C_GRAY, "旧标签体系·历史"),
        ("3模型集成", 0.850, C_GRAY, "旧标签体系·历史"),
        ("ECGFounder 线性探针", 0.905, C_ORANGE, "旧标签体系·历史"),
        ("ECGFounder + MLP", mlp, C_RED, "冻结特征（本文发现层）"),
    ]
    names = [m[0] for m in models]
    aucs = [m[1] for m in models]
    colors = [m[2] for m in models]
    labels = [m[3] for m in models]

    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150)
    bars = ax.barh(names[::-1], aucs[::-1], color=colors[::-1], height=0.62)
    ax.axvline(0.5, color="black", ls="--", lw=0.8, alpha=0.6)
    ax.text(0.505, -0.55, "随机基线 0.50", fontsize=8, color="black", alpha=0.7)
    for i, (b, a, lab) in enumerate(zip(bars, aucs[::-1], labels[::-1])):
        ax.text(a + 0.004, b.get_y() + b.get_height() / 2,
                f"{a:.3f}", va="center", fontsize=9)
        if lab != "从零训练":
            ax.text(a - 0.02, b.get_y() + b.get_height() - 0.14,
                    lab, va="center", ha="right", fontsize=7, color="white")
    ax.set_xlim(0.45, 1.0)
    # 第三轮审查 3D 🟡-1：x 轴按柱区分测试集口径（现役 MLP 柱在患者级 n=3,754 上）
    ax.set_xlabel("test macro AUC（现役柱: 患者级划分 n=3,754；历史柱: 旧标签体系/记录级）",
                  fontsize=10)
    ax.set_title("ECG 发现层成绩单：从零训练 vs 基础模型冻结特征", fontsize=12, pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "scoreboard.png", bbox_inches="tight")
    plt.close()
    print("[OK] scoreboard.png")


def fig_radar():
    # 从磁盘产物读取（检查报告 1.3 修复：不再硬编码）
    with open("outputs/agent_eval/judge_rubric_results.json", encoding="utf-8") as f:
        rubric_res = json.load(f)
    with open("outputs/agent_eval/metrics_main.json", encoding="utf-8") as f:
        metrics = json.load(f)

    arm_labels = {
        "main": "主分支(信号)",
        "report_arm": "报告辅助(+医生报告)",
        "kb_arm": "知识库(+KB)",
    }
    rubric = {}
    for arm, label in arm_labels.items():
        if arm in rubric_res:
            a = rubric_res[arm]
            rubric[label] = [a["correctness"]["mean"], a["completeness"]["mean"],
                             a["grounding"]["mean"], a["total"]["mean"]]
    # 第三轮审查 3D 🟠-1：三臂缺臂时显式报错（旧版静默把"报告质量"能力轴伪造为 0.5）
    missing = [arm for arm in arm_labels if arm not in rubric_res]
    if missing:
        raise SystemExit(
            f"fig_radar: judge_rubric_results.json 缺少臂 {missing}——"
            f"请先重跑 rejudge_with_rubric.py 生成完整三臂判官结果，禁止用默认值填充。")
    rubric_axes = ["正确性", "完整性", "依据性", "总分"]

    tool = metrics["tool_selection"]
    p, r = tool["precision"], tool["recall"]
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    inj = metrics["injection"]
    catch = inj.get("catch_rate_report") or 0.0
    ctrl = inj.get("conflict_flag_rate_control")
    # 2E-Y3 修复：幻觉率 None（无数值声明）时画 0（N/A 口径），不再画成"完美抗幻觉 1.0"
    hallu_rate = metrics["hallucination"].get("rate")
    hallu = 1.0 - hallu_rate if hallu_rate is not None else 0.0
    top5 = metrics["top5_accuracy"]
    rep_q = (rubric_res.get("main", {}).get("total", {}).get("mean", 2.5)) / 5.0

    ability = {
        "Agent 能力": [f1, catch, hallu, top5, rep_q],
    }
    ability_axes = ["工具选择F1", "注入纠错捕获", "抗幻觉率", "发现层Top5", "报告质量(总分/5)"]

    n_main = rubric_res.get("main", {}).get("n", "?")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150,
                                   subplot_kw=dict(polar=True))

    def draw_radar(ax, axes, data, colors, maxv=5.0):
        angles = np.linspace(0, 2 * np.pi, len(axes), endpoint=False).tolist()
        angles += angles[:1]
        for (name, vals), c in zip(data.items(), colors):
            v = vals + vals[:1]
            ax.plot(angles, v, "o-", lw=1.8, color=c, label=name, markersize=4)
            ax.fill(angles, v, color=c, alpha=0.08)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(axes, fontsize=9)
        ax.set_ylim(0, maxv)
        ax.set_yticks([1, 2, 3, 4, 5] if maxv == 5 else [0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels([str(t) for t in
                            ([1, 2, 3, 4, 5] if maxv == 5 else [0.2, 0.4, 0.6, 0.8, 1.0])],
                           fontsize=7)
        ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8)

    draw_radar(ax1, rubric_axes, rubric, [C_BLUE, C_ORANGE, C_GREEN], maxv=5.0)
    ax1.set_title(f"三臂报告质量（rubric 三维, n={n_main}）", fontsize=11, pad=18)

    draw_radar(ax2, ability_axes, ability, [C_RED], maxv=1.0)
    ax2.set_title("Agent 能力五维（主分支）", fontsize=11, pad=18)

    ctrl_txt = (f"注：注入纠错为报告显式冲突标记口径，未注入对照组标记率 {ctrl*100:.1f}%"
                if ctrl is not None else "注：注入纠错为报告显式冲突标记口径")
    fig.text(0.5, 0.02, ctrl_txt, ha="center", fontsize=8, color=C_GRAY)

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(FIG_DIR / "radar.png", bbox_inches="tight")
    plt.close()
    print("[OK] radar.png")


def fig_context_matrix():
    with open("outputs/context_eval/records.jsonl", encoding="utf-8") as f:
        recs = [json.loads(l) for l in f if l.strip()]

    ctx_ids = ["ed_chest_pain", "routine_checkup", "af_followup"]
    ctx_labels = ["急诊胸痛", "常规体检", "房颤随访"]
    urg_map = {"routine": 0, "urgent": 1, "emergent": 2, None: -1}
    risk_map = {"low": 0, "moderate": 1, "high": 2, None: -1}

    # 按 GT 首要标签排序
    def primary(r):
        return sorted(r["gt_dx_names"])[0] if r["gt_dx_names"] else "?"

    recs = sorted(recs, key=primary)

    U = np.zeros((len(recs), 3))
    R = np.zeros((len(recs), 3))
    for i, r in enumerate(recs):
        for j, cid in enumerate(ctx_ids):
            c = r["contexts"].get(cid)
            if isinstance(c, dict):
                U[i, j] = urg_map.get(
                    str(c.get("recommendations", {}).get("urgency", "")).lower(), -1)
                R[i, j] = risk_map.get(
                    str(c.get("risk_assessment", {}).get("level", "")).lower(), -1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8), dpi=150)

    for ax, M, title, labels, cmap in [
        (ax1, U, "紧急度（routine/urgent/emergent）", ["routine", "urgent", "emergent"], "YlOrRd"),
        (ax2, R, "风险分层（low/moderate/high）", ["low", "moderate", "high"], "YlGnBu"),
    ]:
        im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=-0.5, vmax=2.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels(ctx_labels, fontsize=10)
        ax.set_yticks(range(len(recs)))
        ax.set_yticklabels([primary(r) for r in recs], fontsize=6.5)
        ax.set_title(title, fontsize=11)
        cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
        cbar.ax.set_yticklabels(labels, fontsize=8)

    # 第三轮审查 3D 🟡-2：n 动态取值（旧版硬编码 n=36，磁盘 records 实际 40 条）
    fig.suptitle(f"情境敏感性矩阵：同一 ECG × 3 种临床情境（n={len(recs)}）",
                 fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "context_matrix.png", bbox_inches="tight")
    plt.close()
    print("[OK] context_matrix.png")


def fig_kb_pipeline():
    fig, ax = plt.subplots(figsize=(11, 3.4), dpi=150)
    ax.axis("off")

    stages = [
        # 第三轮审查 3G 🟡-4："350 篇文档"无法溯源 → 改为实测口径
        # （sources 条目 276：pubmed 241 / litfl 27 / ecgpedia 8）
        ("① 采集\nCrawling", "PubMed E-utilities\nECGpedia + LITFL\n276 条来源引用", C_BLUE),
        ("② 分级\nGrading", "A 权威指南\nB 同行评审\nC 教育资源", C_ORANGE),
        ("③ 提炼\nDistillation", "LLM JSON schema\n定义/特征/鉴别/处理\n防幻觉约束+重试", C_GREEN),
        ("④ 编译\nCompiling", "master + 27 类\n每条结论带\n[来源,等级] 引用", C_RED),
        ("⑤ 检索锚定\nRetrieval", "总论 + 阳性详情 +\n全 27 类目录注入\n（--use-kb 分支）", "#7B5EA7"),
    ]

    n = len(stages)
    w, h = 0.165, 0.52
    for i, (title, detail, color) in enumerate(stages):
        x = 0.02 + i * (w + 0.035)
        box = FancyBboxPatch((x, 0.28), w, h, boxstyle="round,pad=0.012",
                             linewidth=1.4, edgecolor=color, facecolor=color,
                             alpha=0.92)
        ax.add_patch(box)
        ax.text(x + w / 2, 0.30 + h + 0.03, title, ha="center", va="bottom",
                fontsize=11, fontweight="bold", color=color)
        ax.text(x + w / 2, 0.30 + h / 2, detail, ha="center", va="center",
                fontsize=8.3, color="white")
        if i < n - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.004, 0.30 + h / 2),
                                         (x + w + 0.032, 0.30 + h / 2),
                                         arrowstyle="-|>", mutation_scale=16,
                                         color=C_GRAY, lw=1.8))

    # 5H 审查（🟡-6）：KB 规模从 classes.json 实时读取（旧版硬编码
    # "每类 6-13 来源、12-49 条特征"与编译产物不符——实测 1-13 来源、2-48 特征）
    try:
        with open("data/knowledge/kb/classes.json", encoding="utf-8") as _f:
            _kb = json.load(_f)["classes"]
        _ns = [v.get("n_sources", 0) for v in _kb.values()]
        _nc = [len(v.get("ecg_criteria", [])) for v in _kb.values()]
        _scale = f"每类 {min(_ns)}-{max(_ns)} 来源、{min(_nc)}-{max(_nc)} 条 ECG 诊断特征"
    except Exception:
        _scale = "27 类全覆盖（来源数见 classes.json）"
    ax.text(0.02, 0.16, f"规模：27 类全覆盖（{_scale}）· "
            "检索工具 query_medical_knowledge · 全部结论可追溯原始 URL",
            fontsize=9, color=C_GRAY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "kb_pipeline.png", bbox_inches="tight")
    plt.close()
    print("[OK] kb_pipeline.png")


if __name__ == "__main__":
    fig_scoreboard()
    fig_radar()
    fig_context_matrix()
    fig_kb_pipeline()
    print("全部完成 →", FIG_DIR)

