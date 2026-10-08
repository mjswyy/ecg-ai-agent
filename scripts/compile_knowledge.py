#!/usr/bin/env python3
"""知识库编译器（Phase K3）— 把提炼条目聚合为可供 Agent 读取的医学参考资料。

输入: data/knowledge/distilled/{class_slug}.jsonl
输出:
    data/knowledge/kb/classes.json          # 27 类 + 总论的结构化知识（Agent 工具读取）
    data/knowledge/kb/{class_slug}.md       # 每类单独的人类可读文档
    data/knowledge/kb/master.md             # 总论 + 读图顺序 + 免责声明

聚合策略:
    - definition: 优先 grade B（同行评审）来源中最长的一条
    - ecg_criteria / differential / management: 去重合并（按规范化文本），每条附 [来源, 等级]
    - measurements / clinical_significance: 合并去重
    - 保留全部引用清单（source/grade/url），供追溯

用法:
    python scripts/compile_knowledge.py
"""

import argparse
import json
import logging
import re
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data_pipeline.label_extractor import LabelExtractor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DIST_DIR = BASE_DIR / "data" / "knowledge" / "distilled"
KB_DIR = BASE_DIR / "data" / "knowledge" / "kb"

GRADE_ORDER = {"A": 0, "B": 1, "C": 2, "D": 3}

# 定义/条目选取的来源对口度（越小越优先）：教育页按类名逐页抓取、
# 主题对口；PubMed 摘要是按查询词检索、相关性不可控（2F 🔴-1 修复）。
SOURCE_RANK = {"litfl": 0, "ecgpedia": 1}

# 2026-08-28：general（总论）条目相关性把关词表（2F 🔴 延伸修复）。
# 实测 general 混入导管放置/十二指肠、TDI 超声、胎儿 ECG 等无关内容。
GENERAL_POS = re.compile(
    r"ECG|electrocardi|QRS|P\s?wave|T\s?wave|ST\s?segment|QT\s?interval|PR\s?interval|"
    r"heart rate|rhythm|sinus|axis|conduction|atrial|ventricular|repolarization|"
    r"ischemia|infarction|brady|tachy|lead", re.IGNORECASE)
GENERAL_NEG = re.compile(
    r"duodenum|nasoenteral|feeding tube|fetal|TDI|ultrasound|catheter|"
    r"nasogastric|gastric|oesophageal|esophageal", re.IGNORECASE)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" -–—")


# R4 修复（第二次审查 2F 🟠-2）：占位符值统一过滤，不再进入知识库。
# 第三轮审查 3E-YELLOW-9：补充 not applicable / not provided / no specific 等常见占位。
PLACEHOLDERS = {
    "not stated", "n/a", "na", "not available", "not mentioned",
    "not specified", "unknown", "none", "nil",
    "not applicable", "not provided", "no specific", "not given",
}

# 4I-ORANGE-3 修复：纯短语式占位（整条文本仅由占位措辞构成，如
# "Not explicitly stated in the source."）——精确集合匹配覆盖不了。
# 只匹配"纯占位"（不允许实质性内容续接），避免误杀
# "not explicitly stated, but IRBBB is typically defined by..." 这类实质条目。
_PURE_PLACEHOLDER_PHRASE = re.compile(
    r"^not\s+(explicitly\s+)?(stated|specified|provided|mentioned|discussed|"
    r"described|reported|given|applicable|available)"
    r"(\s+in\s+(the\s+)?(source|provided text|text|abstract|article))?"
    r"[.\s]*$")


def is_placeholder(s) -> bool:
    ns = norm(s)
    if ns in PLACEHOLDERS:
        return True
    return bool(_PURE_PLACEHOLDER_PHRASE.search(ns))


def compile_class(class_slug, entries):
    if not entries:
        return None

    # 按等级排序：grade 高优先
    entries_sorted = sorted(entries, key=lambda e: GRADE_ORDER.get(e["grade"], 3))

    # R4 修复（2F 🔴-1 建议 b/c）：类名关键词与 condition/文本相关性检查
    # 第三轮审查 3E-RED-1：停用词扩展——通用 ECG 词（rhythm/block/wave/interval/
    # lead/rate 等）不具区分度，'rhythm' 是 'arrhythmia' 的子串导致几乎一切
    # 心律失常摘要都通过把关（实测 Pacing Rhythm 类混入 Brugada/LQTS/ACM）。
    class_norm = norm(class_slug.replace("_", " "))
    CLASS_STOPWORDS = {
        "sinus", "premature", "complex", "conduction", "disorder",
        "abnormality", "abnormal", "cardiac", "rhythm", "block", "wave",
        "interval", "lead", "rate", "segment", "prolonged", "incomplete",
        "complete", "first", "degree", "left", "right",
        "nonspecific", "ventricular", "atrial", "supraventricular",
        "arrhythmia", "low",
        # 注：bundle/branch 刻意保留（组合出现即特异，防止 CRBBB 等
        # 全停用词类目回退整名后被过度过滤——第三轮审查 3E-RED-1 回归修复）
    }
    class_keywords = [w for w in class_norm.split()
                      if w not in CLASS_STOPWORDS] or [class_norm]
    # 4I-RED-1 修复：方向词（left/right）是 LBBB/RBBB/CRBBB 与 LAD/RAD 的
    # 核心区分特征——旧版把 left/right 列入停用词，两侧类目关键词退化为相同
    # 集合（['bundle','branch'] / ['axis','deviation']），PubMed 相关性把关
    # 失去方向区分（实测 RBBB 判据 "rsR' pattern" 混入 LBBB 类、LAD 判据
    # 混入 RAD 类）。现把方向词作为必须词：类名含方向时必须命中，且仍需命中
    # 其余任一关键词（防 "left ventricular" 类通用表述误通过）。
    _direction = next((d for d in ("left", "right")
                       if d in class_norm.split()), None)
    # 4I-RED-1 辅助：方向敏感类的方向域词汇（束支/电轴全家族）与对侧方向，
    # 用于 condition 把关与 ecg_criteria 条目级过滤（对侧判据不得注入 LLM 上下文）
    _DIR_FAMILY = ("bundle", "branch", "fascicular", "axis", "deviation")
    _OPP_DIR = ("right" if _direction == "left" else "left") \
        if _direction is not None else None
    # 判别词同义词（序列匹配时按组扩展）——LAFB 论文常用 "left anterior
    # hemiblock (LAH)"，仅认 fascicular 会把有效条目全滤掉
    _KW_SYNONYMS = {"fascicular": ("fascicular", "hemiblock")}

    def _dir_family_hit(t, d):
        # 分隔符容忍连字符（"bundle-branch"/"left-sided" 等变体）
        return any(re.search(rf"\b{d}\b(?:[\s\-]{{1,2}}\w+){{0,2}}[\s\-]{{1,2}}{re.escape(kw)}\b", t)
                   for kw in _DIR_FAMILY)

    def _dir_consistent(t):
        """方向一致性：文本含本方向+家族词，或不含对侧方向+家族词。"""
        if _direction is None:
            return True
        tl = norm(t)  # 4I-RED-1：条目文本可能是原文大小写，必须归一化
        own = _dir_family_hit(tl, _direction)
        opp = _dir_family_hit(tl, _OPP_DIR)
        return own or not opp

    def _condition_matches(cond: str) -> bool:
        cn = norm(cond)
        if not cn or is_placeholder(cond):
            return False
        if cn == class_norm:
            return True
        return cn in class_norm or class_norm in cn

    def _text_matches(text: str) -> bool:
        tl = norm(text)
        # 第三轮审查 3E-RED-1：词边界匹配（旧版裸子串 any(w in tl)，
        # 'rhythm' 命中 'arrhythmia'/'arrhythmogenic' → 无关条目漏网）
        if _direction is not None:
            # 4I-RED-1：方向类要求方向词 + 全部判别词按序相邻（词间 ≤2 词，
            # 覆盖 "left anterior fascicular"/"left bundle-branch" 变体）。
            # 单个判别词匹配不够——"left anterior" 会误命中 RBBB 论文里的
            # "left anterior descending artery"；全序列 "left anterior
            # fascicular" 才特异。
            pattern = rf"\b{_direction}\b"
            for kw in class_keywords:
                alts = _KW_SYNONYMS.get(kw, (kw,))
                grp = "|".join(re.escape(a) for a in alts)
                # 分隔符容忍连字符（"bundle-branch" 等变体）
                pattern += rf"(?:[\s\-]{{1,2}}\w+){{0,2}}[\s\-]{{1,2}}(?:{grp})\b"
            return re.search(pattern, tl) is not None
        kws = [w for w in class_keywords if w != _direction]
        return any(re.search(rf"\b{re.escape(w)}\b", tl) for w in kws)

    def _entry_relevant(e) -> bool:
        """R4 修复（2F 🔴-2）：条目级主题相关性把关。
        教育页（litfl/ecgpedia）按类名逐页抓取，主题对口，无条件保留；
        PubMed 摘要要求 condition 与类名相似或文本含类名关键词，否则剔除。
        2026-08-28 修复：general（总论）不再无条件放行——无类名可比，
        改用 ECG 领域正词 + 非 ECG 领域负词把关（实测剔除导管/十二指肠、
        TDI 超声、胎儿 ECG 等无关条目）。
        """
        if class_slug == "general":
            # 第三轮审查 3E-ORANGE-3：字符串字段先包为单元素列表再 join
            # （旧版对 str 逐字符 join 使 'catheter'→'c a t h e t e r' 绕过 NEG），
            # 且四个字段全部纳入 POS/NEG 检查（旧版漏 differential 等）
            def _as_list(v):
                return [v] if isinstance(v, str) else list(v)

            entry_text = " ".join([
                norm(e["entry"].get("definition", "")),
                " ".join(str(v) for v in _as_list(e["entry"].get("ecg_criteria") or [])),
                " ".join(str(v) for v in _as_list(e["entry"].get("differential") or [])),
                " ".join(str(v) for v in _as_list(e["entry"].get("measurements") or [])),
                " ".join(str(v) for v in _as_list(e["entry"].get("clinical_significance") or [])),
                " ".join(str(v) for v in _as_list(e["entry"].get("management") or [])),
            ])
            return bool(GENERAL_POS.search(entry_text)
                        and not GENERAL_NEG.search(entry_text))
        if SOURCE_RANK.get(e["source"], 2) <= 1:
            return True
        if _condition_matches(e["entry"].get("condition", "")):
            # 4I-RED-1：condition 匹配仍须方向一致——实测 RBBB 论文的提炼
            # condition 为 "Right bundle-branch block (RBBB) in ACS"，对 LBBB/
            # LAFB 类 condition 子串不匹配但方向短语倒挂，旧版仅靠 condition
            # 匹配放行导致 rsR' 判据混入左侧类
            if _direction is not None and not _dir_consistent(
                    e["entry"].get("condition", "")):
                return False
            return True
        entry_text = norm(e["entry"].get("definition", "")) + " " + " ".join(
            str(v) for v in (e["entry"].get("ecg_criteria") or []))
        return _text_matches(entry_text)

    entries_sorted = [e for e in entries_sorted if _entry_relevant(e)]

    def join_list(field):
        out = []
        cites = {}
        for e in entries_sorted:
            vals = e["entry"].get(field) or []
            if isinstance(vals, str):
                vals = [vals]
            for v in vals:
                if not str(v).strip() or is_placeholder(v):
                    continue
                nk = norm(v)
                if nk not in cites:
                    cites[nk] = {"text": str(v).strip(),
                                 "citations": []}
                cites[nk]["citations"].append({
                    "source": e["source"], "grade": e["grade"],
                    "url": e.get("url", ""), "doc_id": e["doc_id"],
                })
        for item in cites.values():
            # 去重引用
            seen = set()
            item["citations"] = [c for c in item["citations"]
                                 if not (c["doc_id"] in seen or seen.add(c["doc_id"]))]
            out.append(item)
        return out

    # definition: 来源对口度优先（教育页 > 论文摘要），组内等级优先、同组内取最长。
    # R4 修复：LITFL/ECGpedia 是按类名逐页抓取的主题对口内容，PubMed 摘要是
    # 按查询词检索、相关性不可控——定义字段优先教育页，避免"窦缓定义=恰加斯病"
    # 式的错位（第二次审查 2F 🔴-1）。SOURCE_RANK 见模块级定义。

    # R4 修复（2F 🔴-1 建议 b/c）：定义选取加"类名/同义词相关性"硬校验——
    # 候选条目必须 condition 与类名一致，或定义文本包含类名关键词，
    # 否则宁可空缺也不取错位定义（_condition_matches/_text_matches 见上文）。

    def _definition(e):
        d = (e["entry"].get("definition") or "").strip()
        if not d or is_placeholder(d):
            return ""
        # 第三轮审查 3E-ORANGE-2：general（总论）无类名可比——定义选取改用
        # ECG 领域正词/负词把关（旧版 class_keywords=['general'] 永不命中
        # → 总论 definition 恒为空）
        if class_slug == "general":
            return d if (GENERAL_POS.search(norm(d))
                         and not GENERAL_NEG.search(norm(d))) else ""
        # 相关性硬校验：condition 一致 或 定义文本包含类名关键词
        if not (_condition_matches(e["entry"].get("condition", "")) or _text_matches(d)):
            return ""
        return d

    definition = ""
    def_src = None
    best_key = None  # (source_rank, grade, -len)
    for e in entries_sorted:
        d = _definition(e)
        if not d:
            continue
        key = (SOURCE_RANK.get(e["source"], 2), e["grade"], -len(d))
        if best_key is None or key < best_key:
            best_key = key
            definition = d
            def_src = {"source": e["source"], "grade": e["grade"], "url": e.get("url", "")}

    kb = {
        "class": entries[0].get("class", class_slug),
        "slug": class_slug,
        "definition": definition,
        "definition_source": def_src,
        "ecg_criteria": join_list("ecg_criteria"),
        # 第三轮审查 3E-YELLOW-3：measurements/clinical_significance 保留
        # {text,citations} 结构（旧版压成纯字符串，来源引用丢失）
        "measurements": join_list("measurements")[:10],
        "differential": join_list("differential"),
        "clinical_significance": join_list("clinical_significance")[:8],
        "management": join_list("management"),
        "n_sources": len(entries_sorted),  # 第三轮审查 3E-YELLOW-2：用过滤后计数
        "sources": [{"source": e["source"], "grade": e["grade"], "url": e.get("url", ""),
                     "doc_id": e["doc_id"],
                     # R4 修复（2F 🟠-3）：透传 raw 真实标题，回退 condition
                     "title": (e.get("title") or e["entry"].get("condition") or "")}
                    for e in entries_sorted],
    }
    if _direction is not None:
        # 4I-RED-1 条目级方向过滤：方向敏感类的 ecg_criteria 只保留方向一致条目
        # （对侧方向+家族词 且 无本侧方向+家族词 → 剔除；如 LAD 类中
        # "marked right axis deviation" 被剔除）。differential 不滤——
        # 对侧疾病作为鉴别对象是临床正确的。
        kb["ecg_criteria"] = [
            it for it in kb["ecg_criteria"]
            if _dir_consistent(str(it.get("text", "")))]
    return kb


def to_markdown(kb, is_general=False):
    # 2F-Y8 修复：is_general 分支接线（旧版参数从未使用，
    # general 编译结果在 master.md 里显示为裸 "# general" 标题）
    title = "总论：ECG 解读方法学" if is_general else kb["class"]
    lines = [f"# {title}", ""]
    if kb.get("definition"):
        src = kb.get("definition_source") or {}
        lines += ["## 定义", "", kb["definition"], "",
                  f"> 来源: {src.get('source')} (grade {src.get('grade')}) {src.get('url','')}",
                  ""]
    for title, field in [("ECG 诊断特征", "ecg_criteria"),
                         ("关联测量参数", "measurements"),
                         ("鉴别诊断", "differential"),
                         ("临床意义", "clinical_significance"),
                         ("处理建议", "management")]:
        items = kb.get(field, [])
        if not items:
            continue
        lines += [f"## {title}", ""]
        for i, it in enumerate(items):
            if isinstance(it, dict):
                cites = ", ".join(f"{c['source']}({c['grade']})" for c in it.get("citations", []))
                lines.append(f"{i+1}. {it['text']}  `[{cites}]`")
            else:
                lines.append(f"- {it}")
        lines.append("")
    lines += ["## 引用来源", ""]
    for s in kb.get("sources", []):
        lines.append(f"- [{s['grade']}] {s['title'] or s['doc_id']} — {s['source']} {s['url']}")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    le = LabelExtractor(num_classes=27)
    KB_DIR.mkdir(parents=True, exist_ok=True)

    # 第三轮审查 3E-YELLOW-6：编译前清理孤儿 .md（旧分类遗留，与 classes.json 不一致）
    active_slugs = {re.sub(r"[^a-z0-9]+", "_", n.lower()).strip("_")
                    for n in le.class_names} | {"general"}
    for old_md in KB_DIR.glob("*.md"):
        if old_md.stem not in active_slugs:
            old_md.unlink()
            logger.info(f"清理孤儿文档: {old_md.name}")

    dist_files = {f.stem: f for f in DIST_DIR.glob("*.jsonl")} if DIST_DIR.exists() else {}
    logger.info(f"提炼文件: {len(dist_files)} 个")

    classes_kb = {}
    for name in le.class_names:
        slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        f = dist_files.get(slug)
        if not f:
            logger.warning(f"缺少提炼文件: {slug} ({name})")
            continue
        raw_entries = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()
                       if l.strip()]
        # 5I 审查（🟠）：按 doc_id 去重（最后一行胜）——蒸馏缓存以追加写，
        # 重抓内容变化重蒸馏后同一 doc_id 陈旧+新鲜条目并存，旧版原样全部
        # 进 KB（当前被 backfill 一次性去重掩盖，属潜伏缺陷）
        _dedup = {}
        for _e in raw_entries:
            _dedup[_e.get("doc_id", id(_e))] = _e
        entries = list(_dedup.values())
        if len(entries) != len(raw_entries):
            logger.warning(f"{slug}: distilled 文件含重复 doc_id，去重 "
                           f"{len(raw_entries)} → {len(entries)}")
        kb = compile_class(slug, entries)
        if kb:
            classes_kb[name] = kb
            (KB_DIR / f"{slug}.md").write_text(to_markdown(kb), encoding="utf-8")
            logger.info(f"{name}: {kb['n_sources']} 来源, "
                        f"{len(kb['ecg_criteria'])} 特征, {len(kb['differential'])} 鉴别")

    # 总论
    general_entries = []
    gf = dist_files.get("general")
    if gf:
        general_entries = [json.loads(l) for l in gf.read_text(encoding="utf-8").splitlines()
                           if l.strip()]
    general_kb = compile_class("general", general_entries) if general_entries else None

    # classes.json（Agent 工具读取）
    with open(KB_DIR / "classes.json", "w", encoding="utf-8") as f:
        json.dump({"classes": {k: v for k, v in classes_kb.items()},
                   "general": general_kb,
                   "disclaimer": ("本知识库由公开来源（PubMed 摘要/ECGpedia 等）经 LLM 提炼生成，"
                                  "仅供研究参考，不构成医疗诊断建议。每条结论保留来源与可信度等级。")},
                  f, ensure_ascii=False, indent=1)

    # master.md
    master_lines = [
        "# ECG 分析参考知识库（Master）", "",
        "> 由公开来源经 LLM 提炼生成，仅供研究参考，不构成医疗诊断建议。", "",
        "## 读图顺序（标准系统化方法）", "",
        "1. 心率与节律 (Rate & Rhythm)", "",
        "2. 间期 (PR / QRS / QT)", "",
        "3. 电轴 (Axis)", "",
        "4. 形态 (P 波 / QRS / ST-T)", "",
        "5. 综合诊断与鉴别", "",
    ]
    if general_kb:
        master_lines += ["## 总论知识", "", to_markdown(general_kb, is_general=True), ""]
    master_lines += ["## 各类别索引", ""]
    for name, kb in classes_kb.items():
        slug = kb["slug"]
        master_lines.append(f"- [{name}]({slug}.md) — {len(kb['ecg_criteria'])} 特征, "
                            f"{kb['n_sources']} 来源")
    (KB_DIR / "master.md").write_text("\n".join(master_lines), encoding="utf-8")

    logger.info(f"完成: {len(classes_kb)} 类 + 总论 → {KB_DIR}")


if __name__ == "__main__":
    main()
