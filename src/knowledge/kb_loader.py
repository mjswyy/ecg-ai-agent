"""知识库检索器 — Agent 工具 query_medical_knowledge 的后端。

从 data/knowledge/kb/classes.json 按诊断名检索知识条目，
格式化为适合注入 LLM 上下文的 Markdown 片段（含来源与等级标注）。

设计:
    - 查找: 精确类名 → 大小写不敏感包含匹配 → 无结果
    - 返回: 定义 + ECG 特征(带引用) + 鉴别 + 处理，截断至 max_chars
    - 附来源等级标记（grade A/B/C），供 LLM 参考可信度
"""

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

KB_PATH = Path(__file__).parent.parent.parent / "data" / "knowledge" / "kb" / "classes.json"


class KnowledgeBase:
    def __init__(self, kb_path=None):
        self.path = Path(kb_path) if kb_path else KB_PATH
        if not self.path.exists():
            raise FileNotFoundError(f"知识库不存在: {self.path}（先跑 compile_knowledge.py）")
        with open(self.path, encoding="utf-8") as f:
            data = json.load(f)
        self.classes = data.get("classes", {})
        self.general = data.get("general")
        self.disclaimer = data.get("disclaimer", "")

    def find(self, name: str):
        """按诊断名查找知识条目，返回 (类名, kb) 或 None。

        R4 修复（2F 🟠-4）：精确相等 → 规范化相等 → 词边界包含匹配，
        不再做任意子串包含（旧版 "Sinus" 会误匹配 "Sinus Bradycardia"、
        单字母 "A" 会命中 "Atrial Fibrillation" 等）。
        第三轮审查 3E-ORANGE-1：词边界对整词前缀仍会误匹配（'Sinus' 命中
        'Sinus Bradycardia'），且歧义前缀静默返回插入序第一个类——
        现改为：命中多个候选时返回 None（歧义需精确），并更正 docstring。
        """
        if not name:
            return None
        if name in self.classes:
            return name, self.classes[name]
        low = re.sub(r"\s+", " ", name.strip().lower())
        for cname, kb in self.classes.items():
            clow = re.sub(r"\s+", " ", cname.lower())
            if low == clow:
                return cname, kb
        # 第三轮审查 3E-YELLOW-1：缩写解析（复用 LabelExtractor 的官方缩写表）
        try:
            from src.data_pipeline.label_extractor import CHALLENGE_CLASSES
            for info in CHALLENGE_CLASSES.values():
                if (info.get("abbreviation") or "").lower() == low:
                    full = info.get("name")
                    if full in self.classes:
                        return full, self.classes[full]
        except Exception:
            pass
        # 词边界包含（整词匹配）；命中多个候选视为歧义 → None
        hits = []
        for cname, kb in self.classes.items():
            clow = re.sub(r"\s+", " ", cname.lower())
            if re.search(rf"\b{re.escape(low)}\b", clow) or \
               re.search(rf"\b{re.escape(clow)}\b", low):
                hits.append(cname)
        if len(hits) == 1:
            return hits[0], self.classes[hits[0]]
        if len(hits) > 1:
            logger.warning(f"知识检索 '{name}' 命中多个类目 {hits}，歧义返回 None（请用全名）")
        return None

    def format_entry(self, kb, max_chars=2800) -> str:
        lines = [f"# {kb.get('class', '')} (医学参考资料, {kb.get('n_sources', 0)} 来源)", ""]
        if kb.get("definition"):
            src = kb.get("definition_source") or {}
            lines += [f"定义: {kb['definition']}",
                      f"  [来源: {src.get('source')} 等级{src.get('grade')}]", ""]
        for title, field in [("ECG 诊断特征", "ecg_criteria"),
                             ("关联测量", "measurements"),
                             ("鉴别诊断", "differential"),
                             ("临床意义", "clinical_significance"),
                             ("处理建议", "management")]:
            items = kb.get(field, [])
            if not items:
                continue
            lines.append(f"{title}:")
            for it in items[:8]:
                if isinstance(it, dict):
                    cites = ",".join(f"{c['source']}({c['grade']})"
                                     for c in it.get("citations", []))
                    # 3E-💡-1：空引用时省略括号后缀（旧版输出裸 "[]"）
                    lines.append(f"  - {it['text']}" + (f"  [{cites}]" if cites else ""))
                else:
                    lines.append(f"  - {it}")
            lines.append("")
        text = "\n".join(lines)
        if len(text) > max_chars:
            text = text[:max_chars] + "\n...(截断)"
        return text

    def query(self, diagnosis_name, max_chars=2800) -> str:
        hit = self.find(diagnosis_name)
        if not hit:
            return f"(知识库中未找到 '{diagnosis_name}' 的条目)"
        text = self.format_entry(hit[1], max_chars=max_chars)
        # 4I-YELLOW 修复：disclaimer 编译入库但从未被读取/注入——
        # 拼接到检索结果头部，LLM 上下文中始终携带免责声明
        if self.disclaimer:
            text = f"> {self.disclaimer}\n\n" + text
        return text

    def general_excerpt(self, max_chars=1500) -> str:
        """总论知识片段（ECG 读图方法论），供检索时拼接注入。

        2F 🟠-9 修复：general 编译入库但从不被检索——现由调用方
        （eval harness / web_demo）在知识片段头部拼接。
        """
        if not self.general:
            return ""
        kb = self.general
        parts = []
        if kb.get("definition"):
            parts.append(f"定义: {kb['definition']}")
        for field, title in (("ecg_criteria", "读图要点"),
                             ("management", "处理原则")):
            items = kb.get(field) or []
            if items:
                texts = [it["text"] if isinstance(it, dict) else str(it)
                         for it in items[:6]]
                parts.append(f"{title}: " + "；".join(texts))
        text = "\n".join(parts)
        return ("# 总论（ECG 读图方法论，通用）\n" + text)[:max_chars]

    def format_catalog(self, max_chars=320) -> str:
        """全 27 类目录：每类一行压缩摘要（定义+前 2 条诊断特征）。

        用户需求（2026-08-28）：不能只给 LLM 分类器阳性诊断的资料，
        应把全部类目资料都给 LLM，使其能对照各诊断的判定标准来
        验证/质疑 ECGFounder 的判断（鉴别诊断与漏报核查）。
        """
        lines = ["# 全部 27 类诊断参考目录（用于对照验证分类器判断）", ""]
        for cname in sorted(self.classes):
            kb = self.classes[cname]
            parts = []
            d = (kb.get("definition") or "").strip()
            if d:
                parts.append(d[:130])
            crit = kb.get("ecg_criteria") or []
            if crit:
                items = [c["text"] if isinstance(c, dict) else str(c)
                         for c in crit[:2]]
                joined = "；".join(items)
                parts.append("特征: " + joined[:170])
            lines.append(f"- {cname}: " + " | ".join(parts)[:max_chars])
        return "\n".join(lines)

    def list_classes(self):
        return sorted(self.classes.keys())


def query_medical_knowledge_tool_schema():
    return {
        "name": "query_medical_knowledge",
        "description": "从分级可信的 ECG 医学知识库检索某诊断的参考资料"
                       "（定义/诊断特征/鉴别/处理，带来源与可信度等级；"
                       "支持常见缩写如 AF/LBBB/CRBBB——3E-💡-1）。",
        "parameters": {
            "type": "object",
            "properties": {
                "diagnosis": {"type": "string", "description": "诊断名，如 Atrial Fibrillation"},
            },
            "required": ["diagnosis"],
        },
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    kb = KnowledgeBase()
    print(f"知识库类目: {len(kb.classes)}")
    print("=" * 60)
    print(kb.query("Atrial Fibrillation", max_chars=1500))
