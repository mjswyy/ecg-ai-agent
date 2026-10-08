#!/usr/bin/env python3
"""知识提炼器（Phase K2）— 用 LLM 把采集的文档提炼为结构化知识条目。

输入: data/knowledge/raw/{class_slug}/doc_*.json
输出: data/knowledge/distilled/{class_slug}.jsonl
    每行: {doc_id, source, grade, url, entry:{...schema...}, retrieved_at}

Schema（英文提炼）:
    condition            规范化疾病/主题名
    definition           定义（1-3 句）
    ecg_criteria         ECG 诊断特征列表
    measurements         关联测量参数（心率/PR/QRS/QT 等）
    differential         鉴别诊断列表
    clinical_significance 临床意义与风险
    management           处理/建议要点
    confidence           提炼保真度自评 high/medium/low

防幻觉: 提炼必须仅基于源文本；缓存按 doc_id；解析失败自动重试一次。

用法:
    python scripts/distill_knowledge.py --classes "Atrial Fibrillation"
    python scripts/distill_knowledge.py                 # 全量
    python scripts/distill_knowledge.py --verify        # 额外一致性校验（2 次调用/篇）
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.agent.llm.llm_interface import LLMInterface

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "knowledge" / "raw"
OUT_DIR = BASE_DIR / "data" / "knowledge" / "distilled"

SYSTEM = ("You are a medical knowledge engineer. Extract structured, faithful "
          "knowledge from ECG-related medical texts. Only use information present "
          "in the source text. Output valid JSON only. Language: English.")

SCHEMA_HINT = """
Output JSON with exactly these fields:
{
  "condition": "canonical condition name",
  "definition": "1-3 sentence definition",
  "ecg_criteria": ["specific ECG diagnostic criteria, each as one item"],
  "measurements": "relevant ECG measurements/parameters (rate, intervals, axis...)",
  "differential": ["differential diagnoses"],
  "clinical_significance": "clinical meaning and risk",
  "management": "management/recommendation points",
  "confidence": "high|medium|low (fidelity of this extraction to the source)"
}
If the source does not cover a field, use an empty list or empty string
(NEVER use placeholder values like 'not stated', 'n/a' or 'unknown')."""

VERIFY_PROMPT = """You are a medical editor checking faithfulness.
Source text:
{source}

Extracted entry:
{entry}

Does every claim in the extracted entry appear in or directly follow from the source?
Answer JSON: {{"faithful": true/false, "issues": ["..."]}}"""  # 5I 审查（🔴-1）：花括号必须转义——旧版 .format() 把 {"faithful"...} 当占位符抛 KeyError，--verify 模式必然崩溃且中断整个批次


def load_docs(class_slug: str):
    d = RAW_DIR / class_slug
    if not d.exists():
        return []
    docs = []
    for f in sorted(d.glob("*.json")):
        if f.name == "crawl_summary.json":
            continue
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            if len(doc.get("text", "").strip()) >= 150:
                docs.append(doc)
        except Exception:
            pass
    return docs


def validate_entry(entry) -> str:
    """2F-O7 修复：提炼 schema 校验（类型/枚举/必填），返回错误描述（空串=通过）。

    旧版只查 isinstance(dict) and "condition" in entry——字段类型漂移
    （ecg_criteria 非 list、measurements 非 str、confidence 越界、缺字段等）
    会直接进缓存和 KB，属于"模型表现而非代码保证"的缺口。
    """
    if not isinstance(entry, dict):
        return "entry 非 dict"
    missing = [f for f in ("condition", "definition", "ecg_criteria",
                           "measurements", "differential",
                           "clinical_significance", "management",
                           "confidence") if f not in entry]
    if missing:
        return f"缺字段 {missing}"
    if not isinstance(entry["condition"], str) or not entry["condition"].strip():
        return "condition 非字符串"
    if not isinstance(entry["definition"], str):
        return "definition 非字符串"
    for field in ("ecg_criteria", "differential"):
        if not isinstance(entry[field], list) or \
                not all(isinstance(x, str) for x in entry[field]):
            return f"{field} 非字符串列表"
    for field in ("measurements", "clinical_significance", "management"):
        if not isinstance(entry[field], str):
            return f"{field} 非字符串"
    if entry["confidence"] not in ("high", "medium", "low"):
        return "confidence 不在 {high,medium,low}"
    return ""


def _chat(llm, prompt, retries=1):
    """带异常兜底与退避重试的 LLM 调用（2F-O5 修复）。

    旧版 llm.chat 抛 API/网络异常（无重试）直接冒泡中断整个批次提炼。
    """
    for a in range(retries + 1):
        try:
            return llm.chat([{"role": "system", "content": SYSTEM},
                             {"role": "user", "content": prompt}],
                            response_format={"type": "json_object"})
        except Exception as e:
            if a >= retries:
                raise
            logger.warning(f"  LLM 调用异常（第{a+1}次）: {e}，退避重试")
            time.sleep(3 * (2 ** a))
    raise RuntimeError("unreachable")


def doc_key(doc):
    """内容级缓存键：文本+标题+URL 哈希（第三轮审查 3E-YELLOW-4：doc_key 从死代码接线——
    重抓同一 doc_id 但内容更新时缓存失效，旧版仅按 doc_id 去重产生陈旧提炼）。
    5I 审查（🟡）：纳入 title/url——旧版只哈希 text，标题/URL 修正不失效缓存
    （title 透传修复失效的根因）。"""
    payload = json.dumps({
        "text": doc.get("text", ""),
        "title": doc.get("title", ""),
        "url": doc.get("url", ""),
    }, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class DistillCache:
    def __init__(self, class_slug):
        self.path = OUT_DIR / f"{class_slug}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.done = {}  # doc_id -> text_hash
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    e = json.loads(line)
                    self.done[e["doc_id"]] = e.get("text_hash", "")
                except json.JSONDecodeError:
                    pass

    def is_done(self, doc):
        """内容级已完成判定：doc_id 存在且文本哈希一致（内容变更则视为未完成）。"""
        return self.done.get(doc["doc_id"]) == doc_key(doc)

    def put(self, record):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def distill_one(llm, doc, verify=False):
    prompt = (f"Source ({doc['source']}, grade {doc['grade']}):\n"
              f"Title: {doc.get('title', '')}\n\n{doc['text'][:3500]}\n\n{SCHEMA_HINT}")
    entry = None
    err = ""
    # 初试 + 一次重试：JSON 解析失败或 schema 校验不过均重试（2F-O5/O7）
    for attempt in range(2):
        try:
            raw = _chat(llm, prompt)
            candidate = json.loads(raw)
        except Exception as e:
            err = f"{type(e).__name__}: {str(e)[:100]}"
            if attempt == 0:
                logger.warning(f"  提炼调用异常（将重试）: {err}")
                time.sleep(2)
                continue
            return None, err
        verr = validate_entry(candidate)
        if not verr:
            entry = candidate
            break
        err = f"schema 校验失败: {verr}"
        if attempt == 0:
            logger.warning(f"  {err}（将重试）")
            time.sleep(2)
    if entry is None:
        return None, err or "未知失败"

    if verify:
        vp = VERIFY_PROMPT.format(source=doc["text"][:2500],
                                  entry=json.dumps(entry, ensure_ascii=False))
        try:
            v = json.loads(_chat(llm, vp))
            entry["_verified"] = bool(v.get("faithful", False))
            entry["_issues"] = v.get("issues", [])
        except Exception as e:
            logger.warning(f"  校验调用失败: {e}")
            entry["_verified"] = False
    return entry, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classes", nargs="+", default=None)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--model", default="deepseek-chat")
    args = parser.parse_args()

    llm = LLMInterface(backend="deepseek", model=args.model)
    if not llm.is_available:
        logger.error("无 API key（.env）")
        sys.exit(1)

    class_dirs = [d.name for d in RAW_DIR.iterdir() if d.is_dir()] if RAW_DIR.exists() else []
    if args.classes:
        from src.data_pipeline.label_extractor import LabelExtractor
        le = LabelExtractor(num_classes=27)
        wanted = {slugify(c) for c in args.classes} | {"general"}
        class_dirs = [d for d in class_dirs if d in wanted]
    class_dirs = sorted(class_dirs)
    logger.info(f"待提炼类目: {len(class_dirs)}")

    for cs in class_dirs:
        docs = load_docs(cs)
        cache = DistillCache(cs)
        # 3E-YELLOW-4：内容级缓存（doc_id + 文本哈希）
        todo = [d for d in docs if not cache.is_done(d)]
        logger.info(f"[{cs}] 共 {len(docs)} 条, 待提炼 {len(todo)}")
        for i, doc in enumerate(todo):
            entry, err = distill_one(llm, doc, verify=args.verify)
            if entry is None:
                logger.warning(f"  [{cs}] {doc['doc_id']} 提炼失败: {err}")
                continue
            cache.put({
                "doc_id": doc["doc_id"], "source": doc["source"],
                "grade": doc["grade"], "url": doc.get("url", ""),
                "class": doc.get("class", cs), "class_slug": cs,
                # R4 修复（2F 🟠-3）：透传 raw 真实标题，供编译端引用追溯
                "title": doc.get("title", ""),
                "text_hash": doc_key(doc),
                "entry": entry,
            })
            if (i + 1) % 10 == 0:
                logger.info(f"  [{cs}] {i+1}/{len(todo)}")
        logger.info(f"[{cs}] 完成")

    logger.info(f"全部完成 → {OUT_DIR}")


def slugify(name: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


if __name__ == "__main__":
    main()
