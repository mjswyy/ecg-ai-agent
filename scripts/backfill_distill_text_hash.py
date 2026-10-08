#!/usr/bin/env python3
"""4I-ORANGE-2 修复：为 distilled/*.jsonl 回填 text_hash（内容级缓存键）。

旧版 DistillCache 按 doc_id 去重、不存 text_hash → 332/332 条记录缺 text_hash，
is_done 恒 False → distill 重跑会全量重蒸馏 + append 重复行。
本脚本按 doc_id 从 raw/ 文档重算文本哈希回填，并去重（同 doc_id 保留最后一行）。
用法: python scripts/backfill_distill_text_hash.py
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

RAW_DIR = Path("data/knowledge/raw")
OUT_DIR = Path("data/knowledge/distilled")


def doc_key(text: str) -> str:
    payload = json.dumps(text, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def raw_text_of(class_slug: str, doc_id: str):
    raw = RAW_DIR / class_slug / f"{doc_id}.json"
    if not raw.exists():
        return None
    try:
        return json.loads(raw.read_text(encoding="utf-8")).get("text", "")
    except Exception:
        return None


total_fixed = 0
total_dedup = 0
for f in sorted(OUT_DIR.glob("*.jsonl")):
    cls = f.stem
    lines = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines()
             if l.strip()]
    # 去重：同 doc_id 保留最后一行
    seen = {}
    for e in lines:
        seen[e.get("doc_id", "")] = e
    n_dup = len(lines) - len(seen)
    fixed = 0
    for e in seen.values():
        if e.get("text_hash"):
            continue
        text = raw_text_of(cls, e.get("doc_id", ""))
        if text is None:
            continue
        e["text_hash"] = doc_key(text)
        fixed += 1
    with open(f, "w", encoding="utf-8") as fo:
        for e in seen.values():
            fo.write(json.dumps(e, ensure_ascii=False) + "\n")
    total_fixed += fixed
    total_dedup += n_dup
    print(f"{cls}: {len(lines)} 行 → 去重 {n_dup} / 回填 hash {fixed}")

print(f"总计: 回填 {total_fixed} / 去重 {total_dedup}")
