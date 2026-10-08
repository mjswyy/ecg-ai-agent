#!/usr/bin/env python3
"""R4 修复验证：占位符清除 / 定义相关性 / title 透传 / find 词边界。"""
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.knowledge.kb_loader import KnowledgeBase

KB = Path("data/knowledge/kb/classes.json")
data = json.loads(KB.read_text(encoding="utf-8"))
classes = data["classes"]

# 4E 定向复查（ε）：与 compile_knowledge.PLACEHOLDERS 同步（旧版为过期子集）
PLACEHOLDERS = {"not stated", "n/a", "na", "not available", "not mentioned",
                "not specified", "unknown", "none", "nil",
                "not applicable", "not provided", "no specific", "not given"}
# 纯短语式占位（与 compile_knowledge._PURE_PLACEHOLDER_PHRASE 一致）
_PURE_PHRASE = re.compile(
    r"^not\s+(explicitly\s+)?(stated|specified|provided|mentioned|discussed|"
    r"described|reported|given|applicable|available)"
    r"(\s+in\s+(the\s+)?(source|provided text|text|abstract|article))?"
    r"[.\s]*$")

def norm(s):
    return re.sub(r"\s+", " ", str(s).strip().lower()).strip(" -–—")

# 1) 占位符扫描（所有字段 + sources title）
bad = []
for cname, kb in classes.items():
    for field in ["definition", "ecg_criteria", "measurements", "differential",
                  "clinical_significance", "management"]:
        v = kb.get(field)
        if isinstance(v, str):
            vals = [v]
        elif isinstance(v, list):
            vals = [x["text"] if isinstance(x, dict) else x for x in v]
        else:
            vals = []
        for x in vals:
            if norm(x) in PLACEHOLDERS or _PURE_PHRASE.search(norm(x)):
                bad.append((cname, field, x))
    for s in kb.get("sources", []):
        if norm(s.get("title", "")) in PLACEHOLDERS:
            bad.append((cname, "sources.title", s.get("title")))
print(f"[1] 占位符残留: {len(bad)}")
for b in bad[:10]:
    print("   ", b)

# 2) 定义缺失/错位
n_def = 0
for cname, kb in classes.items():
    d = kb.get("definition", "")
    if d:
        n_def += 1
        # 定义文本须含类名关键词
        kw = [w for w in norm(cname).split()
              if w not in {"sinus", "premature", "complex", "conduction",
                           "disorder", "abnormality", "abnormal", "cardiac"}]
        if kw and not any(w in norm(d) for w in kw):
            print(f"   [定义弱相关] {cname}: {d[:80]}...")
    else:
        print(f"   [定义缺失] {cname}")
print(f"[2] 有定义类: {n_def}/27")

# 3) title 透传
titles = []
for cname, kb in classes.items():
    for s in kb.get("sources", []):
        titles.append(s.get("title", ""))
n_title = sum(1 for t in titles if t and len(t) > 3)
print(f"[3] sources title 非空: {n_title}/{len(titles)}（示例: {titles[0][:60]} / {titles[-1][:60]}）")

# 4) 词边界匹配
# 第三轮审查 3F-R3-07/08：改为真断言（旧版纯打印、MISMATCH 仍 exit 0，
# 且 ('AF','Atrial Fibrillation') 期望值与注释/实际行为自相矛盾——
# 现 find 对歧义前缀返回 None，AF 缩写无同义词表 → None 为正确期望）
kb = KnowledgeBase()
cases = [
    ("Atrial Fibrillation", "Atrial Fibrillation"),
    ("Sinus", None),                         # 歧义前缀（命中 4 个窦性类）→ None
    ("A", None),                             # 单字母不应命中 AF
    ("AF", "Atrial Fibrillation"),           # 缩写解析（3E-YELLOW-1 接线后命中）
    ("brady", "Bradycardia"),                # 缩写 'Brady'（3E-YELLOW-1）
    ("Sinus Bradycardia", "Sinus Bradycardia"),
]
print("[4] find 词边界:")
n_fail = 0
for q, exp in cases:
    hit = kb.find(q)
    got = hit[0] if hit else None
    ok = got == exp
    n_fail += 0 if ok else 1
    print(f"   {q!r:30} -> {got!r}  {'OK' if ok else 'MISMATCH'}")
assert n_fail == 0, f"find 词边界 {n_fail} 条不匹配"

# 5) 每类来源数/特征数概览
print("[5] 规模:")
for cname, kb in sorted(classes.items()):
    print(f"   {cname}: {kb['n_sources']} 来源 / {len(kb['ecg_criteria'])} 特征 / def={bool(kb.get('definition'))}")

# 6) 相关性抽查：Pacing Rhythm 不得含 Brugada/LQTS/ACM 等无关特征（第三轮 3E-RED-1）
pacing = classes.get("Pacing Rhythm", {})
pacing_text = " ".join(
    c["text"] if isinstance(c, dict) else str(c)
    for c in pacing.get("ecg_criteria", []))
for bad_kw in ("Brugada", "LQTS", "arrhythmogenic", "Interatrial block",
               "Sodium channel blocker"):
    assert bad_kw not in pacing_text, f"Pacing Rhythm 含污染关键词 {bad_kw}"
print("[6] Pacing Rhythm 相关性抽查: 通过（无 Brugada/LQTS/ACM 污染）")

print("\nverify_r4_kb 全部断言通过")  # 4E 定向复查（ε）：去 emoji 防 GBK 控制台崩溃
