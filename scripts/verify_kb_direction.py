# -*- coding: utf-8 -*-
"""4I-RED-1 终验 v2：方向类纯对侧判据检查 + 正类保留。"""
import json
import re

kb = json.load(open("data/knowledge/kb/classes.json", encoding="utf-8"))
cls = kb["classes"]

ok = True
def chk(name, cond, detail=""):
    global ok
    print(("PASS " if cond else "FAIL ") + name + ((" :: " + str(detail)) if detail else ""))
    if not cond:
        ok = False

FAMILY = ("bundle", "branch", "fascicular", "axis", "deviation")

def dir_hit(t, d):
    t = t.lower()
    return any(re.search(rf"\b{d}\b(?:[\s\-]{{1,2}}\w+){{0,2}}[\s\-]{{1,2}}{k}\b", t)
               for k in FAMILY)

def pure_opposite_items(name, own_dir, opp_dir):
    bad = []
    for c in cls[name].get("ecg_criteria", []):
        t = str(c.get("text", ""))
        own = dir_hit(t, own_dir)
        opp = dir_hit(t, opp_dir)
        if opp and not own:
            bad.append(t[:70])
    return bad

pairs = [
    ("Left Bundle Branch Block", "left", "right"),
    ("Left Axis Deviation", "left", "right"),
    ("Left Anterior Fascicular Block", "left", "right"),
    ("Right Bundle Branch Block", "right", "left"),
    ("Right Axis Deviation", "right", "left"),
    ("Complete Right Bundle Branch Block", "right", "left"),
    ("Incomplete Right Bundle Branch Block", "right", "left"),
]
for name, own, opp in pairs:
    bad = pure_opposite_items(name, own, opp)
    chk(f"{name} 无纯对侧判据", not bad, bad)

# rsR' 特查（无方向词但属 RBBB 专有形态）
for name in ("Left Bundle Branch Block", "Left Anterior Fascicular Block"):
    rsr = [str(c.get("text", ""))[:60] for c in cls[name]["ecg_criteria"]
           if "rsr" in str(c.get("text", "")).lower()]
    chk(f"{name} 无 rsR 判据", not rsr, rsr)

# 正类保留
rbbb_srcs = [s["doc_id"] for s in cls["Right Bundle Branch Block"]["sources"]]
chk("RBBB 保留 pubmed_41429745", "pubmed_41429745" in rbbb_srcs)
for name, _, _ in pairs:
    n = len(cls[name].get("ecg_criteria", []))
    chk(f"{name} 判据非空 (n={n})", n > 0)

print("---")
print("ALL PASS" if ok else "SOME FAIL")
