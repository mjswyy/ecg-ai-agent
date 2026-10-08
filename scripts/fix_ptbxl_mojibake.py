#!/usr/bin/env python3
"""2E-Y5 修复：ptbxl_reports.json 的 mojibake 文本修复。

现象："unbest脛tigter bericht"（应为 unbestätigter Bericht）——报告原始字节是
UTF-8 编码（ä=0xC3 0xA4），却被按 GBK 解码（0xC3A4=脛）。修复：把误读字符串
按 GBK 编码回原始字节流，再按 UTF-8 解码。
安全策略：仅修复含可疑字符的字符串；GBK 编码或 UTF-8 解码失败/含 U+FFFD 时保留原文。
"""
import json
from pathlib import Path

SRC = Path("data/physionet2020/processed_5k/ptbxl_reports.json")
BAK = Path("data/physionet2020/processed_5k/ptbxl_reports.json.mojibake_bak")

# GBK 双字节误读 UTF-8 序列的常见结果（CJK 区）
SUSPECT = set("脛脰脜脝脢脧脨脪脮脭脤脥脦脩脫脬脯脡脣脢脠")

def fix_mojibake(s):
    if not s or not any(c in SUSPECT for c in s):
        return s
    try:
        fixed = s.encode("gbk").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s
    if "\ufffd" in fixed:
        return s
    return fixed

data = json.loads(SRC.read_text(encoding="utf-8"))
n_total = 0
n_fixed = 0
examples = []
for k, v in data.items():
    if not isinstance(v, dict) or not v.get("report"):
        continue
    n_total += 1
    orig = v["report"]
    fixed = fix_mojibake(orig)
    if fixed != orig:
        n_fixed += 1
        v["report"] = fixed
        if len(examples) < 5:
            examples.append((orig[:60], fixed[:60]))

# 备份原始文件
if not BAK.exists():
    BAK.write_bytes(SRC.read_bytes())

SRC.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"报告总数: {n_total}, 修复: {n_fixed}")
for o, f in examples:
    print(f"  {o!r} -> {f!r}")

# 验证：不再有 mojibake 特征字符
left = sum(1 for k, v in data.items()
           if isinstance(v, dict) and v.get("report")
           and any(c in SUSPECT for c in v["report"]))
print(f"残留可疑字符报告: {left}")
