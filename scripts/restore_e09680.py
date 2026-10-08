# ⚠️ DEPRECATED（2026-08-24，检查报告第六步）: 一次性修复脚本（E09680 损坏文件恢复，使命已完成，仅存档）
# -*- coding: utf-8 -*-
"""从原始 zip 恢复 georgia/g10/E09680.* 文件（v3：处理只读属性与目录条目）"""
import zipfile
import os
import stat
import sys
from pathlib import Path

# 4C 审查修复：🟡-7 —— 删除硬编码 Desktop\ECG 绝对路径（含 zip 与解压根目录），
# 改用自动探测，探测失败时给出清晰报错。
def _default_zip():
    stem = "classification-of-12-lead-ecgs-the-physionetcomputing-" \
           "in-cardiology-challenge-2020-1.0.2"
    candidates = [
        Path.home() / "Desktop" / "ECG" / "压缩包" / f"{stem}.zip",
        Path.home() / "Desktop" / "ecg资料" / "压缩包" / f"{stem}.zip",
        Path.home() / "Desktop" / "ECG" / f"{stem}.zip",
        Path.home() / "Desktop" / "ecg资料" / f"{stem}.zip",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def _default_raw_root():
    """探测原始数据解压根目录（含 training/ 子目录）。"""
    stem = "classification-of-12-lead-ecgs-the-physionetcomputing-" \
           "in-cardiology-challenge-2020-1.0.2"
    candidates = [
        Path.home() / "Desktop" / "ECG" / stem,
        Path.home() / "Desktop" / "ecg资料" / stem,
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


ZIP = _default_zip()
if not ZIP:
    sys.exit("未探测到原始 zip 压缩包——请修正探测逻辑（4C 审查修复：🟡-7）")
OUT = _default_raw_root()
if not OUT:
    sys.exit("未探测到原始数据解压根目录——请修正探测逻辑（4C 审查修复：🟡-7）")

with zipfile.ZipFile(ZIP) as z:
    names = [n for n in z.namelist() if "09680" in n and not n.endswith("/")]
    print(f"zip 内文件条目: {len(names)}")
    for n in names:
        info = z.getinfo(n)
        print(f"  {n}  {info.file_size} bytes")
        parts = n.split("/")
        rel = "/".join(parts[1:]) if len(parts) > 1 else parts[0]
        dest = OUT + "\\" + rel.replace("/", "\\")
        # 清除只读属性（PhysioNet 解压的文件常带只读）
        if os.path.exists(dest):
            os.chmod(dest, stat.S_IWRITE)
        with z.open(n) as src, open(dest, "wb") as dst:
            dst.write(src.read())
        print(f"  → 已恢复: {dest}")

# 校验
print("\n恢复后文件:")
for f in ("E09680.hea", "E09680.mat"):
    p = os.path.join(OUT, "training", "georgia", "g10", f)
    if os.path.exists(p):
        print(f"  {f}: {os.path.getsize(p)} bytes")
    else:
        print(f"  {f}: 不存在!")
