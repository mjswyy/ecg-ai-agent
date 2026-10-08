#!/usr/bin/env python3
"""第三轮修复后 + glm-4.5-air 三臂重跑（串行）：main → report_arm → kb_arm。
修复面：KB 重编译（Pacing 污染清除/general 定义恢复）、分类器 Youden 阈值 +
positives 全类判定、HRV None 口径、注入率 5.4% 等。"""
import subprocess
import sys
import time

STEPS = [
    ("main", ["python", "scripts/eval_agent_llm.py"]),
    ("report_arm", ["python", "scripts/eval_agent_llm.py", "--report-arm"]),
    ("kb_arm", ["python", "scripts/eval_agent_llm.py", "--use-kb"]),
]

t0 = time.time()
for name, cmd in STEPS:
    print(f"\n===== [{name}] =====", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    print((r.stdout or "")[-2500:], flush=True)
    if r.returncode != 0:
        print(f"[{name}] FAILED rc={r.returncode}", flush=True)
        print((r.stderr or "")[-1500:], flush=True)
        sys.exit(1)
    print(f"[{name}] OK", flush=True)

print(f"\n三臂完成（glm-4.5-air），耗时 {(time.time()-t0)/60:.1f} 分钟", flush=True)
