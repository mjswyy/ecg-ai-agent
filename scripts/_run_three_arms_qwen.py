#!/usr/bin/env python3
"""三臂重跑（deepseek-v4-flash，三进程并行（deepseek-v4-flash））。"""
import subprocess
import sys
import time

STEPS = [
    ("main", ["python", "scripts/eval_agent_llm.py"]),
    ("report_arm", ["python", "scripts/eval_agent_llm.py", "--report-arm"]),
    ("kb_arm", ["python", "scripts/eval_agent_llm.py", "--use-kb"]),
]

t0 = time.time()
procs = {}
for name, cmd in STEPS:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace")
    procs[name] = p
    print(f"[{name}] 启动 pid={p.pid}", flush=True)

for name, p in procs.items():
    out, _ = p.communicate()
    rc = p.returncode
    print(f"\n===== [{name}] rc={rc} =====", flush=True)
    print((out or "")[-2500:], flush=True)

print(f"\n三臂并行完成（deepseek-v4-flash），耗时 {(time.time()-t0)/60:.1f} 分钟", flush=True)
