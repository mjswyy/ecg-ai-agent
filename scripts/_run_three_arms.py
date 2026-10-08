#!/usr/bin/env python3
"""三臂串行重跑（deepseek-chat，2026-08-30）——串行使 llm_cache 跨臂复用
（三臂规划/反射 prompt 相同，后跑臂命中前臂缓存，只新增 report 调用）；
并行三进程会导致缓存互不可见 + 并发限速（2026-08-29 实测 3 倍调用量）。

历史教训（v4-flash 时代）：deepseek-v4-flash 对长 prompt 大量空响应
（main 臂 60/130 空报告），故最终模型定稿 deepseek-chat；ask() 已有
空响应重试+不写缓存守卫。
"""
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

print(f"\n三臂串行完成（deepseek-chat），耗时 {(time.time()-t0)/60:.1f} 分钟", flush=True)
