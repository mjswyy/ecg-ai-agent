# -*- coding: utf-8 -*-
"""web_demo 管线冒烟测试（不启动 GUI）。

5J 审查（🟡-5）：默认 mock 模式（LLMInterface(mock=True)，不花钱）；
--live 显式开关才调用真实 API（旧版 key 存在时静默打真实付费调用）。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--live", action="store_true",
                    help="调用真实 LLM API（默认 mock，不花钱）")
args = parser.parse_args()

# 导入 app 会构建 UI 对象（不 launch），pipeline 可直接调用
import web_demo.app as appmod

p = appmod.pipeline
if not args.live:
    from src.agent.llm.llm_interface import LLMInterface
    p.llm = LLMInterface(backend="deepseek", model="deepseek-chat", mock=True)
    p.llm_ok = True

fig, findings, kb_md, report_md, status = p.run(
    record_id="HR13177", files=None, age=74, sex="Male",
    setting="急诊科", complaint="胸痛 3 天", history="高血压")

print("状态:", status)
print("--- 发现层 ---")
print(findings[:400])
print("--- 双层报告 ---")
print(report_md[:500])
print("--- 知识库 ---")
print(kb_md[:200])
print("波形图:", type(fig).__name__)
