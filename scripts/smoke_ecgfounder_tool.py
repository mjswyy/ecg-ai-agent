# -*- coding: utf-8 -*-
"""ECGFounder 分类工具冒烟测试

4A/4J 审查修复：💡-12 本脚本使用 "data/..." 相对路径，需在项目根目录
（ecg-ai-agent/）运行，否则 FileNotFoundError。
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.agent.tools.ecgfounder_classifier import ECGFounderClassifier

clf = ECGFounderClassifier()
print("工具加载 OK, 设备:", clf.device)

data_dir = Path("data/physionet2020/processed_5k")
with open(data_dir / "test_manifest.json", encoding="utf-8") as f:
    item = json.load(f)["files"][0]
sig = np.load(data_dir / item["signal_file"])
out = clf.predict(sig)
print(f"记录: {item['source']}/{item['record_id']}")
print("摘要:", out["summary"])
print("阳性(Youden阈值):", [d["name"] for d in out["positives"]])

# 4A/4J 审查修复：💡-13 补真实断言（旧版仅打印、无任何 assert，分类器
# 加载为空/无阳性也照常"通过"）
assert len(out["top_k"]) == 5, f"top_k 长度应为 5，实际 {len(out['top_k'])}"
assert out["probs"].shape == (27,), f"probs 形状应为 (27,)，实际 {out['probs'].shape}"
assert np.all((out["probs"] >= 0.0) & (out["probs"] <= 1.0)), \
    "probs 应全部落在 [0, 1] 区间"
print("冒烟断言通过 [PASS]")
