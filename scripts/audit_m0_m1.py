# -*- coding: utf-8 -*-
"""M0/M1 成果审查脚本（抽样检查，流式输出）"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

print("=== 2. 处理时长分布（判断零填充/裁剪比例）===", flush=True)
with open("data/physionet2020/processed_5k/train_manifest.json", encoding="utf-8") as f:
    files = json.load(f)["files"]
durs = [m.get("duration_original") for m in files if m.get("duration_original")]
durs = [d for d in durs if isinstance(d, (int, float))]
d = np.array(durs)
print(f"  时长统计: min={d.min():.2f}s, median={np.median(d):.2f}s, max={d.max():.2f}s", flush=True)
print(f"  <9.5s(零填充): {(d < 9.5).sum()} 条;  >10.5s(裁剪): {(d > 10.5).sum()} 条", flush=True)

print("\n=== 3. 关键产物存在性 ===", flush=True)
products = [
    "outputs/ecgfounder_mlp/metrics.json",
    "outputs/ecgfounder_mlp/mlp_head.pt",
    "outputs/clip_pretrained/clip_projectors.pt",
    "outputs/clip_pretrained/meta.json",
    "outputs/multimodal_fair/metrics.json",
    "data/physionet2020/ecgfounder_features/train_features.npy",
    "data/physionet2020/text_embeddings/report_embeddings.npy",
    "data/physionet2020/text_embeddings/report_ids.json",
    "data/physionet2020/text_embeddings/report_splits.json",
]
for p in products:
    pp = Path(p)
    print(f"  {'OK ' if pp.exists() else '缺失'} {p}  {pp.stat().st_size/1024/1024:.1f}MB" if pp.exists() else f"  缺失 {p}", flush=True)

print("\n=== 4. 指标文件内容核对 ===", flush=True)
mm = json.load(open("outputs/multimodal_fair/metrics.json", encoding="utf-8"))
print(f"  红线声明: {mm.get('red_line')}", flush=True)
for k, v in mm["variants"].items():
    print(f"  {k}: macro_auc={v['macro_auc']}, CI={v['auc_ci']}, Top-5={v['top5']}", flush=True)
clip_meta = json.load(open("outputs/clip_pretrained/meta.json", encoding="utf-8"))
print(f"  CLIP meta: {clip_meta}", flush=True)

print("\n=== 5. 特征与嵌入数值健康度 ===", flush=True)
xf = np.load("data/physionet2020/ecgfounder_features/test_features.npy")
print(f"  ECG特征 test: shape={xf.shape}, mean={xf.mean():.4f}, std={xf.std():.4f}, NaN={np.isnan(xf).sum()}", flush=True)
te = np.load("data/physionet2020/text_embeddings/report_embeddings.npy")
print(f"  文本嵌入: shape={te.shape}, mean={te.mean():.4f}, std={te.std():.4f}, NaN={np.isnan(te).sum()}", flush=True)
tf = np.load("data/physionet2020/ecgfounder_features/train_features.npy")
print(f"  ECG特征 train: shape={tf.shape}, NaN={np.isnan(tf).sum()}", flush=True)

print("\n=== 6. 文本嵌入 id 与划分一致性（红线复检）===", flush=True)
report_splits = json.load(open("data/physionet2020/text_embeddings/report_splits.json", encoding="utf-8"))
from collections import Counter
sc = Counter(report_splits.values())
print(f"  报告 split 分布: {dict(sc)}", flush=True)
# 与 manifests 交叉验证
old_split = {}
for s in ("train", "val", "test"):
    with open(f"data/physionet2020/processed_5k/{s}_manifest.json", encoding="utf-8") as f:
        for m in json.load(f)["files"]:
            if m["source"] == "ptb-xl":
                old_split[m["record_id"]] = s
mismatch = [rid for rid, s in report_splits.items()
            if rid in old_split and old_split[rid] != s]
print(f"  报告自带 split 与 manifest 不一致: {len(mismatch)} 条", flush=True)
if mismatch:
    print(f"    示例: {mismatch[:5]}", flush=True)

print("\n=== 7. 训练对数量核对（CLIP 防泄漏复检）===", flush=True)
for s in ("train", "val", "test"):
    ids = json.load(open(f"data/physionet2020/ecgfounder_features/{s}_ids.json", encoding="utf-8"))
    ptb = [i for i in ids if i.startswith("ptb-xl_")]
    report_ids = set(json.load(open("data/physionet2020/text_embeddings/report_ids.json", encoding="utf-8")))
    with_report = sum(1 for i in ptb if i.split("_", 1)[1] in report_ids)
    print(f"  {s}: ptb-xl 记录 {len(ptb)}, 有报告 {with_report}", flush=True)
