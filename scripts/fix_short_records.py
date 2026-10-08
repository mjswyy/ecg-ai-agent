# -*- coding: utf-8 -*-
"""第三轮审查 3B-DP-T2 修复执行：短记录的信号与特征缓存重建。

5C 审查（🟡-2）：短记录真实数量为 54 条（= 45 条旧列表 + 9 条漏网），
旧 docstring"49 条"与 preprocess 注释"48"均不准确。

步骤：
1. 从原始 WFDB（ECGLoader）重载全部 54 条记录
2. 走 ECGFounder 官方协议（reorder→bandpass→resample→segment→z-score 修复版）
3. 覆盖 processed_5k 的信号 npy
4. 用 ECGFounder 骨干重提取特征，按 id 替换 ecgfounder_features 对应行
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.loader import ECGLoader
from scripts.preprocess_data_5k import (reorder_leads, filter_bandpass,
                                        resample_to_500, segment_to_5000,
                                        official_zscore, _default_raw_dir)

# 4C 审查修复：🟡-7 —— 删除硬编码 Desktop\ECG 绝对路径，改用与
# preprocess_data_5k.py 一致的 _default_raw_dir() 自动探测。
_raw = _default_raw_dir()
if not _raw:
    sys.exit("未探测到原始数据目录——请设置 --raw-dir 或修正探测逻辑（4C 审查修复：🟡-7）")
RAW = Path(_raw)
DATA = Path("data/physionet2020/processed_5k")
FEAT = Path("data/physionet2020/ecgfounder_features")

# 短记录列表（5C 审查 🟠-1 补全：真实短记录共 54 条 = 45 + 9 条漏网。
# 9 条漏网：A1872/A2695/A4229/Q0561 恒定非零填充（旧 z-score bug）、
# A0240/A0626/A4085/A5524/Q0045 缺 1 样本零填充；8 train + 1 val）
SHORT = [
    ("train", "A0798", "cpsc_2018"),
    ("train", "E00841", "georgia"), ("train", "E00843", "georgia"),
    ("train", "E00857", "georgia"), ("train", "E00906", "georgia"),
    ("train", "E00911", "georgia"), ("train", "E00915", "georgia"),
    ("train", "E00947", "georgia"), ("train", "E01064", "georgia"),
    ("train", "E08366", "georgia"), ("train", "E08668", "georgia"),
    ("train", "E08686", "georgia"), ("train", "E08717", "georgia"),
    ("train", "E08744", "georgia"), ("train", "E08767", "georgia"),
    ("train", "E08800", "georgia"), ("train", "E08824", "georgia"),
    ("train", "E08863", "georgia"), ("train", "E08864", "georgia"),
    ("train", "E09052", "georgia"), ("train", "E09063", "georgia"),
    ("train", "E09067", "georgia"), ("train", "E09083", "georgia"),
    ("train", "E09881", "georgia"), ("train", "E09904", "georgia"),
    ("train", "E09923", "georgia"), ("train", "E09943", "georgia"),
    ("train", "E09948", "georgia"), ("train", "E09983", "georgia"),
    ("train", "E09989", "georgia"), ("train", "E10002", "georgia"),
    ("train", "E10015", "georgia"), ("train", "E10035", "georgia"),
    ("train", "E10113", "georgia"), ("train", "E10308", "georgia"),
    ("train", "E10311", "georgia"), ("train", "E10319", "georgia"),
    ("val", "E08822", "georgia"), ("val", "E08876", "georgia"),
    ("val", "E09965", "georgia"), ("val", "E09988", "georgia"),
    ("val", "E10018", "georgia"),
    ("test", "E09971", "georgia"), ("test", "E09980", "georgia"),
    ("test", "E10320", "georgia"),
    # ---- 5C 审查 🟠-1 补全的 9 条（8 train + 1 val）----
    ("train", "A0240", "cpsc_2018"), ("train", "A0626", "cpsc_2018"),
    ("train", "A1872", "cpsc_2018"), ("train", "A2695", "cpsc_2018"),
    ("train", "A4085", "cpsc_2018"), ("train", "A4229", "cpsc_2018"),
    ("train", "Q0045", "cpsc_2018_extra"), ("train", "Q0561", "cpsc_2018_extra"),
    ("val", "A5524", "cpsc_2018"),
]

loader = ECGLoader(RAW)
ok = 0
for split, rid, src in SHORT:
    try:
        sample = loader.load_record(f"{src}/g1/{rid}" if src == "cpsc_2018" else f"{src}/{rid}")
        if sample is None:
            # 尝试裸名（loader 支持 rglob）
            sample = loader.load_record(rid)
    except Exception:
        sample = None
    if sample is None:
        print(f"⚠️ 无法加载 {src}/{rid}，跳过")
        continue
    signal, _ = reorder_leads(sample.signal, sample.lead_names)
    filtered = filter_bandpass(signal, sample.fs)
    resampled = resample_to_500(filtered, sample.fs)
    seg, pl, pr = segment_to_5000(resampled)
    normalized = official_zscore(seg, pl, pr)
    np.save(DATA / f"{src}_{rid}.npy", normalized)
    ok += 1
print(f"信号重建: {ok}/{len(SHORT)}")

# ---- 特征缓存按 id 替换 ----
from src.ecg_models.backbone.ecgfounder_net1d import Net1D
import torch
from src.utils.safe_load import safe_torch_load

model = Net1D(in_channels=12, base_filters=64, ratio=1,
              filter_list=[64, 160, 160, 400, 400, 1024, 1024],
              m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
              kernel_size=16, stride=2, groups_width=16,
              n_classes=150, use_bn=False, use_do=False, verbose=False)
ckpt = safe_torch_load("checkpoints/ECGFounder/12_lead_ECGFounder.pth", map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=True)
model.dense = torch.nn.Identity()
model.eval()

id2loc = {}
for s in ("train", "val", "test"):
    ids = json.load(open(FEAT / f"{s}_ids.json", encoding="utf-8"))
    for i, eid in enumerate(ids):
        id2loc[eid] = (s, i)

captured = []


def hook_fn(module, inp, out):
    captured.append(inp[0].cpu().numpy())


handle = model.dense.register_forward_hook(hook_fn)
n_feat = 0
for split, rid, src in SHORT:
    eid = f"{src}_{rid}"
    if eid not in id2loc:
        print(f"⚠️ 特征缓存无 {eid}，跳过")
        continue
    sig = np.load(DATA / f"{src}_{rid}.npy").astype(np.float32)
    captured.clear()
    with torch.no_grad():
        _ = model(torch.from_numpy(sig).unsqueeze(0))
    feat = np.concatenate(captured, axis=0)
    s, i = id2loc[eid]
    feats = np.load(FEAT / f"{s}_features.npy", mmap_mode="r+")
    feats[i] = feat
    n_feat += 1
handle.remove()
print(f"特征更新: {n_feat} 行（mmap 直接写回磁盘）")
print("完成")
