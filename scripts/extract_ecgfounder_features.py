#!/usr/bin/env python3
"""
ECGFounder 冻结特征提取（里程碑 M0.2）

对 processed_5k 的 train/val/test 全部记录，用 ECGFounder 骨干（冻结、去分类头）
提取 1024 维特征，存为 npy，供后续线性探针/MLP 头/多模态实验直接读取（只算一次）。

用法:
    python scripts/extract_ecgfounder_features.py            # 全量
    python scripts/extract_ecgfounder_features.py --max-records 50 --batch-size 16
"""

from src.utils.safe_load import safe_torch_load
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.ecg_models.backbone.ecgfounder_net1d import Net1D

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_ecgfounder(ckpt_path: str):
    model = Net1D(
        in_channels=12, base_filters=64, ratio=1,
        filter_list=[64, 160, 160, 400, 400, 1024, 1024],
        m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
        kernel_size=16, stride=2, groups_width=16,
        n_classes=150, use_bn=False, use_do=False, verbose=False,
    )
    ckpt = safe_torch_load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.eval()
    feature_dim = ckpt["state_dict"]["dense.weight"].shape[1]
    return model, feature_dim


@torch.no_grad()
def extract_split(model, data_dir, manifest_path, max_records, batch_size):
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)["files"]
    files = manifest if max_records is None else manifest[:max_records]

    feats_list, labels_list, ids_list = [], [], []
    buf_x, buf_ids, buf_labels = [], [], []

    def flush():
        if not buf_x:
            return
        x = torch.from_numpy(np.stack(buf_x))
        _ = model(x)  # 触发 hook 填充 captured
        buf_feats = np.concatenate(captured, axis=0)
        captured.clear()
        feats_list.append(buf_feats)
        labels_list.append(np.stack(buf_labels))
        ids_list.extend(buf_ids)
        buf_x.clear()
        buf_ids.clear()
        buf_labels.clear()

    captured = []
    def hook_fn(module, inp, out):
        captured.append(inp[0].cpu().numpy())

    handle = model.dense.register_forward_hook(hook_fn)

    data_dir = Path(data_dir)
    for item in files:
        p = data_dir / item["signal_file"]
        if not p.exists():
            logger.warning(f"缺失: {p}")
            continue
        x = np.nan_to_num(np.load(p).astype(np.float32),
                          nan=0.0, posinf=0.0, neginf=0.0)
        if x.shape[0] != 12 and x.shape[1] == 12:
            x = x.T
        buf_x.append(x)
        buf_ids.append(f"{item['source']}_{item['record_id']}")
        buf_labels.append(np.asarray(item.get("labels", [0] * 27), dtype=np.float32))
        if len(buf_x) >= batch_size:
            flush()
    flush()
    handle.remove()

    return (np.concatenate(feats_list, axis=0),
            np.concatenate(labels_list, axis=0),
            ids_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/physionet2020/processed_5k")
    parser.add_argument("--output-dir", default="data/physionet2020/ecgfounder_features")
    parser.add_argument("--ckpt", default="checkpoints/ECGFounder/12_lead_ECGFounder.pth")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("加载 ECGFounder...")
    model, feature_dim = load_ecgfounder(args.ckpt)
    logger.info(f"特征维度: {feature_dim}")

    for split in args.splits:
        logger.info(f"提取 {split} ...")
        feats, labels, ids = extract_split(
            model, args.data_dir,
            Path(args.data_dir) / f"{split}_manifest.json",
            args.max_records, args.batch_size)
        np.save(out_dir / f"{split}_features.npy", feats.astype(np.float32))
        np.save(out_dir / f"{split}_labels.npy", labels.astype(np.float32))
        with open(out_dir / f"{split}_ids.json", "w", encoding="utf-8") as f:
            json.dump(ids, f)
        logger.info(f"  {split}: features {feats.shape}, labels {labels.shape}")

    logger.info(f"完成。输出目录: {out_dir}")


if __name__ == "__main__":
    main()
