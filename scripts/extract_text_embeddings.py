#!/usr/bin/env python3
"""M1.1 — 预计算 PubMedBERT 文本嵌入（15,110 条 PTB-XL 报告，一次性）。

产出:
    data/physionet2020/text_embeddings/report_embeddings.npy  (15110, 768)
    data/physionet2020/text_embeddings/report_ids.json        [record_id, ...]
    data/physionet2020/text_embeddings/report_splits.json     {record_id: split}

CLIP 训练循环不再加载 BERT，只读本缓存（训练快 10 倍+）。
首次运行需下载 PubMedBERT (~440MB)，建议设 HF_ENDPOINT=https://hf-mirror.com

⚠️ 4C 审查修复：🟡-6 数据现状说明（不改逻辑）——
    现役 ptbxl_reports.json 为 21566 条，而本缓存 report_embeddings.npy 形状
    (15110, 768)、report_ids.json 15110 条，二者脱节：report_ids 中 185 条是
    已被 relabel 剔除的孤儿 id；reports 中 6641 条无嵌入；manifest ptb-xl 中仅
    约 14925/21604 条有文本嵌入（部分 PTB-XL 无嵌入）。CLIP/多模态按 report_ids
    对齐时只能覆盖约 7 成 PTB-XL 报告。如需覆盖全部 21566 条需重跑本脚本。
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
import src.utils.torchvision_stub  # noqa: F401  # 绕过本机损坏的 torchvision（纯文本任务用不到图像功能）
from src.context_modeling.text_encoder import TextEncoder

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", default="data/physionet2020/processed_5k/ptbxl_reports.json")
    parser.add_argument("--output-dir", default="data/physionet2020/text_embeddings")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--model", default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.reports, encoding="utf-8") as f:
        reports = json.load(f)
    ids = sorted(reports.keys())
    if args.max_records:
        ids = ids[:args.max_records]
    logger.info(f"报告数: {len(ids)}")

    logger.info("加载 PubMedBERT ...")
    enc = TextEncoder(model_name=args.model, freeze=True, pooling="mean")
    if enc.encoder is None:
        logger.error("文本编码器加载失败，请检查网络/HF_ENDPOINT")
        sys.exit(1)
    enc.eval()

    all_emb = []
    for i in tqdm(range(0, len(ids), args.batch_size), desc="Encoding"):
        batch_ids = ids[i:i + args.batch_size]
        # 清洗：空串/None → "none"（Rust tokenizer 拒收空输入；数据中已知 3 条异常报告）
        texts = [(str(reports[r].get("report") or "").strip() or "none")
                 for r in batch_ids]
        with torch.no_grad():
            emb = enc(texts)
        all_emb.append(emb.cpu().numpy().astype(np.float32))

    embs = np.concatenate(all_emb, axis=0)
    logger.info(f"嵌入矩阵: {embs.shape}")

    np.save(out_dir / "report_embeddings.npy", embs)
    with open(out_dir / "report_ids.json", "w", encoding="utf-8") as f:
        json.dump(ids, f)
    splits = {r: reports[r].get("split", "unknown") for r in ids}
    with open(out_dir / "report_splits.json", "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=1)

    # 抽查
    print("抽查:", embs[:2, :5], flush=True)
    print(f"范数均值: {np.linalg.norm(embs, axis=1).mean():.2f}", flush=True)
    logger.info(f"完成 → {out_dir}")


if __name__ == "__main__":
    main()
