#!/usr/bin/env python3
"""M1.2 — ECG-Text CLIP 对比预训练（公平版：文本只做监督，推理不碰文本）。

输入:
    ECG 特征缓存   data/physionet2020/ecgfounder_features/{split}_features.npy + ids.json
    文本嵌入缓存   data/physionet2020/text_embeddings/report_embeddings.npy + report_ids.json + report_splits.json

设计:
    - 训练对只用 train 划分的 (ECG, report) 对（防泄漏红线）
    - val 对只用于检索评估（不反向传播）
    - test 对完全不参与（脚本会打印红线检查）
    - 产出两个投影头: ECG 1024→512, Text 768→512（下游 M1.3 复用）

用法:
    python scripts/train_ecg_text_clip.py
    python scripts/train_ecg_text_clip.py --epochs 50 --lr 3e-4 --batch-size 256
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.context_modeling.contrastive.losses import InfoNCELoss
from src.evaluation.metrics.classification import retrieval_recall
from src.utils.device_utils import detect_device

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class CLIPPairProjectors(nn.Module):
    """缓存特征上的 CLIP 投影头。"""

    def __init__(self, ecg_dim=1024, text_dim=768, proj_dim=512, temperature=0.07):
        super().__init__()
        self.ecg_proj = nn.Sequential(
            nn.Linear(ecg_dim, proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, proj_dim), nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.loss_fn = InfoNCELoss(temperature=temperature)

    def forward(self, ecg_feat, text_feat):
        ecg_emb = nn.functional.normalize(self.ecg_proj(ecg_feat), dim=-1)
        text_emb = nn.functional.normalize(self.text_proj(text_feat), dim=-1)
        loss = self.loss_fn(ecg_emb, text_emb)
        return ecg_emb, text_emb, loss

    @torch.no_grad()
    def project_ecg(self, ecg_feat):
        return nn.functional.normalize(self.ecg_proj(ecg_feat), dim=-1)

    @torch.no_grad()
    def project_text(self, text_feat):
        return nn.functional.normalize(self.text_proj(text_feat), dim=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat-dir", default="data/physionet2020/ecgfounder_features")
    parser.add_argument("--text-dir", default="data/physionet2020/text_embeddings")
    parser.add_argument("--output-dir", default="outputs/clip_pretrained")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--proj-dim", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    # 4E/4F 审查修复：💡-5 —— 旧版无 --device 恒定 CPU 训练；新增设备参数并接入。
    parser.add_argument("--device", default=detect_device())
    args = parser.parse_args()

    device = torch.device(detect_device(args.device))
    logger.info(f"训练设备: {device}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    feat_dir = Path(args.feat_dir)
    text_dir = Path(args.text_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. 加载特征与 id ----
    ecg_feat = {s: np.load(feat_dir / f"{s}_features.npy") for s in ("train", "val", "test")}
    ecg_ids = {s: json.load(open(feat_dir / f"{s}_ids.json", encoding="utf-8"))
               for s in ("train", "val", "test")}
    text_emb = np.load(text_dir / "report_embeddings.npy")
    report_ids = json.load(open(text_dir / "report_ids.json", encoding="utf-8"))
    report_splits = json.load(open(text_dir / "report_splits.json", encoding="utf-8"))

    # ---- 2. 红线检查：报告与 ECG 记录的分割一致性 ----
    text_id2idx = {rid: i for i, rid in enumerate(report_ids)}
    for s in ("train", "val", "test"):
        n_missing = sum(1 for rid in ecg_ids[s]
                        if rid.startswith("ptb-xl_") and rid.split("_", 1)[1] not in text_id2idx)
        logger.info(f"[红线] {s} 中 ptb-xl 记录缺报告文本: {n_missing}")

    def build_pairs(split):
        feats, ids, skipped = [], [], 0
        for i, eid in enumerate(ecg_ids[split]):
            if not eid.startswith("ptb-xl_"):
                continue
            rid = eid.split("_", 1)[1]
            if rid in text_id2idx:
                # 报告自带的 split 应与 manifest 一致（双保险）
                # 4E/4F 审查修复：🟠-1 —— 旧版不一致时仅 warning 仍继续加入训练对，
                # 红线无约束力；现改为跳过该记录并计数（红线拦截），双保险真正生效。
                if report_splits.get(rid, "unknown") != split:
                    logger.warning(f"[红线] {rid} 报告 split={report_splits.get(rid)} "
                                   f"!= manifest {split}，已跳过")
                    skipped += 1
                    continue
                feats.append(i)
                ids.append(rid)
        if skipped:
            logger.warning(f"[红线] {split} 划分不一致，跳过 {skipped} 条记录")
        return feats, ids, skipped

    train_idx, train_rids, skip_train = build_pairs("train")
    val_idx, val_rids, skip_val = build_pairs("val")
    test_idx, test_rids, skip_test = build_pairs("test")
    logger.info(f"训练对 {len(train_idx)}, 验证对 {len(val_idx)}, 测试对 {len(test_idx)} (仅记录，不参与)")

    # ---- 3. 模型与优化器 ----
    model = CLIPPairProjectors(proj_dim=args.proj_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    x_train = torch.from_numpy(ecg_feat["train"][train_idx]).to(device)
    t_train = torch.from_numpy(text_emb[[text_id2idx[r] for r in train_rids]]).to(device)
    x_val = torch.from_numpy(ecg_feat["val"][val_idx]).to(device)
    t_val = torch.from_numpy(text_emb[[text_id2idx[r] for r in val_rids]]).to(device)

    # ---- 4. 训练 ----
    best_recall, best_state = 0.0, None
    n = len(x_train)
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        losses = []
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size]
            _, _, loss = model(x_train[idx], t_train[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        model.eval()
        with torch.no_grad():
            ve = model.project_ecg(x_val).cpu().numpy()
            vt = model.project_text(t_val).cpu().numpy()
        r_t2s = retrieval_recall(vt, ve, val_rids, val_rids, ks=(1, 5, 10))
        r_s2t = retrieval_recall(ve, vt, val_rids, val_rids, ks=(1, 5, 10))
        logger.info(f"epoch {epoch}: loss={np.mean(losses):.4f} "
                    f"t2s r1/r5/r10={r_t2s['r1']}/{r_t2s['r5']}/{r_t2s['r10']} "
                    f"s2t r1={r_s2t['r1']}")
        if r_t2s["r1"] + r_s2t["r1"] > best_recall:
            best_recall = r_t2s["r1"] + r_s2t["r1"]
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    # 第三轮审查 3F-R3-05：best_state None 兜底
    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save({"ecg_proj": model.ecg_proj.state_dict(),
                "text_proj": model.text_proj.state_dict(),
                "proj_dim": args.proj_dim},
               out_dir / "clip_projectors.pt")
    meta = {"n_train_pairs": len(train_idx), "n_val_pairs": len(val_idx),
            "best_sum_r1": round(best_recall, 4), "proj_dim": args.proj_dim,
            # 4E/4F 审查修复：🟠-1 —— 落盘划分不一致跳过条数，供事后审计
            "n_split_mismatch_skipped": skip_train + skip_val + skip_test}
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"完成 → {out_dir / 'clip_projectors.pt'} (best sum r1={best_recall:.4f})")


if __name__ == "__main__":
    main()
