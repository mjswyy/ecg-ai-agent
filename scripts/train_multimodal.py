"""⚠️ 已废弃（检查报告 1.7）：本脚本是含标签泄漏的旧版多模态实验。

问题: 末尾打印的 "Test macro_auc" 实际用 val_loader（同一验证集既选最优模型又做
"测试"评估，双重乐观）——旧 0.971 泄漏实验数字来自此脚本。
公平版替代: `train_multimodal_fair.py`（M1.3，信号-only 推理，无报告文本泄漏）。
仅保留作历史对照，勿再用于新实验。

4E/4F 审查修复：🟡-1/2/3 —— 本脚本依赖的 metadata_encoder / cross_attention 已下线
（import 即抛 RuntimeError）；且 fusion 参数 nhead 与签名 num_heads 不符、--no-text 并不真正
关闭文本分支。即 import 即崩、已不可运行，仅供存档，勿用。

架构: ECGFounder(冻结) + PubMedBERT(冻结) + CrossAttentionFusion → 27类
数据: PTB-XL 15,110 ECG-文本对
"""
from src.utils.safe_load import safe_torch_load
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from src.ecg_models.backbone.ecgfounder_net1d import Net1D
from src.context_modeling.text_encoder import TextEncoder
from src.context_modeling.metadata_encoder import MetadataEncoder
from src.context_modeling.fusion.cross_attention import CrossAttentionFusion

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================
# Multimodal Model
# ============================================================

class ECGFounderMultimodal(nn.Module):
    """ECGFounder ECG + PubMedBERT Text + CrossAttention → 27 classes."""

    def __init__(self, ecgfounder_ckpt, n_classes=27,
                 text_model="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                 fusion="cross_attention", dropout=0.2):
        super().__init__()

        # ---- ECG: ECGFounder backbone (frozen) ----
        self.ecg_encoder = Net1D(
            in_channels=12, base_filters=64, ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16, stride=2, groups_width=16,
            n_classes=150, use_bn=False, use_do=False, verbose=False,
        )
        ckpt = safe_torch_load(ecgfounder_ckpt, map_location="cpu")
        self.ecg_encoder.load_state_dict(ckpt["state_dict"], strict=True)
        # Remove classification head, keep backbone
        self.ecg_encoder.dense = nn.Identity()
        ecg_dim = ckpt["state_dict"]["dense.weight"].shape[1]  # 1024

        for p in self.ecg_encoder.parameters():
            p.requires_grad = False
        logger.info(f"ECGFounder frozen, feature dim: {ecg_dim}")

        # ---- Text: PubMedBERT (frozen) ----
        text_dim = 768
        try:
            self.text_encoder = TextEncoder(model_name=text_model, freeze=True, pooling="mean")
            text_dim = self.text_encoder.output_dim
            logger.info(f"Text encoder loaded, dim: {text_dim}")
        except Exception as e:
            logger.warning(f"Text encoder failed ({e}), using zero placeholder")
            self.text_encoder = None

        # ---- Metadata ----
        meta_dim = 128
        self.metadata_encoder = MetadataEncoder(output_dim=meta_dim)

        # ---- Fusion ----
        self.fusion = CrossAttentionFusion(
            ecg_dim=ecg_dim, text_dim=text_dim, meta_dim=meta_dim,
            hidden_dim=256, nhead=8, dropout=dropout,
        )

        # ---- Classifier ----
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, ecg, texts, ages=None, sexes=None, sources=None):
        # Pad ECG 4096 → 5000
        if ecg.shape[-1] < 5000:
            pad = 5000 - ecg.shape[-1]
            ecg = nn.functional.pad(ecg, (pad // 2, pad - pad // 2))
        elif ecg.shape[-1] > 5000:
            ecg = ecg[..., :5000]

        ecg_feat = self.ecg_encoder(ecg)  # (B, 1024)

        if self.text_encoder is not None and texts is not None and len(texts) > 0:
            text_feat = self.text_encoder(texts)  # (B, 768)
        else:
            text_feat = torch.zeros(ecg.shape[0], 768, device=ecg.device)

        meta_feat = self.metadata_encoder(ages if ages is not None else None,
                                          sexes if sexes is not None else None,
                                          sources if sources is not None else None)

        fused = self.fusion(ecg_feat, text_feat, meta_feat)
        return self.classifier(fused)

    def predict(self, x, texts=None):
        with torch.no_grad():
            return torch.sigmoid(self.forward(x, texts))


# ============================================================
# Dataset
# ============================================================

class ECGFounderTextDataset(torch.utils.data.Dataset):
    """ECG + Text + Labels 三元组."""

    def __init__(self, manifest_path, reports_path, data_dir, label_extractor):
        with open(manifest_path) as f:
            manifest = json.load(f)
        with open(reports_path, encoding="utf-8") as f:
            reports = json.load(f)

        self.data_dir = Path(data_dir)
        self.le = label_extractor
        self.samples = []

        for item in manifest["files"]:
            rid = item["record_id"]
            if rid in reports:
                self.samples.append({
                    "path": str(self.data_dir / item["signal_file"]),
                    "report": reports[rid]["report"],
                    "dx_codes": item.get("dx_codes", []),
                    "age": item.get("age"),
                    "sex": item.get("sex"),
                })

        logger.info(f"Dataset: {len(self.samples)} ECG-text pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        ecg = np.nan_to_num(np.load(s["path"]).astype(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
        if ecg.shape[0] != 12 and ecg.shape[1] == 12:
            ecg = ecg.T

        labels = self.le.encode(s["dx_codes"])
        age = float(s["age"]) if s["age"] and s["age"] > 0 else -1.0
        sex_idx = {"Male": 1, "Female": 0}.get(s.get("sex", ""), 2)

        return (torch.from_numpy(ecg),
                s["report"],
                torch.FloatTensor([age]),
                torch.LongTensor([sex_idx]),
                torch.FloatTensor(labels))


# ============================================================
# Training
# ============================================================

from src.ecg_models.trainer import ECGTrainer
from src.data_pipeline.label_extractor import LabelExtractor
from src.utils.device_utils import detect_device


def collate_fn(batch):
    ecgs, texts, ages, sexes, labels = zip(*batch)
    return (torch.stack(ecgs), list(texts),
            torch.cat(ages, dim=0), torch.cat(sexes, dim=0),
            torch.stack(labels, dim=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ecgfounder-ckpt", default="checkpoints/ECGFounder/12_lead_ECGFounder.pth")
    parser.add_argument("--data-dir", default="data/physionet2020/processed")
    parser.add_argument("--device", default=detect_device())
    parser.add_argument("--output-dir", default="/cache/output/multimodal")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-text", action="store_true",
                        help="Skip text encoder (ablation: ECG-only)")
    args = parser.parse_args()

    device = args.device
    logger.info(f"Device: {device}")

    # Verify text encoder works
    if not args.no_text:
        try:
            te = TextEncoder(model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                             freeze=True, pooling="mean")
            test_emb = te(["sinus rhythm"])
            logger.info(f"Text encoder OK, output dim: {test_emb.shape[-1]}")
            del te
        except Exception as e:
            logger.error(f"Text encoder failed: {e}")
            logger.error("Add --no-text to proceed without text branch")
            sys.exit(1)

    # Load data
    le = LabelExtractor()
    data_dir = Path(args.data_dir)
    reports_path = data_dir / "ptbxl_reports.json"

    logger.info("Loading datasets...")
    train_ds = ECGFounderTextDataset(data_dir / "train_manifest.json", reports_path, data_dir, le)
    val_ds = ECGFounderTextDataset(data_dir / "val_manifest.json", reports_path, data_dir, le)
    test_ds = ECGFounderTextDataset(data_dir / "test_manifest.json", reports_path, data_dir, le)
    logger.info(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )

    # Build model
    model = ECGFounderMultimodal(args.ecgfounder_ckpt, dropout=args.dropout)
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {trainable:,}")

    # Trainer
    trainer = ECGTrainer(model, device=device, output_dir=args.output_dir,
                         use_amp=not args.no_amp)

    # Custom training loop (handles text inputs)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for ecgs, texts, ages, sexes, labels in train_loader:
            ecgs, labels = ecgs.to(device), labels.to(device)
            ages = ages.to(device) if ages is not None else None
            sexes = sexes.to(device) if sexes is not None else None

            logits = model(ecgs, texts, ages, sexes)
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = train_loss / max(n_batches, 1)

        # Validation
        from sklearn.metrics import roc_auc_score
        model.eval()
        val_probs, val_labels = [], []
        with torch.no_grad():
            for ecgs, texts, ages, sexes, labels in val_loader:
                ecgs = ecgs.to(device)
                logits = model(ecgs, texts, ages.to(device) if ages is not None else None,
                               sexes.to(device) if sexes is not None else None)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_probs.append(probs)
                val_labels.append(labels.numpy())

        val_probs = np.concatenate(val_probs, axis=0)
        val_labels = np.concatenate(val_labels, axis=0)

        aucs = []
        for c in range(27):
            if 0 < val_labels[:, c].sum() < len(val_labels):
                aucs.append(roc_auc_score(val_labels[:, c], val_probs[:, c]))
        val_auc = float(np.mean(aucs)) if aucs else 0.0

        logger.info(f"Epoch {epoch+1}/{args.epochs} | loss={avg_loss:.4f} | val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_auc": val_auc,
            }, Path(args.output_dir) / "best_model.pt")

    logger.info(f"\nBest val_auc: {best_val_auc:.4f}")

    # Test
    logger.info("Test evaluation...")
    # Load best model
    # 3F-YELLOW-8 修复：与全仓 safe_torch_load 口径一致（旧版裸 torch.load）
    best_ckpt = safe_torch_load(Path(args.output_dir) / "best_model.pt",
                                map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    model.eval()

    test_probs, test_labels = [], []
    with torch.no_grad():
        for ecgs, texts, ages, sexes, labels in val_loader:  # use val since test doesn't have reports
            ecgs = ecgs.to(device)
            logits = model(ecgs, texts, ages.to(device), sexes.to(device))
            test_probs.append(torch.sigmoid(logits).cpu().numpy())
            test_labels.append(labels.numpy())

    test_probs = np.concatenate(test_probs, axis=0)
    test_labels = np.concatenate(test_labels, axis=0)

    aucs = []
    for c in range(27):
        if 0 < test_labels[:, c].sum() < len(test_labels):
            aucs.append(roc_auc_score(test_labels[:, c], test_probs[:, c]))
    test_auc = float(np.mean(aucs)) if aucs else 0.0
    logger.warning("⚠️ 以下 'Test macro_auc' 实为验证集分数（本脚本已废弃，"
                   "见脚本头说明；公平版请用 train_multimodal_fair.py）")
    logger.info(f"Val-as-test macro_auc: {test_auc:.4f}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
