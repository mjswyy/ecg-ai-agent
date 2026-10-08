#!/usr/bin/env python3
"""M1.3 — 公平下游评测（信号-only 推理，文本绝不进入分类器）。

变体对照（全部只用缓存特征 + 元数据）:
    A:  ECG 特征(1024) → MLP                      [基线，对照 M0.2 的 0.9214]
    B:  ECG 特征 → CLIP 投影(512) → MLP           [M1.2 预训练增益]
    C:  ECG 特征(1024) + 元数据(10维含数据源) → MLP  [情境元数据增益]
    C2: ECG 特征(1024) + 临床元数据(4维: 年龄+性别) → MLP  [排除数据源，纯临床情境]
    D:  CLIP 投影(512) + 元数据(10维) → MLP
    D2: CLIP 投影(512) + 临床元数据(4维) → MLP

防泄漏红线:
    - 文本嵌入只在 CLIP 预训练(train 对)使用过一次；本脚本推理时无任何文本输入
    - 划分沿用旧版 manifests；按 record 级对齐
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
# 5F 审查（🟠-1）：safe_torch_load 的 import 必须在 sys.path.insert 之后
# （旧版位于其前，`python scripts/train_multimodal_fair.py` 直接 ModuleNotFoundError）
from src.utils.safe_load import safe_torch_load
from src.evaluation.metrics.classification import evaluate, youden_thresholds

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SOURCES = ["cpsc_2018", "cpsc_2018_extra", "georgia", "ptb", "ptb-xl",
           "st_petersburg_incart"]


# ------------------------------------------------------------
# 元数据编码（年龄/性别/数据源，纯表格信息，无文本）
# meta_mode: "full"=年龄+性别+数据源(10维)；"clinical"=仅年龄+性别(4维，无数据源，排除数据集成员泄漏质疑)
# ------------------------------------------------------------
def build_metadata_matrix(manifest_dir: Path, meta_mode="full"):
    metas = {}
    for s in ("train", "val", "test"):
        with open(manifest_dir / f"{s}_manifest.json", encoding="utf-8") as f:
            for m in json.load(f)["files"]:
                key = f"{m['source']}_{m['record_id']}"
                age = m.get("age")
                # 4E/4F 审查修复：💡-4 —— 字符串型年龄（如 JSON "57"）此前被
                # isinstance 判为 False → 全部静默归一化为 -1.0（未知）。改为对
                # 数值字符串做 float() 兜底，仅真正缺失/非法才判未知。
                try:
                    age_norm = float(age) / 100.0
                except (TypeError, ValueError):
                    age_norm = -1.0
                sex = (m.get("sex") or "Unknown").lower()
                sex_male = 1.0 if sex in ("male", "m") else 0.0
                sex_female = 1.0 if sex in ("female", "f") else 0.0
                sex_unknown = 1.0 if sex_male == 0 and sex_female == 0 else 0.0
                base = [age_norm, sex_male, sex_female, sex_unknown]
                if meta_mode == "full":
                    src = m["source"]
                    base += [1.0 if src == sname else 0.0 for sname in SOURCES]
                metas[key] = base
    return metas


class Head(nn.Module):
    """2 层 MLP 分类头（BCE + 标签平滑训练）。"""

    def __init__(self, in_dim, n_classes=27, hidden=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_head(x_train, y_train, x_val, y_val, lr=1e-3, epochs=50,
               batch_size=256, seed=42, label=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    x_train = torch.from_numpy(x_train)
    y_train = torch.from_numpy(y_train)
    x_val = torch.from_numpy(x_val)
    y_val = torch.from_numpy(y_val)
    model = Head(x_train.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    loss_fn = nn.BCEWithLogitsLoss()
    n = len(x_train)
    best_val, best_state, patience = 0.0, None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            logits = model(x_train[idx])
            # 第三轮审查 3F-R3-13：均匀标签平滑 ε=0.1（正类 0.95/负类 0.05）
            loss = loss_fn(logits, y_train[idx] * 0.9 + 0.05)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            vp = torch.sigmoid(model(x_val)).numpy()
        from sklearn.metrics import roc_auc_score
        aucs = [roc_auc_score(y_val[:, c].numpy(), vp[:, c])
                for c in range(y_val.shape[1])
                if 0 < y_val[:, c].sum() < len(y_val)]
        val_auc = float(np.mean(aucs)) if aucs else 0.0
        if val_auc > best_val + 1e-4:
            best_val, patience = val_auc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                logger.info(f"[{label}] 早停 @ epoch {epoch} (best val_auc={best_val:.4f})")
                break
    # 第三轮审查 3F-R3-05：best_state None 兜底
    if best_state is None:
        best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_val


def run_variant(name, x_train, x_val, x_test, y_train, y_val, y_test):
    model, best_val = train_head(x_train, y_train, x_val, y_val, label=name)
    model.eval()
    with torch.no_grad():
        vp = torch.sigmoid(model(torch.from_numpy(x_val))).numpy()
        tp = torch.sigmoid(model(torch.from_numpy(x_test))).numpy()
    ths = youden_thresholds(y_val, vp)
    m = evaluate(y_test, tp, thresholds=ths)
    logger.info(f"[{name}] test macro_auc={m['macro_auc']} (CI {m['auc_ci']}), "
                f"F1@Youden={m['macro_f1_youden']}, Top-5={m['top5']}")
    return {"best_val_auc": round(best_val, 4), **m}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat-dir", default="data/physionet2020/ecgfounder_features")
    parser.add_argument("--manifest-dir", default="data/physionet2020/processed_5k")
    parser.add_argument("--clip-ckpt", default="outputs/clip_pretrained/clip_projectors.pt")
    parser.add_argument("--output", default="outputs/multimodal_fair/metrics.json")
    args = parser.parse_args()

    feat_dir = Path(args.feat_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    x = {s: np.load(feat_dir / f"{s}_features.npy") for s in ("train", "val", "test")}
    y = {s: np.load(feat_dir / f"{s}_labels.npy") for s in ("train", "val", "test")}
    ids = {s: json.load(open(feat_dir / f"{s}_ids.json", encoding="utf-8"))
           for s in ("train", "val", "test")}

    # 两类元数据：full（含数据源 10 维）与 clinical（仅年龄+性别 4 维）
    meta_full = build_metadata_matrix(Path(args.manifest_dir), meta_mode="full")
    meta_clin = build_metadata_matrix(Path(args.manifest_dir), meta_mode="clinical")

    meta = {}
    for s in ("train", "val", "test"):
        mf = np.array([meta_full.get(k, [-1, 0, 0, 1] + [0] * 6) for k in ids[s]],
                      dtype=np.float32)
        mc = np.array([meta_clin.get(k, [-1, 0, 0, 1]) for k in ids[s]],
                      dtype=np.float32)
        meta[s] = {"full": mf, "clinical": mc}
        logger.info(f"[元数据] {s}: full {mf.shape}, clinical {mc.shape}, "
                    f"缺元数据比例 {(mf[:, 0] < 0).mean():.1%}")

    results = {}

    # ---- A: 基线 ECG 特征 ----
    results["A_ecg_mlp"] = run_variant("A", x["train"], x["val"], x["test"],
                                       y["train"], y["val"], y["test"])

    # ---- C/C2: 元数据变体（不依赖 CLIP，第三轮审查 3F-R3-04 移出门控，
    #      旧版放 if clip_path.exists() 内导致干净环境静默丢失）----
    results["C_ecg_meta_mlp"] = run_variant(
        "C", np.concatenate([x["train"], meta["train"]["full"]], 1),
        np.concatenate([x["val"], meta["val"]["full"]], 1),
        np.concatenate([x["test"], meta["test"]["full"]], 1),
        y["train"], y["val"], y["test"])

    results["C2_ecg_clinical_mlp"] = run_variant(
        "C2", np.concatenate([x["train"], meta["train"]["clinical"]], 1),
        np.concatenate([x["val"], meta["val"]["clinical"]], 1),
        np.concatenate([x["test"], meta["test"]["clinical"]], 1),
        y["train"], y["val"], y["test"])

    # ---- B/D: 依赖 CLIP 投影 ----
    clip_path = Path(args.clip_ckpt)
    if clip_path.exists():
        ckpt = safe_torch_load(clip_path, map_location="cpu")
        proj_dim = ckpt["proj_dim"]
        # 4E/4F 审查修复：💡-3 —— 不再硬编码 ECG 特征维 1024，改为从缓存特征
        # 维度推导，避免特征维度变更时 load_state_dict 形状不匹配崩溃。
        ecg_dim = int(x["train"].shape[1])
        ecg_proj = nn.Sequential(nn.Linear(ecg_dim, proj_dim), nn.ReLU(inplace=True),
                                 nn.Linear(proj_dim, proj_dim))
        ecg_proj.load_state_dict(ckpt["ecg_proj"])
        ecg_proj.eval()
        with torch.no_grad():
            xp = {s: nn.functional.normalize(
                ecg_proj(torch.from_numpy(x[s])), dim=-1).numpy()
                for s in ("train", "val", "test")}

        results["B_clip_ecg_mlp"] = run_variant(
            "B", xp["train"], xp["val"], xp["test"],
            y["train"], y["val"], y["test"])

        results["D_clip_meta_mlp"] = run_variant(
            "D", np.concatenate([xp["train"], meta["train"]["full"]], 1),
            np.concatenate([xp["val"], meta["val"]["full"]], 1),
            np.concatenate([xp["test"], meta["test"]["full"]], 1),
            y["train"], y["val"], y["test"])

        results["D2_clip_clinical_mlp"] = run_variant(
            "D2", np.concatenate([xp["train"], meta["train"]["clinical"]], 1),
            np.concatenate([xp["val"], meta["val"]["clinical"]], 1),
            np.concatenate([xp["test"], meta["test"]["clinical"]], 1),
            y["train"], y["val"], y["test"])
    else:
        logger.warning(f"未找到 CLIP 投影 {clip_path}，跳过 B/D 变体（C/C2 不受影响）")

    out = {"red_line": "signal-only inference; no report text in any variant",
           "variants": results}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"结果保存 → {out_path}")


if __name__ == "__main__":
    main()
