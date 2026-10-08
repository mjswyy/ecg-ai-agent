"""ECGFounder-MLP 诊断工具 — Agent 的 classify_arrhythmia 升级版（M2.2）。

加载: ECGFounder 骨干(冻结) + 训练好的 MLP 头（outputs/ecgfounder_mlp/mlp_head.pt）
输入: 预处理后的信号 (12, 5000)（processed_5k 协议）
输出: 27 类诊断概率 + Top-K 解读 + 150 类标签映射（可选）

与 Agent 工具的差异: 用 ECGFounder 冻结特征 + MLP（test macro_auc 0.9456，
官方 27 评分类），替代旧的自训集成（0.850）。
"""

from src.utils.safe_load import safe_torch_load
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 检查报告 1.9 修复：文件位于 src/agent/tools/，需四级 parent 才到项目根
# （旧版少一级 parent 只到 src/，从任意目录直接运行时 ImportError）
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ecg_models.backbone.ecgfounder_net1d import Net1D
from src.data_pipeline.label_extractor import LabelExtractor

logger = logging.getLogger(__name__)


class ECGFounderClassifier:
    """ECGFounder(冻结) + MLP 头的 27 类 ECG 分类器。"""

    def __init__(
        self,
        ckpt_path: str = "checkpoints/ECGFounder/12_lead_ECGFounder.pth",
        head_path: str = "outputs/ecgfounder_mlp/mlp_head.pt",
        thresholds_path: str = None,   # 可选: metrics.json 里的逐类 Youden 阈值
        device: str = "cpu",
    ):
        # 第三轮审查 AGENT-LLM-O5：thresholds_path 缺省时默认加载主产物阈值
        # （旧版缺省 0.5 与逐类 Youden 阈值 0.06~0.66 脱节 → 系统性漏报）
        if thresholds_path is None:
            candidate = Path("outputs/ecgfounder_mlp/metrics.json")
            if candidate.exists():
                thresholds_path = str(candidate)
        self.device = torch.device(device)
        self.le = LabelExtractor(num_classes=27)

        # ---- ECGFounder 骨干（冻结）----
        self.backbone = Net1D(
            in_channels=12, base_filters=64, ratio=1,
            filter_list=[64, 160, 160, 400, 400, 1024, 1024],
            m_blocks_list=[2, 2, 2, 3, 3, 4, 4],
            kernel_size=16, stride=2, groups_width=16,
            n_classes=150, use_bn=False, use_do=False, verbose=False,
        )
        ckpt = safe_torch_load(ckpt_path, map_location="cpu")
        self.backbone.load_state_dict(ckpt["state_dict"], strict=True)
        self.backbone.dense = nn.Identity()  # 去分类头，保留 1024 维特征
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        # ---- MLP 头 ----
        self.head = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, 27),
        )
        sd = safe_torch_load(head_path, map_location="cpu")
        # 兼容 train_ecgfounder_head.py 的 MLPHead 包装（键带 "net." 前缀）
        if any(k.startswith("net.") for k in sd):
            sd = {k[len("net."):]: v for k, v in sd.items()}
        self.head.load_state_dict(sd)
        self.head.eval()

        # ---- 逐类 Youden 阈值（验证集定的，来自 metrics.json）----
        self.thresholds = np.full(27, 0.5, dtype=np.float32)
        if thresholds_path and Path(thresholds_path).exists():
            with open(thresholds_path, encoding="utf-8") as f:
                data = json.load(f)
            ths = data.get("metrics", {}).get("thresholds")
            if ths and len(ths) == 27:
                self.thresholds = np.asarray(ths, dtype=np.float32)

        self.backbone.to(self.device)
        self.head.to(self.device)
        logger.info(f"ECGFounderClassifier 就绪 (device={device})")

    @torch.no_grad()
    def predict(self, signal: np.ndarray = None, top_k: int = 5, **kwargs) -> dict:
        """预测 27 类概率。

        Args:
            signal: (12, 5000) 预处理信号（processed_5k 协议）。
            top_k: 返回前 K 诊断。
            **kwargs: 兼容 Agent registry 路径——memory 注入 ecg_signal/
                patient_info 等键（4A 定向复查 γ：旧签名只认 signal 导致
                registry 接入报 TypeError，旗舰分类器接入 Agent 仍断）。

        Returns:
            {probs: (27,), top_k: [{name, snomed, prob, positive}], summary}
        """
        if signal is None:
            signal = kwargs.get("ecg_signal")
        # 2D-O7 修复：坏输入（NaN/Inf）显式失败，不再静默清零后产出"可信"概率
        # 第三轮审查 AGENT-LLM-T2：补充形状/类型校验（1D/非 12 通道/None 不再崩溃）
        if signal is None or not isinstance(signal, np.ndarray):
            return {"error": "输入信号缺失或类型错误（需 np.ndarray）",
                    "top_k": [], "positives": [], "probs": None}
        if signal.ndim != 2 or signal.shape[0] != 12:
            return {"error": f"输入信号形状 {signal.shape} 非法（需 (12, L)）",
                    "top_k": [], "positives": [], "probs": None}
        if not np.isfinite(signal).all():
            return {"error": "输入信号含 NaN/Inf（信号质量异常），拒绝分类",
                    "top_k": [], "positives": [], "probs": None}
        x = torch.from_numpy(signal).unsqueeze(0).float().to(self.device)
        if x.shape[-1] < 5000:
            pad = 5000 - x.shape[-1]
            x = torch.nn.functional.pad(x, (pad // 2, pad - pad // 2))
        elif x.shape[-1] > 5000:
            x = x[..., :5000]

        feats = self.backbone(x)          # (1, 1024)
        logits = self.head(feats)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

        topk_list = []
        for idx in np.argsort(-probs)[:top_k]:
            name = self.le.class_names[idx] if idx < len(self.le.class_names) else f"C{idx}"
            snomed = self.le.idx_to_snomed.get(idx)
            topk_list.append({
                "name": name,
                "snomed": snomed,
                "prob": float(probs[idx]),
                "positive": bool(probs[idx] >= self.thresholds[idx]),
            })
        # 第三轮审查 AGENT-LLM-O5：positives 对全部 27 类按阈值判定
        # （旧版仅从 top_k 子集派生 → 概率≥阈值但排名第 6 名后的类被丢弃）
        positives = [
            {"name": (self.le.class_names[i]
                      if i < len(self.le.class_names) else f"C{i}"),
             "snomed": self.le.idx_to_snomed.get(i),
             "prob": float(probs[i]),
             "positive": True}
            for i in range(len(probs))
            if probs[i] >= self.thresholds[i]
        ]
        return {
            "probs": probs,
            "top_k": topk_list,
            "positives": positives,
            "summary": "; ".join(f"{d['name']}({d['prob']:.0%})" for d in topk_list),
        }

    def tool_schema(self) -> dict:
        return {
            "name": "classify_arrhythmia",
            "description": "27 类心律失常/传导/形态多标签分类（ECGFounder 基础模型冻结特征 + MLP 头，"
                           "test macro_auc 0.9456，官方 PhysioNet 2020 评分类）。输入 12 导联预处理信号。",
            "parameters": {
                "type": "object",
                "properties": {
                    "signal": {"type": "array", "description": "(12, 5000) 预处理信号"},
                    "top_k": {"type": "integer", "default": 5},
                },
            },
        }


if __name__ == "__main__":
    # 冒烟: 随机取测试集一条
    import json as _json
    logging.basicConfig(level=logging.INFO)

    data_dir = Path("data/physionet2020/processed_5k")
    with open(data_dir / "test_manifest.json", encoding="utf-8") as f:
        item = _json.load(f)["files"][0]
    sig = np.load(data_dir / item["signal_file"])
    clf = ECGFounderClassifier()
    out = clf.predict(sig)
    print(f"记录: {item['source']}/{item['record_id']}  真实标签: {item.get('dx_codes')}")
    for d in out["top_k"]:
        flag = "✓" if d["positive"] else " "
        print(f"  {flag} {d['name']:35s} {d['prob']:.3f}")
