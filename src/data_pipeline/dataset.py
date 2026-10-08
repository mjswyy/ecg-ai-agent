"""
PyTorch Dataset 和数据模块 — 3种数据集 + 1个DataModule。

提供:
    - ECGDataset: 有监督多标签分类训练（加载.npy + manifest JSON）
    - ECGContrastiveDataset: SimCLR 对比预训练（生成两个增强视图）
    - ECGDatasetForAgent: Agent 推理/评估（返回完整病人上下文）
    - ECGDataModule: 统一 DataLoader 管理（兼容独立训练和 PyTorch Lightning）

使用示例:
    # 有监督训练
    ds = ECGDataset("data/physionet2020/processed", split="train", augment=True)
    signal, labels = ds[0]  # → (12,4096) tensor, (27,) tensor

    # 对比学习
    cs = ECGContrastiveDataset(".../processed", augmentor, target_length=4096)
    view1, view2 = cs[0]  # 同一信号的两个增强视图

    # Agent 推理
    dsa = ECGDatasetForAgent(".../processed", "test_manifest.json", le)
    item = dsa[0]  # → {record_id, signal, age, sex, dx_codes, labels}

    # DataLoader 管理
    dm = ECGDataModule(".../processed", batch_size=128, augmentor=aug, label_extractor=le)
    dm.setup()
    for signals, labels in dm.train_dataloader(): ...
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)


class ECGDataset(Dataset):
    """有监督多标签 ECG 分类数据集。

    从预处理后的 .npy 文件和 manifest JSON 中加载数据。
    自动处理形状校正、padding/crop、和在线数据增强。

    参数:
        data_dir: 预处理数据目录。
        split: 数据集划分 ("train" / "val" / "test")。
        manifest_file: 自定义 manifest 路径（默认自动从 split 推断）。
        augment: 是否启用数据增强。
        augmentor: ECGAugmentor 实例。
        label_extractor: LabelExtractor 实例（用于在线标签编码）。
        target_length: 目标信号长度（默认 4096）。
        return_metadata: 是否返回元数据。
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        manifest_file: Optional[str] = None,
        augment: bool = False,
        augmentor: Optional["ECGAugmentor"] = None,
        label_extractor: Optional["LabelExtractor"] = None,
        target_length: int = 4096,
        return_metadata: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment
        self.augmentor = augmentor
        self.label_extractor = label_extractor
        self.target_length = target_length
        self.return_metadata = return_metadata

        # 4B 审查修复：🟠-4 全零标签告警节流开关（每个数据集实例只告警一次）
        self._warned_allzero = False
        # 5B 审查（🟠-2）：静默居中裁剪告警节流——权威信号 (12,5000) 被默认
        # target_length=4096 裁剪时每侧丢 452 样本，旧版无任何提示
        self._warned_crop = False

        # 加载 manifest JSON
        if manifest_file:
            manifest_path = Path(manifest_file)
        else:
            manifest_path = self.data_dir / f"{split}_manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest 文件不存在: {manifest_path}。请先运行预处理。"
            )

        with open(manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.file_list = self.manifest.get("files", [])
        logger.info(f"ECGDataset [{split}]: 加载了 {len(self.file_list)} 个样本")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, Dict],
    ]:
        """获取一个样本。

        返回:
            如果 return_metadata=False: (signal, labels) 张量
            如果 return_metadata=True:  (signal, labels, metadata) 元组
        """
        record = self.file_list[idx]
        signal_path = self.data_dir / record["signal_file"]

        # 加载信号，替换 NaN/Inf 为 0（NPU 上 NaN 会污染梯度）
        signal = np.nan_to_num(np.load(signal_path).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # === 形状校正 ===
        # 正确的: (12, L) → 不处理
        # 转置的: (L, 12) → 自动转回来
        # 异常的: (1, L) 等 → 抛出 ValueError
        # 4B 审查修复：💡-3 通道数合理性判断——合法通道数为 12（标准）或 1
        # （单导联，在下方 elif 抛错）。仅当 shape[1]==12 且 shape[0] 明显为
        # 时间轴（≥100 采样，与 ECGLoader 校验阈值一致）才转置；否则 (13,12)
        # 这类"13 导联/12 采样"畸形信号会被误转置为 (12,13) 通过校验。
        if signal.ndim == 2 and signal.shape[1] == 12 and signal.shape[0] >= 100:
            signal = signal.T
        elif signal.ndim != 2 or signal.shape[0] != 12:
            raise ValueError(
                f"期望 (12, L) 形状的信号，实际得到 {signal.shape}。"
                f"文件: {record.get('signal_file', 'unknown')}"
            )

        # === 对齐到目标长度 ===
        if signal.shape[1] != self.target_length:
            if signal.shape[1] > self.target_length:
                # 5B 审查（🟠-2）：居中裁剪显式告警一次（权威协议 5000 样本；
                # target_length=4096 是历史训练协议，每侧丢弃 (L-4096)//2 样本）
                if not self._warned_crop:
                    self._warned_crop = True
                    logger.warning(
                        f"ECGDataset 将信号从 {signal.shape[1]} 居中裁剪到 "
                        f"{self.target_length}（每侧丢弃 "
                        f"{(signal.shape[1] - self.target_length) // 2} 样本）——"
                        f"target_length=4096 为历史训练协议，权威预处理为 5000，"
                        f"请确认所用模型协议一致（仅告警一次）")
                start = (signal.shape[1] - self.target_length) // 2
                signal = signal[:, start:start + self.target_length]
            else:
                pad = self.target_length - signal.shape[1]
                signal = np.pad(
                    signal, ((0, 0), (pad // 2, pad - pad // 2)),
                    mode="constant",
                )

        # === 数据增强（仅训练时） ===
        if self.augment and self.augmentor is not None and self.split == "train":
            signal = self.augmentor(signal)

        signal_tensor = torch.from_numpy(signal)

        # === 加载标签 ===
        labels = np.array(record.get("labels", []), dtype=np.float32)
        if len(labels) == 0 and "dx_codes" in record and self.label_extractor:
            # 如果没有预编码标签但有原始代码，在线编码
            labels = self.label_extractor.encode(
                record["dx_codes"], format="multi_hot"
            )
        if len(labels) == 0:
            # 第三轮审查 3B-DP-Y1：标签缺失且无法编码时回退全零向量并告警
            # （旧版产出形状 (0,) 张量，下游 BCE 损失崩溃）
            n_cls = getattr(self.label_extractor, "num_classes", None) or 27
            logger.warning(f"{record.get('record_id', '?')}: 无标签且无法编码，回退全零 {n_cls} 维")
            labels = np.zeros(n_cls, dtype=np.float32)

        # 4B 审查修复：🟠-3 预编码 labels 长度校验（3B 审查编号 3B-DP-Y1）——
        # manifest 预编码 labels 若来自旧标签体系（旧 24 类/自定义 27 类/150 类），
        # 会与 27 维 MLP 头 shape 不匹配或维度错位；长度不符直接抛错。
        n_cls = getattr(self.label_extractor, "num_classes", None) or 27
        if len(labels) != n_cls:
            raise ValueError(
                f"标签维度 {tuple(labels.shape)} 与 num_classes={n_cls} 不一致，"
                f"记录 {record.get('record_id', '?')}：manifest 预编码 labels 可能来自"
                f"旧标签体系，请重新 relabel 或移除预编码字段。"
            )

        # 4B 审查修复：🟠-4 dx_codes 全部为非评分类 → encode 返回全零 27 维，
        # 该记录被当作"27 类全阴性"参与 BCE（把"无评分类标签"与"全阴性"混同）。
        # 旧版此路径无任何告警；现节流告警一次（每数据集实例仅一次）。
        # 4B 定向复查（α）：显式 dx_codes=[] 也走全零路径，一并覆盖告警
        if float(np.sum(labels)) == 0.0 and not self._warned_allzero:
            self._warned_allzero = True
            logger.warning(
                "存在全零标签向量（dx_codes 缺失或全部为非评分类），"
                "此类记录被当作全阴性样本参与训练；与\"无标签\"语义混同，仅告警一次。"
            )

        labels_tensor = torch.from_numpy(labels)

        if self.return_metadata:
            metadata = {
                "record_id": record.get("record_id", ""),
                "source": record.get("source", "unknown"),
                "age": record.get("age"),
                "sex": record.get("sex", "Unknown"),
                "dx_codes": record.get("dx_codes", []),
            }
            return signal_tensor, labels_tensor, metadata

        return signal_tensor, labels_tensor


class ECGContrastiveDataset(Dataset):
    """SimCLR 风格的对比预训练数据集。

    对同一 ECG 信号生成两个不同的增强视图作为正样本对。
    负样本来自 batch 内其他记录（由 InfoNCE loss 处理）。

    参数:
        data_dir: 预处理数据目录（扫描所有 .npy 文件）。
        augmentor: ECGAugmentor 实例（必须启用 segment_shuffle）。
        target_length: 目标长度（自动 cropping/padding）。
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        augmentor: "ECGAugmentor",
        target_length: int = 4096,
        split: str = "train",
        allow_unscoped_scan: bool = False,
    ):
        self.data_dir = Path(data_dir)
        self.augmentor = augmentor
        self.target_length = target_length
        # 复检A（🟡-4）：裁剪告警节流开关（与 ECGDataset 一致）
        self._warned_crop = False

        # 检查报告 1.5 修复：默认只读 train 划分（泄漏红线）。
        # 旧版 rglob 扫描全部 .npy（train+val+test），val/test 信号进入
        # 对比预训练。有 manifest 时按 signal_file 精确取 train 记录。
        manifest = self.data_dir / f"{split}_manifest.json"
        if manifest.exists():
            with open(manifest, "r", encoding="utf-8") as f:
                manifest_data = json.load(f)
            self.file_list = [str(self.data_dir / m["signal_file"])
                              for m in manifest_data.get("files", [])]
            logger.info(f"ECGContrastiveDataset [{split}]: "
                        f"{len(self.file_list)} 个样本（仅 {split} 划分）")
        else:
            # 4B-ORANGE-5 修复：缺 manifest 时不再静默扫描全部 .npy——旧版
            # 把 val/test 混入对比预训练，泄漏红线形同虚设。现显式失败；
            # 仅显式 allow_unscoped_scan=True 且 split=="train" 时保留旧
            # 目录（NPU 时代 /cache 无 manifest）回退。
            if not (allow_unscoped_scan and split == "train"):
                raise FileNotFoundError(
                    f"{manifest} 不存在：ECGContrastiveDataset 拒绝无划分扫描"
                    f"（防止 val/test 泄漏进对比预训练）。请提供 manifest，"
                    f"或确认旧目录环境后显式传 allow_unscoped_scan=True。")
            logger.warning(
                f"{manifest} 不存在，显式 allow_unscoped_scan=True → "
                f"回退扫描全部 .npy（仅限旧目录环境，泄漏红线由调用方负责）")
            signal_files = sorted(self.data_dir.rglob("*.npy"))
            self.file_list = [str(f) for f in signal_files]
            logger.info(f"ECGContrastiveDataset: {len(self.file_list)} 个样本")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回同一信号的两个增强视图作为正样本对。"""
        signal = np.nan_to_num(np.load(self.file_list[idx]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # 形状校正（含异常形状检测）
        # 4B 审查修复：💡-3 通道数合理性判断——仅当 shape[1]==12 且 shape[0]
        # 明显为时间轴（≥100 采样）才转置，避免 (13,12) 被误转置通过校验。
        if signal.ndim == 2 and signal.shape[1] == 12 and signal.shape[0] >= 100:
            signal = signal.T
        elif signal.ndim != 2 or signal.shape[0] != 12:
            raise ValueError(
                f"期望 (12, L) 信号，实际得到 {signal.shape}。"
                f"文件: {self.file_list[idx]}"
            )

        # Crop/pad 到目标长度
        if signal.shape[1] > self.target_length:
            # 复检A（🟡-4）：与 ECGDataset 一致的裁剪告警（对比预训练默认 4096
            # 裁剪权威 (12,5000)，每侧丢 452 样本，旧版静默）
            if not self._warned_crop:
                self._warned_crop = True
                logger.warning(
                    f"ECGContrastiveDataset 将信号从 {signal.shape[1]} 居中裁剪到 "
                    f"{self.target_length}（每侧丢弃 "
                    f"{(signal.shape[1] - self.target_length) // 2} 样本）——"
                    f"target_length=4096 为历史训练协议，权威预处理为 5000，"
                    f"请确认协议一致（仅告警一次）")
            start = (signal.shape[1] - self.target_length) // 2
            signal = signal[:, start:start + self.target_length]
        elif signal.shape[1] < self.target_length:
            pad = self.target_length - signal.shape[1]
            signal = np.pad(
                signal, ((0, 0), (pad // 2, pad - pad // 2)),
                mode="constant",
            )

        # 生成两个独立的增强视图
        view1 = self.augmentor(signal)
        view2 = self.augmentor(signal)

        return torch.from_numpy(view1), torch.from_numpy(view2)


class ECGDatasetForAgent(Dataset):
    """Agent 推理/评估专用数据集。

    返回完整的病人上下文（信号 + 元数据 + 标签），
    供 AI Agent 进行完整的诊断流程。

    参数:
        data_dir: 预处理数据目录。
        manifest_file: manifest JSON 路径。
        label_extractor: LabelExtractor 实例。
        target_length: 目标信号长度。
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        manifest_file: str,
        label_extractor: "LabelExtractor",
        target_length: int = 4096,
    ):
        self.data_dir = Path(data_dir)
        self.label_extractor = label_extractor
        self.target_length = target_length
        # 复检A（🟡-4）：裁剪告警节流开关（与 ECGDataset/Contrastive 一致）
        self._warned_crop = False

        with open(manifest_file, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        # 验证 manifest 格式
        if not isinstance(manifest, dict):
            raise TypeError(
                f"Manifest 必须是 JSON 对象，实际类型: {type(manifest).__name__}"
            )

        self.file_list = manifest.get("files", [])

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Dict:
        """返回完整的样本信息字典。"""
        record = self.file_list[idx]
        signal = np.nan_to_num(np.load(self.data_dir / record["signal_file"]).astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

        # 形状校正
        # 4B 审查修复：💡-3 通道数合理性判断——仅当 shape[1]==12 且 shape[0]
        # 明显为时间轴（≥100 采样）才转置，避免 (13,12) 被误转置通过校验。
        if signal.ndim == 2 and signal.shape[1] == 12 and signal.shape[0] >= 100:
            signal = signal.T
        # 第三轮审查 3B-DP-O2：严格形状校验（与 ECGDataset/Contrastive 一致；
        # 旧版畸形信号（如 6 导联）被静默送入 crop/pad 返回错误形状）
        if signal.ndim != 2 or signal.shape[0] != 12:
            raise ValueError(
                f"ECGDatasetForAgent 输入信号形状 {signal.shape} 非法（需 (12, L)）")

        # Crop/pad
        if signal.shape[1] > self.target_length:
            # 复检A（🟡-4）：与 ECGDataset/Contrastive 一致的裁剪告警
            if not self._warned_crop:
                self._warned_crop = True
                logger.warning(
                    f"ECGDatasetForAgent 将信号从 {signal.shape[1]} 居中裁剪到 "
                    f"{self.target_length}（每侧丢弃 "
                    f"{(signal.shape[1] - self.target_length) // 2} 样本）——"
                    f"target_length=4096 为历史协议，权威预处理为 5000，"
                    f"请确认协议一致（仅告警一次）")
            start = (signal.shape[1] - self.target_length) // 2
            signal = signal[:, start:start + self.target_length]
        elif signal.shape[1] < self.target_length:
            pad = self.target_length - signal.shape[1]
            signal = np.pad(
                signal, ((0, 0), (pad // 2, pad - pad // 2)),
                mode="constant",
            )

        # 4B 审查修复：🟡-9 修复 label_extractor 存储却从未使用——manifest 无
        # labels 字段但有 dx_codes 时，用 label_extractor.encode 在线编码（与
        # ECGDataset 一致），否则 labels 恒为空张量 shape=(0,)。有预编码 labels
        # 则直接透传。
        # 5B 审查（🟡-4）：dx_codes 也缺失时回退 27 维全零（与 ECGDataset 一致，
        # 旧版返回 shape (0,) 与下游不一致）
        labels = np.array(record.get("labels", []), dtype=np.float32)
        if len(labels) == 0 and "dx_codes" in record and self.label_extractor:
            labels = self.label_extractor.encode(record["dx_codes"], format="multi_hot")
        if len(labels) == 0:
            n_cls = getattr(self.label_extractor, "num_classes", None) or 27
            labels = np.zeros(n_cls, dtype=np.float32)

        return {
            "record_id": record.get("record_id", ""),
            "signal": torch.from_numpy(signal),
            "age": record.get("age"),
            "sex": record.get("sex", "Unknown"),
            "dx_codes": record.get("dx_codes", []),
            "labels": torch.from_numpy(labels),
            "source": record.get("source", "unknown"),
        }


class ECGDataModule:
    """统一的数据模块，提供 train/val/test 三个 DataLoader。

    兼容独立训练循环和 PyTorch Lightning（duck-typing）。

    参数:
        data_dir: 预处理数据目录。
        batch_size: 批大小。
        num_workers: DataLoader 工作进程数。
        pin_memory: 是否 pin 内存（GPU 训练时加速）。
        augment_train: 是否对训练集做数据增强。
        augmentor: ECGAugmentor 实例。
        label_extractor: LabelExtractor 实例。
        target_length: 目标信号长度。
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        batch_size: int = 128,
        num_workers: int = 4,
        pin_memory: bool = True,
        augment_train: bool = True,
        augmentor: Optional["ECGAugmentor"] = None,
        label_extractor: Optional["LabelExtractor"] = None,
        target_length: int = 4096,
        seed: Optional[int] = None,  # 第三轮审查 3B-DP-T3：显式主种子
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment_train = augment_train
        self.augmentor = augmentor
        self.label_extractor = label_extractor
        self.target_length = target_length
        self.seed = seed

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """初始化三个数据集。

        如果 augment_train=True 但未提供 augmentor，发出警告。
        第三轮审查 3B-DP-T3：显式 seed 时固定主随机种子（worker 种子
        派生自 torch.initial_seed()，主种子是复现的前提）。
        """
        if self.seed is not None:
            import random as _random
            _random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            # 4B 审查修复：🟡-10 补充 CUDA 种子与 cudnn 确定性开关（仅在 CUDA
            # 可用时）。旧版只固定 CPU 主种子，GPU 训练下不可完全复现；现设
            # manual_seed_all 并开 deterministic / 关 benchmark。
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.seed)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            # 5B 审查（🟡-5）：num_workers=0 时增强在主进程执行——augmentor 的
            # Generator 不在上述种子覆盖内，增强不可复现；显式重播种。
            # 复检A（🟠-1）：属性名是 rng（augmentor.py 存 self.rng），旧版写
            # default_rng 只是新增一个无人读的属性、静默 no-op
            if self.augment_train and self.augmentor is not None:
                try:
                    self.augmentor.rng = np.random.default_rng(self.seed)
                except AttributeError:
                    logger.warning("augmentor 无 rng 属性，无法重播种")

        if self.augment_train and self.augmentor is None:
            logger.warning(
                "augment_train=True 但未提供 augmentor。增强将被跳过。"
            )

        common_kwargs = dict(
            data_dir=self.data_dir,
            label_extractor=self.label_extractor,
            target_length=self.target_length,
        )

        self._train_dataset = ECGDataset(
            split="train", augment=self.augment_train,
            augmentor=self.augmentor, **common_kwargs,
        )
        self._val_dataset = ECGDataset(split="val", **common_kwargs)
        self._test_dataset = ECGDataset(split="test", **common_kwargs)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_dataset, batch_size=self.batch_size,
            shuffle=True, num_workers=self.num_workers,
            pin_memory=self.pin_memory, drop_last=True,
            worker_init_fn=self._worker_init_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self._test_dataset, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            worker_init_fn=self._worker_init_fn,
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """训练结束后的清理（供 PyTorch Lightning 兼容）。"""
        pass

    def _worker_init_fn(self, worker_id: int) -> None:
        """为每个 DataLoader worker 设置独立的随机种子。

        在多进程数据加载时保证增强的多样性和可复现性。
        2A 🟠 修复：旧版只 seed np.random（legacy 全局状态），对
        np.random.default_rng（ECGAugmentor 用 Generator）无效 →
        所有 worker 的增强序列相同。现同时重置：
        legacy np.random / python random / augmentor 的 Generator。
        """
        import random
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        if self.augmentor is not None:
            # worker 进程内重绑 Generator（fork 语义：不影响主进程）
            self.augmentor.rng = np.random.default_rng(worker_seed)
