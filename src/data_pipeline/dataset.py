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

        # 加载信号
        signal = np.load(signal_path).astype(np.float32)

        # === 形状校正 ===
        # 正确的: (12, L) → 不处理
        # 转置的: (L, 12) → 自动转回来
        # 异常的: (1, L) 等 → 抛出 ValueError
        if signal.ndim == 2 and signal.shape[1] == 12 and signal.shape[0] != 12:
            signal = signal.T
        elif signal.ndim != 2 or signal.shape[0] != 12:
            raise ValueError(
                f"期望 (12, L) 形状的信号，实际得到 {signal.shape}。"
                f"文件: {record.get('signal_file', 'unknown')}"
            )

        # === 对齐到目标长度 ===
        if signal.shape[1] != self.target_length:
            if signal.shape[1] > self.target_length:
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
    ):
        self.data_dir = Path(data_dir)
        self.augmentor = augmentor
        self.target_length = target_length

        # 使用全部 .npy 文件（无标签也可用于对比学习）
        signal_files = sorted(self.data_dir.rglob("*.npy"))
        self.file_list = [str(f) for f in signal_files]

        logger.info(f"ECGContrastiveDataset: {len(self.file_list)} 个样本")

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回同一信号的两个增强视图作为正样本对。"""
        signal = np.load(self.file_list[idx]).astype(np.float32)

        # 形状校正（含异常形状检测）
        if signal.ndim == 2 and signal.shape[1] == 12 and signal.shape[0] != 12:
            signal = signal.T
        elif signal.ndim != 2 or signal.shape[0] != 12:
            raise ValueError(
                f"期望 (12, L) 信号，实际得到 {signal.shape}。"
                f"文件: {self.file_list[idx]}"
            )

        # Crop/pad 到目标长度
        if signal.shape[1] > self.target_length:
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
        signal = np.load(self.data_dir / record["signal_file"]).astype(np.float32)

        # 形状校正
        if signal.ndim == 2 and signal.shape[1] == 12 and signal.shape[0] != 12:
            signal = signal.T

        # Crop/pad
        if signal.shape[1] > self.target_length:
            start = (signal.shape[1] - self.target_length) // 2
            signal = signal[:, start:start + self.target_length]
        elif signal.shape[1] < self.target_length:
            pad = self.target_length - signal.shape[1]
            signal = np.pad(
                signal, ((0, 0), (pad // 2, pad - pad // 2)),
                mode="constant",
            )

        return {
            "record_id": record.get("record_id", ""),
            "signal": torch.from_numpy(signal),
            "age": record.get("age"),
            "sex": record.get("sex", "Unknown"),
            "dx_codes": record.get("dx_codes", []),
            "labels": torch.from_numpy(
                np.array(record.get("labels", []), dtype=np.float32)
            ),
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
    ):
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.augment_train = augment_train
        self.augmentor = augmentor
        self.label_extractor = label_extractor
        self.target_length = target_length

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        """初始化三个数据集。

        如果 augment_train=True 但未提供 augmentor，发出警告。
        """
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

    @staticmethod
    def _worker_init_fn(worker_id: int) -> None:
        """为每个 DataLoader worker 设置独立的随机种子。

        在多进程数据加载时保证增强的多样性和可复现性。
        """
        worker_seed = torch.initial_seed() % (2**32)
        np.random.seed(worker_seed)
