"""
ECG 数据加载器 — 读取 PhysioNet Challenge 2020 WFDB 格式文件。

数据格式说明:
    .hea 头部文件: 记录名、12导联、采样率、样本数、患者元数据
    .mat 信号文件: WFDB format 16 二进制格式（注意：不是 MATLAB 格式！）
        - 16位有符号整数，一阶差分编码（每个值表示与前一值的差值）
        - 12 通道交错存储
        - 解码方式: 累积求和(cumsum) → 除以增益(1000) → 毫伏

参考文献:
    https://physionetchallenges.github.io/2020/
"""

import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import wfdb  # PhysioNet 官方 WFDB 库，自动处理差分解码

logger = logging.getLogger(__name__)


class ECGSample:
    """单条 ECG 记录的数据容器。

    封装了原始信号和解析后的元数据，提供便捷的属性访问。

    属性:
        record_id: 唯一记录标识，如 "A0001"
        signal: 12导联 ECG 信号，形状 (12, 采样点数)，单位 毫伏(mV)
        fs: 原始采样频率，单位 Hz
        metadata: 解析后的元数据字典 {Age, Sex, Dx, Rx, Hx, Sx}
        source: 数据来源名称，如 "cpsc_2018"
        lead_names: 12个导联名称列表
    """

    def __init__(
        self,
        record_id: str,
        signal: np.ndarray,
        fs: float,
        metadata: Dict[str, str],
        source: str,
        lead_names: Optional[List[str]] = None,
    ):
        self.record_id = record_id
        self.signal = signal          # 形状 (12, L)，单位 mV
        if fs <= 0:
            raise ValueError(f"无效的采样频率: {fs} Hz，必须大于0")
        self.fs = fs
        self.metadata = metadata
        self.source = source
        # 默认使用标准12导联名称
        self.lead_names = lead_names or [
            "I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6",
        ]

    @property
    def duration(self) -> float:
        """信号时长（秒）。

        安全处理 fs≤0 的损坏头文件，返回 0.0。
        """
        if self.fs <= 0:
            return 0.0
        return self.signal.shape[1] / self.fs

    @property
    def num_leads(self) -> int:
        """导联数（通常为12）。"""
        return self.signal.shape[0]

    @property
    def num_samples(self) -> int:
        """每导联的采样点数。"""
        return self.signal.shape[1]

    @property
    def age(self) -> Optional[float]:
        """患者年龄（岁）。

        自动处理各种异常值：
        - NaN、inf → 返回 None
        - 负数 → 返回 None
        - 超过122岁（人类最长寿命）→ 返回 None
        - 正常值 → 返回 float

        返回 None 表示"未知"，不会毒化下游统计计算。
        """
        raw = self.metadata.get("Age")
        if raw is None:
            return None
        try:
            val = float(raw)
            # NaN/inf/异常范围统一返回 None
            if np.isnan(val) or np.isinf(val) or val < 0 or val > 122:
                return None
            return val
        except (ValueError, TypeError):
            return None

    @property
    def sex(self) -> str:
        """患者性别，标准化为 "Male"/"Female"/"Unknown"。

        处理不同数据源的编码不一致：
        - cpsc_2018: "Male"/"Female"
        - ptb: "M"/"F"
        - georgia: "NaN" → "Unknown"
        """
        raw = self.metadata.get("Sex", "Unknown").strip()
        if raw in ("Male", "M", "male"):
            return "Male"
        elif raw in ("Female", "F", "female"):
            return "Female"
        return "Unknown"

    @property
    def dx_codes(self) -> List[str]:
        """SNOMED CT 诊断码列表。

        只返回纯数字的代码（过滤掉非代码文本如 "Unknown"）。
        空诊断码或 "Unknown" 返回空列表。
        """
        raw = self.metadata.get("Dx", "")
        if not raw or raw == "Unknown":
            return []
        return [code.strip() for code in raw.split(",") if code.strip().isdigit()]

    def __repr__(self) -> str:
        return (
            f"ECGSample(id={self.record_id}, source={self.source}, "
            f"shape={self.signal.shape}, fs={self.fs}, "
            f"age={self.age}, sex={self.sex})"
        )


class ECGLoader:
    """ECG 数据批量加载器。

    负责从 PhysioNet Challenge 2020 数据集加载原始 ECG 记录。
    自动处理 WFDB format 16 格式（.hea + .mat 配对文件）。

    使用示例:
        loader = ECGLoader("/path/to/training")
        sample = loader.load_record("cpsc_2018/g1/A0001")
        # 或迭代所有记录
        for sample in loader.iter_records(sources=["cpsc_2018"]):
            process(sample)
    """

    # 标准12导联 ECG 配置
    STANDARD_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF",
                      "V1", "V2", "V3", "V4", "V5", "V6"]

    # .hea 文件注释行中的元数据字段名
    METADATA_FIELDS = ["Age", "Sex", "Dx", "Rx", "Hx", "Sx"]

    def __init__(self, raw_dir: Union[str, Path]):
        """初始化加载器。

        参数:
            raw_dir: 原始数据根目录路径（包含各数据源子目录）。

        异常:
            FileNotFoundError: 如果目录不存在。
        """
        self.raw_dir = Path(raw_dir)
        if not self.raw_dir.exists():
            raise FileNotFoundError(f"原始数据目录不存在: {self.raw_dir}")

        # 自动发现所有数据源子目录
        self.sources = self._discover_sources()
        logger.info(f"ECGLoader 初始化完成: 在 {self.raw_dir} 中发现 "
                     f"{len(self.sources)} 个数据源")

    def _discover_sources(self) -> List[str]:
        """自动扫描 raw_dir 下的子目录，找出包含 .hea 文件的数据源。

        只检查一级子目录（如 cpsc_2018, georgia 等）。
        """
        sources = []
        for entry in sorted(self.raw_dir.iterdir()):
            # 跳过隐藏目录和非目录文件
            if entry.is_dir() and not entry.name.startswith("."):
                # 用 any() 短路：找到第一个 .hea 即确认该目录为数据源
                has_hea = any(entry.rglob("*.hea"))
                if has_hea:
                    sources.append(entry.name)
        return sources

    def _find_record_path(self, record_ref: str) -> Tuple[Path, str]:
        """将记录引用解析为 .hea 文件的完整路径。

        参数:
            record_ref: 可以是以下任一形式：
                - 完整的 .hea 文件路径
                - 记录名如 "A0001"
                - 相对路径如 "cpsc_2018/g1/A0001"

        返回:
            (hea文件路径, 记录名) 元组。

        异常:
            FileNotFoundError: 如果找不到对应记录。
        """
        ref_path = Path(record_ref)

        # 情况1: 已经是 .hea 文件的完整路径
        if ref_path.suffix == ".hea":
            if ref_path.exists():
                return ref_path, ref_path.stem
            # 也尝试在 raw_dir 下查找
            hea_under_raw = self.raw_dir / ref_path
            if hea_under_raw.exists():
                return hea_under_raw, hea_under_raw.stem

        # 情况2: 无扩展名的相对路径或记录名
        if not ref_path.suffix:
            # 先作为 raw_dir 下的相对路径尝试
            hea_path = self.raw_dir / f"{record_ref}.hea"
            if hea_path.exists():
                return hea_path, ref_path.name

            # 再作为裸记录名在整个 raw_dir 下搜索
            record_name = ref_path.name
            for hea_file in self.raw_dir.rglob(f"{record_name}.hea"):
                return hea_file, record_name

        raise FileNotFoundError(f"找不到记录: {record_ref}")

    def load_record(
        self,
        record_ref: str,
        validate: bool = True,
    ) -> Optional[ECGSample]:
        """加载单条 ECG 记录。

        参数:
            record_ref: 记录标识（路径名、记录名或完整路径）。
            validate: 如果为 True，跳过信号形状异常的记录。

        返回:
            ECGSample 对象，如果加载失败或验证不通过则返回 None。
        """
        try:
            hea_path, record_name = self._find_record_path(record_ref)

            # wfdb.rdrecord() 自动读取配对的 .hea 和 .mat 文件
            # 内部处理 WFDB format 16 的差分编码解码
            record = wfdb.rdrecord(str(hea_path.with_suffix("")))

            # wfdb 返回 (L, 12)，转置为 (12, L) 方便 PyTorch 卷积处理
            signal = record.p_signal.T.astype(np.float32)

            # 验证：必须12导联，至少100个采样点
            if validate and (signal.shape[0] != 12 or signal.shape[1] < 100):
                logger.warning(f"跳过 {record_name}: 信号形状异常 {signal.shape}")
                return None

            # 从 .hea 注释行解析元数据
            metadata = self._parse_metadata(record.comments)

            # 从文件路径推断数据来源
            source = self._infer_source(hea_path)

            return ECGSample(
                record_id=record_name,
                signal=signal,
                fs=record.fs,
                metadata=metadata,
                source=source,
                lead_names=list(self.STANDARD_LEADS),
            )

        except Exception as e:
            logger.error(f"加载记录失败 {record_ref}: {e}")
            return None

    def _parse_metadata(self, comments: List[str]) -> Dict[str, str]:
        """解析 WFDB 头文件中的注释行元数据。

        .hea 文件注释行格式:
            # Age: 74
            # Sex: Male
            # Dx: 59118001
            # Rx: Unknown
            # Hx: Unknown
            # Sx: Unknown

        注意: wfdb 库会自动去掉 "# " 前缀，
        所以 comments 中的内容是 "Age: 74" 而非 "# Age: 74"。
        """
        # 预编译正则模式（只编译一次，43K条记录复用）
        if not hasattr(self, '_meta_patterns'):
            self._meta_patterns = [
                (field, re.compile(rf"^{field}\s*:\s*(.*)", re.IGNORECASE))
                for field in self.METADATA_FIELDS
            ]
        metadata = {}
        for comment in comments:
            for field, pattern in self._meta_patterns:
                match = pattern.search(comment)
                if match:
                    metadata[field] = match.group(1).strip()
                    break
        return metadata

    def _infer_source(self, hea_path: Path) -> str:
        """从文件路径推断数据源名称。

        例:
            .../training/cpsc_2018/g1/A0001.hea → "cpsc_2018"
        """
        try:
            rel = hea_path.relative_to(self.raw_dir)
            parts = rel.parts
            if len(parts) >= 2:
                return parts[0]  # 第一层目录名即为数据源
        except ValueError:
            pass
        return "unknown"

    def iter_records(
        self,
        sources: Optional[List[str]] = None,
        max_records: Optional[int] = None,
    ):
        """迭代所有 ECG 记录（生成器，内存友好）。

        参数:
            sources: 要包含的数据源列表，None 表示全部。
            max_records: 最大返回记录数，用于快速测试。

        产出:
            ECGSample 对象。
        """
        target_sources = sources or self.sources
        count = 0

        for source in target_sources:
            source_dir = self.raw_dir / source
            if not source_dir.exists():
                logger.warning(f"数据源目录不存在: {source_dir}")
                continue

            for hea_file in sorted(source_dir.rglob("*.hea")):
                if max_records and count >= max_records:
                    return

                # 跳过隐藏文件
                if hea_file.name.startswith("."):
                    continue

                sample = self.load_record(str(hea_file))
                if sample is not None:
                    yield sample
                    count += 1

    def get_record_count(self, sources: Optional[List[str]] = None) -> int:
        """统计记录总数（基于 .hea 文件数量，近似值）。

        使用懒计数方式，不构建中间列表以节省内存。
        """
        target_sources = sources or self.sources
        total = 0
        for source in target_sources:
            source_dir = self.raw_dir / source
            if source_dir.exists():
                # 懒计数：不 materialize 整个列表
                total += sum(1 for _ in source_dir.rglob("*.hea"))
        return total

    def get_statistics(self, max_records: int = 500) -> Dict[str, any]:
        """通过抽样快速获取数据集统计信息。

        返回字典包含: 各数据源记录数、采样率分布、时长统计、
        年龄统计、性别分布、标签频率 等。
        """
        stats = {
            "sources": {},
            "fs_values": set(),
            "durations": [],
            "ages": [],
            "sexes": {"Male": 0, "Female": 0, "Unknown": 0},
            "label_counts": {},
            "samples_per_record": [],
        }

        for sample in self.iter_records(max_records=max_records):
            src = sample.source
            stats["sources"][src] = stats["sources"].get(src, 0) + 1
            stats["fs_values"].add(sample.fs)
            stats["durations"].append(sample.duration)
            stats["samples_per_record"].append(sample.num_samples)

            if sample.age is not None:
                stats["ages"].append(sample.age)
            stats["sexes"][sample.sex] += 1

            for code in sample.dx_codes:
                stats["label_counts"][code] = stats["label_counts"].get(code, 0) + 1

        stats["fs_values"] = sorted(stats["fs_values"])
        stats["total_sampled"] = sum(stats["sources"].values())

        if stats["durations"]:
            durations = np.array(stats["durations"])
            stats["duration_stats"] = {
                "min": float(np.min(durations)),
                "max": float(np.max(durations)),
                "mean": float(np.mean(durations)),
                "median": float(np.median(durations)),
            }

        if stats["ages"]:
            ages = np.array(stats["ages"])
            stats["age_stats"] = {
                "count": len(stats["ages"]),
                "min": float(np.min(ages)),
                "max": float(np.max(ages)),
                "mean": float(np.mean(ages)),
                "nan_rate": 1.0 - len(stats["ages"]) / stats["total_sampled"],
            }

        return stats
