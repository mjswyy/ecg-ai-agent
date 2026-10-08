"""
标签提取器 — SNOMED CT 诊断码到 PhysioNet Challenge 27 类的映射。

PhysioNet 2020 Challenge 定义了 27 个诊断类别，每个类别由一个或多个
SNOMED CT 代码表示。本模块处理:
    - 规范 SNOMED CT 代码 → Challenge 类别索引的映射
    - 等价代码的多对一映射（临床同义词、子类型等）
    - 多标签编码/解码
    - 类别名称解析和统计

注意（类别索引顺序，4B 审查修复 🟠-1）:
    类别索引按 SNOMED CT 码字符串的字典序建立
    （LabelExtractor.__init__ 中的 sorted(CHALLENGE_CLASSES.items())），
    与官方 dx_mapping_scored.csv 的行序（按诊断名首字母排序）不同。
    二者 27 个码的"集合"完全一致、码级映射一致；阈值/标签/class_names
    均按 class_names 键名对齐，而非 CSV 行号。切勿按 CSV 行号索引 class_names。

参考文献:
    Reyna et al. "Classification of 12-lead ECGs: the PhysioNet/Computing
    in Cardiology Challenge 2020"

使用示例:
    extractor = LabelExtractor()
    labels = extractor.encode(["426783006", "164889003"])
    # → np.array([1, 1, 0, ...])  形状 (27,) 的 multi-hot 向量
"""

import collections
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

# ============================================================================
# 27 个官方 PhysioNet 2020 Challenge 评分类
# 来源: https://github.com/physionetchallenges/evaluation-2020
#       dx_mapping_scored.csv（官方评分集，27 行）
# 4B 审查修复：🟠-1 类序按 SNOMED 码字符串字典序（sorted()）建立，与官方
# CSV 行序不同；27 码集合一致、码级映射一致，阈值/标签均按 class_names
# 键名对齐而非 CSV 行号。切勿按 CSV 行号索引 class_names。
# 第二次审查 R1 修复（2026-08-24）：旧版 27 类为自定义体系且存在多处
# 码-名错配（如 427084000 官方窦速被标为 ST Elevation）；本版严格按
# 官方评分类重写。官方注记"将 X 与 Y 计为同一诊断"的 3 对
# （RBBB/CRBBB、PAC/SVPB、PVC/VPB）按官方表保留为独立类。
# ============================================================================

CHALLENGE_CLASSES: Dict[str, Dict] = {
    # ------ 节律类 (11) ------
    "164889003": {"name": "Atrial Fibrillation",             "abbreviation": "AF",    "category": "rhythm"},
    "164890007": {"name": "Atrial Flutter",                  "abbreviation": "AFL",   "category": "rhythm"},
    "426177001": {"name": "Sinus Bradycardia",               "abbreviation": "SB",    "category": "rhythm"},
    "426627000": {"name": "Bradycardia",                     "abbreviation": "Brady", "category": "rhythm"},
    "426783006": {"name": "Sinus Rhythm",                    "abbreviation": "NSR",   "category": "rhythm"},
    "427084000": {"name": "Sinus Tachycardia",               "abbreviation": "STach", "category": "rhythm"},
    "427393009": {"name": "Sinus Arrhythmia",                "abbreviation": "SA",    "category": "rhythm"},
    "284470004": {"name": "Premature Atrial Contraction",    "abbreviation": "PAC",   "category": "rhythm"},
    "63593006":  {"name": "Supraventricular Premature Beats","abbreviation": "SVPB",  "category": "rhythm"},
    "427172004": {"name": "Premature Ventricular Contractions","abbreviation": "PVC", "category": "rhythm"},
    "17338001":  {"name": "Ventricular Premature Beats",     "abbreviation": "VPB",   "category": "rhythm"},

    # ------ 传导类 (7) ------
    "270492004": {"name": "First Degree AV Block",           "abbreviation": "IAVB",  "category": "conduction"},
    "164909002": {"name": "Left Bundle Branch Block",        "abbreviation": "LBBB",  "category": "conduction"},
    "59118001":  {"name": "Right Bundle Branch Block",       "abbreviation": "RBBB",  "category": "conduction"},
    "713426002": {"name": "Incomplete Right Bundle Branch Block","abbreviation": "IRBBB","category": "conduction"},
    "713427006": {"name": "Complete Right Bundle Branch Block","abbreviation": "CRBBB","category": "conduction"},
    "445118002": {"name": "Left Anterior Fascicular Block",  "abbreviation": "LAFB", "category": "conduction"},  # 5I 审查（🟡）：标准缩写 LAFB（旧值 LAnFB 非通用，检索解析失败）
    "698252002": {"name": "Nonspecific Intraventricular Conduction Disorder","abbreviation": "NSIVCB", "category": "conduction"},

    # ------ 形态 / 轴 / 间期 / 起搏类 (9) ------
    "164917005": {"name": "Q Wave Abnormal",                 "abbreviation": "QAb",   "category": "morphology"},
    "164934002": {"name": "T Wave Abnormal",                 "abbreviation": "TAb",   "category": "morphology"},
    "59931005":  {"name": "T Wave Inversion",                "abbreviation": "TInv",  "category": "morphology"},
    "251146004": {"name": "Low QRS Voltages",                "abbreviation": "LQRSV", "category": "morphology"},
    "39732003":  {"name": "Left Axis Deviation",             "abbreviation": "LAD",   "category": "axis"},
    "47665007":  {"name": "Right Axis Deviation",            "abbreviation": "RAD",   "category": "axis"},
    "111975006": {"name": "Prolonged QT Interval",           "abbreviation": "LQT",   "category": "interval"},
    "164947007": {"name": "Prolonged PR Interval",           "abbreviation": "LPR",   "category": "interval"},
    "10370003":  {"name": "Pacing Rhythm",                   "abbreviation": "PR",    "category": "pacing"},
}

# ============================================================================
# SNOMED CT 等价映射表（官方评分类体系下为空）
# 官方 dx_mapping_scored.csv 的 27 个评分类即全部标签空间；不在评分集内的
# 诊断码（ST 段改变、心肌梗死、缺血等）不参与评分，重贴标签时将被丢弃。
# ============================================================================

SNOMED_CT_EQUIVALENTS: Dict[str, str] = {}


class LabelExtractor:
    """SNOMED CT 诊断码到 Challenge 27 类的映射器。

    参数:
        num_classes: 输出类别数（默认 27）。
    """

    def __init__(self, num_classes: int = 27):
        self.num_classes = num_classes

        # SNOMED CT → 类别索引 (0-26) 映射
        self.snomed_to_idx: Dict[str, int] = {}
        self.idx_to_snomed: Dict[int, str] = {}
        self.class_names: List[str] = []

        # 4B 审查修复：🟡-2 num_classes≠27 时告警——字典序末位类别被静默丢弃，
        # 会导致标签空间与模型头不一致（如 num_classes=26 丢 713427006 CRBBB）。
        # 4B 定向复查（α）：>27 时实际 27 类全保留、多余维为死零维，消息措辞已修正
        if num_classes != 27:
            if num_classes < 27:
                logger.warning(
                    f"LabelExtractor(num_classes={num_classes}) < 27：按 SNOMED 字典序"
                    f"截断，末尾 {27 - num_classes} 个类别被丢弃；请确认与模型头一致。"
                )
            else:
                logger.warning(
                    f"LabelExtractor(num_classes={num_classes}) > 27：27 个类别全保留，"
                    f"多出 {num_classes - 27} 维为死零维；请确认与模型头一致。"
                )

        # 按代码排序建立映射（保证确定性）
        for idx, (snomed, info) in enumerate(sorted(CHALLENGE_CLASSES.items())):
            if idx >= num_classes:
                break
            self.snomed_to_idx[snomed] = idx
            self.idx_to_snomed[idx] = snomed
            self.class_names.append(info["name"])

        # 构建完整映射: 规范代码 + 等价代码 → 类别索引
        self._full_mapping: Dict[str, int] = dict(self.snomed_to_idx)
        for code, canonical in SNOMED_CT_EQUIVALENTS.items():
            if canonical in self.snomed_to_idx and code not in self._full_mapping:
                self._full_mapping[code] = self.snomed_to_idx[canonical]

        # 4B 审查修复：🟡-1 记录所有未匹配代码及其真实出现次数（Counter），
        # 供 get_unmapped_codes 返回真实计数而非恒 0。
        self._unmapped_codes: collections.Counter = collections.Counter()

        logger.info(
            f"LabelExtractor: {len(self.snomed_to_idx)} 个规范类别, "
            f"{len(self._full_mapping)} 个 SNOMED CT 代码已映射"
        )

    def encode(
        self, dx_codes: List[str], format: str = "multi_hot",
    ) -> Union[np.ndarray, List[int]]:
        """将 SNOMED CT 代码编码为标签向量。

        参数:
            dx_codes: SNOMED CT 代码字符串列表。
            format: 输出格式 —
                "multi_hot": 二进制向量 (num_classes,)  ← 默认
                "indices":   活跃类别索引列表
                "names":     类别名称列表

        返回:
            编码后的标签。
        """
        active_indices = []
        for code in dx_codes:
            code = code.strip()
            if code in self._full_mapping:
                active_indices.append(self._full_mapping[code])
            elif code.isdigit():
                # 记录未匹配的代码以便后续添加映射（计数真实出现次数）
                self._unmapped_codes[code] += 1

        if format == "multi_hot":
            vec = np.zeros(self.num_classes, dtype=np.float32)
            for idx in active_indices:
                vec[idx] = 1.0
            return vec

        elif format == "indices":
            return sorted(set(active_indices))

        elif format == "names":
            return [self.class_names[i] for i in sorted(set(active_indices))]

        else:
            raise ValueError(f"未知的格式: {format}")

    def decode(
        self, labels: np.ndarray, threshold: float = 0.5,
    ) -> List[Tuple[str, str, float]]:
        """将概率向量解码为人类可读的诊断列表。

        参数:
            labels: 概率向量，形状 (num_classes,)
            threshold: 判定阈值

        返回:
            [(snomed_code, class_name, probability), ...] 列表
        """
        results = []
        for idx in range(min(len(labels), self.num_classes)):
            prob = float(labels[idx])
            if prob >= threshold:
                snomed = self.idx_to_snomed.get(idx, f"unknown_{idx}")
                name = self.class_names[idx] if idx < len(self.class_names) else "Unknown"
                results.append((snomed, name, prob))
        # 按概率降序排列
        return sorted(results, key=lambda x: x[2], reverse=True)

    def get_class_name(self, snomed_code: str) -> Optional[str]:
        """获取 SNOMED CT 代码对应的人类可读类别名。

        同时查询规范代码和等价代码映射。
        """
        idx = self._full_mapping.get(snomed_code, self.snomed_to_idx.get(snomed_code))
        if idx is not None and idx < len(self.class_names):
            return self.class_names[idx]
        return None

    def get_category(self, snomed_code: str) -> Optional[str]:
        """获取 SNOMED CT 代码所属类别（节律/传导/形态）。"""
        info = CHALLENGE_CLASSES.get(snomed_code)
        return info["category"] if info else None

    def get_class_distribution(self, all_dx_codes: List[List[str]]) -> Dict[str, int]:
        """计算类别频率分布（使用完整映射）。"""
        distribution = {name: 0 for name in self.class_names}
        for dx_list in all_dx_codes:
            for code in dx_list:
                idx = self._full_mapping.get(code.strip())
                if idx is not None:
                    distribution[self.class_names[idx]] += 1
        return distribution

    def get_rare_classes(self, distribution: Dict[str, int], min_samples: int = 100) -> List[str]:
        """识别样本数少于 min_samples 的稀有类别。"""
        return [name for name, count in distribution.items() if count < min_samples]

    def save_mapping(self, filepath: Union[str, Path]) -> None:
        """保存 SNOMED CT → 类别映射到 JSON 文件（含等价映射）。"""
        mapping = {
            "num_classes": self.num_classes,
            "snomed_to_idx": self.snomed_to_idx,
            "idx_to_snomed": {str(k): v for k, v in self.idx_to_snomed.items()},
            "class_names": self.class_names,
            "full_mapping": self._full_mapping,  # 持久化等价映射
            "class_details": {
                code: CHALLENGE_CLASSES.get(code, {})
                for code in self.snomed_to_idx
            },
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
        logger.info(f"标签映射已保存至 {filepath}")

    def get_unmapped_codes(self) -> Dict[str, int]:
        """返回所有遇到的未映射 SNOMED CT 代码及其真实出现次数。

        4B 审查修复：🟡-1 旧版恒返回 {code: 0}；现返回 Counter 计数的真实频次。
        """
        return dict(sorted(self._unmapped_codes.items()))

    @classmethod
    def load_mapping(cls, filepath: Union[str, Path]) -> "LabelExtractor":
        """从 JSON 文件加载 LabelExtractor（含等价映射恢复）。"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        extractor = cls(num_classes=data["num_classes"])
        extractor.snomed_to_idx = data["snomed_to_idx"]
        extractor.idx_to_snomed = {int(k): v for k, v in data["idx_to_snomed"].items()}
        extractor.class_names = data["class_names"]
        if "full_mapping" in data:
            extractor._full_mapping = data["full_mapping"]
        return extractor
