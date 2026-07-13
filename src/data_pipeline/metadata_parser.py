"""
元数据解析器 — 解析和标准化患者人口统计学信息。

处理来自不同数据源的不一致元数据格式:
    - Age:  "74" / "NaN" / "inf" → 归一化 [0,1] 或 -1.0 (未知)
    - Sex:  "Male"/"M"/"Female"/"F"/"NaN" → 0/1/2 编码
    - Dx:   "59118001,270492004" → SNOMED CT代码列表
    - Rx/Hx/Sx: "Unknown" / 自由文本 → 关键词标志字典
"""

import logging
import math
from typing import Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class MetadataParser:
    """解析和标准化 ECG 记录中的患者元数据。

    使用示例:
        parser = MetadataParser()
        parsed = parser.parse(sample.metadata)
        # → {"age": 0.617, "age_raw": 74, "sex": 1, "dx_codes": [...], ...}
    """

    # 性别值映射（不同数据源编码不同）
    SEX_MALE_VALUES = {"male", "m", "1", "man"}
    SEX_FEMALE_VALUES = {"female", "f", "0", "woman"}
    AGE_UNKNOWN_SENTINEL = -1.0

    # 中英文症状关键词
    SYMPTOM_KEYWORDS = {
        "chest_pain": ["chest pain", "chest tightness", "胸痛", "胸闷"],
        "palpitations": ["palpitation", "心悸"],
        "dizziness": ["dizziness", "dizzy", "头晕", "眩晕"],
        "shortness_of_breath": ["shortness of breath", "dyspnea", "呼吸困难", "气短"],
        "syncope": ["syncope", "fainting", "晕厥"],
        "fatigue": ["fatigue", "tired", "疲劳", "乏力"],
    }

    # 中英文病史关键词
    HISTORY_KEYWORDS = {
        "hypertension": ["hypertension", "htn", "高血压"],
        "diabetes": ["diabetes", "dm", "糖尿病"],
        "af_history": ["atrial fibrillation", "af", "afib", "房颤"],
        "cad": ["coronary artery disease", "cad", "冠心病"],
        "mi_history": ["myocardial infarction", "mi", "心肌梗死"],
        "heart_failure": ["heart failure", "hf", "心力衰竭", "心衰"],
        "stroke": ["stroke", "cva", "中风"],
    }

    def __init__(self, age_normalize: bool = True, age_default: float = 60.0):
        self.age_normalize = age_normalize
        self.age_default = age_default

    def parse(self, metadata: Dict[str, str]) -> Dict:
        """解析原始元数据字典为结构化格式。"""
        return {
            "age": self.parse_age(metadata.get("Age", "NaN")),
            "age_raw": self._parse_age_raw(metadata.get("Age", "NaN")),
            "sex": self.parse_sex(metadata.get("Sex", "Unknown")),
            "dx_codes": self.parse_dx(metadata.get("Dx", "")),
            "rx_info": self.parse_text_field(metadata.get("Rx", "Unknown")),
            "hx_info": self.parse_text_field(metadata.get("Hx", "Unknown")),
            "sx_info": self.parse_text_field(metadata.get("Sx", "Unknown")),
        }

    def parse_age(self, age_str: str) -> float:
        """解析年龄字符串为归一化浮点数。

        NaN/inf/负数/>122岁 → -1.0 (未知标记)。
        """
        age = self._parse_age_raw(age_str)
        if age is None:
            return self.AGE_UNKNOWN_SENTINEL
        if self.age_normalize:
            return age / 120.0  # 归一化到 [0, 1]
        return age

    def _parse_age_raw(self, age_str: str) -> Optional[float]:
        """解析年龄为原始浮点值（岁）。

        包含 NaN/inf 检查，防止毒化下游统计。
        """
        if not age_str or age_str.strip().lower() in ("nan", "unknown", "", "none"):
            return None
        try:
            age = float(age_str.strip())
            if math.isnan(age) or math.isinf(age) or age < 0 or age > 122:
                return None
            return age
        except (ValueError, TypeError):
            return None

    def parse_sex(self, sex_str: str) -> int:
        """解析性别字符串为整数编码: 0=Female, 1=Male, 2=Unknown。"""
        if not sex_str:
            return 2
        cleaned = sex_str.strip().lower()
        if cleaned in self.SEX_FEMALE_VALUES:    return 0
        elif cleaned in self.SEX_MALE_VALUES:    return 1
        return 2

    def parse_sex_str(self, sex_str: str) -> str:
        return {0: "Female", 1: "Male", 2: "Unknown"}[self.parse_sex(sex_str)]

    def parse_dx(self, dx_str: str) -> List[str]:
        """解析 SNOMED CT 诊断码字符串。

        只返回纯数字代码（过滤掉非代码文本）。
        """
        if not dx_str or dx_str.strip().lower() in ("unknown", "", "none", "nan"):
            return []
        return [code.strip() for code in dx_str.split(",") if code.strip().isdigit()]

    def parse_text_field(self, text: str) -> Dict[str, bool]:
        """解析自由文本医学字段 (Rx/Hx/Sx) 为关键词标志。

        同时检查症状和病史关键词字典。
        Rx/Hx/Sx 在 PhysioNet 数据集中几乎全为 "Unknown"，此功能实用性有限。
        """
        result = {}
        text_lower = (text or "").lower()
        if text_lower in ("unknown", "", "none", "nan"):
            return result
        for keyword_dict in [self.SYMPTOM_KEYWORDS, self.HISTORY_KEYWORDS]:
            for key, patterns in keyword_dict.items():
                result[key] = any(p in text_lower for p in patterns)
        return result

    def encode_metadata_vector(self, parsed: Dict, include_dx: bool = False) -> np.ndarray:
        """将解析后的元数据编码为固定维度的特征向量。

        当前输出 5 维: [age, age_unknown_flag, sex_female, sex_male, sex_unknown]
        注意: include_dx 参数尚未实现。
        """
        features = []
        age = parsed["age"]
        features.append(age)
        features.append(1.0 if age == self.AGE_UNKNOWN_SENTINEL else 0.0)

        sex_code = parsed["sex"]
        features.extend([
            1.0 if sex_code == 0 else 0.0,  # Female
            1.0 if sex_code == 1 else 0.0,  # Male
            1.0 if sex_code == 2 else 0.0,  # Unknown
        ])
        return np.array(features, dtype=np.float32)
