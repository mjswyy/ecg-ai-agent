"""
ECG 信号预处理器 — 完整的信号处理管道。

处理流程（6步）:
    1. 带通滤波 (0.5-45 Hz) — 去除基线漂移和高频噪声
    2. 陷波滤波 (50/60 Hz) — 去除工频干扰
    3. 重采样至目标采样率 — 统一不同数据源的采样率
    4. 分段至固定长度 — 居中裁剪或对称零填充
    5. 逐导联归一化 (z-score) — 标准化幅度
    6. 异常值裁剪 (5σ) — 限制极端值

关键技术:
    - SOS (Second-Order Sections) 滤波形式，比传统 ba 形式数值更稳定
    - resample_poly 多相滤波重采样，内置抗混叠滤波，保留 QRS 波形形态
    - 归一化时只对信号区域计算统计量，不受零填充影响
"""

import logging
from typing import Literal, Optional, Tuple, Union

import numpy as np
from scipy import signal

logger = logging.getLogger(__name__)


class ECGPreprocessor:
    """ECG 信号预处理管道。

    使用示例:
        preprocessor = ECGPreprocessor(target_fs=500, target_length=4096)
        clean = preprocessor(raw_signal, original_fs=500)
        # 批量分段处理长记录:
        segments = preprocessor.segment(long_ecg, segment_length=2048, overlap=0.5)
    """

    def __init__(
        self,
        target_fs: float = 500.0,       # 目标采样率 (Hz)
        target_length: int = 4096,       # 目标信号长度 (~8.2秒)
        bandpass_low: float = 0.5,       # 带通滤波低截止频率 (Hz)
        bandpass_high: float = 45.0,     # 带通滤波高截止频率 (Hz)
        notch_freq: float = 50.0,        # 陷波滤波频率 (Hz)
        filter_order: int = 4,           # Butterworth滤波阶数
        normalization: Literal["zscore", "minmax", "none"] = "zscore",
        outlier_threshold: float = 5.0,  # 异常值裁剪阈值(σ倍数)
    ):
        self.target_fs = target_fs
        self.target_length = target_length
        self.bandpass_low = bandpass_low
        self.bandpass_high = bandpass_high
        self.notch_freq = notch_freq
        self.filter_order = filter_order
        self.normalization = normalization
        self.outlier_threshold = outlier_threshold

        # 滤波器系数（延迟设计，首次使用时根据实际采样率计算）
        self._bp_sos = None      # 带通滤波器 SOS 系数
        self._notch_sos = None   # 陷波滤波器 SOS 系数
        self._design_fs = None   # 上次设计的采样率（用于缓存）

    def _design_filters(self, fs: float) -> None:
        """设计 Butterworth 带通和陷波滤波器（SOS 形式，数值稳定）。

        SOS (Second-Order Sections) 将高阶滤波器分解为多个二阶节的级联，
        相比传统 ba (分子/分母多项式) 形式，数值误差更小，
        尤其对窄带或低频滤波器效果显著。

        安全机制:
            - 带通上限超过 0.99×Nyquist → 抛出 ValueError
            - 陷波频率超过 0.99×Nyquist → 自动跳过
        """
        if self._design_fs == fs and self._bp_sos is not None:
            return  # 滤波器已设计，复用缓存

        nyquist = fs / 2.0  # 奈奎斯特频率

        # === Nyquist 边界检查 ===
        if self.bandpass_high >= nyquist * 0.99:
            raise ValueError(
                f"带通上限 ({self.bandpass_high} Hz) 必须小于 Nyquist 频率 "
                f"({nyquist:.1f} Hz)，当前 fs={fs} Hz"
            )
        if self.bandpass_low <= 0:
            raise ValueError(
                f"带通下限 ({self.bandpass_low} Hz) 必须大于 0"
            )

        # === 带通滤波器 ===
        # 将频率归一化到 [0, 1]，并钳制以避免边界问题
        low = max(self.bandpass_low / nyquist, 1e-6)
        high = min(self.bandpass_high / nyquist, 0.999)
        self._bp_sos = signal.butter(
            self.filter_order, [low, high], btype="band", output="sos"
        )

        # === 陷波滤波器 ===
        if self.notch_freq >= nyquist * 0.99:
            logger.warning(
                f"陷波频率 ({self.notch_freq} Hz) 过于接近 Nyquist 频率 "
                f"({nyquist:.1f} Hz)，fs={fs} Hz。跳过陷波滤波。"
            )
            self._notch_sos = None
        else:
            notch = self.notch_freq / nyquist
            q = 30.0  # 品质因数（带宽 = 中心频率/Q）
            b, a = signal.iirnotch(notch, q)
            # 转换为 SOS 形式保持一致性
            self._notch_sos = signal.tf2sos(b, a)

        self._design_fs = fs

    def __call__(
        self,
        ecg_signal: np.ndarray,
        original_fs: float,
        return_meta: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """执行完整的预处理管道。

        参数:
            ecg_signal: 原始 ECG 信号，形状 (12, L)。
            original_fs: 原始采样频率 (Hz)。
            return_meta: 是否返回预处理元数据。

        返回:
            预处理后的信号 (12, target_length) float32，
            如果 return_meta=True 则返回 (signal, meta) 元组。
        """
        meta = {"original_shape": ecg_signal.shape, "original_fs": original_fs}

        # 步骤1: 带通滤波 (0.5-45 Hz)
        ecg = self._apply_bandpass(ecg_signal, original_fs)
        meta["bandpass_applied"] = True

        # 步骤2: 陷波滤波 (50 Hz)
        ecg = self._apply_notch(ecg, original_fs)
        meta["notch_applied"] = True

        # 步骤3: 重采样至目标频率（如果采样率不同）
        if abs(original_fs - self.target_fs) > 1e-3:
            ecg = self._resample(ecg, original_fs, self.target_fs)
            meta["resampled"] = f"{original_fs}→{self.target_fs}Hz"
        else:
            meta["resampled"] = "no change"

        # 步骤4: 提取固定长度片段（过长裁剪，过短零填充）
        ecg, segment_info = self._extract_segment(ecg)
        meta.update(segment_info)

        # 步骤5: 逐导联归一化
        # 关键: 只对信号区域计算统计量，不受零填充区域影响
        ecg, norm_info = self._normalize(
            ecg,
            pad_left=segment_info.get("pad_left", 0),
            pad_right=segment_info.get("pad_right", 0),
        )
        meta.update(norm_info)

        # 步骤6: 异常值裁剪
        ecg = self._clip_outliers(ecg)

        meta["final_shape"] = ecg.shape

        if return_meta:
            return ecg.astype(np.float32), meta
        return ecg.astype(np.float32)

    def _apply_bandpass(self, ecg: np.ndarray, fs: float) -> np.ndarray:
        """应用 Butterworth 带通滤波器 (SOS 形式，零相位)。

        使用 sosfiltfilt 实现零相位滤波（前后向各一次），
        不会引入相位失真，对 ECG 波形形态保护至关重要。
        """
        self._design_filters(fs)
        return signal.sosfiltfilt(self._bp_sos, ecg, axis=1)

    def _apply_notch(self, ecg: np.ndarray, fs: float) -> np.ndarray:
        """应用陷波滤波器去除工频干扰。

        如果陷波频率过于接近 Nyquist（跳过设计），直接返回原信号。
        """
        self._design_filters(fs)
        if self._notch_sos is None:
            return ecg  # 陷波被跳过
        return signal.sosfiltfilt(self._notch_sos, ecg, axis=1)

    def _resample(
        self, ecg: np.ndarray, orig_fs: float, target_fs: float
    ) -> np.ndarray:
        """使用多相滤波重采样 ECG 信号。

        scipy.signal.resample_poly 的优势:
            - 先上采样再下采样，内置抗混叠低通滤波
            - 保留 QRS 波群等高频成分的形态（线性插值会模糊）
            - up/down 因子自动约简 (math.gcd)

        参数:
            ecg: 形状 (12, L) 的原始信号。
            orig_fs: 原始采样率。
            target_fs: 目标采样率。

        返回:
            形状 (12, L_new) 的重采样信号。
        """
        orig_len = ecg.shape[1]

        # 退化情况：信号太短无法重采样
        if orig_len < 2:
            logger.warning(f"信号太短 ({orig_len} 采样点)，无法重采样")
            return ecg

        # 计算重采样比例并约简
        from math import gcd
        up = int(target_fs)
        down = int(orig_fs)
        g = gcd(up, down)
        up //= g
        down //= g

        # 限制 up/down 因子上限防止内存溢出
        max_factor = 100
        while up > max_factor or down > max_factor:
            up = (up + 1) // 2
            down = (down + 1) // 2

        # 计算目标长度
        target_len = round(orig_len * target_fs / orig_fs)
        resampled = np.zeros((ecg.shape[0], target_len), dtype=np.float32)

        for i in range(ecg.shape[0]):
            # resample_poly 保真度高但速度较慢
            out = signal.resample_poly(
                ecg[i].astype(np.float64), up, down
            ).astype(np.float32)
            # 对齐长度（resample_poly 可能产生略多/略少的样本）
            actual_len = len(out)
            if actual_len >= target_len:
                resampled[i] = out[:target_len]
            else:
                resampled[i, :actual_len] = out

        return resampled

    def _extract_segment(self, ecg: np.ndarray) -> Tuple[np.ndarray, dict]:
        """从 ECG 信号中提取固定长度片段。

        策略:
            - L == target: 不变
            - L > target:  居中裁剪
            - L < target:  对称零填充

        返回:
            (segment, info_dict)，info_dict 包含 pad_left/pad_right
            供后续归一化步骤使用。
        """
        L = ecg.shape[1]
        target = self.target_length

        if L == target:
            return ecg, {"segment_mode": "exact", "original_length": L,
                         "pad_left": 0, "pad_right": 0}

        if L > target:
            # 居中裁剪
            start = (L - target) // 2
            return ecg[:, start:start + target], {
                "segment_mode": "crop", "original_length": L,
                "crop_start": start, "pad_left": 0, "pad_right": 0,
            }
        else:
            # 对称零填充（记录填充区域以便归一化时排除）
            pad_total = target - L
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            padded = np.pad(
                ecg, ((0, 0), (pad_left, pad_right)),
                mode="constant", constant_values=0.0,
            )
            return padded, {
                "segment_mode": "pad", "original_length": L,
                "pad_left": pad_left, "pad_right": pad_right,
            }

    def _normalize(
        self, ecg: np.ndarray,
        pad_left: int = 0,
        pad_right: int = 0,
    ) -> Tuple[np.ndarray, dict]:
        """逐导联归一化。

        关键修复: 统计量计算只针对非填充的信号区域。
        如果对整个包含零填充的数组计算 mean/std，
        填充会稀释标准差 → 归一化后信号幅度放大 ~2x。

        参数:
            ecg: 可能包含零填充的信号。
            pad_left: 左侧零填充样本数。
            pad_right: 右侧零填充样本数。
        """
        if self.normalization == "none":
            return ecg, {"norm_method": "none"}

        L = ecg.shape[1]
        # 信号区域的起止索引（排除零填充）
        signal_start = pad_left
        signal_end = L - pad_right if pad_right > 0 else L

        normalized = np.zeros_like(ecg)
        means, stds = [], []

        for i in range(ecg.shape[0]):
            lead = ecg[i]
            # 只对信号区域计算统计量
            valid_region = lead[signal_start:signal_end] if signal_end > signal_start else lead

            if self.normalization == "zscore":
                mean = np.mean(valid_region)
                std = np.std(valid_region)
                if std < 1e-6:
                    normalized[i] = lead - mean  # 几乎平坦的信号，只去均值
                else:
                    normalized[i] = (lead - mean) / std
                means.append(float(mean))
                stds.append(float(std))

        info = {"norm_method": self.normalization}
        if means:
            info["per_lead_mean"] = means
            info["per_lead_std"] = stds

        return normalized, info

    def _clip_outliers(self, ecg: np.ndarray) -> np.ndarray:
        """裁剪异常值（超过 threshold*std 的值）。

        先 copy 再 clip，不修改原数组。
        注意: 在归一化后执行，裁剪的是归一化后的极端值。
        """
        if self.outlier_threshold <= 0:
            return ecg

        result = ecg.copy()
        for i in range(ecg.shape[0]):
            mean = np.mean(result[i])
            std = np.std(result[i])
            if std > 1e-8:
                upper = mean + self.outlier_threshold * std
                lower = mean - self.outlier_threshold * std
                result[i] = np.clip(result[i], lower, upper)

        return result

    def segment(
        self,
        ecg: np.ndarray,
        segment_length: Optional[int] = None,
        overlap: float = 0.0,
    ) -> np.ndarray:
        """将长 ECG 信号分割为多个重叠/非重叠窗口。

        参数:
            ecg: 形状 (12, L) 的信号。
            segment_length: 窗口长度（默认: target_length）。
            overlap: 窗口重叠比例 [0, 1)，0=不重叠，0.5=50%重叠。

        返回:
            形状 (N, 12, segment_length) 的分段数组。

        异常:
            ValueError: 如果 overlap 不在 [0, 1) 范围内。
        """
        if not 0.0 <= overlap < 1.0:
            raise ValueError(f"overlap 必须在 [0, 1) 范围内，当前值: {overlap}")

        seg_len = segment_length or self.target_length
        L = ecg.shape[1]

        # 信号短于目标长度：零填充
        if L < seg_len:
            pad_total = seg_len - L
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            padded = np.pad(
                ecg, ((0, 0), (pad_left, pad_right)),
                mode="constant", constant_values=0.0,
            )
            return padded[np.newaxis, ...]

        # 滑动窗口分段
        stride = int(seg_len * (1.0 - overlap))
        stride = max(1, stride)

        segments = []
        for start in range(0, L - seg_len + 1, stride):
            segments.append(ecg[:, start:start + seg_len])

        return np.stack(segments, axis=0).astype(np.float32)
