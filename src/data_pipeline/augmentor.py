"""
ECG 数据增强器 — 6种面向心电图信号的时序增强方法。

全部增强方法均来自 ECG 增强相关文献，可用于训练数据扩充
和对比学习的正样本对生成。

增强方法:
    1. 随机基线漂移 — 模拟呼吸伪影（正弦波 0.1-0.5 Hz）
    2. 高斯噪声 — 模拟传感器/电极噪声（σ=0.01-0.05×信号std）
    3. 时间扭曲 — 模拟心率变异性（拉伸/压缩 0.8-1.2×）
    4. 幅度缩放 — 模拟电极阻抗变化（每导联独立 0.8-1.2×）
    5. 导联丢失 — 模拟电极脱落（随机置零 1-2 导联）
    6. 片段打乱 — 对比学习专用（随机排列时间片段）

参考文献:
    - Clifford et al. "Signal processing methods for heart rate variability"
    - Um et al. "Data augmentation of wearable sensor data..."
    - Strodthoff et al. "PTB-XL, a large publicly available ECG dataset"

使用示例:
    augmentor = ECGAugmentor(random_seed=42)
    augmented = augmentor(ecg_signal)  # 随机应用全部增强
    # 单独调用:
    noisy = augmentor.add_gaussian_noise(ecg, sigma=0.02)
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ECGAugmentor:
    """ECG 数据增强管道。

    每种增强可以独立启用/禁用和参数化，方便消融实验。

    关键设计:
        - 使用 np.random.Generator（替代老式 RandomState）
          → 在 DataLoader 多进程环境下更安全
        - 支持单样本 (12,L) 和批量 (N,12,L) 两种模式
        - 每种增强独立 apply_prob 控制
    """

    def __init__(
        self,
        baseline_wander: bool = True,         # 是否启用基线漂移
        gaussian_noise: bool = True,           # 是否启用高斯噪声
        time_warp: bool = True,               # 是否启用时间扭曲
        amplitude_scale: bool = True,          # 是否启用幅度缩放
        lead_dropout: bool = True,             # 是否启用导联丢失
        segment_shuffle: bool = False,         # 片段打乱（默认关闭，对比学习专用）
        wander_freq_range: Tuple[float, float] = (0.1, 0.5),   # 基线漂移频率范围
        wander_amplitude: float = 0.3,         # 基线漂移幅度（×信号std）
        noise_sigma_range: Tuple[float, float] = (0.01, 0.05), # 噪声标准差范围
        warp_scale_range: Tuple[float, float] = (0.8, 1.2),    # 时间扭曲缩放范围
        amp_scale_range: Tuple[float, float] = (0.8, 1.2),     # 幅度缩放范围
        max_drop_leads: int = 2,               # 最大置零导联数
        shuffle_chunk_seconds: float = 1.0,     # 片段打乱每段秒数
        apply_prob: float = 1.0,               # 默认每种增强的独立应用概率
        random_seed: Optional[int] = None,     # 随机种子（用于可复现性）
    ):
        self.apply_prob = apply_prob
        self.wander_freq_range = wander_freq_range
        self.wander_amplitude = wander_amplitude
        self.noise_sigma_range = noise_sigma_range
        self.warp_scale_range = warp_scale_range
        self.amp_scale_range = amp_scale_range
        self.max_drop_leads = max_drop_leads
        self.shuffle_chunk_seconds = shuffle_chunk_seconds

        # 增强开关（字典形式便于消融实验配置）
        self.augmentations = {
            "baseline_wander": baseline_wander,
            "gaussian_noise": gaussian_noise,
            "time_warp": time_warp,
            "amplitude_scale": amplitude_scale,
            "lead_dropout": lead_dropout,
            "segment_shuffle": segment_shuffle,
        }

        # 新版 NumPy Generator（fork-safe，替代 RandomState）
        self.rng = np.random.default_rng(random_seed)

    def __call__(
        self,
        ecg: np.ndarray,
        fs: float = 500.0,
        apply_prob: Optional[float] = None,
    ) -> np.ndarray:
        """应用随机数据增强。

        参数:
            ecg: ECG 信号，形状 (12, L) 或 (N, 12, L)。
            fs: 采样频率 (Hz)。
            apply_prob: 每种增强独立应用的概率（None=使用实例默认值）。

        返回:
            增强后的信号，形状与输入相同。
        """
        if apply_prob is None:
            apply_prob = self.apply_prob
        # 统一处理为 3D 批量格式
        single = (ecg.ndim == 2)
        if single:
            ecg = ecg[np.newaxis, ...]  # (12, L) → (1, 12, L)

        augmented = ecg.copy()
        for i in range(augmented.shape[0]):
            sample = augmented[i]

            # 每种增强独立随机决定是否应用
            if self.augmentations["baseline_wander"] and self.rng.random() < apply_prob:
                sample = self.add_baseline_wander(sample, fs)
            if self.augmentations["gaussian_noise"] and self.rng.random() < apply_prob:
                sample = self.add_gaussian_noise(sample)
            if self.augmentations["time_warp"] and self.rng.random() < apply_prob:
                sample = self.apply_time_warp(sample)
            if self.augmentations["amplitude_scale"] and self.rng.random() < apply_prob:
                sample = self.apply_amplitude_scaling(sample)
            if self.augmentations["lead_dropout"] and self.rng.random() < apply_prob:
                sample = self.apply_lead_dropout(sample)
            if self.augmentations["segment_shuffle"] and self.rng.random() < apply_prob:
                # 根据 chunk_seconds 和采样率自动计算分段数
                num_segments = max(2, int(sample.shape[1] / (fs * self.shuffle_chunk_seconds)))
                sample = self.apply_segment_shuffle(sample, fs, num_segments=num_segments)

            augmented[i] = sample

        if single:
            return augmented[0]
        return augmented

    # ------------------------------------------------------------------
    # 各增强方法实现
    # ------------------------------------------------------------------

    def add_baseline_wander(self, ecg: np.ndarray, fs: float) -> np.ndarray:
        """添加随机正弦基线漂移（模拟呼吸伪影）。

        每导联独立生成不同频率、幅度和相位的正弦波。
        幅度使用均匀分布（而非正态分布）以获得可控的最大幅值。

        参数:
            ecg: 形状 (12, L) 的信号。
            fs: 采样频率。

        返回:
            添加了基线漂移的信号。
        """
        L = ecg.shape[1]
        t = np.arange(L) / fs       # 时间轴（秒）
        result = ecg.copy()

        for i in range(ecg.shape[0]):
            freq = self.rng.uniform(*self.wander_freq_range)      # 随机频率
            amplitude = self.rng.uniform(0, self.wander_amplitude) * np.std(ecg[i])  # 随机幅度
            phase = self.rng.uniform(0, 2 * np.pi)                # 随机相位
            wander = amplitude * np.sin(2 * np.pi * freq * t + phase)
            result[i] += wander

        return result

    def add_gaussian_noise(self, ecg: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """添加高斯噪声（模拟传感器/电极噪声）。

        参数:
            ecg: 形状 (12, L) 的信号。
            sigma: 噪声标准差（×信号std）。None 表示随机采样。

        返回:
            添加了噪声的信号。
        """
        if sigma is None:
            sigma = self.rng.uniform(*self.noise_sigma_range)

        noise = self.rng.standard_normal(size=ecg.shape).astype(np.float32)

        # 每导联独立缩放噪声
        result = ecg.copy()
        for i in range(ecg.shape[0]):
            lead_std = np.std(ecg[i])
            if lead_std > 1e-8:
                result[i] += noise[i] * sigma * lead_std

        return result

    def apply_time_warp(self, ecg: np.ndarray, scale: Optional[float] = None) -> np.ndarray:
        """应用时间扭曲（拉伸/压缩时间轴）。

        通过插值来拉伸或压缩信号的时间轴，模拟心率变异性。
        边界处理使用边界值填充（而非外推）防止产生伪影。

        参数:
            ecg: 形状 (12, L) 的信号。
            scale: 扭曲因子（<1=压缩, >1=拉伸），None 表示随机。

        返回:
            时间扭曲后的信号，形状不变。
        """
        if scale is None:
            scale = self.rng.uniform(*self.warp_scale_range)

        L = ecg.shape[1]
        if L < 2:
            return ecg.copy()  # 信号太短无法插值

        result = np.zeros_like(ecg)
        orig_positions = np.arange(L)

        # 2A 🟠 修复：scale<1（压缩）时旧版 warped_positions 定义域仅到
        # (L-1)*scale，查询超出部分全部落在尾部 fill_value → 尾部最长
        # (1-scale)·L 为常量平台（信息丢失）。改为"内容居中承载"：
        # 原信号线性插值压缩到 keep 个采样点置于信号中心，两端补边界值，
        # 常量仅剩两端各 (1-scale)/2 比例。
        if scale < 1.0:
            keep = max(int(L * scale), 2)
            half = (L - keep) // 2
            xs = np.linspace(0.0, L - 1.0, keep)
            for i in range(ecg.shape[0]):
                ys = np.interp(xs, orig_positions, ecg[i])
                result[i, half:half + keep] = ys
                result[i, :half] = ecg[i][0]
                result[i, half + keep:] = ecg[i][-1]
            return result

        # 4B 审查修复：🟠-2 scale>1（拉伸）改为与压缩对称的"内容居中承载"。
        # 旧版在 orig_positions(0..L-1) 上查询被拉伸域 [0,(L-1)*scale]，等价于
        # "取前 1/scale 部分拉伸铺满"，系统性丢弃信号尾部（scale=1.2 丢末尾 ~17%）。
        # 现将原始信号沿原定义域 [0, L-1] 等距拉伸到 stretched_len 个采样点，
        # 再居中截取 L 个承载段——复检1 措辞终稿：头端让出 (stretched_len-L)//2、
        # 尾端让出剩余（stretched_len-L 为奇数时尾端多让 1），较旧版"只丢尾"
        # 更合理，无外推伪影（np.interp 查询域恰为 [0, L-1]）。
        stretched_len = int(round((L - 1) * scale)) + 1
        xs = np.linspace(0.0, L - 1.0, stretched_len)
        start = (stretched_len - L) // 2
        for i in range(ecg.shape[0]):
            stretched = np.interp(xs, orig_positions, ecg[i])
            result[i] = stretched[start:start + L]

        return result

    def apply_amplitude_scaling(self, ecg: np.ndarray, per_lead: bool = True) -> np.ndarray:
        """应用幅度缩放（模拟电极阻抗变化）。

        参数:
            ecg: 形状 (12, L) 的信号。
            per_lead: True=每导联独立缩放, False=全局统一缩放。

        返回:
            幅度缩放后的信号。
        """
        result = ecg.copy()
        if per_lead:
            for i in range(ecg.shape[0]):
                scale = self.rng.uniform(*self.amp_scale_range)
                result[i] *= scale
        else:
            scale = self.rng.uniform(*self.amp_scale_range)
            result *= scale
        return result

    def apply_lead_dropout(self, ecg: np.ndarray, num_drop: Optional[int] = None) -> np.ndarray:
        """随机置零导联（模拟电极脱落）。

        安全机制: num_drop 自动钳制到 [1, n_leads-1] 范围内，
        保证至少保留一条导联有信号。

        参数:
            ecg: 形状 (12, L) 的信号。
            num_drop: 置零的导联数。None 表示随机 (1~max_drop_leads)。

        返回:
            部分导联被置零的信号。
        """
        if num_drop is None:
            num_drop = self.rng.integers(1, self.max_drop_leads + 1)

        # 钳制：不能置零所有导联
        num_drop = min(num_drop, ecg.shape[0] - 1)
        if num_drop <= 0:
            return ecg.copy()

        result = ecg.copy()
        drop_indices = self.rng.choice(ecg.shape[0], size=num_drop, replace=False)
        result[drop_indices] = 0.0
        return result

    def apply_segment_shuffle(
        self, ecg: np.ndarray, fs: float, num_segments: int = 4,
    ) -> np.ndarray:
        """打乱时间片段（对比学习专用增强）。

        将信号均匀分成 num_segments 段，随机排列顺序。
        所有导联同步打乱以保持导联间关系。

        参数:
            ecg: 形状 (12, L) 的信号。
            fs: 采样频率（用于计算分段数）。
            num_segments: 分段数。

        返回:
            时间片段被随机排列的信号。
        """
        L = ecg.shape[1]
        seg_len = L // num_segments
        if seg_len < 10:
            # 4B 审查修复：💡-1 返回副本，保持"不修改入参"语义一致
            return ecg.copy()  # 分段太短，跳过

        # 分割
        segments = []
        for s in range(num_segments):
            start = s * seg_len
            end = start + seg_len if s < num_segments - 1 else L
            segments.append(ecg[:, start:end])

        # 随机排列
        order = self.rng.permutation(num_segments)
        # 4B 审查修复：💡-1 删除不可达的 >L/<L 对齐分支——segments 精确铺满
        # [0, L]（末段 end=L），concatenate 后长度恒等于 L。
        shuffled = np.concatenate([segments[i] for i in order], axis=1)

        return shuffled

    def get_params(self) -> Dict:
        """获取当前增强参数（用于日志记录和可复现性）。"""
        return {
            "wander_freq_range": self.wander_freq_range,
            "wander_amplitude": self.wander_amplitude,
            "noise_sigma_range": self.noise_sigma_range,
            "warp_scale_range": self.warp_scale_range,
            "amp_scale_range": self.amp_scale_range,
            "max_drop_leads": self.max_drop_leads,
            "enabled_augmentations": [k for k, v in self.augmentations.items() if v],
        }
