"""
统一设备检测与AMP适配工具 / Unified Device Detection & AMP Adapter.

支持 NVIDIA GPU (CUDA) 和 Huawei Ascend NPU 910B 的自动检测与AMP适配。
Provides auto-detection for CUDA / Ascend NPU / CPU and wraps AMP imports
so the rest of the codebase stays device-agnostic.

使用示例 / Usage:
    from src.utils.device_utils import detect_device, GradScaler, autocast

    device = detect_device()                          # auto-detect npu/cuda/cpu
    scaler = GradScaler(enabled=(device != "cpu"))   # works on both CUDA & NPU

    with autocast(enabled=(device != "cpu")):
        loss = model(data)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
"""

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# ============================================================================
# 设备检测 / Device Detection
# ============================================================================

def detect_device(device_str: str = "auto") -> str:
    """自动检测最佳可用计算设备 / Auto-detect best available compute device.

    检测优先级: NPU > CUDA > CPU
    Priority order: Ascend NPU > NVIDIA GPU > CPU

    参数 / Args:
        device_str: 指定设备名或 "auto" / Explicit device name or "auto".

    返回 / Returns:
        设备字符串: "npu", "cuda", 或 "cpu" / Device string.

    使用示例 / Usage:
        device = detect_device()          # "npu" on Ascend, "cuda" on GPU
        device = detect_device("cuda")    # "cuda" (forced)
    """
    if device_str == "auto":
        # 优先检测 Ascend NPU / Check NPU first
        if _has_npu():
            device = "npu"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
        logger.info(f"自动检测设备: {device}")
        return device

    # 验证用户指定的设备 / Validate user-specified device
    if device_str == "npu" and not _has_npu():
        logger.warning("指定了 NPU 设备但 torch.npu 不可用，回退到 CPU")
        return "cpu"
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("指定了 CUDA 设备但 torch.cuda 不可用，回退到 CPU")
        return "cpu"
    # 5E 审查（🟠-1）：显式多卡序号 "cuda:N"/"npu:N" 同样校验可用性与越界——
    # 旧版直接原样返回，训练脚本 torch.device(detect_device(...)) 绕过
    # get_device_object 的越界保护。复检B（🟡-1）：仅校验"加速器可用"不够，
    # 加速器可用但序号越界（如 1-GPU 机上 "cuda:3"）仍会延迟到 .to() 崩溃
    for prefix, avail, count_fn in (
            ("cuda:", torch.cuda.is_available, torch.cuda.device_count),
            ("npu:", _has_npu, None)):
        if device_str.startswith(prefix):
            if not avail():
                logger.warning(f"指定 {device_str} 但 {prefix[:-1]} 不可用，回退到 CPU")
                return "cpu"
            try:
                idx = int(device_str[len(prefix):])
            except ValueError:
                logger.warning(f"无法解析设备序号 {device_str}，回退到 CPU")
                return "cpu"
            if count_fn is not None and idx >= count_fn():
                logger.warning(f"{device_str} 超出可见设备数({count_fn()})，回退到 CPU")
                return "cpu"
            return device_str
    return device_str


def _has_npu() -> bool:
    """检查 Ascend NPU 是否可用 / Check if Ascend NPU is available."""
    try:
        import torch_npu  # noqa: F401
        return torch.npu.is_available()
    except (ImportError, AttributeError):
        return False


def get_device_object(device_str: str = "auto") -> torch.device:
    """获取 torch.device 对象 / Get torch.device object.

    参数 / Args:
        device_str: 设备名或 "auto" / Device string or "auto".

    返回 / Returns:
        torch.device 实例 / torch.device instance.
    """
    resolved = detect_device(device_str)
    if resolved == "npu":
        return torch.device("npu:0")
    elif resolved == "cuda":
        return torch.device("cuda:0")
    elif resolved.startswith("cuda:") or resolved.startswith("npu:"):
        # 4E/4F 审查修复：🟠-4 —— 显式多卡序号（"cuda:N"/"npu:N"）此前走 else
        # 分支静默回退 CPU 且无告警。现解析索引并校验范围：越界/解析失败时告警回退 CPU。
        # 4E 定向复查（δ）：npu 越界同样校验（旧版只校验 cuda，NPU 机器上
        # 越界 npu 索引仍静默返回）
        try:
            dev = torch.device(resolved)
            idx = dev.index or 0
            if dev.type == "cuda" and idx >= torch.cuda.device_count():
                logger.warning(
                    f"{resolved} 超出可见 GPU 数({torch.cuda.device_count()})，回退 CPU")
                return torch.device("cpu")
            if dev.type == "npu":
                try:
                    npu_count = torch.npu.device_count()
                    if idx >= npu_count:
                        logger.warning(
                            f"{resolved} 超出可见 NPU 数({npu_count})，回退 CPU")
                        return torch.device("cpu")
                except (AttributeError, RuntimeError):
                    logger.warning(f"{resolved} 无法查询 NPU 数量，回退 CPU")
                    return torch.device("cpu")
            return dev
        except (RuntimeError, ValueError) as e:
            logger.warning(f"无法解析设备 '{resolved}'（{e}），回退 CPU")
            return torch.device("cpu")
    else:
        return torch.device("cpu")


def is_accelerator(device: torch.device) -> bool:
    """检查设备是否为加速器（GPU或NPU）/ Check if device is an accelerator."""
    return device.type in ("cuda", "npu")


# ============================================================================
# AMP 适配器 / AMP Adapter (Auto-select CUDA or NPU backend)
# ============================================================================

GradScaler = None    # type: ignore
autocast = None      # type: ignore


def _init_amp():
    """初始化 AMP 后端 / Initialize AMP backend based on available hardware.

    在模块加载时自动调用，优先使用 Ascend NPU AMP，
    回退到 CUDA AMP，最后回退到纯 FP32。

    Called automatically at module import. Prefers Ascend NPU AMP,
    falls back to CUDA AMP, then to pure FP32.
    """
    global GradScaler, autocast

    # 优先尝试 Ascend NPU AMP / Try NPU AMP first
    try:
        import torch_npu  # noqa: F401
        from torch.npu.amp import GradScaler as NPUGradScaler
        from torch.npu.amp import autocast as NPUAutocast
        GradScaler = NPUGradScaler
        autocast = NPUAutocast
        logger.debug("AMP: 使用 Ascend NPU 后端")
        return
    except (ImportError, AttributeError):
        pass

    # 回退到 CUDA AMP / Fall back to CUDA AMP
    # 第三轮审查 3C-AMP-1：GradScaler 优先 torch.amp（无 FutureWarning）；
    # autocast 保持 torch.cuda.amp（torch.amp.autocast 需 device_type 位置参数，
    # 与现有 enabled= 调用不兼容）
    try:
        from torch.amp import GradScaler as _TAGradScaler
        GradScaler = _TAGradScaler
    except (ImportError, AttributeError):
        from torch.cuda.amp import GradScaler as CUDAGradScaler
        GradScaler = CUDAGradScaler
    try:
        from torch.cuda.amp import autocast as CUDAAutocast
        autocast = CUDAAutocast
        logger.debug("AMP: 使用 CUDA 后端")
        return
    except (ImportError, AttributeError):
        pass

    # 纯 CPU 模式 / CPU-only fallback
    logger.warning("AMP 不可用（无 NPU/CUDA），将使用纯 FP32 训练")
    GradScaler = _DummyGradScaler
    autocast = _DummyAutocast


class _DummyGradScaler:
    """AMP 不可用时的空操作 GradScaler / No-op when AMP is unavailable."""

    def __init__(self, enabled: bool = False, **kwargs):
        self._enabled = enabled

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        pass

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        pass


class _DummyAutocast:
    """AMP 不可用时的空操作 autocast / No-op context manager."""

    def __init__(self, enabled: bool = True, **kwargs):
        self._enabled = enabled

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


# 模块导入时自动初始化 AMP 后端 / Auto-init on import
_init_amp()
