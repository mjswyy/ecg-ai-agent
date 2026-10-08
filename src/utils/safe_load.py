"""安全检查点加载 — 2B 🟠-5 修复。

torch 2.6+ 默认 weights_only=True（禁止 pickle 任意对象，防恶意检查点代码执行）。
官方 ECGFounder ckpt（保存于 numpy 1.x 环境）含 numpy.core.multiarray.scalar 与
numpy.dtype 全局，白名单注册后即可在 weights_only=True 下加载，无需全局关闭。

用法:
    from src.utils.safe_load import safe_torch_load
    ckpt = safe_torch_load(path, map_location="cpu")
"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

_WHITELIST_REGISTERED = False


class _NumpyScalarShim:
    """扮演 pickle 中的 numpy.core.multiarray.scalar（numpy 1.x 模块名）：

    torch 的 weights_only allowlist 按 module.qualname 匹配；numpy 2.x 该
    符号在 numpy._core.multiarray 下，故用 shim 转发到真实实现。
    """

    def __new__(cls, *args, **kwargs):
        from numpy._core.multiarray import scalar as _scalar
        return _scalar(*args, **kwargs)


_NumpyScalarShim.__module__ = "numpy.core.multiarray"
_NumpyScalarShim.__qualname__ = "scalar"


class _NumpyScalarShim2:
    """同上——numpy 2.x 模块名 numpy._core.multiarray.scalar（4E-ORANGE-2 修复：
    旧白名单只注册 1.x 名，numpy 2.x 环境下标量实例仍 UnpicklingError）。"""

    def __new__(cls, *args, **kwargs):
        from numpy._core.multiarray import scalar as _scalar
        return _scalar(*args, **kwargs)


_NumpyScalarShim2.__module__ = "numpy._core.multiarray"
_NumpyScalarShim2.__qualname__ = "scalar"


def register_safe_globals() -> None:
    """注册加载本项目检查点所需的 numpy 全局（幂等）。"""
    global _WHITELIST_REGISTERED
    if _WHITELIST_REGISTERED:
        return
    try:
        import numpy.dtypes as nd
    except ImportError:  # numpy < 1.20 无 dtypes 子模块
        nd = None
    globs = [_NumpyScalarShim, _NumpyScalarShim2,
             np.dtype, np.ndarray,
             np.float64, np.int64, np.float32]
    # numpy ndarray 反序列化依赖 _reconstruct：torch≥2.6 的二元组格式为
    # (可调用对象, 全路径字符串)——把 numpy 1.x 的 pickle 名映射到 2.x 真实实现
    # （覆盖新旧两种 pickle 产出的检查点）
    try:
        from numpy._core.multiarray import _reconstruct as _np_reconstruct
        globs.append((_np_reconstruct, "numpy.core.multiarray._reconstruct"))
        globs.append((_np_reconstruct, "numpy._core.multiarray._reconstruct"))
    except ImportError:
        pass
    if nd is not None:
        for name in ("Float64DType", "Int64DType", "Float32DType", "Int32DType"):
            cls = getattr(nd, name, None)
            if cls is not None:
                globs.append(cls)
    try:
        torch.serialization.add_safe_globals(globs)
        _WHITELIST_REGISTERED = True
    except Exception as e:  # 旧版 torch 无 add_safe_globals
        # 5E 审查（🟡-3）：旧日志声称"回退 weights_only=False"与事实不符——
        # 本函数从不回退；白名单未注册时后续加载可能失败（由 safe_torch_load
        # 的显式 fallback 参数决定），如实修正日志措辞
        logger.warning(f"add_safe_globals 不可用（{e}）：白名单未注册，"
                       f"weights_only=True 加载 numpy 检查点可能失败")


def safe_torch_load(path, map_location=None, weights_only: bool = True,
                    fallback: bool = False):
    """torch.load 的安全封装：默认 weights_only=True + 本项目白名单。

    4E-RED-1 修复：加载失败不再静默回退 weights_only=False——旧版对任何异常
    （含恶意 pickle 触发白名单拒绝）都回退完整 pickle 反序列化，使"安全加载"
    对不可信检查点形同虚设。现仅当调用方显式 fallback=True（已确认检查点
    来源可信）才允许回退；否则抛出带说明的 RuntimeError。

    weights_only=False 仅用于明确需要完整对象图的旧代码路径。
    """
    if weights_only:
        register_safe_globals()
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only)
    except Exception as e:
        if weights_only and fallback:
            logger.warning(f"weights_only=True 加载失败({e})；"
                           f"显式 fallback=True → 回退 weights_only=False（仅限可信检查点）")
            return torch.load(path, map_location=map_location, weights_only=False)
        raise RuntimeError(
            f"安全加载失败（{type(e).__name__}: {e}）。若已确认该检查点来源可信，"
            f"可显式传 fallback=True 允许完整 pickle 反序列化；"
            f"否则请检查白名单或文件完整性。") from e
