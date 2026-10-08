"""torchvision 桩模块 — 绕过本机损坏的 torchvision 安装（仅限纯文本任务）。

背景:
    本机 torchvision 安装损坏（ops/__init__.py 与 drop_block.py 版本错配），
    `import torchvision` 直接抛 ImportError。
    transformers.image_utils 在模块顶部 `from torchvision.io import ...`，
    导致任何模型加载（包括纯文本 BERT）都失败。

用法:
    在 import transformers 之前:
        import src.utils.torchvision_stub  # noqa: F401

注意:
    - 本桩仅提供 transformers 导入期需要的符号，不提供任何真实图像功能；
      调用图像相关函数会抛 RuntimeError。
    - 图像任务（MEETI 图片、GEM 式多模态）前必须重装 torchvision:
      pip install --force-reinstall --no-deps torchvision==0.22.1
"""

import sys
import types
import importlib.machinery

_STUBBED = False


def _mk(name, attrs):
    m = types.ModuleType(name)
    m.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


def _fail(*args, **kwargs):
    raise RuntimeError("torchvision 桩模块不支持实际图像操作，请重装 torchvision")


def install():
    global _STUBBED
    if _STUBBED:
        return
    if "torchvision" in sys.modules:
        return  # 已导入（可能环境已修复）
    # 真实 torchvision 可正常导入 → 无需桩，直接让位
    try:
        import torchvision  # noqa: F401
        return
    except Exception:
        pass

    tv = _mk("torchvision", {})
    tv.io = _mk("torchvision.io", {
        "ImageReadMode": type("ImageReadMode", (), {"RGB": 0}),
        "decode_image": _fail,
    })
    # torchvision.transforms.InterpolationMode 枚举成员（transformers.image_utils 构建映射所需）
    _interp = type("InterpolationMode", (), {
        "NEAREST": 0, "NEAREST_EXACT": 1, "BOX": 2, "BILINEAR": 3,
        "HAMMING": 4, "BICUBIC": 5, "LANCZOS": 6,
    })
    tv.transforms = _mk("torchvision.transforms", {"InterpolationMode": _interp})
    tv.transforms.functional = _mk("torchvision.transforms.functional", {
        "pil_to_tensor": _fail,
    })

    sys.modules["torchvision"] = tv
    sys.modules["torchvision.io"] = tv.io
    sys.modules["torchvision.transforms"] = tv.transforms
    sys.modules["torchvision.transforms.functional"] = tv.transforms.functional
    _STUBBED = True


install()
