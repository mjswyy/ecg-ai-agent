#!/usr/bin/env python3
"""
ModelArts Training Entry Point — ECG Backbone Model Training on Ascend NPU.

Usage:
    Submit via ModelArts console or API as a training job.
    This script handles:
        1. OBS data sync → local cache
        2. Environment validation (NPU/CANN available)
        3. Training execution (standard train_backbone.py)
        4. Output sync → OBS

Environment variables (set by ModelArts):
    MA_INPUT_DIR  — OBS input data mount path (default: /cache/data)
    MA_OUTPUT_DIR — OBS output path for checkpoints/logs (default: /cache/output)
    BACKBONE      — Model backbone: inception_time | xresnet1d_101 | ecg_transformer
    EPOCHS        — Number of training epochs (default: 50)
    BATCH_SIZE    — Batch size (default: 128)
    LR            — Learning rate (default: 1e-4)
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def find_data_dir():
    """自动发现数据目录 — 扫描 /cache 下哪里能找到 train_manifest.json。

    ModelArts 将 OBS 数据挂载到容器后，具体路径取决于输入参数名和挂载方式。
    不用猜路径，直接扫描 manifest 文件定位。
    """
    from pathlib import Path

    search_root = Path("/cache/data")
    manifest_files = list(search_root.rglob("train_manifest.json"))

    if manifest_files:
        data_dir = str(manifest_files[0].parent)
        logger.info(f"发现数据目录: {data_dir} (找到 train_manifest.json)")
        return data_dir

    logger.warning("未找到 train_manifest.json，使用默认路径 /cache/data/processed")
    return "/cache/data/processed"


def validate_environment():
    """Verify NPU/CANN environment is working correctly."""
    import torch

    # Check NPU availability
    try:
        import torch_npu  # noqa: F401
        npu_count = torch.npu.device_count()
        if npu_count > 0:
            device_name = torch.npu.get_device_name(0)
            logger.info(f"Ascend NPU: {npu_count} device(s) — [{device_name}]")
        else:
            logger.warning("torch_npu loaded but no NPU devices found! "
                         "Check ASCEND_RT_VISIBLE_DEVICES.")
    except ImportError:
        logger.warning("torch_npu not installed. Will fall back to CPU/CUDA.")
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("No accelerator detected (NPU or CUDA). Training on CPU.")

    # Check CANN environment (Ascend only)
    ascend_home = os.environ.get("ASCEND_HOME", "")
    if ascend_home:
        logger.info(f"CANN: ASCEND_HOME={ascend_home}")
    else:
        logger.info("ASCEND_HOME not set (non-Ascend environment or CPU mode).")


def run_training(data_dir: str):
    """Execute the standard training script with ModelArts-compatible paths."""
    backbone = os.environ.get("BACKBONE", "inception_time")
    epochs = os.environ.get("EPOCHS", "50")
    batch_size = os.environ.get("BATCH_SIZE", "128")
    lr = os.environ.get("LR", "1e-4")
    output_dir = os.environ.get("MA_OUTPUT_DIR", "/cache/output")

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "scripts/train_backbone.py",
        "--backbone", backbone,
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--epochs", epochs,
        "--batch-size", batch_size,
        "--lr", lr,
        "--device", "auto",
    ]

    logger.info(f"Training: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        logger.error(f"Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    logger.info("Training completed successfully.")


def sync_output_to_obs():
    """Sync training outputs from local cache to OBS via moxing."""
    output_dir = os.environ.get("MA_OUTPUT_DIR", "/cache/output")
    local_dir = "/cache/output"

    try:
        import moxing as mox
        logger.info(f"Syncing output: {local_dir} → {output_dir}")
        mox.file.copy_parallel(local_dir, output_dir)
        logger.info("Output sync complete.")
    except ImportError:
        logger.info("moxing not available. Outputs remain at /cache/output.")
    except Exception as e:
        logger.error(f"Output sync failed: {e}")
        logger.info("Outputs remain at /cache/output (may be lost after job ends).")


def install_dependencies():
    """安装训练依赖。NumPy 必须先降级，再安装 requirements.txt。"""
    # NumPy 必须先降级（PyTorch 2.1 + CANN 8.0 基于 NumPy 1.x 编译）
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "numpy<2", "--quiet"],
        check=False,
    )
    # 安装项目全部依赖
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt", "--quiet"],
        check=False,
    )


def main():
    logger.info("=" * 60)
    logger.info("ECG AI Agent — ModelArts Training Job")
    logger.info("=" * 60)

    # 0. Install dependencies (must be before any torch import)
    install_dependencies()

    # 1. Validate NPU environment
    validate_environment()

    # 2. Auto-detect data directory
    data_dir = find_data_dir()

    # 3. Run training
    run_training(data_dir)

    # 4. Sync outputs to OBS
    sync_output_to_obs()

    logger.info("ModelArts training job finished.")


if __name__ == "__main__":
    main()
