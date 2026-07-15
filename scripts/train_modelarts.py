#!/usr/bin/env python3
"""
ModelArts Training Entry Point — ECG Backbone Model Training on Ascend NPU.

Usage:
    Submit via ModelArts console or API as a training job.
    This script handles:
        1. Dependency installation (numpy<2 + requirements.txt)
        2. Auto-detect code & data directories
        3. Environment validation (NPU/CANN available)
        4. Training execution (standard train_backbone.py)
        5. Output sync → OBS

Environment variables (set by ModelArts):
    MA_JOB_DIR   — Code download directory (default: /home/ma-user/modelarts/user-job-dir)
    MA_INPUT_DIR — OBS input data mount path (default: /cache/data)
    MA_OUTPUT_DIR— OBS output path for checkpoints/logs (default: /cache/output)
    BACKBONE     — Model backbone: inception_time | xresnet1d_101 | ecg_transformer
    EPOCHS       — Number of training epochs (default: 50)
    BATCH_SIZE   — Batch size (default: 128)
    LR           — Learning rate (default: 1e-4)
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def find_project_root():
    """Find the project root directory (where scripts/ and requirements.txt live).

    ModelArts downloads code to ${MA_JOB_DIR}/ecg-ai-agent/.
    We also check relative to this script's location.
    """
    # Primary: MA_JOB_DIR (ModelArts code download directory)
    job_dir = os.environ.get("MA_JOB_DIR", "")
    if job_dir:
        candidate = Path(job_dir) / "ecg-ai-agent"
        if (candidate / "scripts" / "train_backbone.py").exists():
            logger.info(f"项目目录: {candidate}")
            return candidate

    # Fallback: relative to this script
    script_dir = Path(__file__).resolve().parent.parent
    if (script_dir / "scripts" / "train_backbone.py").exists():
        logger.info(f"项目目录: {script_dir}")
        return script_dir

    logger.error("找不到项目目录（scripts/train_backbone.py 不存在）")
    sys.exit(1)


def find_and_fix_data():
    """Auto-detect data directory and fix manifest paths.

    ModelArts mounts OBS data to unpredictable paths. This function:
    1. Scans common mount points for train_manifest.json
    2. Fixes signal_file paths in manifests (OBS upload can add directory prefixes)
    3. Returns the data directory

    All .npy files are guaranteed to be in the same directory as the manifest,
    so we strip any directory prefix from signal_file paths.
    """
    search_roots = [
        Path("/home/ma-user/modelarts/inputs"),
        Path("/cache/data"),
        Path("/cache"),
    ]

    import json

    for root in search_roots:
        if not root.exists():
            continue
        manifests = list(root.rglob("train_manifest.json"))
        if not manifests:
            continue

        manifest_path = manifests[0]
        data_dir = str(manifest_path.parent)
        logger.info(f"发现数据目录: {data_dir}")

        # Fix all manifests: strip directory prefix from signal_file paths
        with open(manifest_path) as f:
            last_manifest = json.load(f)
        for split in ["train", "val", "test"]:
            mp = Path(data_dir) / f"{split}_manifest.json"
            if not mp.exists():
                continue
            with open(mp) as f:
                manifest = json.load(f)
            last_manifest = manifest
            fixed = 0
            for rec in manifest["files"]:
                old = rec["signal_file"]
                if "/" in old:
                    rec["signal_file"] = Path(old).name
                    fixed += 1
            if fixed > 0:
                logger.info(f"  修复 {split}_manifest.json: {fixed} 条路径")
                with open(mp, "w") as f:
                    json.dump(manifest, f, indent=2)

        # Verify a sample file loads
        import numpy as np
        sample_file = last_manifest["files"][0]["signal_file"]
        sample_path = Path(data_dir) / sample_file
        if sample_path.exists():
            np.load(sample_path)
            logger.info(f"  数据验证通过: {sample_file}")
            return data_dir
        logger.warning(f"  文件不存在: {sample_path}")

    logger.error("未找到可用的训练数据")
    sys.exit(1)


def validate_environment():
    """Verify NPU/CANN environment is working correctly."""
    import torch

    try:
        import torch_npu  # noqa: F401
        npu_count = torch.npu.device_count()
        if npu_count > 0:
            device_name = torch.npu.get_device_name(0)
            logger.info(f"Ascend NPU: {npu_count} device(s) — [{device_name}]")
        else:
            logger.warning("torch_npu loaded but no NPU devices found!")
    except ImportError:
        logger.warning("torch_npu not installed. Will fall back to CPU/CUDA.")
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("No accelerator detected (NPU or CUDA). Training on CPU.")


def install_dependencies(project_root: Path):
    """Install training dependencies. NumPy must be downgraded first."""
    req_path = project_root / "requirements.txt"

    # NumPy must be downgraded before any torch import
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "numpy<2", "--quiet"],
        check=False,
    )

    # Install from requirements.txt (absolute path)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(req_path), "--quiet"],
        check=False,
    )


def run_training(project_root: Path, data_dir: str):
    """Execute the standard training script."""
    backbone = os.environ.get("BACKBONE", "inception_time")
    epochs = os.environ.get("EPOCHS", "50")
    batch_size = os.environ.get("BATCH_SIZE", "128")
    lr = os.environ.get("LR", "1e-4")
    output_dir = os.environ.get("MA_OUTPUT_DIR", "/cache/output")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    train_script = project_root / "scripts" / "train_backbone.py"

    cmd = [
        sys.executable, str(train_script),
        "--backbone", backbone,
        "--data-dir", data_dir,
        "--output-dir", output_dir,
        "--epochs", epochs,
        "--batch-size", batch_size,
        "--lr", lr,
        "--device", "auto",
    ]

    logger.info(f"Training: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(project_root), check=False)

    if result.returncode != 0:
        logger.error(f"Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)

    logger.info("Training completed successfully.")


def sync_output_to_obs():
    """Sync training outputs to OBS."""
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


def main():
    logger.info("=" * 60)
    logger.info("ECG AI Agent — ModelArts Training Job")
    logger.info("=" * 60)

    # 0. Find project root & install dependencies
    project_root = find_project_root()
    install_dependencies(project_root)

    # 1. Validate NPU environment
    validate_environment()

    # 2. Auto-detect data directory & fix manifest paths
    data_dir = find_and_fix_data()

    # 3. Run training
    run_training(project_root, data_dir)

    # 4. Sync outputs to OBS
    sync_output_to_obs()

    logger.info("ModelArts training job finished.")


if __name__ == "__main__":
    main()
