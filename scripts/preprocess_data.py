#!/usr/bin/env python3
# ⚠️ DEPRECATED（2026-08-24，检查报告第六步）: 4096 旧协议预处理；主用 preprocess_data_5k.py。本目录 processed/ 的 train/val/test manifest 仍是划分事实来源，保留勿删
"""Data Preprocessing Script - Convert raw PhysioNet 2020 data to processed .npy files.

This script:
    1. Reads all ECG records from the raw dataset
    2. Applies preprocessing (filtering, resampling, normalization)
    3. Extracts labels via SNOMED CT mapping
    4. Creates train/val/test splits
    5. Saves processed .npy files and manifest JSON

Usage:
    python scripts/preprocess_data.py --raw-dir path/to/training
    python scripts/preprocess_data.py --max-records 1000  # Quick test run
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.loader import ECGLoader
from src.data_pipeline.preprocessor import ECGPreprocessor
from src.data_pipeline.label_extractor import LabelExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def _default_raw_dir():
    """3F-Y5 修复：探测原始数据目录（旧版硬编码互不一致的绝对路径
    Desktop\\ECG vs Desktop\\ecg资料，换机即失效）。探测失败返回 None，
    由调用方要求显式传 --raw-dir。"""
    repo = Path(__file__).parent.parent
    stem = "classification-of-12-lead-ecgs-the-physionetcomputing-" \
           "in-cardiology-challenge-2020-1.0.2"
    candidates = [
        repo.parent / stem / "training",
        Path.home() / "Desktop" / "ecg资料" / stem / "training",
        Path.home() / "Desktop" / "ECG" / stem / "training",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def main():
    parser = argparse.ArgumentParser(description="Preprocess PhysioNet 2020 ECG data")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=_default_raw_dir(),
        help="Path to raw training data directory（不传时自动探测；探测失败必须显式指定）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/physionet2020/processed",
        help="Output directory for processed files",
    )
    parser.add_argument(
        "--target-fs", type=float, default=500.0,
        help="Target sampling rate (Hz)",
    )
    parser.add_argument(
        "--target-length", type=int, default=4096,
        help="Target signal length (samples)",
    )
    parser.add_argument(
        "--max-records", type=int, default=None,
        help="Maximum records to process (for quick testing)",
    )
    parser.add_argument(
        "--sources", type=str, nargs="+", default=None,
        help="Data sources to include (default: all)",
    )
    args = parser.parse_args()

    if not args.raw_dir:
        sys.exit("未探测到原始数据目录——请用 --raw-dir 显式指定（3F-Y5）")
    raw_dir = Path(args.raw_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components
    loader = ECGLoader(raw_dir)
    preprocessor = ECGPreprocessor(
        target_fs=args.target_fs,
        target_length=args.target_length,
    )
    label_extractor = LabelExtractor(num_classes=27)

    # Collect all records
    logger.info(f"Scanning records from {raw_dir}...")
    total = loader.get_record_count(args.sources)
    logger.info(f"Found ~{total} records")
    if args.max_records:
        logger.info(f"Limiting to {args.max_records} records")

    # Process records
    manifest: List[Dict] = []
    skipped = 0
    failed_sources = set()

    for sample in tqdm(
        loader.iter_records(sources=args.sources, max_records=args.max_records),
        total=min(total, args.max_records or total),
        desc="Processing",
    ):
        try:
            # Preprocess signal
            signal_clean = preprocessor(
                sample.signal,
                original_fs=sample.fs,
            )

            # Encode labels
            labels = label_extractor.encode(sample.dx_codes, format="multi_hot")
            labels_list = labels.tolist()

            # Generate output filename
            safe_name = f"{sample.source}_{sample.record_id}"
            signal_file = f"{safe_name}.npy"
            signal_path = output_dir / signal_file

            # Save
            np.save(signal_path, signal_clean)

            manifest.append({
                "record_id": sample.record_id,
                "source": sample.source,
                "signal_file": signal_file,
                "fs_original": float(sample.fs),
                "fs_target": args.target_fs,
                "duration_original": sample.duration,
                "signal_shape": list(signal_clean.shape),
                "age": sample.age if (sample.age is not None and not (isinstance(sample.age, float) and (np.isnan(sample.age) or np.isinf(sample.age)))) else None,
                "sex": sample.sex,
                "dx_codes": sample.dx_codes,
                "labels": labels_list,
                "has_labels": len(sample.dx_codes) > 0,
            })

        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as e:
            logger.warning(f"Failed to process {sample.record_id}: {e}")
            skipped += 1
            failed_sources.add(str(e)[:200])

    # Create train/val/test splits
    train_ratio, val_ratio = 0.70, 0.15
    rng = np.random.RandomState(42)

    # Split by data source to prevent leakage
    sources = sorted(set(m["source"] for m in manifest))
    train_files, val_files, test_files = [], [], []

    for source in sources:
        source_records = [m for m in manifest if m["source"] == source]
        indices = rng.permutation(len(source_records))
        n = len(source_records)
        if n >= 3:
            # 3F-💡-3：小源 floor 保护（train 优先、val 至少 1、test 让渡兜底）
            n_train = max(int(n * train_ratio), 1)
            n_val = max(int(n * val_ratio), 1)
            n_test = n - n_train - n_val
            if n_test < 1 and n_val > 1:
                n_val -= min(n_val - 1, 1 - n_test)
                n_test = n - n_train - n_val
            if n_test < 1 and n_train > 1:
                n_train -= 1
                n_test = 1
        else:
            # 极端小源（n<3）：全进训练集，避免负计数
            n_train, n_val, n_test = n, 0, 0

        train_files.extend([source_records[i] for i in indices[:n_train]])
        val_files.extend([source_records[i] for i in indices[n_train:n_train + n_val]])
        test_files.extend([source_records[i] for i in indices[n_train + n_val:]])

    # Save manifests
    for split_name, split_files in [
        ("train", train_files),
        ("val", val_files),
        ("test", test_files),
    ]:
        manifest_path = output_dir / f"{split_name}_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"files": split_files, "count": len(split_files)}, f, indent=2)
        logger.info(f"  {split_name}: {len(split_files)} records → {manifest_path}")

    # Save label mapping (ensure labels directory exists)
    labels_dir = output_dir.parent / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    label_extractor.save_mapping(labels_dir / "snomed_to_class.json")

    # Summary
    total_processed = len(manifest)
    logger.info(f"\n{'='*50}")
    logger.info(f"Preprocessing complete!")
    logger.info(f"  Total processed: {total_processed}")
    logger.info(f"  Skipped: {skipped}")
    if failed_sources:
        logger.warning(f"  Failure reasons: {list(failed_sources)[:5]}")
    logger.info(f"  Output: {output_dir}")
    logger.info(f"  Train/Val/Test: {len(train_files)}/{len(val_files)}/{len(test_files)}")

    # Label statistics
    label_counts = {}
    for m in manifest:
        for i, v in enumerate(m["labels"]):
            if v > 0:
                class_name = label_extractor.class_names[i] if i < len(label_extractor.class_names) else f"class_{i}"
                label_counts[class_name] = label_counts.get(class_name, 0) + 1

    logger.info(f"  Classes with data: {len(label_counts)}/27")
    logger.info(f"  Top 5 classes: {sorted(label_counts.items(), key=lambda x: -x[1])[:5]}")

    if label_counts:
        rare = [(k, v) for k, v in label_counts.items() if v < 50]
        if rare:
            logger.warning(f"  Rare classes (<50 samples): {rare}")


if __name__ == "__main__":
    main()
