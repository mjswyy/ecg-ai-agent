"""Tests for ECG Data Loader.

第三轮审查 3G 🟠-1 修复：旧版硬编码外部绝对路径不存在时静默 return
（0 个 assert 被评估却打印"✅ 全部通过"——假阳性测试）。
现改为：数据缺失时显式失败（抛错），并支持 DATA_RAW_DIR 环境变量覆盖路径。
"""

import os
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.loader import ECGLoader, ECGSample

RAW_DIR = Path(os.environ.get(
    "DATA_RAW_DIR",
    "c:/Users/llyun/Desktop/ecg资料/"
    "classification-of-12-lead-ecgs-the-physionetcomputing-"
    "in-cardiology-challenge-2020-1.0.2/training",
))


def _require_raw_dir():
    if not RAW_DIR.exists():
        raise AssertionError(
            f"原始数据目录不存在: {RAW_DIR}（设置 DATA_RAW_DIR 环境变量指向"
            " PhysioNet 2020 training 目录后重跑；测试绝不静默跳过）")


def test_loader_init():
    """Test that loader discovers data sources."""
    _require_raw_dir()
    loader = ECGLoader(RAW_DIR)
    print(f"Sources found: {loader.sources}")
    print(f"Estimated records: {loader.get_record_count()}")
    assert len(loader.sources) > 0, "No sources found!"


def test_load_single_record():
    """Test loading a single record."""
    _require_raw_dir()
    loader = ECGLoader(RAW_DIR)
    # Load first record from cpsc_2018
    sample = loader.load_record("cpsc_2018/g1/A0001")
    assert sample is not None, "Failed to load sample record"
    assert sample.signal.shape[0] == 12, f"Expected 12 leads, got {sample.signal.shape[0]}"
    assert sample.fs == 500, f"Expected 500Hz, got {sample.fs}"

    print(f"Loaded: {sample}")
    print(f"  Signal shape: {sample.signal.shape}")
    print(f"  Duration: {sample.duration:.1f}s")
    print(f"  Age: {sample.age}")
    print(f"  Sex: {sample.sex}")
    print(f"  Dx codes: {sample.dx_codes}")


def test_iter_records():
    """Test iterating over records."""
    _require_raw_dir()
    loader = ECGLoader(RAW_DIR)
    count = 0
    for sample in loader.iter_records(sources=["cpsc_2018"], max_records=10):
        assert isinstance(sample, ECGSample)
        count += 1
        print(f"  [{count}] {sample.record_id}: {len(sample.dx_codes)} labels")

    print(f"Successfully iterated {count} records")
    assert count > 0, "No records iterated!"


def test_get_statistics():
    """Test statistics collection."""
    _require_raw_dir()
    loader = ECGLoader(RAW_DIR)
    stats = loader.get_statistics(max_records=50)
    print("Statistics:")
    print(f"  Sources: {stats['sources']}")
    print(f"  FS values: {stats['fs_values']}")
    print(f"  Duration stats: {stats.get('duration_stats', {})}")
    print(f"  Sex distribution: {stats['sexes']}")
    print(f"  Top labels: {sorted(stats['label_counts'].items(), key=lambda x: -x[1])[:5]}")

    # 4A/4J 审查修复：🟡-9 补真实断言（旧版只打印、无一条 assert，数据目录
    # 存在即"通过"——统计逻辑出错也测不出）。数据缺失时由 _require_raw_dir
    # 显式失败，绝不静默跳过。
    assert stats.get("sources"), "sources 字段缺失或为空"
    assert stats.get("fs_values"), "fs_values 字段缺失或为空"
    assert stats.get("label_counts"), "label_counts 字段缺失或为空"
    assert sum(stats["label_counts"].values()) > 0, \
        "label_counts 汇总计数应为正（统计逻辑疑似出错）"


if __name__ == "__main__":
    n_pass = 0
    for fn in (test_loader_init, test_load_single_record,
               test_iter_records, test_get_statistics):
        fn()
        n_pass += 1
        print(f"  ✓ {fn.__name__}")
    print(f"\n✅ All {n_pass} loader tests passed（{n_pass} 个断言组均真实执行）!")
