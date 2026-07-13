"""Tests for ECG Data Loader."""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.loader import ECGLoader, ECGSample


def test_loader_init():
    """Test that loader discovers data sources."""
    raw_dir = Path("c:/Users/llyun/Desktop/ecg资料/"
                    "classification-of-12-lead-ecgs-the-physionetcomputing-"
                    "in-cardiology-challenge-2020-1.0.2/training")

    if not raw_dir.exists():
        print(f"SKIP: Raw data directory not found at {raw_dir}")
        return

    loader = ECGLoader(raw_dir)
    print(f"Sources found: {loader.sources}")
    print(f"Estimated records: {loader.get_record_count()}")
    assert len(loader.sources) > 0, "No sources found!"


def test_load_single_record():
    """Test loading a single record."""
    raw_dir = Path("c:/Users/llyun/Desktop/ecg资料/"
                    "classification-of-12-lead-ecgs-the-physionetcomputing-"
                    "in-cardiology-challenge-2020-1.0.2/training")

    if not raw_dir.exists():
        print(f"SKIP: Raw data directory not found at {raw_dir}")
        return

    loader = ECGLoader(raw_dir)
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
    raw_dir = Path("c:/Users/llyun/Desktop/ecg资料/"
                    "classification-of-12-lead-ecgs-the-physionetcomputing-"
                    "in-cardiology-challenge-2020-1.0.2/training")

    if not raw_dir.exists():
        print(f"SKIP: Raw data directory not found at {raw_dir}")
        return

    loader = ECGLoader(raw_dir)
    count = 0
    for sample in loader.iter_records(sources=["cpsc_2018"], max_records=10):
        assert isinstance(sample, ECGSample)
        count += 1
        print(f"  [{count}] {sample.record_id}: {len(sample.dx_codes)} labels")

    print(f"Successfully iterated {count} records")
    assert count > 0, "No records iterated!"


def test_get_statistics():
    """Test statistics collection."""
    raw_dir = Path("c:/Users/llyun/Desktop/ecg资料/"
                    "classification-of-12-lead-ecgs-the-physionetcomputing-"
                    "in-cardiology-challenge-2020-1.0.2/training")

    if not raw_dir.exists():
        print(f"SKIP: Raw data directory not found at {raw_dir}")
        return

    loader = ECGLoader(raw_dir)
    stats = loader.get_statistics(max_records=50)
    print("Statistics:")
    print(f"  Sources: {stats['sources']}")
    print(f"  FS values: {stats['fs_values']}")
    print(f"  Duration stats: {stats.get('duration_stats', {})}")
    print(f"  Sex distribution: {stats['sexes']}")
    print(f"  Top labels: {sorted(stats['label_counts'].items(), key=lambda x: -x[1])[:5]}")


if __name__ == "__main__":
    test_loader_init()
    test_load_single_record()
    test_iter_records()
    test_get_statistics()
    print("\n✅ All loader tests passed!")
