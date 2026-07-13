"""Tests for Preprocessor, Augmentor, LabelExtractor, and MetadataParser."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_pipeline.preprocessor import ECGPreprocessor
from src.data_pipeline.augmentor import ECGAugmentor
from src.data_pipeline.label_extractor import LabelExtractor
from src.data_pipeline.metadata_parser import MetadataParser


def test_preprocessor():
    """Test ECG preprocessing pipeline."""
    preprocessor = ECGPreprocessor(
        target_fs=500,
        target_length=4096,
        normalization="zscore",
    )

    # Generate synthetic ECG-like signal (sine wave + noise)
    fs = 500
    t = np.linspace(0, 10, 5000)
    synthetic = np.zeros((12, 5000))
    for i in range(12):
        synthetic[i] = np.sin(2 * np.pi * 1.2 * t) + 0.1 * np.random.randn(5000)

    # Process
    processed = preprocessor(synthetic, original_fs=fs)
    assert processed.shape == (12, 4096), f"Expected (12, 4096), got {processed.shape}"
    assert processed.dtype == np.float32, f"Expected float32, got {processed.dtype}"
    print(f"✅ Preprocessor: input (12, 5000) → output {processed.shape}")

    # Test with metadata
    processed, meta = preprocessor(synthetic, original_fs=fs, return_meta=True)
    print(f"   Metadata keys: {list(meta.keys())}")
    assert "final_shape" in meta

    # Test segment
    long_signal = np.random.randn(12, 15000)
    segments = preprocessor.segment(long_signal)
    print(f"   Segmented: (12, 15000) → {segments.shape}")
    assert segments.ndim == 3
    assert segments.shape[1] == 12


def test_augmentor():
    """Test data augmentation."""
    augmentor = ECGAugmentor(random_seed=42)

    # Synthetic signal
    ecg = np.random.randn(12, 4096).astype(np.float32) * 0.5

    # Test each augmentation individually
    print("Testing individual augmentations:")

    augmented = augmentor.add_baseline_wander(ecg.copy(), fs=500)
    assert augmented.shape == ecg.shape
    assert not np.allclose(augmented, ecg), "Baseline wander had no effect"
    print("  ✅ baseline_wander")

    augmented = augmentor.add_gaussian_noise(ecg.copy(), sigma=0.05)
    assert augmented.shape == ecg.shape
    assert not np.allclose(augmented, ecg)
    print("  ✅ gaussian_noise")

    augmented = augmentor.apply_time_warp(ecg.copy(), scale=0.9)
    assert augmented.shape == ecg.shape
    print("  ✅ time_warp")

    augmented = augmentor.apply_amplitude_scaling(ecg.copy())
    assert augmented.shape == ecg.shape
    assert not np.allclose(augmented, ecg)
    print("  ✅ amplitude_scale")

    augmented = augmentor.apply_lead_dropout(ecg.copy(), num_drop=2)
    assert augmented.shape == ecg.shape
    zero_leads = np.sum(np.all(augmented == 0, axis=1))
    assert zero_leads == 2, f"Expected 2 zeroed leads, got {zero_leads}"
    print("  ✅ lead_dropout")

    # Test combined augmentation
    augmented = augmentor(ecg)
    assert augmented.shape == ecg.shape
    print("  ✅ combined augmentations")

    # Test batch mode
    batch = np.random.randn(4, 12, 4096).astype(np.float32)
    augmented_batch = augmentor(batch)
    assert augmented_batch.shape == batch.shape
    print("  ✅ batch augmentation")


def test_label_extractor():
    """Test label extraction."""
    extractor = LabelExtractor(num_classes=27)

    # Test encoding
    labels = extractor.encode(["426783006", "164889003"])
    assert labels.shape == (27,)
    assert labels.sum() == 2, f"Expected 2 active labels, got {labels.sum()}"

    # Test decoding
    decoded = extractor.decode(labels)
    print(f"  Decoded labels: {[(d[1], d[2]) for d in decoded]}")
    assert len(decoded) == 2

    # Test empty
    empty = extractor.encode([])
    assert empty.sum() == 0

    # Test format options
    indices = extractor.encode(["426783006"], format="indices")
    assert isinstance(indices, list)
    names = extractor.encode(["426783006"], format="names")
    assert names[0] == "Sinus Rhythm"

    # Test distribution
    dist = extractor.get_class_distribution([["426783006"], ["164889003"], ["426783006", "164889003"]])
    print(f"  Distribution: {dist}")


def test_metadata_parser():
    """Test metadata parsing."""
    parser = MetadataParser()

    # Normal record
    parsed = parser.parse({
        "Age": "74",
        "Sex": "Male",
        "Dx": "59118001,270492004",
        "Rx": "Unknown",
        "Hx": "Unknown",
        "Sx": "chest pain",
    })
    assert parsed["age"] == 74.0 / 120.0
    assert parsed["sex"] == 1  # Male
    assert len(parsed["dx_codes"]) == 2
    assert parsed["sx_info"]["chest_pain"] == True
    print(f"✅ Parsed normal record: {parsed}")

    # Missing data
    parsed = parser.parse({})
    assert parsed["age"] == -1.0
    assert parsed["sex"] == 2  # Unknown
    assert parsed["dx_codes"] == []
    print("✅ Parsed empty record")

    # Sex variants
    assert parser.parse_sex_str("Male") == "Male"
    assert parser.parse_sex_str("F") == "Female"


if __name__ == "__main__":
    test_preprocessor()
    test_augmentor()
    test_label_extractor()
    test_metadata_parser()
    print("\n✅ All pipeline tests passed!")
