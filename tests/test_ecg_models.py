"""Tests for Phase 2: ECG Models."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_r_peak_detector():
    from src.ecg_models.feature_extraction.r_peak_detector import RPeakDetector

    detector = RPeakDetector(method="pan_tompkins")

    # Generate synthetic ECG-like signal with clear QRS peaks at ~72 bpm
    fs = 500
    duration = 10
    L = duration * fs
    hr = 72  # bpm
    beat_interval = int(fs * 60 / hr)

    # Baseline + noise
    t = np.linspace(0, duration, L)
    ecg = 0.1 * np.sin(2 * np.pi * 0.3 * t)  # Baseline wander
    ecg += 0.02 * np.random.randn(L)

    # Add sharp QRS-like triangular peaks
    qrs_width = 6  # samples (~12ms)
    qrs_amplitude = 2.0
    for i in range(beat_interval, L - qrs_width, beat_interval):
        # Triangular QRS
        ramp = np.linspace(0, qrs_amplitude, qrs_width // 2)
        ecg[i : i + qrs_width // 2] += ramp
        ecg[i + qrs_width // 2 : i + qrs_width] += qrs_amplitude - ramp

    result = detector.detect(ecg, fs)
    assert result["num_beats"] > 0, f"No beats detected (neurokit2 missing?)"
    if result["num_beats"] >= 2:
        assert 50 < result["heart_rate"] < 150, f"HR out of range: {result['heart_rate']}"
    print(f"  OK: {result['num_beats']} beats, HR={result['heart_rate']} bpm")

    # Multi-lead (12-lead array)
    multi = np.tile(ecg, (12, 1))
    result2 = detector.detect_multi_lead(multi, fs)
    assert result2["num_beats"] > 0
    print(f"  OK: Multi-lead consensus HR={result2['heart_rate']} bpm")


def test_hrv_analyzer():
    from src.ecg_models.feature_extraction.hrv_analyzer import HRVAnalyzer

    analyzer = HRVAnalyzer()
    rr = 1000 * np.ones(100) + 50 * np.random.randn(100)  # ~1000ms RR = 60bpm
    metrics = analyzer.analyze(rr)
    assert metrics["sdnn"] > 0
    assert metrics["rmssd"] > 0
    print(f"  OK: SDNN={metrics['sdnn']:.1f}, RMSSD={metrics['rmssd']:.1f}, "
          f"LF/HF={metrics['lf_hf_ratio']:.2f}")


def test_qt_analyzer():
    from src.ecg_models.feature_extraction.qt_analyzer import QTAnalyzer

    analyzer = QTAnalyzer()
    # Simple: no beats = empty result
    result = analyzer.analyze(np.zeros(1000), [], fs=500)
    assert result["qt_interpretation"] == "insufficient_data"
    print(f"  OK: Empty signal → {result['qt_interpretation']}")


def test_feature_bank():
    from src.ecg_models.feature_extraction.feature_bank import FeatureBank

    bank = FeatureBank()
    ecg = np.random.randn(12, 5000).astype(np.float32) * 0.5
    features = bank.extract_all(ecg, fs=500)
    assert "heart_rate" in features
    vector = bank.to_vector(features)
    assert vector.shape == (bank.feature_dim,)
    print(f"  OK: {bank.feature_dim} features extracted")


def test_backbones():
    from src.ecg_models.backbone.resnet1d import resnet1d_34
    from src.ecg_models.backbone.xresnet1d import xresnet1d_101
    from src.ecg_models.backbone.inception_time import inception_time
    from src.ecg_models.backbone.transformer_encoder import ecg_transformer

    x = torch.randn(2, 12, 4096)

    for name, fn in [
        ("ResNet1D-34", resnet1d_34),
        ("xResNet1D-101", xresnet1d_101),
        ("InceptionTime", inception_time),
        ("ECG Transformer", ecg_transformer),
    ]:
        model = fn(in_channels=12)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, model.feature_dim), \
            f"{name}: expected (2, {model.feature_dim}), got {out.shape}"
        params = sum(p.numel() for p in model.parameters())
        print(f"  OK: {name} → ({out.shape[1]},) — {params:,} params")


def test_classifier():
    from src.ecg_models.backbone.inception_time import inception_time
    from src.ecg_models.classifiers.arrhythmia_classifier import (
        ArrhythmiaClassifier, AsymmetricLoss, FocalLoss,
    )

    backbone = inception_time(in_channels=12)
    model = ArrhythmiaClassifier(backbone, num_classes=27)
    x = torch.randn(4, 12, 4096)
    logits = model(x)
    assert logits.shape == (4, 27)

    # Loss functions
    targets = torch.randint(0, 2, (4, 27)).float()

    loss_asl = AsymmetricLoss()(logits, targets)
    assert loss_asl.item() > 0
    print(f"  OK: ASL loss={loss_asl.item():.4f}")

    loss_focal = FocalLoss()(logits, targets)
    assert loss_focal.item() > 0
    print(f"  OK: Focal loss={loss_focal.item():.4f}")

    # Predict
    probs = model.predict(x)
    assert probs.shape == (4, 27)
    assert (probs >= 0).all() and (probs <= 1).all()
    print(f"  OK: Predict probs in [0,1]")


def test_anomaly_detector():
    from src.ecg_models.backbone.inception_time import inception_time
    from src.ecg_models.classifiers.anomaly_detector import ECGAnomalyDetector

    backbone = inception_time(in_channels=12)
    detector = ECGAnomalyDetector(backbone, latent_dim=64)
    x = torch.randn(4, 12, 4096)

    recon, mu, logvar, z, features = detector(x)
    assert recon.shape == features.shape
    assert z.shape == (4, 64)
    print(f"  OK: VAE latent={z.shape[1]}, recon shape={recon.shape[1]}")

    # Loss
    total, recon_loss, kl_loss = detector.loss_fn(recon, features, mu, logvar)
    assert total.item() > 0
    print(f"  OK: VAE total_loss={total.item():.4f} (recon={recon_loss.item():.4f}, kl={kl_loss.item():.4f})")

    # Anomaly score
    scores = detector.anomaly_score(x)
    assert scores.shape == (4,)
    print(f"  OK: Anomaly scores: {scores.tolist()}")


if __name__ == "__main__":
    print("=== Phase 2 Tests ===\n")
    test_r_peak_detector()
    test_hrv_analyzer()
    test_qt_analyzer()
    test_feature_bank()
    test_backbones()
    test_classifier()
    test_anomaly_detector()
    print("\n=== ALL TESTS PASSED ===")
