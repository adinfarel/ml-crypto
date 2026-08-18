import pytest
import numpy as np
import pandas as pd
from ml_crypto.data.drift_detector import DriftDetector


def test_drift_detector_no_drift():
    np.random.seed(42)
    ref = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})
    curr = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})

    detector = DriftDetector()
    report = detector.detect_drift(ref, curr, ["feature1"])

    assert report["has_drift"] is False
    assert report["feature_report"]["feature1"]["psi"] < 0.1


def test_drift_detector_with_drift():
    np.random.seed(42)
    ref = pd.DataFrame({"feature1": np.random.normal(0, 1, 1000)})
    curr = pd.DataFrame({"feature1": np.random.normal(3.5, 1, 1000)})

    detector = DriftDetector()
    report = detector.detect_drift(ref, curr, ["feature1"])

    assert report["has_drift"] is True
    assert report["feature_report"]["feature1"]["drift_detected"] is True
    assert report["feature_report"]["feature1"]["psi"] > 0.25