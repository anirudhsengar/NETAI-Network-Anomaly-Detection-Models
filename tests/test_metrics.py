"""Tests for evaluation metrics."""

import numpy as np

from netai_anomaly.utils.metrics import compute_anomaly_metrics, find_optimal_threshold


class TestMetrics:
    def test_perfect_detection(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        metrics = compute_anomaly_metrics(y_true, y_scores, threshold=0.5)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_no_anomalies_detected(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])
        metrics = compute_anomaly_metrics(y_true, y_scores, threshold=0.9)
        assert metrics["recall"] == 0.0

    def test_all_flagged(self):
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.6, 0.7, 0.8, 0.9])
        metrics = compute_anomaly_metrics(y_true, y_scores, threshold=0.0)
        assert metrics["recall"] == 1.0

    def test_roc_auc(self):
        y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
        metrics = compute_anomaly_metrics(y_true, y_scores, threshold=0.5)
        assert metrics["roc_auc"] == 1.0

    def test_optimal_threshold(self):
        y_true = np.array([0, 0, 0, 1, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        threshold = find_optimal_threshold(y_true, y_scores)
        assert 0.3 <= threshold <= 0.8

    def test_single_class_fallback(self):
        y_true = np.array([0, 0, 0, 0])
        y_scores = np.array([0.1, 0.2, 0.3, 0.4])
        threshold = find_optimal_threshold(y_true, y_scores)
        assert isinstance(threshold, float)
