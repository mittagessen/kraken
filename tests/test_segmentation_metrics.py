# -*- coding: utf-8 -*-
import unittest

import torch

from kraken.lib import segmentation_metrics


def _hline(y: float, x0: float = 0.0, x1: float = 10.0) -> torch.Tensor:
    return torch.tensor([[x0, y], [x1, y]], dtype=torch.float32)


class TestSegmentationMetrics(unittest.TestCase):
    def test_compute_detection_metrics_empty_cases(self):
        both_empty = segmentation_metrics.compute_detection_metrics([], [], tol=1.0)
        self.assertEqual(both_empty['precision'], 1.0)
        self.assertEqual(both_empty['recall'], 1.0)
        self.assertEqual(both_empty['f1'], 1.0)
        self.assertEqual(both_empty['num_pred'], 0)
        self.assertEqual(both_empty['num_gt'], 0)

        no_pred = segmentation_metrics.compute_detection_metrics([], [_hline(0.0)], tol=1.0)
        self.assertEqual(no_pred['precision'], 0.0)
        self.assertEqual(no_pred['recall'], 0.0)
        self.assertEqual(no_pred['f1'], 0.0)

        no_gt = segmentation_metrics.compute_detection_metrics([_hline(0.0)], [], tol=1.0)
        self.assertEqual(no_gt['precision'], 0.0)
        self.assertEqual(no_gt['recall'], 0.0)
        self.assertEqual(no_gt['f1'], 0.0)

    def test_compute_detection_metrics_simple_match(self):
        pred = [_hline(0.0)]
        gt = [_hline(0.0)]

        metrics = segmentation_metrics.compute_detection_metrics(pred, gt, tol=1.0)

        self.assertAlmostEqual(metrics['precision'], 1.0, places=6)
        self.assertAlmostEqual(metrics['recall'], 1.0, places=6)
        self.assertAlmostEqual(metrics['f1'], 1.0, places=6)

    def test_compute_detection_metrics_extra_prediction_penalizes_precision(self):
        pred = [_hline(0.0), _hline(50.0)]
        gt = [_hline(0.0)]

        metrics = segmentation_metrics.compute_detection_metrics(pred, gt, tol=1.0)

        self.assertAlmostEqual(metrics['precision'], 0.5, places=6)
        self.assertAlmostEqual(metrics['recall'], 1.0, places=6)
        self.assertAlmostEqual(metrics['f1'], 2.0 / 3.0, places=6)

    def test_compute_detection_metrics_monotonicity(self):
        gt = [_hline(0.0)]
        good_pred = [_hline(0.0)]
        bad_pred = [_hline(2.5)]

        good = segmentation_metrics.compute_detection_metrics(good_pred, gt, tol=1.0)
        bad = segmentation_metrics.compute_detection_metrics(bad_pred, gt, tol=1.0)

        self.assertGreater(good['precision'], bad['precision'])
        self.assertGreater(good['recall'], bad['recall'])
        self.assertGreater(good['f1'], bad['f1'])

    def test_aggregate_detection_metrics(self):
        page_metrics = [
            {'precision': 1.0, 'recall': 0.5, 'f1': 2.0 / 3.0, 'num_pred': 1, 'num_gt': 2},
            {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'num_pred': 1, 'num_gt': 1},
        ]

        agg = segmentation_metrics.aggregate_detection_metrics(page_metrics)

        self.assertAlmostEqual(agg['precision'], 0.5, places=6)
        self.assertAlmostEqual(agg['recall'], 0.25, places=6)
        self.assertAlmostEqual(agg['f1'], 1.0 / 3.0, places=6)
