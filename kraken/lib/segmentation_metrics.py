#
# Copyright 2026 Benjamin Kiessling
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
"""
kraken.lib.segmentation_metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Baseline detection evaluation metrics adapted from the Transkribus Baseline
Evaluation Scheme, using optimal (Hungarian) matching.
"""
import torch
import logging

from scipy.optimize import linear_sum_assignment

logger = logging.getLogger(__name__)

__all__ = ['interpolate_polyline',
           'compute_detection_metrics',
           'aggregate_detection_metrics']


def interpolate_polyline(points: torch.Tensor, spacing: float = 5.0) -> torch.Tensor:
    """
    Resample a polyline to approximately uniform point spacing.

    Args:
        points: Tensor of shape (N, 2) representing polyline vertices.
        spacing: Target distance between consecutive points in pixels.

    Returns:
        Tensor of shape (M, 2) with uniformly spaced points.
    """
    if points.shape[0] < 2:
        return points

    diffs = points[1:] - points[:-1]
    seg_lengths = torch.norm(diffs, dim=-1)
    cum_lengths = torch.cat([torch.zeros(1, device=points.device, dtype=points.dtype),
                             torch.cumsum(seg_lengths, dim=0)])
    total_length = cum_lengths[-1]

    if total_length < 1e-6:
        return points[:1]

    num_points = max(2, int(torch.round(total_length / spacing).item()))
    target_lengths = torch.linspace(0, total_length.item(), num_points,
                                    device=points.device, dtype=points.dtype)

    indices = torch.searchsorted(cum_lengths, target_lengths).clamp(1, len(cum_lengths) - 1)

    seg_start = cum_lengths[indices - 1]
    seg_end = cum_lengths[indices]
    seg_len = seg_end - seg_start
    t = torch.where(seg_len > 1e-8,
                    (target_lengths - seg_start) / seg_len,
                    torch.zeros_like(target_lengths))

    p0 = points[indices - 1]
    p1 = points[indices]
    return p0 + t.unsqueeze(-1) * (p1 - p0)


def _point_scores(min_dists: torch.Tensor, tol: float) -> torch.Tensor:
    """
    Per-point scoring with tolerance falloff.

    Args:
        min_dists: Minimum distances from each point to the other polyline.
        tol: Tolerance in pixels.

    Returns:
        Scores in [0, 1] for each point.
    """
    return torch.where(
        min_dists <= tol,
        torch.ones_like(min_dists),
        torch.where(
            min_dists < 3 * tol,
            (3 * tol - min_dists) / (2 * tol),
            torch.zeros_like(min_dists),
        ),
    )


def baseline_score(pred_points: torch.Tensor,
                   gt_points: torch.Tensor,
                   tol: float) -> float:
    """
    Directed score from one polyline to another.

    For each point in pred, finds the minimum distance to any point in gt,
    applies the point score, and returns the mean.

    Args:
        pred_points: (M, 2) uniformly spaced points on prediction polyline.
        gt_points: (N, 2) uniformly spaced points on GT polyline.
        tol: Tolerance in pixels.

    Returns:
        Mean point score (directed, pred -> gt).
    """
    dists = torch.cdist(pred_points.unsqueeze(0), gt_points.unsqueeze(0)).squeeze(0)
    min_dists = dists.min(dim=1).values
    return _point_scores(min_dists, tol).mean().item()


def match_baselines(pred_polylines: list[torch.Tensor],
                    gt_polylines: list[torch.Tensor],
                    tol: float) -> tuple[torch.Tensor, list[tuple[int, int]], torch.Tensor]:
    """
    Build a symmetric score matrix and solve the optimal assignment.

    Args:
        pred_polylines: List of P predicted polylines, each (M_i, 2).
        gt_polylines: List of G ground truth polylines, each (N_j, 2).
        tol: Tolerance in pixels.

    Returns:
        score_matrix: (P, G) symmetric baseline scores.
        matches: List of (pred_idx, gt_idx) pairs from assignment.
        match_scores: Scores for each matched pair.
    """
    n_pred = len(pred_polylines)
    n_gt = len(gt_polylines)

    score_matrix = torch.zeros(n_pred, n_gt)
    for i, pred in enumerate(pred_polylines):
        for j, gt in enumerate(gt_polylines):
            s_pg = baseline_score(pred, gt, tol)
            s_gp = baseline_score(gt, pred, tol)
            score_matrix[i, j] = (s_pg + s_gp) / 2.0

    cost_matrix = 1.0 - score_matrix.numpy()
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = list(zip(row_ind.tolist(), col_ind.tolist()))
    match_scores = score_matrix[row_ind, col_ind]

    return score_matrix, matches, match_scores


def compute_detection_metrics(pred_polylines: list[torch.Tensor],
                              gt_polylines: list[torch.Tensor],
                              tol: float) -> dict[str, float]:
    """
    Compute precision, recall, and F1 for one page.

    Args:
        pred_polylines: List of P predicted polylines.
        gt_polylines: List of G GT polylines.
        tol: Tolerance in pixels.

    Returns:
        Dict with 'precision', 'recall', 'f1', 'num_pred', 'num_gt'.
    """
    n_pred = len(pred_polylines)
    n_gt = len(gt_polylines)

    if n_pred == 0 and n_gt == 0:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
                'num_pred': 0, 'num_gt': 0}
    if n_pred == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'num_pred': 0, 'num_gt': n_gt}
    if n_gt == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0,
                'num_pred': n_pred, 'num_gt': 0}

    _, matches, match_scores = match_baselines(pred_polylines, gt_polylines, tol)

    precision = match_scores.sum().item() / n_pred
    recall = match_scores.sum().item() / n_gt

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return {'precision': precision, 'recall': recall, 'f1': f1,
            'num_pred': n_pred, 'num_gt': n_gt}


def aggregate_detection_metrics(page_metrics: list[dict[str, float]]) -> dict[str, float]:
    """
    Macro-average per-page detection metrics across pages.

    Args:
        page_metrics: List of per-page metric dicts from compute_detection_metrics.

    Returns:
        Aggregated metric dict with 'precision', 'recall', 'f1'.
    """
    if not page_metrics:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    n = len(page_metrics)

    precision = sum(m['precision'] for m in page_metrics) / n
    recall = sum(m['recall'] for m in page_metrics) / n
    f1 = sum(m['f1'] for m in page_metrics) / n

    return {'precision': precision, 'recall': recall, 'f1': f1}
