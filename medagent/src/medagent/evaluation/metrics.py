from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def calculate_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Computes Accuracy, Precision, Recall, and F1-score for classification predictions.

    Recall is weighted as the primary metric of clinical interest, since this
    application prioritizes minimizing false negatives over precision.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def calculate_iou(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Computes Intersection over Union (IoU) for two bounding boxes in [x, y, w, h] format."""
    boxA = np.asarray(boxA, dtype=np.float64)
    boxB = np.asarray(boxB, dtype=np.float64)

    ax1, ay1 = boxA[0], boxA[1]
    ax2, ay2 = boxA[0] + boxA[2], boxA[1] + boxA[3]
    bx1, by1 = boxB[0], boxB[1]
    bx2, by2 = boxB[0] + boxB[2], boxB[1] + boxB[3]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0.0, boxA[2]) * max(0.0, boxA[3])
    area_b = max(0.0, boxB[2]) * max(0.0, boxB[3])
    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return float(intersection / union)