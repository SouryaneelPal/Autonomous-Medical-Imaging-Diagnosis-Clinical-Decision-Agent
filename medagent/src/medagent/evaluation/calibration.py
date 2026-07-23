import numpy as np


def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, num_bins: int = 10) -> float:
    """
    Compute the Expected Calibration Error (ECE) for binary probabilistic predictions.

    Partitions [0, 1] into `num_bins` equal-width bins, and computes the
    weighted average of the absolute gap between each bin's average predicted
    probability (confidence) and its average true label rate (accuracy):

        ECE = sum_m (|B_m| / n) * |acc(B_m) - conf(B_m)|

    Args:
        y_true: Binary ground truth labels, shape (n,), values in {0, 1}.
        y_prob: Predicted probabilities, shape (n,), values in [0, 1].
        num_bins: Number of equal-width bins to partition [0, 1] into.

    Returns:
        Scalar ECE value in [0, 1]; lower is better calibrated.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.clip(np.asarray(y_prob, dtype=np.float64), 0.0, 1.0)
    n = y_true.shape[0]

    bin_indices = np.minimum((y_prob * num_bins).astype(np.int64), num_bins - 1)

    bin_counts = np.bincount(bin_indices, minlength=num_bins)
    bin_conf_sum = np.bincount(bin_indices, weights=y_prob, minlength=num_bins)
    bin_acc_sum = np.bincount(bin_indices, weights=y_true, minlength=num_bins)

    nonempty = bin_counts > 0
    avg_confidence = bin_conf_sum[nonempty] / bin_counts[nonempty]
    avg_accuracy = bin_acc_sum[nonempty] / bin_counts[nonempty]
    bin_weights = bin_counts[nonempty] / n

    return float(np.sum(bin_weights * np.abs(avg_accuracy - avg_confidence)))