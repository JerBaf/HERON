""" Forecasting and regime-recovery metrics (paper App. 10.2). """
import numpy as np
from torch import Tensor


# Forecasting Metrics

def compute_displacement_errors(pred: Tensor, true: Tensor, mask: Tensor) -> tuple:
    """ Compute the per-window ADE (Eq. 26), FDE and validity [B] from positions [B, H, N, P] and a mask [B, H, N]. """
    # L2 error averaged over evaluated nodes
    errors = (pred - true).norm(dim=-1)  # [B, H, N]
    weights = mask.to(errors.dtype)
    num_nodes = weights.sum(-1)
    per_step = (errors * weights).sum(-1) / num_nodes.clamp_min(1.0)  # [B, H]
    has_nodes = num_nodes > 0
    # Aggregate over horizon steps
    ade = (per_step * has_nodes).sum(-1) / has_nodes.sum(-1).clamp_min(1)  # [B]
    fde = per_step[:, -1].masked_fill(~has_nodes[:, -1], float("nan"))     # [B], nan without final nodes
    return ade, fde, has_nodes.any(-1)


# Regime Metrics

def compute_regime_mapping(pred: np.ndarray, labels: np.ndarray, num_regimes: int) -> np.ndarray:
    """ Compute the Hungarian matching of learned regimes to labels (fit on validation, then frozen). """
    from scipy.optimize import linear_sum_assignment
    # Contingency matrix
    classes = np.unique(labels)
    confusion = np.zeros((num_regimes, len(classes)))
    np.add.at(confusion, (pred, np.searchsorted(classes, labels)), 1)
    # Optimal assignment
    rows, cols = linear_sum_assignment(-confusion)
    mapping = np.full(num_regimes, -1)  # -1 for unmatched regimes
    mapping[rows] = classes[cols]       # mapping[k] = label assigned to regime k
    return mapping


def _compute_auc(score: np.ndarray, positive: np.ndarray) -> float:
    """ Compute the binary ROC AUC of the scores from their ranks (Mann-Whitney U statistic). """
    from scipy.stats import rankdata
    num_positives, num_negatives = positive.sum(), (~positive).sum()
    ranks = rankdata(score)
    return (ranks[positive].sum() - num_positives * (num_positives + 1) / 2) / (num_positives * num_negatives)


def compute_classification_metrics(probs: np.ndarray, labels: np.ndarray, mapping: np.ndarray) -> dict:
    """ Compute the macro-F1, one-vs-rest macro AUC and accuracy of regime probabilities [n, K] after alignment. """
    # Aligned predictions
    pred = mapping[probs.argmax(1)]
    all_f1, all_auc = [], []
    for label in np.unique(labels):
        # F1 score
        true_positives, false_positives = (((pred == label) & (labels == label)).sum(),
                                           ((pred == label) & (labels != label)).sum())
        false_negatives = ((pred != label) & (labels == label)).sum()
        all_f1.append(2 * true_positives / max(2 * true_positives + false_positives + false_negatives, 1))
        # One-vs-rest AUC
        positive = labels == label
        if 0 < positive.sum() < len(labels):
            all_auc.append(_compute_auc(probs[:, mapping == label].sum(1), positive))
    return {"macro_f1": float(np.mean(all_f1)), "auc": float(np.mean(all_auc)) if all_auc else float("nan"),
            "accuracy": float((pred == labels).mean())}
