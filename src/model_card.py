"""Headline metrics for the model card, computed from saved predictions on held-out customers."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             precision_score, recall_score, roc_auc_score)


def held_out_metrics(y_true, proba, threshold: float) -> dict:
    """Metrics on customers the model never saw during training.

    Probabilities come from the saved scores (rounded to 4 decimals), so values can differ
    from reports/model_evaluation.md in the 4th decimal at most.
    """
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba, dtype=float)
    pred = (p >= threshold).astype(int)
    return {
        "customers": int(len(y)),
        "churn_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "accuracy": float(accuracy_score(y, pred)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "flagged": int(pred.sum()),
    }