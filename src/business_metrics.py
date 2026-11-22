"""Business-value metrics: what a retention campaign earns or costs.

All numbers depend on the assumptions in config.py, so they are estimates, not facts.
"""
import numpy as np
import pandas as pd

from src.config import OFFER_COST, OFFER_SUCCESS_RATE, VALUE_HORIZON_MONTHS


def customer_value(monthly_charges) -> np.ndarray:
    """Revenue we keep if a customer is saved (monthly charge x horizon)."""
    return np.asarray(monthly_charges, dtype=float) * VALUE_HORIZON_MONTHS


def evaluate_flags(y_true, flagged, monthly_charges) -> dict:
    """Score a contact list: who we flag for an offer vs who actually churned."""
    y = np.asarray(y_true).astype(int)
    flagged = np.asarray(flagged).astype(bool)
    value = customer_value(monthly_charges)

    caught = (y == 1) & flagged
    n_flagged = int(flagged.sum())
    n_churners = int((y == 1).sum())
    benefit = OFFER_SUCCESS_RATE * value[caught].sum()
    cost = OFFER_COST * n_flagged
    return {
        "flagged": n_flagged,
        "flagged_pct": round(100 * n_flagged / len(y), 1),
        "recall": caught.sum() / n_churners if n_churners else 0.0,
        "precision": caught.sum() / n_flagged if n_flagged else 0.0,
        "net_benefit": float(benefit - cost),
    }


def best_threshold(y_true, proba, monthly_charges, grid=None) -> tuple[float, pd.DataFrame]:
    """Probability cut-off that maximises net benefit. Returns (threshold, full sweep table)."""
    p = np.asarray(proba, dtype=float)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    rows = []
    for threshold in grid:
        row = {"threshold": float(threshold)}
        row.update(evaluate_flags(y_true, p >= threshold, monthly_charges))
        rows.append(row)
    sweep = pd.DataFrame(rows)
    best = sweep.loc[sweep["net_benefit"].idxmax()]
    return float(best["threshold"]), sweep