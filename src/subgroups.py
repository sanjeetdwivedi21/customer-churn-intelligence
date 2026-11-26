"""Subgroup check: how does the model perform for different groups of customers?

This is a descriptive check, not a certification of fairness. Groups in a 1,409-customer test
set are small, so every recall comes with a Wilson 95% interval and small groups are flagged.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data_preprocessing import TENURE_BAND_ORDER

MIN_CUSTOMERS = 50   # smaller groups are shown but marked unreliable
MIN_CHURNERS = 20    # fewer churners than this makes recall too noisy to compare

# Display name -> (column in the labelled frame, fixed order or None for alphabetical)
GROUPINGS = {
    "Gender": ("gender", None),
    "Senior citizen": ("senior_label", ["Senior", "Not senior"]),
    "Partner": ("partner_label", ["Has partner", "No partner"]),
    "Dependents": ("dependents_label", ["Has dependents", "No dependents"]),
    "Contract": ("contract", ["Month-to-month", "One year", "Two year"]),
    "Internet service": ("internet_service", ["DSL", "Fiber optic", "No"]),
    "Tenure band": ("tenure_band_label", TENURE_BAND_ORDER),
    "Payment method": ("payment_method", None),
}


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion. Returns (nan, nan) when n is 0."""
    if n == 0:
        return float("nan"), float("nan")
    p = successes / n
    denominator = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denominator
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denominator
    return max(0.0, centre - half), min(1.0, centre + half)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Readable group labels for the 0/1 columns and the tenure band."""
    out = df.copy()
    out["senior_label"] = out["senior_citizen"].map({1: "Senior", 0: "Not senior"})
    out["partner_label"] = out["has_partner"].map({1: "Has partner", 0: "No partner"})
    out["dependents_label"] = out["has_dependents"].map({1: "Has dependents", 0: "No dependents"})
    out["tenure_band_label"] = out["tenure_band"].astype(str)
    return out


def group_metrics(df: pd.DataFrame, column: str, threshold: float, order=None,
                  min_customers: int = MIN_CUSTOMERS, min_churners: int = MIN_CHURNERS) -> pd.DataFrame:
    """Metrics for each value of one column. df needs: churn, churn_probability and the column."""
    values = order if order is not None else sorted(df[column].dropna().unique())
    rows = []
    for value in values:
        part = df[df[column] == value]
        if part.empty:
            continue
        y = part["churn"].to_numpy().astype(int)
        p = part["churn_probability"].to_numpy(dtype=float)
        flagged = p >= threshold
        n, churners = len(part), int(y.sum())
        caught = int((flagged & (y == 1)).sum())
        n_flagged = int(flagged.sum())
        low, high = wilson_interval(caught, churners)
        reliable = n >= min_customers and churners >= min_churners
        rows.append({
            "group": str(value),
            "customers": n,
            "churners": churners,
            "actual_churn_pct": 100 * churners / n,
            "avg_predicted_pct": 100 * float(p.mean()),
            "calibration_gap_pts": 100 * (float(p.mean()) - churners / n),
            "flagged_pct": 100 * n_flagged / n,
            "recall": caught / churners if churners else float("nan"),
            "recall_low": low,
            "recall_high": high,
            "precision": caught / n_flagged if n_flagged else float("nan"),
            "roc_auc": float(roc_auc_score(y, p)) if reliable and 0 < churners < n else float("nan"),
            "reliable": reliable,
        })
    return pd.DataFrame(rows)


def all_group_metrics(df: pd.DataFrame, threshold: float, **limits) -> pd.DataFrame:
    """Metrics for every grouping in GROUPINGS. df must already have the labels (add_labels)."""
    parts = []
    for name, (column, order) in GROUPINGS.items():
        part = group_metrics(df, column, threshold, order=order, **limits)
        part.insert(0, "grouping", name)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def gap_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    """Largest recall and calibration gaps within each grouping, using reliable groups only."""
    rows = []
    for name, part in metrics.groupby("grouping", sort=False):
        part = part[part["reliable"]]
        if len(part) < 2:
            continue
        best = part.loc[part["recall"].idxmax()]
        worst = part.loc[part["recall"].idxmin()]
        overlap = bool(worst["recall_high"] >= best["recall_low"])
        rows.append({
            "grouping": name,
            "highest_recall_group": best["group"],
            "lowest_recall_group": worst["group"],
            "recall_gap_pts": round(100 * (best["recall"] - worst["recall"]), 1),
            "highest_recall_churn_pct": round(float(best["actual_churn_pct"]), 1),
            "lowest_recall_churn_pct": round(float(worst["actual_churn_pct"]), 1),
            "intervals_overlap": "Yes (could be noise)" if overlap else "No (worth a look)",
            "largest_calibration_gap_pts": round(float(part["calibration_gap_pts"].abs().max()), 1),
        })
    return pd.DataFrame(rows)