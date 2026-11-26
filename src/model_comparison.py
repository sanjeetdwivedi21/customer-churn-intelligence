"""Compare the tuned models using out-of-fold predictions on the TRAINING set.

Every customer in oof_predictions.csv was scored by a model that never saw that customer, so
the comparison is honest, and the test set is not involved. Net benefit depends on the business
assumptions (offer cost, success rate, horizon), which can be changed here.
"""
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss, precision_recall_curve,
                             roc_auc_score, roc_curve)

from src.business_metrics import best_threshold
from src.config import MODELS_DIR

OOF_PATH = MODELS_DIR / "oof_predictions.csv"
NON_MODEL_COLUMNS = ["row_index", "churn", "monthly_charges"]
# Lower = simpler. Same values as model_tuning.COMPLEXITY (a test keeps them identical).
COMPLEXITY = {"Logistic Regression": 1, "Random Forest": 2, "Gradient Boosting": 3, "XGBoost": 3}


def load_oof(path=OOF_PATH):
    """Out-of-fold predictions, or None if model_tuning has not produced them yet."""
    return pd.read_csv(path) if path.exists() else None


def model_names(oof: pd.DataFrame) -> list[str]:
    return [column for column in oof.columns if column not in NON_MODEL_COLUMNS]


def metrics_table(oof: pd.DataFrame) -> pd.DataFrame:
    """Threshold-free quality of each model's probabilities."""
    rows = [{"model": name,
             "roc_auc": roc_auc_score(oof["churn"], oof[name]),
             "pr_auc": average_precision_score(oof["churn"], oof[name]),
             "brier": brier_score_loss(oof["churn"], oof[name])}
            for name in model_names(oof)]
    return pd.DataFrame(rows)


def roc_table(oof: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for name in model_names(oof):
        fpr, tpr, _ = roc_curve(oof["churn"], oof[name])
        parts.append(pd.DataFrame({"model": name, "fpr": fpr, "tpr": tpr}))
    return pd.concat(parts, ignore_index=True)


def pr_table(oof: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for name in model_names(oof):
        precision, recall, _ = precision_recall_curve(oof["churn"], oof[name])
        parts.append(pd.DataFrame({"model": name, "recall": recall, "precision": precision}))
    return pd.concat(parts, ignore_index=True)


def calibration_table(oof: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    """Average predicted probability vs actual churn rate in equal-sized probability bins."""
    parts = []
    for name in model_names(oof):
        groups = pd.qcut(oof[name], bins, duplicates="drop")
        table = (oof.groupby(groups, observed=True)
                    .agg(mean_predicted=(name, "mean"), actual_rate=("churn", "mean"))
                    .reset_index(drop=True))
        parts.append(table.assign(model=name))
    return pd.concat(parts, ignore_index=True)


def net_benefit_table(oof: pd.DataFrame, **assumptions) -> pd.DataFrame:
    """Estimated net benefit at every threshold for every model."""
    parts = []
    for name in model_names(oof):
        _, sweep = best_threshold(oof["churn"], oof[name], oof["monthly_charges"], **assumptions)
        parts.append(sweep[["threshold", "net_benefit"]].assign(model=name))
    return pd.concat(parts, ignore_index=True)


def selection_table(oof: pd.DataFrame, **assumptions) -> pd.DataFrame:
    """Each model at its own best threshold, under the given assumptions."""
    rows = []
    for name in model_names(oof):
        threshold, sweep = best_threshold(oof["churn"], oof[name], oof["monthly_charges"],
                                          **assumptions)
        best = sweep[sweep["threshold"] == threshold].iloc[0]
        rows.append({"model": name, "best_threshold": threshold,
                     "net_benefit": float(best["net_benefit"]),
                     "recall": float(best["recall"]), "precision": float(best["precision"]),
                     "flagged_pct": float(best["flagged_pct"])})
    return pd.DataFrame(rows).sort_values("net_benefit", ascending=False).reset_index(drop=True)


def annotate(table: pd.DataFrame, tolerance: float) -> pd.DataFrame:
    """Add complexity and whether each model is within the tolerance of the best net benefit."""
    out = table.copy()
    best_value = out["net_benefit"].max()
    out["complexity"] = out["model"].map(COMPLEXITY).fillna(99).astype(int)
    out["within_tolerance"] = (out["net_benefit"] >= (1 - tolerance) * best_value) & (best_value > 0)
    return out


def choose_model(table: pd.DataFrame, tolerance: float):
    """The simplest model within the tolerance of the best, or None if nothing beats doing nothing.

    Same rule as model_tuning.select_model.
    """
    annotated = annotate(table, tolerance)
    close = annotated[annotated["within_tolerance"]]
    if close.empty:
        return None
    return close.sort_values(["complexity", "net_benefit"], ascending=[True, False]).iloc[0]["model"]


def agreement_tables(oof: pd.DataFrame, top_share: float = 0.10):
    """How similar are the models? Returns (Spearman rank correlation, top-group overlap).

    Overlap[a][b] = share of the top `top_share` customers (by model a) that model b also puts
    in its top group.
    """
    names = model_names(oof)
    spearman = oof[names].corr(method="spearman")
    k = max(1, int(len(oof) * top_share))
    top = {name: set(oof[name].nlargest(k).index) for name in names}
    overlap = pd.DataFrame([[len(top[a] & top[b]) / k for b in names] for a in names],
                           index=names, columns=names)
    return spearman, overlap