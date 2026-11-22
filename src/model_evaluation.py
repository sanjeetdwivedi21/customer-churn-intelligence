"""Phase 5b: evaluate the saved final model on the held-out TEST set, once.

Run from the project root:  python -m src.model_evaluation
Run model_tuning first. Do not re-run tuning just to improve the test numbers.
"""
import json
import logging
import sys

import joblib
import matplotlib
matplotlib.use("Agg")  # draw to files only, no pop-up windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
                             accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, roc_auc_score)

from src.business_metrics import best_threshold, evaluate_flags
from src.config import FIGURES_DIR, MODELS_DIR, REPORTS_DIR
from src.model_training import load_split, rule_based_predictions

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def classification_metrics(y, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
    }


def md_table(df: pd.DataFrame) -> str:
    """Small markdown-table writer (avoids needing the 'tabulate' package)."""
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def save(fig, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", bbox_inches="tight", dpi=120)
    plt.close(fig)


def plot_all(y, proba, pred, rule, threshold, charges, model_name) -> None:
    # Confusion matrix at the business threshold
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ConfusionMatrixDisplay.from_predictions(
        y, pred, display_labels=["Stayed", "Churned"], ax=ax, colorbar=False, values_format="d")
    ax.set_title(f"Confusion matrix (threshold {threshold:.2f})")
    save(fig, "09_confusion_matrix")

    # ROC curve, with the SQL rule as a single point
    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_predictions(y, proba, ax=ax, name=model_name)
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
    fpr_rule = ((rule == 1) & (y == 0)).sum() / (y == 0).sum()
    ax.scatter([fpr_rule], [recall_score(y, rule)], color="#c44e52", zorder=5, label="SQL rule")
    ax.legend(loc="lower right")
    ax.set_title("ROC curve (test set)")
    save(fig, "10_roc_curve")

    # Precision-Recall curve
    fig, ax = plt.subplots(figsize=(5.5, 5))
    PrecisionRecallDisplay.from_predictions(y, proba, ax=ax, name=model_name)
    ax.axhline(y.mean(), ls="--", color="grey", lw=1, label="No-skill (churn rate)")
    ax.scatter([recall_score(y, rule)], [precision_score(y, rule)], color="#c44e52",
               zorder=5, label="SQL rule")
    ax.scatter([recall_score(y, pred)], [precision_score(y, pred)], color="black",
               marker="x", zorder=5, label=f"Chosen threshold {threshold:.2f}")
    ax.legend(loc="upper right")
    ax.set_title("Precision-Recall curve (test set)")
    save(fig, "11_precision_recall_curve")

    # Calibration: do predicted probabilities match real churn rates?
    frac_pos, mean_pred = calibration_curve(y, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="grey", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", label=model_name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Actual churn rate")
    ax.set_title("Calibration (test set)")
    ax.legend()
    save(fig, "12_calibration")

    # Net benefit across thresholds (threshold was chosen on training data, marked here)
    _, sweep = best_threshold(y, proba, charges)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(sweep["threshold"], sweep["net_benefit"])
    ax.axvline(threshold, color="#c44e52", ls="--", label=f"Chosen threshold {threshold:.2f}")
    ax.axhline(0, color="grey", lw=1)
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Net benefit (USD, assumption-based)")
    ax.set_title("Net benefit vs threshold (test set)")
    ax.legend()
    save(fig, "13_net_benefit_vs_threshold")


def write_report(meta, test_default, test_chosen, strategies, calib, tuning) -> None:
    a = meta["assumptions"]
    text = f"""# Model Evaluation Report

## Summary
- **Selected model:** {meta['model_name']} (chosen on training data only)
- **Decision threshold:** {meta['threshold']:.2f}, chosen by maximising estimated net benefit on
  out-of-fold training predictions. Risk tiers: Medium >= {meta['risk_thresholds']['medium']:.2f},
  High >= {meta['risk_thresholds']['high']:.2f}.
- **Test set:** evaluated once, after model and threshold were fixed.

## Business assumptions (not from the dataset)
Offer cost ${a['offer_cost']:.0f}, offer success rate {a['offer_success_rate']:.0%},
value of a saved customer = {a['value_horizon_months']} months of their monthly charge.
All net-benefit figures depend on these and are estimates.

## Model comparison (training data, out-of-fold)
{md_table(tuning)}

## Test-set metrics
{md_table(pd.DataFrame([test_default, test_chosen]).round(3))}

The first row uses the default 0.5 threshold, the second the business threshold.
ROC-AUC {test_chosen['roc_auc']:.3f}, PR-AUC {test_chosen['pr_auc']:.3f}, Brier score {calib['brier']:.3f}.

## Strategy comparison (test set, estimated net benefit)
{md_table(strategies)}

## Calibration check
Expected churners from model probabilities: {calib['expected_churners']:.0f}; actual: {calib['actual_churners']}.
Expected monthly revenue at risk: ${calib['expected_revenue']:,.0f}; actual churned: ${calib['actual_revenue']:,.0f}.
If these are close, probability-based "revenue at risk" is trustworthy.

## Limitations
- Associations in this dataset, not causes.
- The dataset is a snapshot with no usage or complaint data.
- Offer cost, success rate and horizon are assumptions.
- One 80/20 split: test metrics carry sampling noise (about +/- 0.02).

## Figures
`reports/figures/`: 09_confusion_matrix, 10_roc_curve, 11_precision_recall_curve,
12_calibration, 13_net_benefit_vs_threshold.
"""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "model_evaluation.md").write_text(text, encoding="utf-8")


def main() -> int:
    try:
        model_path = MODELS_DIR / "final_model.joblib"
        meta_path = MODELS_DIR / "model_metadata.json"
        if not model_path.exists() or not meta_path.exists():
            raise FileNotFoundError("Run first: python -m src.model_tuning")

        model = joblib.load(model_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        _, X_test, _, y_test = load_split()
        y = y_test.to_numpy()
        charges = X_test["monthly_charges"].to_numpy()

        proba = model.predict_proba(X_test)[:, 1]
        threshold = meta["threshold"]
        pred = (proba >= threshold).astype(int)
        rule = rule_based_predictions(X_test).to_numpy()

        shared = {"roc_auc": roc_auc_score(y, proba), "pr_auc": average_precision_score(y, proba)}
        test_default = {"setting": "threshold 0.50", **classification_metrics(y, proba, 0.5), **shared}
        test_chosen = {"setting": "business threshold", **classification_metrics(y, proba, threshold), **shared}

        def strategy_row(label, flagged):
            r = evaluate_flags(y, flagged, charges)
            return {"strategy": label, "customers_flagged": r["flagged"],
                    "recall": round(r["recall"], 3), "precision": round(r["precision"], 3),
                    "net_benefit_usd": f"{r['net_benefit']:,.0f}"}

        strategies = pd.DataFrame([
            strategy_row("Contact nobody", np.zeros(len(y), dtype=bool)),
            strategy_row("Contact everyone", np.ones(len(y), dtype=bool)),
            strategy_row("SQL rule baseline", rule == 1),
            strategy_row(f"{meta['model_name']} @ {threshold:.2f}", pred == 1),
        ])

        calib = {
            "brier": brier_score_loss(y, proba),
            "expected_churners": float(proba.sum()), "actual_churners": int(y.sum()),
            "expected_revenue": float((proba * charges).sum()),
            "actual_revenue": float(charges[y == 1].sum()),
        }

        plot_all(y, proba, pred, rule, threshold, charges, meta["model_name"])
        tuning = pd.read_csv(MODELS_DIR / "tuning_summary.csv")
        write_report(meta, test_default, test_chosen, strategies, calib, tuning)

        pd.set_option("display.width", 200)
        print(f"\nFINAL MODEL: {meta['model_name']} (test set, {len(y)} customers)\n")
        print(pd.DataFrame([test_default, test_chosen]).round(3).to_string(index=False))
        print("\nStrategy comparison:\n")
        print(strategies.to_string(index=False))
        print(f"\nCalibration: expected churners {calib['expected_churners']:.0f} vs actual "
              f"{calib['actual_churners']} | expected revenue at risk ${calib['expected_revenue']:,.0f} "
              f"vs actual ${calib['actual_revenue']:,.0f} | Brier {calib['brier']:.3f}")
        logger.info("Report: reports/model_evaluation.md | Figures: reports/figures/")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logger.error("Failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())