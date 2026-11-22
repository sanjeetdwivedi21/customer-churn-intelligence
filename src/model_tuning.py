"""Phase 5a: tune the shortlisted models, pick one by business value, save it.

Run from the project root:  python -m src.model_tuning     (takes several minutes)
Uses the TRAINING set only. The test set is not touched here.
"""
import json
import logging
import sys
from datetime import datetime, timezone

import joblib
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict

from src.business_metrics import best_threshold
from src.config import (CV_FOLDS, MODELS_DIR, OFFER_COST, OFFER_SUCCESS_RATE, RANDOM_SEED,
                        RISK_HIGH_THRESHOLD, SELECTION_TOLERANCE, TUNING_ITERATIONS,
                        VALUE_HORIZON_MONTHS)
from src.feature_engineering import FEATURE_COLUMNS
from src.model_training import build_pipeline, load_split

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = MODELS_DIR / "final_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Lower number = simpler model. Used only to break near-ties.
COMPLEXITY = {"Logistic Regression": 1, "Random Forest": 2, "Gradient Boosting": 3, "XGBoost": 3}


def search_spaces() -> dict:
    """Model + hyperparameter ranges. No class weights: the threshold handles imbalance."""
    spaces = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
            {"model__C": loguniform(1e-3, 1e2)},
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_SEED),
            {"model__max_depth": randint(3, 12),
             "model__min_samples_leaf": randint(3, 40),
             "model__max_features": ["sqrt", 0.3, 0.5]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_SEED),
            {"model__n_estimators": randint(100, 400),
             "model__learning_rate": loguniform(0.01, 0.2),
             "model__max_depth": randint(2, 5),
             "model__subsample": uniform(0.6, 0.4),
             "model__min_samples_leaf": randint(10, 60)},
        ),
    }
    if XGBClassifier is not None:
        spaces["XGBoost"] = (
            XGBClassifier(eval_metric="logloss", n_jobs=-1, random_state=RANDOM_SEED),
            {"model__n_estimators": randint(100, 500),
             "model__learning_rate": loguniform(0.01, 0.2),
             "model__max_depth": randint(2, 6),
             "model__subsample": uniform(0.6, 0.4),
             "model__colsample_bytree": uniform(0.6, 0.4),
             "model__min_child_weight": randint(1, 10),
             "model__reg_lambda": loguniform(0.5, 20)},
        )
    return spaces


def to_native(params: dict) -> dict:
    """Convert numpy numbers to plain Python so the dict can be saved as JSON."""
    return {k: (v.item() if hasattr(v, "item") else v) for k, v in params.items()}


def select_model(summary: pd.DataFrame) -> str:
    """Simplest model whose net benefit is within the tolerance of the best one."""
    best_value = summary["net_benefit"].max()
    if best_value <= 0:
        raise RuntimeError("No model has a positive net benefit under the current "
                           "business assumptions. Check the values in config.py.")
    close = summary[summary["net_benefit"] >= (1 - SELECTION_TOLERANCE) * best_value].copy()
    close["complexity"] = close["model"].map(COMPLEXITY)
    close = close.sort_values(["complexity", "net_benefit"], ascending=[True, False])
    return close.iloc[0]["model"]


def main() -> int:
    try:
        X_train, _, y_train, _ = load_split()  # test set is deliberately ignored
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        charges = X_train["monthly_charges"]

        fitted, sweeps, rows = {}, {}, []
        for name, (model, space) in search_spaces().items():
            logger.info("Tuning %s (%d combinations x %d folds) ...",
                        name, TUNING_ITERATIONS, CV_FOLDS)
            search = RandomizedSearchCV(
                build_pipeline(model), space, n_iter=TUNING_ITERATIONS, scoring="roc_auc",
                cv=cv, random_state=RANDOM_SEED, n_jobs=1, refit=True)
            search.fit(X_train, y_train)

            # Out-of-fold probabilities: every customer is scored by a model that never saw them.
            oof = cross_val_predict(search.best_estimator_, X_train, y_train,
                                    cv=cv, method="predict_proba")[:, 1]
            threshold, sweep = best_threshold(y_train, oof, charges)
            at_best = sweep[sweep["threshold"] == threshold].iloc[0]

            fitted[name], sweeps[name] = search.best_estimator_, sweep
            rows.append({
                "model": name,
                "cv_roc_auc": round(search.best_score_, 4),
                "oof_pr_auc": round(average_precision_score(y_train, oof), 4),
                "oof_brier": round(brier_score_loss(y_train, oof), 4),
                "threshold": threshold,
                "recall": round(at_best["recall"], 3),
                "precision": round(at_best["precision"], 3),
                "flagged_pct": at_best["flagged_pct"],
                "net_benefit": round(at_best["net_benefit"], 0),
                "best_params": to_native(search.best_params_),
            })

        summary = pd.DataFrame(rows).sort_values("net_benefit", ascending=False)
        chosen = select_model(summary)
        chosen_row = summary[summary["model"] == chosen].iloc[0]
        threshold = float(chosen_row["threshold"])
        high = max(RISK_HIGH_THRESHOLD, round(threshold + 0.10, 2))

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(fitted[chosen], MODEL_PATH)
        sweeps[chosen].to_csv(MODELS_DIR / "threshold_sweep_oof.csv", index=False)
        summary.drop(columns="best_params").to_csv(MODELS_DIR / "tuning_summary.csv", index=False)

        metadata = {
            "model_name": chosen,
            "best_params": chosen_row["best_params"],
            "threshold": threshold,
            "risk_thresholds": {"medium": threshold, "high": high},
            "assumptions": {"offer_cost": OFFER_COST, "offer_success_rate": OFFER_SUCCESS_RATE,
                            "value_horizon_months": VALUE_HORIZON_MONTHS},
            "selection_tolerance": SELECTION_TOLERANCE,
            "random_seed": RANDOM_SEED,
            "feature_columns": FEATURE_COLUMNS,
            "trained_on_rows": int(len(X_train)),
            "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        pd.set_option("display.width", 200)
        print("\nTuned models, TRAINING data only (out-of-fold, each at its own best threshold):\n")
        print(summary.drop(columns="best_params").to_string(index=False))
        print(f"\nSelected: {chosen} | threshold {threshold:.2f} | "
              f"risk tiers: Medium >= {threshold:.2f}, High >= {high:.2f}")
        logger.info("Saved %s and %s", MODEL_PATH.name, METADATA_PATH.name)
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logger.error("Failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())