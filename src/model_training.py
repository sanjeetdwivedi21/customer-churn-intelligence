"""Phase 4: split the data, compare baseline models with cross-validation.

Run from the project root:  python -m src.model_training
The test set is NOT used here. It stays untouched until Phase 5.
"""
import logging
import sys

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import MODELS_DIR, RANDOM_SEED, TEST_SIZE
from src.data_preprocessing import load_customers
from src.feature_engineering import ID_COLUMN, TARGET, build_features, build_preprocessor

try:
    from xgboost import XGBClassifier
except ImportError:  # the project still runs without XGBoost
    XGBClassifier = None

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

SCORING = ["roc_auc", "recall", "precision", "f1", "accuracy"]


def load_split():
    """Stratified train/test split. Same seed, so Phase 5 gets the exact same split."""
    X, y = build_features(load_customers())
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)


def get_models(scale_pos_weight: float = 1.0) -> dict:
    """The five candidate models with sensible, not yet tuned, settings."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=5, min_samples_leaf=20, class_weight="balanced", random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=5, class_weight="balanced",
            n_jobs=-1, random_state=RANDOM_SEED),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_SEED),
    }
    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
            eval_metric="logloss", n_jobs=-1, random_state=RANDOM_SEED)
    else:
        logger.warning("xgboost is not installed, skipping XGBoost")
    return models


def build_pipeline(model) -> Pipeline:
    """Preprocessing and model in one object, so scaling is fit on training data only."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def rule_based_predictions(X: pd.DataFrame) -> pd.Series:
    """Phase 2 rule: month-to-month, 12 months or less, electronic check."""
    return ((X["contract"] == "Month-to-month")
            & (X["tenure_months"] <= 12)
            & (X["payment_method"] == "Electronic check")).astype(int)


def sanity_checks(X_train, X_test, y_train, y_test) -> None:
    """Fail loudly if something that would cause leakage or bad input slipped in."""
    forbidden = {ID_COLUMN, TARGET}
    if forbidden & set(X_train.columns):
        raise RuntimeError(f"Forbidden columns in features: {forbidden & set(X_train.columns)}")
    if set(X_train.index) & set(X_test.index):
        raise RuntimeError("Train and test sets overlap")
    if X_train.isna().any().any() or X_test.isna().any().any():
        raise RuntimeError("Missing values found in features")
    logger.info("Train: %d rows (churn %.2f%%) | Test: %d rows (churn %.2f%%)",
                len(X_train), 100 * y_train.mean(), len(X_test), 100 * y_test.mean())
    logger.info("Feature columns before encoding: %d", X_train.shape[1])


def compare_models(X_train, y_train) -> pd.DataFrame:
    """5-fold stratified cross-validation on the training set only."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    rows = []
    for name, model in get_models(scale_pos_weight).items():
        logger.info("Cross-validating %s ...", name)
        result = cross_validate(build_pipeline(model), X_train, y_train,
                                cv=cv, scoring=SCORING, n_jobs=1)
        row = {"model": name}
        row.update({metric: result[f"test_{metric}"].mean() for metric in SCORING})
        row["roc_auc_std"] = result["test_roc_auc"].std()
        rows.append(row)

    # Rule baseline: no learning involved, so we score it directly on the training set.
    rule = rule_based_predictions(X_train)
    rows.append({
        "model": "Rule baseline (SQL profile)",
        "roc_auc": float("nan"),
        "recall": recall_score(y_train, rule),
        "precision": precision_score(y_train, rule),
        "f1": f1_score(y_train, rule),
        "accuracy": accuracy_score(y_train, rule),
        "roc_auc_std": float("nan"),
    })
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False, na_position="last")


def main() -> int:
    try:
        X_train, X_test, y_train, y_test = load_split()
        sanity_checks(X_train, X_test, y_train, y_test)

        results = compare_models(X_train, y_train)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        results.round(4).to_csv(MODELS_DIR / "phase4_cv_results.csv", index=False)

        pd.set_option("display.width", 200)
        print("\nCross-validation on the TRAINING set (threshold 0.5 for recall/precision/f1):\n")
        print(results.round(3).to_string(index=False))

        # Fit and save the simplest model as a first saved artifact (also tests save/load).
        baseline = build_pipeline(get_models()["Logistic Regression"]).fit(X_train, y_train)
        joblib.dump(baseline, MODELS_DIR / "baseline_logreg.joblib")
        logger.info("Features after encoding: %d", baseline[:-1].transform(X_train).shape[1])
        logger.info("Saved %s", MODELS_DIR / "baseline_logreg.joblib")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        logger.error("Failed: %s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())