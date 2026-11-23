"""Integration tests on the real database and saved artifacts.

They are skipped (not failed) when the files do not exist yet, so a fresh clone
can run pytest before building anything.
"""
from pathlib import Path

import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.config import DB_PATH, MODELS_DIR, PROCESSED_DIR

ROOT = Path(__file__).resolve().parent.parent
needs_db = pytest.mark.skipif(not DB_PATH.exists(), reason="Run python -m src.db_setup first")
needs_model = pytest.mark.skipif(
    not (MODELS_DIR / "final_model.joblib").exists() or not DB_PATH.exists(),
    reason="Run python -m src.model_tuning first")
needs_scores = pytest.mark.skipif(
    not (PROCESSED_DIR / "scored_customers.csv").exists() or not DB_PATH.exists(),
    reason="Run python -m src.explainability first")


@needs_db
def test_database_matches_known_totals():
    from src.data_preprocessing import load_customers
    df = load_customers()
    assert len(df) == 7043
    assert int(df["churn"].sum()) == 1869
    assert int(df["total_charges"].isna().sum()) == 11
    assert df["customer_id"].is_unique


@needs_db
def test_train_test_split_is_clean_and_stratified():
    from src.model_training import load_split
    X_train, X_test, y_train, y_test = load_split()
    assert len(X_train) + len(X_test) == 7043
    assert abs(len(X_test) / 7043 - 0.2) < 0.01
    assert not set(X_train.index) & set(X_test.index)
    assert abs(y_train.mean() - y_test.mean()) < 0.002
    for forbidden in ("customer_id", "churn", "total_charges"):
        assert forbidden not in X_train.columns


@needs_model
def test_saved_model_metadata_matches_code():
    from src.feature_engineering import FEATURE_COLUMNS
    from src.prediction import load_artifacts
    _, meta = load_artifacts()
    assert meta["feature_columns"] == FEATURE_COLUMNS
    assert meta["risk_thresholds"]["medium"] < meta["risk_thresholds"]["high"]


@needs_model
def test_scoring_is_valid_deterministic_and_monotonic():
    from src.data_preprocessing import load_customers
    from src.prediction import load_artifacts, score_customers
    model, meta = load_artifacts()
    customers = load_customers().head(300)
    first = score_customers(customers, model, meta)
    second = score_customers(customers, model, meta)

    assert first["churn_probability"].between(0, 1).all()
    assert set(first["risk_level"]) <= {"Low", "Medium", "High"}
    assert first["churn_probability"].equals(second["churn_probability"])

    # A higher probability must never get a lower risk level
    rank = first["risk_level"].map({"Low": 0, "Medium": 1, "High": 2})
    ordered = rank.loc[first["churn_probability"].sort_values().index]
    assert ordered.is_monotonic_increasing


@needs_model
def test_model_has_real_signal():
    """Smoke check, not an evaluation: in-sample AUC on all customers."""
    from src.data_preprocessing import load_customers
    from src.prediction import load_artifacts, score_customers
    model, meta = load_artifacts()
    customers = load_customers()
    scored = score_customers(customers, model, meta)
    assert roc_auc_score(customers["churn"], scored["churn_probability"]) > 0.80


@needs_scores
def test_saved_scores_match_database_and_math():
    from src.data_preprocessing import load_customers
    customers = load_customers()
    scored = pd.read_csv(PROCESSED_DIR / "scored_customers.csv")
    assert scored["customer_id"].tolist() == customers["customer_id"].tolist()
    gap = (scored["revenue_at_risk"]
           - scored["churn_probability"] * scored["monthly_charges"]).abs().max()
    assert gap < 0.02


@needs_scores
def test_saved_shap_values_align_with_scores():
    scored = pd.read_csv(PROCESSED_DIR / "scored_customers.csv")
    shap_df = pd.read_csv(PROCESSED_DIR / "shap_values.csv")
    assert scored["customer_id"].tolist() == shap_df["customer_id"].tolist()
    assert not shap_df.drop(columns="customer_id").isna().any().any()


@needs_scores
def test_dashboard_starts_without_exception():
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    app = AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    assert not app.exception, [e.value for e in app.exception]