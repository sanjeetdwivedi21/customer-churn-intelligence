"""Uploading the original dataset must reproduce the saved scores (needs the built artifacts)."""
import numpy as np
import pandas as pd
import pytest

from src.config import DB_PATH, MODELS_DIR, PROCESSED_DIR, RAW_CSV_PATH

pytestmark = pytest.mark.skipif(
    not (DB_PATH.exists() and RAW_CSV_PATH.exists()
         and (MODELS_DIR / "final_model.joblib").exists()
         and (PROCESSED_DIR / "scored_customers.csv").exists()),
    reason="Build the project first (db_setup, model_tuning, explainability)")


@pytest.fixture(scope="module")
def batch():
    from src.batch_scoring import score_batch
    from src.explainability import build_explainer
    from src.model_training import load_split
    from src.prediction import load_artifacts
    from src.upload_validation import read_upload, validate_and_convert

    raw = read_upload(str(RAW_CSV_PATH)).head(300)
    clean = validate_and_convert(raw)
    model, metadata = load_artifacts()
    explainer = build_explainer(model, load_split()[0])
    return score_batch(clean, model, metadata, explainer)


def test_upload_reproduces_saved_scores(batch):
    result, _, _ = batch
    saved = pd.read_csv(PROCESSED_DIR / "scored_customers.csv").head(300)
    assert result["customer_id"].tolist() == saved["customer_id"].tolist()
    assert np.allclose(result["churn_probability"], saved["churn_probability"], atol=1e-4)
    assert (result["risk_level"].to_numpy() == saved["risk_level"].to_numpy()).all()


def test_shap_columns_and_alignment(batch):
    from src.feature_engineering import FEATURE_COLUMNS
    result, X, grouped = batch
    assert list(grouped.columns) == FEATURE_COLUMNS
    assert len(result) == len(X) == len(grouped) == 300


def test_recommendations_are_rule_based_and_complete(batch):
    from src.ai_recommendations import SOURCE_RULES
    from src.batch_scoring import build_recommendations
    result, X, grouped = batch
    report = build_recommendations(result, X, grouped, result.index[:5])
    assert len(report) == 5
    assert set(report["source"]) == {SOURCE_RULES}
    assert report["suggested_action"].str.len().gt(0).all()
