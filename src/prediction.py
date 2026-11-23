"""Inference only: load the saved model and score customers. Nothing is trained here."""
import json

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS_DIR
from src.feature_engineering import ID_COLUMN, TARGET, build_features

MODEL_PATH = MODELS_DIR / "final_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"


def load_artifacts():
    """Return (pipeline, metadata) saved by src.model_tuning."""
    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        raise FileNotFoundError("Model files not found. Run first: python -m src.model_tuning")
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return model, metadata


def prepare_features(customers: pd.DataFrame) -> pd.DataFrame:
    """Validate the input and build model features. Works with or without a churn column."""
    if ID_COLUMN not in customers.columns:
        raise ValueError(f"Input must contain the '{ID_COLUMN}' column")
    # The target is never used for scoring; build_features only requires the column to exist.
    data = customers if TARGET in customers.columns else customers.assign(**{TARGET: 0})
    X, _ = build_features(data)
    return X


def assign_risk(probabilities, metadata) -> np.ndarray:
    """Low / Medium / High from the thresholds stored with the model."""
    tiers = metadata["risk_thresholds"]
    p = np.asarray(probabilities, dtype=float)
    return np.select([p >= tiers["high"], p >= tiers["medium"]], ["High", "Medium"], default="Low")


def score_customers(customers: pd.DataFrame, model, metadata) -> pd.DataFrame:
    """Churn probability, risk level and expected monthly revenue at risk per customer."""
    X = prepare_features(customers)
    model_columns = [c for c in X.columns if c != "monthly_charges"]
    if model_columns != metadata["feature_columns"]:
        raise ValueError("Feature columns do not match the ones the model was trained on")

    probability = model.predict_proba(X)[:, 1]
    scored = pd.DataFrame({
        "customer_id": customers[ID_COLUMN].to_numpy(),
        "churn_probability": probability.round(4),
        "risk_level": assign_risk(probability, metadata),
        "monthly_charges": X["monthly_charges"].to_numpy(),
        # Expected monthly revenue lost = probability x monthly charge (an estimate)
        "revenue_at_risk": (probability * X["monthly_charges"].to_numpy()).round(2),
    }, index=customers.index)
    if TARGET in customers.columns:
        scored["actual_churn"] = customers[TARGET].to_numpy()
    return scored