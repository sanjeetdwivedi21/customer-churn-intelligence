"""Data loading for the dashboard. Reads saved outputs only: no training, no SHAP here."""
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make "src" importable when Streamlit runs dashboard/app.py
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DB_PATH, MODELS_DIR, PROCESSED_DIR  # noqa: E402
from src.data_preprocessing import load_features  # noqa: E402


def ensure_database() -> None:
    """Build the SQLite database from the raw CSV if it is missing (for example on a fresh deploy)."""
    if not DB_PATH.exists():
        from src.db_setup import main as build_database
        if build_database() != 0:
            raise RuntimeError("Could not build the database from data/raw/telco_churn.csv")


@st.cache_data(show_spinner="Loading data...")
def load_dashboard_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Return (customers merged with scores, SHAP values by customer, model metadata)."""
    scored_path = PROCESSED_DIR / "scored_customers.csv"
    shap_path = PROCESSED_DIR / "shap_values.csv"
    meta_path = MODELS_DIR / "model_metadata.json"
    for path in (scored_path, shap_path, meta_path):
        if not path.exists():
            raise FileNotFoundError(
                f"{path.name} not found. Run: python -m src.model_tuning, "
                "then python -m src.explainability")

    ensure_database()
    features = load_features()
    scored = pd.read_csv(scored_path).drop(columns=["monthly_charges"])
    data = features.merge(scored, on="customer_id", how="inner", validate="one_to_one")
    if len(data) != len(features):
        raise RuntimeError("Scores do not match the database. Re-run: python -m src.explainability")

    data["status"] = data["churn"].map({0: "Active", 1: "Churned"})
    shap_df = pd.read_csv(shap_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return data, shap_df, meta