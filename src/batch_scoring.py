"""Score a batch of customers (for example an uploaded CSV): probabilities, SHAP, recommendations.

Inference only. Nothing is trained here: it uses the saved model and the explainer
built from the training set.
"""
import pandas as pd

from src.ai_recommendations import build_facts, generate_recommendation
from src.explainability import compute_shap, group_shap
from src.prediction import prepare_features, score_customers

PROFILE_COLUMNS = ["tenure_months", "contract", "internet_service", "payment_method"]


def score_batch(clean: pd.DataFrame, model, metadata: dict, explainer):
    """Return (result, X, grouped_shap), all with the same 0..n-1 row index.

    result: customer_id, churn_probability, risk_level, monthly_charges, revenue_at_risk,
            actual_churn (only if the input had a churn column) and a few profile columns.
    """
    customers = clean.reset_index(drop=True)
    X = prepare_features(customers).reset_index(drop=True)
    scored = score_customers(customers, model, metadata)
    _, shap_t, _ = compute_shap(model, explainer, X)
    grouped = group_shap(shap_t)
    result = pd.concat([scored, X[PROFILE_COLUMNS]], axis=1)
    return result, X, grouped


def build_recommendations(result: pd.DataFrame, X: pd.DataFrame, grouped: pd.DataFrame,
                          indices) -> pd.DataFrame:
    """Rule-based recommendations for the given rows (no LLM calls, so no API cost)."""
    rows = [generate_recommendation(build_facts(i, X, result, grouped), use_llm=False)
            for i in indices]
    report = pd.DataFrame(rows)
    if not report.empty:
        report["main_reasons"] = report["main_reasons"].apply("; ".join)
    return report
