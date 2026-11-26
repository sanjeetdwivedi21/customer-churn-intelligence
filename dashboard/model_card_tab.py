"""Model card tab: what the model is, how it was built, how well it works, where not to trust it."""
import sys
from pathlib import Path

import pandas as pd
import sklearn
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.campaign_tab import test_customer_ids  # noqa: E402
from src.config import MODELS_DIR, REPORTS_DIR  # noqa: E402
from src.model_card import held_out_metrics  # noqa: E402


def _read_csv(path: Path):
    return pd.read_csv(path) if path.exists() else None


def render(data: pd.DataFrame, metadata: dict) -> None:
    st.markdown("### Model card")
    st.caption("A plain-language summary of what this model is, how it was built, how well it "
               "works and where it should not be trusted.")

    pool = data[data["customer_id"].isin(test_customer_ids())]
    m = held_out_metrics(pool["churn"], pool["churn_probability"], metadata["threshold"])

    st.subheader("Overview")
    o1, o2, o3, o4 = st.columns(4)
    o1.markdown("<div style='font-size:.85rem;color:#6b7280'>Model</div>"
                f"<div style='font-size:1.4rem;font-weight:600'>{metadata['model_name']}</div>",
                unsafe_allow_html=True)
    o2.metric("Trained on", f"{metadata['trained_on_rows']:,} customers")
    o3.metric("Decision threshold", f"{metadata['threshold']:.2f}")
    o4.metric("Features", len(metadata["feature_columns"]))
    st.caption(f"Trained {metadata['trained_at_utc'][:10]} | random seed {metadata['random_seed']} | "
               f"scikit-learn {sklearn.__version__}")

    st.subheader("Intended use")
    left, right = st.columns(2)
    left.markdown("**Intended**\n"
                  "- Rank telecom customers by churn risk to decide who to contact first\n"
                  "- Estimate monthly revenue at risk\n"
                  "- Explain which factors the model associates with higher risk\n"
                  "- Support a human decision about retention offers")
    right.markdown("**Not intended**\n"
                   "- Automated decisions about a customer without human review\n"
                   "- Claims about what causes churn\n"
                   "- Scoring customers of other industries or very different markets\n"
                   "- Decisions about credit, eligibility or pricing for individuals")

    st.subheader("Training data")
    st.markdown("IBM Telco Customer Churn sample dataset: 7,043 customers, 26.5% churned. "
                "Split 80/20 with the same churn share in both parts (stratified, seed "
                f"{metadata['random_seed']}). The test part was used once, after the model and "
                "threshold were fixed. It has no usage, complaint or price-change history.")
    with st.expander("Features used by the model"):
        st.write(", ".join(metadata["feature_columns"]))
        st.caption("Not used: customer ID, gender, monthly charges and total charges. Monthly "
                   "charges only feed the revenue-at-risk estimate.")

    st.subheader("Performance on held-out customers")
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    p1.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    p2.metric("PR-AUC", f"{m['pr_auc']:.3f}")
    p3.metric("Recall", f"{m['recall']:.1%}", help="Share of churners flagged at the decision threshold")
    p4.metric("Precision", f"{m['precision']:.1%}", help="Share of flagged customers who churned")
    p5.metric("Brier score", f"{m['brier']:.3f}", help="Lower is better. Measures how accurate the probabilities are")
    p6.metric("Customers", f"{m['customers']:,}")
    st.caption(f"At threshold {metadata['threshold']:.2f} the model flags {m['flagged']:,} customers. "
               "The threshold is low on purpose: an offer costs far less than a lost customer is "
               "worth (under the business assumptions below), so missing churners costs more than "
               "contacting a few loyal customers. Metrics carry about +/- 0.02 of sampling noise.")

    comparison = _read_csv(MODELS_DIR / "tuning_summary.csv")
    if comparison is not None:
        st.markdown("**Models compared** (training data, out-of-fold)")
        st.dataframe(comparison[["model", "cv_roc_auc", "oof_pr_auc", "threshold", "recall",
                                 "precision", "net_benefit"]], hide_index=True)
        st.caption("The simplest model whose net benefit was within 2% of the best was chosen.")

    st.subheader("Business assumptions (not from the dataset)")
    a = metadata["assumptions"]
    st.markdown(f"Offer cost ${a['offer_cost']:.0f}, offer success rate {a['offer_success_rate']:.0%}, "
                f"a saved customer is worth {a['value_horizon_months']} months of their bill. "
                "Change them in the Campaign tab. Risk tiers: Medium from "
                f"{metadata['risk_thresholds']['medium']:.2f}, High from "
                f"{metadata['risk_thresholds']['high']:.2f}.")

    st.subheader("What drives the predictions")
    drivers = _read_csv(REPORTS_DIR / "shap_global_importance.csv")
    if drivers is not None:
        st.dataframe(drivers.head(5).rename(columns={"mean_abs_shap": "mean |SHAP|",
                                                     "share_pct": "share (%)"}), hide_index=True)
    st.caption("SHAP values describe what the model does, not what causes churn.")

    st.subheader("Limitations and responsible use")
    st.markdown(
        "- The data shows associations, not causes. A real change may not have the same effect "
        "as the model predicts.\n"
        "- It is a snapshot of one telecom dataset. Accuracy on other companies' customers is not "
        "guaranteed.\n"
        "- Scores on the dashboard are in-sample for the customers the model trained on, so they "
        "are slightly optimistic. The numbers above use customers it never saw.\n"
        "- Senior citizen status is a model input; gender is not. Performance for different "
        "customer groups has not been evaluated in this version.\n"
        "- Offer cost, success rate and value horizon are assumptions. Test a campaign on a "
        "sample with a control group before relying on the estimated benefit.\n"
        "- A person should review recommendations before any customer is contacted.")