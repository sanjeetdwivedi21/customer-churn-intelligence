"""Upload & Score tab: score your own customers with the saved model.

The model was trained on the IBM Telco dataset, so the file must use the same columns.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import charts  # noqa: E402
from src.ai_recommendations import build_facts, generate_recommendation  # noqa: E402
from src.batch_scoring import build_recommendations, score_batch  # noqa: E402
from src.explainability import build_explainer, describe_feature  # noqa: E402
from src.model_training import load_split  # noqa: E402
from src.prediction import load_artifacts  # noqa: E402
from src.upload_validation import (MAX_ROWS, REQUIRED_COLUMNS, UploadError,  # noqa: E402
                                   read_upload, sample_template, validate_and_convert)


def _show(fig, key: str) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def _table(frame: pd.DataFrame) -> None:
    try:
        st.dataframe(frame, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(frame, use_container_width=True, hide_index=True)


def _kpi(column, label: str, value: str, sub: str, color: str) -> None:
    column.markdown(f'<div class="kpi" style="--c:{color}"><div class="l">{label}</div>'
                    f'<div class="v">{value}</div><div class="s">{sub}</div></div>',
                    unsafe_allow_html=True)


@st.cache_resource(show_spinner="Preparing the model...")
def get_scoring_assets():
    """Model, metadata and SHAP explainer (background = training set), built once."""
    model, metadata = load_artifacts()
    explainer = build_explainer(model, load_split()[0])
    return model, metadata, explainer


@st.cache_data(show_spinner="Scoring customers...", max_entries=3)
def cached_batch(clean: pd.DataFrame):
    model, metadata, explainer = get_scoring_assets()
    return score_batch(clean, model, metadata, explainer)


def render() -> None:
    st.markdown("### Score your own customers")
    st.caption("Upload a CSV and get churn probabilities, risk levels, explanations and a report. "
               "The model was trained on the IBM Telco dataset, so accuracy on other companies' "
               "customers is not guaranteed. The file is processed in memory and not saved by "
               "this app. Please do not upload real personal data to a public demo.")

    with st.expander("Required file format"):
        st.write(f"A CSV with up to {MAX_ROWS:,} customers and the same columns as the "
                 "IBM Telco Customer Churn dataset. Required columns:")
        st.code(", ".join(REQUIRED_COLUMNS))
        st.write("Optional: `Churn` (Yes/No) adds a performance check. `gender` and "
                 "`TotalCharges` are ignored. Values must match the dataset exactly "
                 "(for example `Month-to-month`, `Fiber optic`, `Electronic check`).")
        st.download_button("Download a sample file", sample_template(),
                           file_name="sample_customers.csv", mime="text/csv")

    uploaded = st.file_uploader("Upload a CSV", type=["csv"])
    if uploaded is None:
        st.info("Upload a CSV to start. You can try the original `data/raw/telco_churn.csv`, "
                "or the sample file above.")
        return

    try:
        clean = validate_and_convert(read_upload(uploaded))
    except UploadError as error:
        st.error("This file cannot be used:")
        st.code(str(error), language=None)
        return

    try:
        result, X, grouped = cached_batch(clean)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        st.error(f"Scoring failed: {error}")
        return

    has_labels = "actual_churn" in result.columns
    pool = result[result["actual_churn"] == 0] if has_labels else result
    st.success(f"Scored {len(result):,} customers.")
    if has_labels:
        st.caption("Your file has a Churn column, so KPIs and the contact list below cover "
                   "customers who have NOT churned.")

    k = st.columns(4)
    _kpi(k[0], "Customers scored", f"{len(result):,}", "rows in your file", "#4c72b0")
    _kpi(k[1], "High risk", f"{int((pool['risk_level'] == 'High').sum()):,}",
         "can still be retained", "#c44e52")
    _kpi(k[2], "Medium risk", f"{int((pool['risk_level'] == 'Medium').sum()):,}",
         "above the model threshold", "#dd8452")
    _kpi(k[3], "Monthly revenue at risk", f"${pool['revenue_at_risk'].sum():,.0f}",
         "probability x charge (estimate)", "#1f2a44")

    if not pool.empty:
        left, right = st.columns([1, 2])
        with left:
            _show(charts.risk_donut(pool), "upload_risk_donut")
        with right:
            _show(charts.revenue_treemap(pool), "upload_treemap")

    if has_labels and result["actual_churn"].nunique() == 2:
        _, metadata, _ = get_scoring_assets()
        flagged = result["churn_probability"] >= metadata["threshold"]
        churned = result["actual_churn"] == 1
        st.markdown("**Performance on your labels**")
        p = st.columns(3)
        p[0].metric("ROC-AUC", f"{roc_auc_score(result['actual_churn'], result['churn_probability']):.3f}")
        p[1].metric("Recall at model threshold", f"{(flagged & churned).sum() / churned.sum():.1%}")
        p[2].metric("Precision at model threshold",
                    f"{(flagged & churned).sum() / max(flagged.sum(), 1):.1%}")
        st.caption("If your file contains customers the model was trained on (such as the original "
                   "Telco file), these numbers are optimistic. The honest test-set result is "
                   "ROC-AUC 0.840.")

    st.divider()
    st.subheader("Customer detail")
    ids = result.sort_values("churn_probability", ascending=False)["customer_id"].tolist()
    chosen = st.selectbox("Customer (highest risk first)", ids, key="upload_customer")
    i = int(result.index[result["customer_id"] == chosen][0])
    facts = build_facts(i, X, result, grouped)
    d1, d2, d3 = st.columns(3)
    d1.metric("Churn probability", f"{facts['churn_probability']:.0%}")
    d2.metric("Risk level", facts["risk_level"])
    d3.metric("Revenue at risk", f"${facts['revenue_at_risk']:,.2f}")
    x = X.iloc[i]
    descriptions = {f: describe_feature(f, x[f]) for f in grouped.columns}
    _show(charts.shap_waterfall(grouped.iloc[i], descriptions), "upload_waterfall")
    rec = generate_recommendation(facts, use_llm=False)
    st.markdown(f"**RECOMMENDATION** `{rec['source']}` | priority: **{rec['priority']}**")
    st.write(rec["recommendation"])
    st.write(f"**Suggested action:** {rec['suggested_action']}")
    st.write(f"**Expected objective:** {rec['expected_objective']}")

    st.divider()
    st.subheader("Contact list: highest revenue at risk")
    candidates = pool[pool["risk_level"].isin(["High", "Medium"])]
    if candidates.empty:
        st.info("No High or Medium risk customers in this file.")
    else:
        n = st.slider("Number of customers", 1, min(500, len(candidates)),
                      min(20, len(candidates)), key="upload_contact_n")
        top = candidates.nlargest(n, "revenue_at_risk")
        report = build_recommendations(result, X, grouped, top.index)
        _table(report[["customer_id", "churn_probability", "risk_level", "priority",
                       "revenue_at_risk", "suggested_action", "expected_objective", "source"]])
        st.download_button("Download recommendation report (CSV)", report.to_csv(index=False),
                           file_name="upload_recommendations.csv", mime="text/csv")

    st.download_button("Download all scores (CSV)", result.to_csv(index=False),
                       file_name="upload_scores.csv", mime="text/csv")
