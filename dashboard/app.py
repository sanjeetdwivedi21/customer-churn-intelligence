"""Customer Churn Intelligence dashboard.

Run from the project root:  python -m streamlit run dashboard/app.py
"""
import sys
from pathlib import Path
from dashboard import upload_tab  # noqa: E402
from dashboard import whatif_tab  # noqa: E402

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard import charts  # noqa: E402
from dashboard.data import load_dashboard_data  # noqa: E402
from src.ai_recommendations import generate_recommendation, load_env  # noqa: E402
from src.explainability import describe_feature  # noqa: E402
from src.prediction import prepare_features  # noqa: E402

st.set_page_config(page_title="Customer Churn Intelligence", page_icon="📉", layout="wide")
load_env()

CSS = """
<style>
.block-container {padding-top: 1.4rem; max-width: 1400px;}
html, body, [class*="css"] {font-family: 'Inter', 'Segoe UI', sans-serif;}

/* Sidebar */
[data-testid="stSidebar"] {background: #111827;}
[data-testid="stSidebar"] * {color: #e5e7eb;}
[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] small {color: #9ca3af !important;}
[data-testid="stSidebar"] input {color: #111827 !important;}

/* Hero */
.hero {background: linear-gradient(120deg, #1f2a44 0%, #6d3a5c 55%, #c44e52 100%); color: #fff;
       padding: 1.5rem 2rem; border-radius: 18px; margin-bottom: 1.2rem;
       box-shadow: 0 10px 30px rgba(31,42,68,.25);}
.hero h1 {margin: 0; font-size: 1.9rem; color: #fff; letter-spacing: -.01em;}
.hero p {margin: .4rem 0 0; opacity: .85; font-size: .93rem;}

/* KPI cards */
.kpi {background: #fff; border: 1px solid #eceef3; border-top: 4px solid var(--c);
      border-radius: 14px; padding: .9rem 1.05rem; transition: all .18s ease;
      box-shadow: 0 2px 10px rgba(0,0,0,.05);}
.kpi:hover {transform: translateY(-3px); box-shadow: 0 10px 22px rgba(0,0,0,.10);}
.kpi .i {font-size: 1.15rem;}
.kpi .l {font-size: .72rem; color: #6b7280; text-transform: uppercase; letter-spacing: .06em;}
.kpi .v {font-size: 1.7rem; font-weight: 700; color: #111827; line-height: 1.25;}
.kpi .s {font-size: .74rem; color: #9ca3af;}

/* Pill tabs */
.stTabs [data-baseweb="tab-list"] {gap: .4rem; border-bottom: none;}
.stTabs [data-baseweb="tab"] {background: #f3f4f6; border-radius: 999px; padding: .5rem 1.1rem;
      font-weight: 600; height: auto;}
.stTabs [aria-selected="true"] {background: #c44e52 !important; color: #fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {display: none;}

/* Chart cards */
[data-testid="stPlotlyChart"] {background: #fff; border: 1px solid #eceef3; border-radius: 14px;
      padding: .4rem; box-shadow: 0 2px 10px rgba(0,0,0,.04);}

/* Section headers and badge */
.sec {margin: 1.2rem 0 .4rem;}
.sec h3 {margin: 0; font-size: 1.15rem; color: #111827;}
.sec span {font-size: .82rem; color: #6b7280;}
.badge {display: inline-block; padding: .25rem .8rem; border-radius: 999px; color: #fff;
        font-weight: 600; font-size: .85rem; vertical-align: middle;}
.foot {text-align: center; color: #9ca3af; font-size: .78rem; margin-top: 2rem;
       padding-top: 1rem; border-top: 1px solid #eceef3;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ---------- Small helpers ----------
def show(fig) -> None:
    """Plotly chart that works on both old and new Streamlit versions."""
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_table(frame: pd.DataFrame) -> None:
    try:
        st.dataframe(frame, width="stretch", hide_index=True)
    except TypeError:
        st.dataframe(frame, use_container_width=True, hide_index=True)


def kpi(column, label: str, value: str, sub: str, color: str, icon: str = "") -> None:
    column.markdown(f'<div class="kpi" style="--c:{color}"><div class="i">{icon}</div>'
                    f'<div class="l">{label}</div><div class="v">{value}</div>'
                    f'<div class="s">{sub}</div></div>', unsafe_allow_html=True)


def section(title: str, subtitle: str = "") -> None:
    st.markdown(f'<div class="sec"><h3>{title}</h3><span>{subtitle}</span></div>',
                unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def build_features_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return prepare_features(frame).reset_index(drop=True)


try:
    data, shap_df, meta = load_dashboard_data()
except (FileNotFoundError, RuntimeError) as error:
    st.error(str(error))
    st.stop()

SHAP_COLUMNS = [c for c in shap_df.columns if c != "customer_id"]
MIN_IMPACT = 0.05
MEDIUM_T, HIGH_T = meta["risk_thresholds"]["medium"], meta["risk_thresholds"]["high"]
X_all = build_features_frame(data)

# ---------- Sidebar ----------
st.sidebar.header("Filters")
contracts = st.sidebar.multiselect("Contract", sorted(data["contract"].unique()),
                                   default=sorted(data["contract"].unique()))
internet = st.sidebar.multiselect("Internet service", sorted(data["internet_service"].unique()),
                                  default=sorted(data["internet_service"].unique()))
tenure_max = int(data["tenure_months"].max())
tenure_range = st.sidebar.slider("Tenure (months)", 0, tenure_max, (0, tenure_max),
                                 key="tenure_filter_v2")

filtered = data[data["contract"].isin(contracts) & data["internet_service"].isin(internet)
                & data["tenure_months"].between(*tenure_range)]
if filtered.empty:
    st.warning("No customers match these filters.")
    st.stop()
active = filtered[filtered["churn"] == 0]
st.sidebar.caption(f"{len(filtered):,} of {len(data):,} customers selected")

st.sidebar.divider()
st.sidebar.header("Selected customer")
typed = st.sidebar.text_input("Customer ID", placeholder="e.g. 5150-ITWWB").strip()
ranked_ids = filtered.sort_values("churn_probability", ascending=False)["customer_id"].tolist()
chosen = typed if typed else st.sidebar.selectbox("...or pick (highest risk first)", ranked_ids)
if chosen not in set(data["customer_id"]):
    st.sidebar.error(f"Customer '{chosen}' not found.")
    st.stop()
st.sidebar.caption(f"Model: {meta['model_name']} | threshold {meta['threshold']:.2f}")

st.markdown('<div class="hero"><h1>📉 Customer Churn Intelligence</h1>'
            '<p>Predict who may leave, understand why, and decide who to contact first. '
            'Scores come from a statistical model and show associations, not causes.</p></div>',
            unsafe_allow_html=True)

tab_a, tab_b, tab_c, tab_d, tab_e, tab_f, tab_g = st.tabs(
    ["📊 Executive Overview", "🔍 Customer Analytics", "🎯 Churn Prediction",
     "🧠 Explainable AI", "💡 AI Retention Advisor", "📤 Upload & Score", "🧪 What-if"])


def customer_facts(customer_id: str) -> dict:
    """Facts for one customer from saved scores and SHAP (same shape as Phase 7)."""
    i = data.index[data["customer_id"] == customer_id][0]
    x, contrib, row = X_all.iloc[i], shap_df.iloc[i][SHAP_COLUMNS], data.iloc[i]

    def pick(series):
        return [{"feature": f, "description": describe_feature(f, x[f]), "impact": round(float(v), 3)}
                for f, v in series.items()]

    return {
        "customer_id": customer_id,
        "churn_probability": float(row["churn_probability"]),
        "risk_level": str(row["risk_level"]),
        "monthly_charges": float(row["monthly_charges"]),
        "revenue_at_risk": float(row["revenue_at_risk"]),
        "profile": {"tenure_months": int(x["tenure_months"]), "contract": str(x["contract"]),
                    "internet_service": str(x["internet_service"]),
                    "payment_method": str(x["payment_method"]),
                    "has_tech_support": bool(x["tech_support"]),
                    "has_online_security": bool(x["online_security"])},
        "drivers_up": pick(contrib[contrib >= MIN_IMPACT].sort_values(ascending=False).head(5)),
        "drivers_down": pick(contrib[contrib <= -MIN_IMPACT].sort_values().head(3)),
        "_contrib": contrib, "_x": x,
    }


def public(facts: dict) -> dict:
    return {k: v for k, v in facts.items() if not k.startswith("_")}


# ---------- A. Executive Overview ----------
with tab_a:
    k = st.columns(5)
    kpi(k[0], "Total customers", f"{len(filtered):,}", "in current filter", "#4c72b0", "👥")
    kpi(k[1], "Churned", f"{int(filtered['churn'].sum()):,}", "already left", "#c44e52", "🚪")
    kpi(k[2], "Churn rate", f"{100 * filtered['churn'].mean():.1f}%", "churned / total", "#dd8452", "📉")
    kpi(k[3], "High-risk active", f"{int((active['risk_level'] == 'High').sum()):,}",
        "can still be retained", "#c44e52", "🔥")
    kpi(k[4], "Monthly revenue at risk", f"${active['revenue_at_risk'].sum():,.0f}",
        "probability x charge (estimate)", "#1f2a44", "💰")
    st.caption("High-risk and revenue at risk cover ACTIVE customers. Scores for customers the "
               "model was trained on are slightly optimistic.")
    left, right = st.columns([1, 2])
    with left:
        show(charts.risk_donut(active))
    with right:
        show(charts.revenue_treemap(active))
    left, right = st.columns(2)
    with left:
        show(charts.gain_curve(filtered))
    with right:
        show(charts.churn_rate_bar(filtered, "contract", "Churn rate by contract"))
    st.caption("Gain curve is computed on all customers in the filter, so it is slightly optimistic "
               "(the honest test-set result is in reports/model_evaluation.md).")

# ---------- B. Customer Analytics ----------
with tab_b:
    c1, c2 = st.columns(2)
    with c1:
        show(charts.contract_tenure_heatmap(filtered))
    with c2:
        show(charts.churn_distribution(filtered))
    show(charts.risk_scatter(active))
    c3, c4 = st.columns(2)
    with c3:
        show(charts.churn_rate_bar(filtered, "internet_service", "Churn by internet service"))
    with c4:
        show(charts.churn_rate_bar(filtered, "payment_method", "Churn by payment method"))
    c5, c6 = st.columns(2)
    with c5:
        show(charts.churn_rate_bar(filtered, "tenure_band", "Churn by tenure band"))
    with c6:
        show(charts.tenure_curve(filtered))
    show(charts.revenue_by_contract(filtered))

# ---------- C. Churn Prediction ----------
facts = customer_facts(chosen)
risk_color = {"High": "#c44e52", "Medium": "#dd8452", "Low": "#4c72b0"}[facts["risk_level"]]
with tab_c:
    st.markdown(f"### Customer {chosen} "
                f'<span class="badge" style="background:{risk_color}">{facts["risk_level"]} risk</span>',
                unsafe_allow_html=True)
    st.caption("MODEL PREDICTION (statistical model)")
    left, right = st.columns([1, 1])
    with left:
        show(charts.probability_gauge(facts["churn_probability"], MEDIUM_T, HIGH_T))
    with right:
        m1, m2 = st.columns(2)
        kpi(m1, "Monthly charge", f"${facts['monthly_charges']:,.2f}", "current bill", "#4c72b0")
        kpi(m2, "Revenue at risk", f"${facts['revenue_at_risk']:,.2f}", "per month, estimate", "#c44e52")
        st.write("")
        st.markdown("**Main factors raising the risk**")
        for d in facts["drivers_up"] or [{"description": "None above the impact threshold"}]:
            st.write(f"🔺 {d['description']}")
        for d in facts["drivers_down"]:
            st.write(f"🔹 {d['description']} (lowers risk)")
    st.caption(f"Risk tiers: Medium at {MEDIUM_T:.2f} or above, High at {HIGH_T:.2f} or above.")
    if int(data.loc[data["customer_id"] == chosen, "churn"].iloc[0]) == 1:
        st.info("This customer has already churned in the dataset.")

# ---------- D. Explainable AI ----------
with tab_d:
    left, right = st.columns(2)
    with left:
        show(charts.shap_global_bar(shap_df))
    with right:
        descriptions = {f: describe_feature(f, facts["_x"][f]) for f in SHAP_COLUMNS}
        show(charts.shap_waterfall(facts["_contrib"], descriptions))
    st.caption("SHAP values are in log-odds and describe what the MODEL does, not what causes churn. "
               f"Right chart: customer {chosen}.")

# ---------- E. AI Retention Advisor ----------
with tab_e:
    st.subheader(f"Recommendation for {chosen}")
    use_llm = st.checkbox("Use LLM if an API key is configured", value=True)
    rec = generate_recommendation(public(facts), use_llm)
    st.markdown(f"**RECOMMENDATION** `{rec['source']}`  |  priority: **{rec['priority']}**")
    st.write(rec["recommendation"])
    st.write(f"**Suggested action:** {rec['suggested_action']}")
    st.write(f"**Expected objective:** {rec['expected_objective']}")

    st.divider()
    st.subheader("Contact list: highest revenue at risk")
    n = st.slider("Number of customers", 5, 100, 20, key="contact_n")
    top = active[active["risk_level"].isin(["High", "Medium"])].nlargest(n, "revenue_at_risk")
    rows = [generate_recommendation(public(customer_facts(cid)), False) for cid in top["customer_id"]]
    report = pd.DataFrame(rows)
    if report.empty:
        st.info("No High or Medium risk active customers in the current filter.")
    else:
        report["main_reasons"] = report["main_reasons"].apply("; ".join)
        show_table(report[["customer_id", "churn_probability", "risk_level", "priority",
                           "revenue_at_risk", "suggested_action", "expected_objective", "source"]])
        st.download_button("⬇ Download recommendation report (CSV)", report.to_csv(index=False),
                           file_name="retention_recommendations.csv", mime="text/csv")

st.markdown(f'<div class="foot">Model: {meta["model_name"]} | threshold {meta["threshold"]:.2f} | '
            'Business assumptions (offer cost, success rate) are estimates, not dataset facts | '
            'Associations, not causes</div>', unsafe_allow_html=True)


# ---------- F. Upload & Score ----------
with tab_f:
    upload_tab.render()


# ---------- G. What-if ----------
with tab_g:
    whatif_tab.render(data, chosen)