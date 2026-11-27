"""Survival & Value tab: how long do customers stay, and what is an active customer worth?"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.subgroups import add_labels  # noqa: E402
from src.survival import expected_months_kept, group_summary, group_survival  # noqa: E402

GROUP_COLUMNS = {"Contract": "contract", "Internet service": "internet_service",
                 "Payment method": "payment_method", "Senior citizen": "senior_label",
                 "Dependents": "dependents_label"}
PALETTE = ["#c44e52", "#4c72b0", "#55a868", "#dd8452", "#8172b3"]


def _show(fig, key: str) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def render(data: pd.DataFrame, customer_id: str) -> None:
    st.markdown("### Survival and customer value")
    st.caption("How long do customers stay, and what is an active customer worth over the next "
               "months? Method: Kaplan-Meier with tenure as the time and churn as the event; "
               "customers who have not churned are counted up to their current tenure. The Telco "
               "dataset describes churn as 'left within the last month', which does not strictly "
               "fit this design, so treat the curves as a descriptive illustration, not a forecast. "
               "Values are revenue, not profit, and are not discounted.")

    labelled = add_labels(data)
    t_max = int(labelled["tenure_months"].max())
    left, right = st.columns([1, 2])
    choice = left.selectbox("Compare by", list(GROUP_COLUMNS), key="surv_group")
    horizon = right.slider("Value horizon (months)", 3, 36, 12, key="surv_horizon")
    column = GROUP_COLUMNS[choice]

    curves = group_survival(labelled, column, t_max)
    months = list(range(t_max + 1))
    fig = go.Figure()
    for i, (group, curve) in enumerate(curves.items()):
        color = PALETTE[i % len(PALETTE)]
        fig.add_scatter(x=months, y=curve["ci_high"], mode="lines", line={"width": 0, "shape": "hv"},
                        showlegend=False, hoverinfo="skip")
        fig.add_scatter(x=months, y=curve["ci_low"], mode="lines", line={"width": 0, "shape": "hv"},
                        fill="tonexty", fillcolor=_rgba(color, 0.15), showlegend=False,
                        hoverinfo="skip")
        fig.add_scatter(x=months, y=curve["survival"], mode="lines",
                        line={"color": color, "width": 3, "shape": "hv"},
                        name=f"{group} (n={curve['customers']:,})")
    fig.update_layout(template="plotly_white", height=420, title=f"Share of customers still active, by {choice.lower()} (shaded = 95% interval)",
                      xaxis_title="Tenure (months)", yaxis_title="Share still active",
                      yaxis_range=[0, 1.02], margin={"l": 10, "r": 10, "t": 55, "b": 10},
                      legend_title_text="")
    _show(fig, "surv_curves")

    summary = group_summary(labelled, column, horizon, t_max)
    st.dataframe(pd.DataFrame({
        "group": summary["group"],
        "customers": summary["customers"],
        "churned": summary["churned"],
        "still active at 12 months": summary["retained_12m"].map("{:.0%}".format),
        "at 24 months": summary["retained_24m"].map("{:.0%}".format),
        "at 36 months": summary["retained_36m"].map("{:.0%}".format),
        "median tenure (months)": summary["median_months"].map(
            lambda v: "not reached" if pd.isna(v) else f"{v:.0f}"),
    }), hide_index=True)
    st.caption("'Not reached' means fewer than half of the group churned within the longest "
               "observed tenure. Wide shaded bands mean few customers, so be careful reading "
               "differences there.")

    st.subheader(f"What is an active customer worth over the next {horizon} months?")
    value_fig = px.bar(summary, x="group", y="active_avg_revenue_scaled", text_auto="$,.0f",
                       color_discrete_sequence=["#4c72b0"],
                       labels={"active_avg_revenue_scaled": "Expected revenue ($)", "group": ""},
                       title=f"Average expected revenue per active customer, next {horizon} months")
    value_fig.update_layout(template="plotly_white", height=360,
                            margin={"l": 10, "r": 10, "t": 55, "b": 10})
    _show(value_fig, "surv_value")
    st.dataframe(pd.DataFrame({
        "group": summary["group"],
        "avg share of countable months kept": summary["active_avg_retention"].map(
            lambda v: "n/a" if pd.isna(v) else f"{v:.0%}"),
        f"avg expected revenue, next {horizon} months ($)": summary["active_avg_revenue_scaled"].round(0),
        "share with fewer months counted": summary["active_truncated_pct"].map("{:.0f}%".format),
    }), hide_index=True)
    st.caption("For each customer: expected months kept = average of S(tenure + k) / S(tenure) over "
               f"the next months, using the group's curve. Only months up to the longest observed "
               f"tenure ({t_max}) can be counted, which mostly affects long-tenure customers (many "
               "two-year customers). To keep groups comparable, revenue is scaled to the full "
               "horizon: monthly charge x horizon x the share of countable months the customer is "
               "expected to stay. That assumes retention after the last observed month is like "
               "retention in the counted months. Customers already at the end of the data get their "
               "group's average share. The Campaign tab assumes a saved customer is worth 12 months of their bill; "
               "compare that with these numbers.")

    st.subheader(f"Selected customer: {customer_id}")
    row = labelled[labelled["customer_id"] == customer_id].iloc[0]
    if int(row["churn"]) == 1:
        st.info("This customer has already churned in the dataset.")
        return
    curve = curves[str(row[column])]
    kept, counted = expected_months_kept(curve["survival"], [int(row["tenure_months"])], horizon)
    months_counted = int(counted[0])
    if months_counted == 0 or pd.isna(kept[0]):
        st.info("This customer's tenure is at the end of the observed data, so no future months "
                "can be estimated.")
        return
    charge = float(row["monthly_charges"])
    retention = float(kept[0]) / months_counted
    k1, k2, k3 = st.columns(3)
    k1.metric(f"Expected months kept (of {months_counted} counted)", f"{kept[0]:.1f}")
    k2.metric("Monthly charge", f"${charge:,.2f}")
    k3.metric(f"Expected revenue, next {horizon} months", f"${charge * horizon * retention:,.0f}")
    st.caption(f"Based on the curve for {choice.lower()} = {row[column]}. Tenure "
               f"{int(row['tenure_months'])} months; {months_counted} of the next {horizon} months can "
               "be counted, and the result is scaled to the full horizon. It uses the group's "
               "curve, not the customer's own churn probability.")