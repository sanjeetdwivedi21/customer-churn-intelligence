"""What-if tab: change a customer's setup and see how the model's prediction moves."""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction import load_artifacts  # noqa: E402
from src.whatif import (ADDONS, CONTRACTS, PAYMENT_METHODS, score_options,  # noqa: E402
                        single_changes)


@st.cache_resource(show_spinner=False)
def get_model():
    return load_artifacts()


def _show(fig, key: str) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def render(data: pd.DataFrame, customer_id: str) -> None:
    row = data[data["customer_id"] == customer_id].iloc[0]
    model, metadata = get_model()

    st.markdown(f"### What-if for customer {customer_id}")
    st.caption("Change the setup below and see how the MODEL's churn probability moves. "
               "This is a prediction from a statistical model, not a promise: the data shows "
               "associations, not causes, so a real change may not have the same effect. "
               "Monthly charges are kept fixed.")
    if int(row["churn"]) == 1:
        st.info("This customer has already churned in the dataset. The what-if is only illustrative.")

    has_internet = row["internet_service"] != "No"
    left, right = st.columns(2)
    with left:
        contract = st.selectbox("Contract", CONTRACTS, index=CONTRACTS.index(row["contract"]),
                                key=f"wi_contract_{customer_id}")
        payment = st.selectbox("Payment method", PAYMENT_METHODS,
                               index=PAYMENT_METHODS.index(row["payment_method"]),
                               key=f"wi_payment_{customer_id}")
        paperless = st.checkbox("Paperless billing", value=bool(row["paperless_billing"]),
                                key=f"wi_paperless_{customer_id}")
    with right:
        addon_choice = {}
        for column, name in ADDONS.items():
            addon_choice[column] = st.checkbox(f"Has {name}", value=row[column] == "Yes",
                                               disabled=not has_internet,
                                               key=f"wi_{column}_{customer_id}")
        if not has_internet:
            st.caption("This customer has no internet service, so add-ons do not apply.")

    changes = {}
    if contract != row["contract"]:
        changes["contract"] = contract
    if payment != row["payment_method"]:
        changes["payment_method"] = payment
    if int(paperless) != int(row["paperless_billing"]):
        changes["paperless_billing"] = int(paperless)
    if has_internet:
        for column, wanted in addon_choice.items():
            if wanted != (row[column] == "Yes"):
                changes[column] = "Yes" if wanted else "No"

    st.divider()
    if not changes:
        st.info("Change something above to see the effect on this customer.")
    else:
        scenario = score_options(row, [("Your scenario", changes)], model, metadata)
        before, after = scenario.iloc[0], scenario.iloc[1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn probability now", f"{before['churn_probability']:.0%}",
                  help=f"Risk level: {before['risk_level']}")
        m2.metric("Churn probability with changes", f"{after['churn_probability']:.0%}",
                  delta=f"{after['change_pct_points']:+.1f} points", delta_color="inverse",
                  help=f"Risk level: {after['risk_level']}")
        revenue_change = after["revenue_at_risk"] - before["revenue_at_risk"]
        sign = "-" if revenue_change < 0 else "+"
        m3.metric("Revenue at risk (per month)", f"${after['revenue_at_risk']:,.2f}",
                  delta=f"{sign}${abs(revenue_change):,.2f}", delta_color="inverse")
        st.caption(f"Risk level: {before['risk_level']} to {after['risk_level']}")

    st.subheader("Which single change moves this customer's prediction most?")
    options = score_options(row, single_changes(row), model, metadata).iloc[1:]
    ranked = options.sort_values("change_pct_points")
    fig = px.bar(ranked, x="change_pct_points", y="change", orientation="h",
                 color="change_pct_points", color_continuous_scale="RdBu_r",
                 color_continuous_midpoint=0,
                 labels={"change_pct_points": "Change in churn probability (points)", "change": ""},
                 title="Model-predicted change from each single change (negative = lower risk)")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(coloraxis_showscale=False, template="plotly_white",
                      height=140 + 36 * len(ranked), margin={"l": 10, "r": 10, "t": 55, "b": 10})
    _show(fig, "whatif_options")
    table = ranked.assign(churn_probability=(100 * ranked["churn_probability"]).round(1))
    st.dataframe(table[["change", "churn_probability", "risk_level", "change_pct_points"]]
                 .rename(columns={"churn_probability": "new churn probability (%)",
                                  "change_pct_points": "change (points)"}),
                 hide_index=True)