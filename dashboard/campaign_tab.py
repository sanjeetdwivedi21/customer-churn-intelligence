"""Campaign tab: what is a retention campaign worth under different business assumptions?"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.business_metrics import evaluate_flags  # noqa: E402
from src.campaign import break_even_probability, simulate  # noqa: E402
from src.data_preprocessing import load_customers  # noqa: E402
from src.model_training import load_split  # noqa: E402

TEST_SET = "Held-out test set (honest)"
ALL_CUSTOMERS = "All customers (in-sample, optimistic)"


@st.cache_data(show_spinner=False)
def test_customer_ids() -> list[str]:
    """IDs of the customers the model never saw during training."""
    _, X_test, _, _ = load_split()
    return load_customers().loc[X_test.index, "customer_id"].tolist()


def _show(fig, key: str) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def render(data: pd.DataFrame, metadata: dict) -> None:
    st.markdown("### Campaign simulator")
    st.caption("Change the business assumptions and see what a retention campaign would be worth. "
               "This only re-prices the model's predictions. It does not retrain the model or "
               "change its saved threshold. Offer cost, success rate and horizon are assumptions, "
               "not facts from the dataset.")

    defaults = metadata["assumptions"]
    pool_choice = st.radio("Customers to evaluate", [TEST_SET, ALL_CUSTOMERS], horizontal=True,
                           key="campaign_pool")
    pool = data[data["customer_id"].isin(test_customer_ids())] if pool_choice == TEST_SET else data

    c1, c2, c3, c4 = st.columns(4)
    offer_cost = c1.slider("Offer cost ($)", 5, 300, int(defaults["offer_cost"]), step=5,
                           key="campaign_cost")
    success_rate = c2.slider("Offer success rate (%)", 5, 80,
                             int(round(100 * defaults["offer_success_rate"])),
                             key="campaign_success") / 100
    horizon = c3.slider("Value horizon (months)", 3, 36, int(defaults["value_horizon_months"]),
                        key="campaign_horizon")
    threshold = c4.slider("Probability threshold", 0.05, 0.95, float(metadata["threshold"]),
                          step=0.01, key="campaign_threshold",
                          help="The saved model uses this threshold by default.")

    table, sweep, best = simulate(pool, threshold, offer_cost, success_rate, horizon)
    y, charges = pool["churn"].to_numpy(), pool["monthly_charges"].to_numpy()
    selected = evaluate_flags(y, pool["churn_probability"].to_numpy() >= threshold, charges,
                              offer_cost=offer_cost, success_rate=success_rate,
                              horizon_months=horizon)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Customers contacted", f"{selected['flagged']:,}", help=f"{selected['flagged_pct']}% of the group")
    k2.metric("Churners reached", f"{selected['recall']:.0%}")
    k3.metric("Customers expected to be saved", f"{selected['expected_saved']:,.0f}")
    k4.metric("Estimated net benefit", f"${selected['net_benefit']:,.0f}")
    k5.metric("Best threshold for these assumptions", f"{best:.2f}")

    best_net = float(sweep["net_benefit"].max())
    if best_net <= 0:
        st.warning("With these assumptions no threshold beats contacting nobody (net benefit 0). "
                   "Offers cost more than they are expected to save.")

    average_charge = float(charges.mean()) if len(charges) else 0.0
    break_even = break_even_probability(average_charge, offer_cost, success_rate, horizon)
    if break_even > 1:
        st.info(f"A customer paying the average ${average_charge:,.2f} a month is never worth "
                "contacting at these settings, even if they were certain to leave.")
    else:
        st.info(f"A customer paying the average ${average_charge:,.2f} a month is worth contacting "
                f"when their churn probability is above about {break_even:.0%}.")

    fig = go.Figure()
    fig.add_scatter(x=sweep["threshold"], y=sweep["net_benefit"], mode="lines",
                    line={"color": "#c44e52", "width": 3}, name="Net benefit")
    fig.add_hline(y=0, line_color="#9ca3af")
    fig.add_vline(x=threshold, line_dash="dash", line_color="#1f2a44",
                  annotation_text=f"selected {threshold:.2f}", annotation_position="top left")
    fig.add_vline(x=best, line_dash="dot", line_color="#4c72b0",
                  annotation_text=f"best {best:.2f}", annotation_position="top right")
    fig.update_layout(template="plotly_white", height=380, title="Net benefit by threshold (USD)",
                      xaxis_title="Probability threshold", yaxis_title="Estimated net benefit",
                      margin={"l": 10, "r": 10, "t": 55, "b": 10})
    _show(fig, "campaign_curve")

    st.subheader("Strategies compared")
    st.dataframe(table, hide_index=True)
    st.caption(f"Group: {len(pool):,} customers, {int(y.sum()):,} churned. "
               "The best threshold is found on this same group, so its result is slightly "
               "optimistic. The saved model's threshold was chosen on training data only. "
               "The expected-value rule contacts a customer when probability x success rate x "
               "monthly charge x horizon exceeds the offer cost.")