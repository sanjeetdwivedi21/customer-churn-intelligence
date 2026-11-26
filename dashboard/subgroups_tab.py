"""Subgroups tab: how does the model perform for different groups of customers?"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.campaign_tab import test_customer_ids  # noqa: E402
from src.subgroups import GROUPINGS, add_labels, all_group_metrics, gap_summary  # noqa: E402

TEST_SET = "Held-out test set (honest, small groups)"
ALL_CUSTOMERS = "All customers (in-sample, optimistic, larger groups)"


def _show(fig, key: str) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def _percent(value: float) -> str:
    return "n/a" if pd.isna(value) else f"{100 * value:.1f}%"


def render(data: pd.DataFrame, metadata: dict) -> None:
    st.markdown("### Subgroup check")
    st.caption("Does the model work equally well for different groups of customers? This is a "
               "descriptive check, not a certification of fairness. Groups are small, so every "
               "recall comes with a 95% interval, and groups with fewer than 50 customers or 20 "
               "churners are marked as small and left out of the gap summary.")

    pool_choice = st.radio("Customers to evaluate", [TEST_SET, ALL_CUSTOMERS], horizontal=True,
                           key="subgroup_pool")
    pool = data[data["customer_id"].isin(test_customer_ids())] if pool_choice == TEST_SET else data
    threshold = metadata["threshold"]
    metrics = all_group_metrics(add_labels(pool), threshold)

    st.subheader("Largest gaps between groups")
    summary = gap_summary(metrics)
    st.dataframe(summary.rename(columns={
        "highest_recall_group": "highest recall", "lowest_recall_group": "lowest recall",
        "recall_gap_pts": "recall gap (points)",
        "highest_recall_churn_pct": "churn rate of highest-recall group (%)",
        "lowest_recall_churn_pct": "churn rate of lowest-recall group (%)",
        "intervals_overlap": "95% intervals overlap?",
        "largest_calibration_gap_pts": "largest calibration gap (points)"}), hide_index=True)
    st.caption("Recall = share of real churners the model flags at the decision threshold "
               f"({threshold:.2f}). Calibration gap = average predicted churn rate minus actual "
               "churn rate. If the intervals overlap, the difference between the best and worst "
               "group may just be noise. Large recall gaps for contract, internet service, tenure "
               "and payment are expected: those groups have very different churn rates, and the model "
               "gives low-churn groups lower probabilities, so fewer of their rare churners cross the "
               "threshold. The calibration gap is the more useful column: it shows whether the "
               "probabilities themselves are right for each group.")

    choice = st.selectbox("Look at one grouping", list(GROUPINGS), key="subgroup_choice")
    part = metrics[metrics["grouping"] == choice].copy()
    part["label"] = np.where(part["reliable"], part["group"], part["group"] + " (small group)")

    overall_recall = float(((pool["churn_probability"] >= threshold) & (pool["churn"] == 1)).sum()
                           / max(int(pool["churn"].sum()), 1))
    recall_fig = px.bar(
        part.assign(recall_pct=100 * part["recall"],
                    up=100 * (part["recall_high"] - part["recall"]),
                    down=100 * (part["recall"] - part["recall_low"])),
        x="label", y="recall_pct", error_y="up", error_y_minus="down",
        labels={"recall_pct": "Recall (%)", "label": ""},
        title=f"Recall by group: {choice} (bars show the 95% interval)",
        color_discrete_sequence=["#c44e52"])
    recall_fig.add_hline(y=100 * overall_recall, line_dash="dash", line_color="#9ca3af",
                         annotation_text="overall", annotation_position="top right")
    recall_fig.update_layout(template="plotly_white", height=380,
                             margin={"l": 10, "r": 10, "t": 55, "b": 10})

    calibration = part.melt(id_vars="label", value_vars=["avg_predicted_pct", "actual_churn_pct"],
                            var_name="measure", value_name="churn_pct")
    calibration["measure"] = calibration["measure"].map(
        {"avg_predicted_pct": "Average predicted", "actual_churn_pct": "Actual"})
    calibration_fig = px.bar(
        calibration, x="label", y="churn_pct", color="measure", barmode="group",
        labels={"churn_pct": "Churn rate (%)", "label": "", "measure": ""},
        title=f"Predicted vs actual churn rate: {choice}",
        color_discrete_map={"Average predicted": "#4c72b0", "Actual": "#c44e52"})
    calibration_fig.update_layout(template="plotly_white", height=380,
                                  margin={"l": 10, "r": 10, "t": 55, "b": 10})

    left, right = st.columns(2)
    with left:
        _show(recall_fig, "subgroup_recall")
    with right:
        _show(calibration_fig, "subgroup_calibration")

    table = pd.DataFrame({
        "group": part["label"],
        "customers": part["customers"],
        "churners": part["churners"],
        "actual churn (%)": part["actual_churn_pct"].round(1),
        "avg predicted (%)": part["avg_predicted_pct"].round(1),
        "calibration gap (pts)": part["calibration_gap_pts"].round(1),
        "flagged (%)": part["flagged_pct"].round(1),
        "recall (95% interval)": [f"{_percent(r)} ({_percent(lo)} to {_percent(hi)})"
                                  for r, lo, hi in zip(part["recall"], part["recall_low"],
                                                       part["recall_high"])],
        "precision": part["precision"].map(_percent),
        "ROC-AUC": part["roc_auc"].round(3),
    })
    st.dataframe(table, hide_index=True)
    st.caption("Gender is not a model input and is shown only to check for differences. Senior "
               "citizen status is a model input. A gap in recall does not by itself show unfair "
               "treatment, and equal recall is only one of several possible fairness criteria. "
               "If the interval for a group is wide, there is not enough data to say.")