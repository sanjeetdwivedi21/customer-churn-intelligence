"""Model Comparison tab: why was this model chosen over the others?"""
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_comparison import (agreement_tables, annotate, calibration_table,  # noqa: E402
                                  choose_model, load_oof, metrics_table, net_benefit_table,
                                  pr_table, roc_table, selection_table)

COLORS = {"Logistic Regression": "#c44e52", "Random Forest": "#4c72b0",
          "Gradient Boosting": "#55a868", "XGBoost": "#dd8452"}


@st.cache_data(show_spinner=False)
def cached_oof():
    return load_oof()


def _show(fig, key: str) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except TypeError:
        st.plotly_chart(fig, use_container_width=True, key=key)


def _style(fig, title: str, height: int = 380):
    fig.update_layout(template="plotly_white", title=title, height=height,
                      margin={"l": 10, "r": 10, "t": 55, "b": 10}, legend_title_text="")
    return fig


def render(metadata: dict) -> None:
    st.markdown("### Model comparison")
    oof = cached_oof()
    if oof is None:
        st.info("Run `python -m src.model_tuning` to create the out-of-fold predictions "
                "this tab needs.")
        return

    st.caption("Why was this model chosen? All four tuned models are compared on the TRAINING "
               "customers using out-of-fold predictions: every customer is scored by a model that "
               "never saw them, so the comparison is honest and the test set is not involved.")

    defaults = metadata["assumptions"]
    tolerance = metadata["selection_tolerance"]
    c1, c2, c3 = st.columns(3)
    offer_cost = c1.slider("Offer cost ($)", 5, 300, int(defaults["offer_cost"]), step=5,
                           key="cmp_cost")
    success_rate = c2.slider("Offer success rate (%)", 5, 80,
                             int(round(100 * defaults["offer_success_rate"])),
                             key="cmp_success") / 100
    horizon = c3.slider("Value horizon (months)", 3, 36, int(defaults["value_horizon_months"]),
                        key="cmp_horizon")
    assumptions = {"offer_cost": offer_cost, "success_rate": success_rate, "horizon_months": horizon}

    selection = annotate(selection_table(oof, **assumptions), tolerance)
    chosen = choose_model(selection, tolerance)
    saved = metadata["model_name"]

    if chosen is None:
        st.warning("With these assumptions no model beats contacting nobody.")
    elif chosen == saved:
        st.success(f"Under these assumptions the selection rule picks **{chosen}**, the same model "
                   "that is saved and used everywhere in this app.")
    else:
        st.info(f"Under these assumptions the selection rule would pick **{chosen}**. The saved "
                f"model is **{saved}**, chosen with the default assumptions. Changing the saved "
                "model means re-running model_tuning.")

    metrics = metrics_table(oof)
    table = (metrics.merge(selection, on="model")
                    .sort_values("net_benefit", ascending=False))
    display = pd.DataFrame({
        "model": table["model"],
        "ROC-AUC": table["roc_auc"].round(3),
        "PR-AUC": table["pr_auc"].round(3),
        "Brier": table["brier"].round(3),
        "best threshold": table["best_threshold"],
        "recall": table["recall"].round(3),
        "precision": table["precision"].round(3),
        "net benefit ($)": table["net_benefit"].round(0).astype(int),
        f"within {tolerance:.0%} of best": table["within_tolerance"].map({True: "Yes", False: "No"}),
        "complexity (1 = simplest)": table["complexity"],
    })
    st.dataframe(display, hide_index=True)
    spread = float(metrics["roc_auc"].max() - metrics["roc_auc"].min())
    st.caption(f"The four models are within {spread:.3f} ROC-AUC of each other. The rule: pick the "
               f"simplest model whose net benefit is within {tolerance:.0%} of the best, because "
               "extra complexity is only worth paying for if it earns more. Net benefit is "
               "assumption-based; each model is shown at its own best threshold, found on these "
               "same training predictions.")

    left, right = st.columns(2)
    with left:
        fig = px.line(roc_table(oof), x="fpr", y="tpr", color="model", color_discrete_map=COLORS,
                      labels={"fpr": "False positive rate", "tpr": "True positive rate (recall)"})
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "#9ca3af"})
        _show(_style(fig, "ROC curves (out-of-fold)"), "cmp_roc")
    with right:
        fig = px.line(pr_table(oof), x="recall", y="precision", color="model",
                      color_discrete_map=COLORS)
        fig.add_hline(y=float(oof["churn"].mean()), line_dash="dash", line_color="#9ca3af",
                      annotation_text="no skill", annotation_position="bottom right")
        _show(_style(fig, "Precision-recall curves (out-of-fold)"), "cmp_pr")

    left, right = st.columns(2)
    with left:
        fig = px.line(calibration_table(oof), x="mean_predicted", y="actual_rate", color="model",
                      markers=True, color_discrete_map=COLORS,
                      labels={"mean_predicted": "Average predicted probability",
                              "actual_rate": "Actual churn rate"})
        fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dash", "color": "#9ca3af"})
        _show(_style(fig, "Calibration (closer to the dashed line is better)"), "cmp_cal")
    with right:
        fig = px.line(net_benefit_table(oof, **assumptions), x="threshold", y="net_benefit",
                      color="model", color_discrete_map=COLORS,
                      labels={"threshold": "Probability threshold",
                              "net_benefit": "Estimated net benefit ($)"})
        fig.add_hline(y=0, line_color="#9ca3af")
        _show(_style(fig, "Net benefit by threshold (under the assumptions above)"), "cmp_net")

    st.subheader("How similar are the models?")
    spearman, overlap = agreement_tables(oof)
    left, right = st.columns(2)
    with left:
        fig = px.imshow(spearman, text_auto=".2f", zmin=0.5, zmax=1, color_continuous_scale="Blues",
                        aspect="auto")
        _show(_style(fig, "Rank correlation of predicted risk", 340), "cmp_spearman")
    with right:
        fig = px.imshow(overlap, text_auto=".0%", zmin=0, zmax=1, color_continuous_scale="Blues",
                        aspect="auto")
        _show(_style(fig, "Overlap of the top 10% riskiest customers", 340), "cmp_overlap")
    st.caption("High values mean the models rank customers almost the same way, which is why a "
               "simple model can match a complex one on this dataset. Learning-curve evidence "
               "(reports/figures/18_learning_curve.png) points the same way: more rows or new "
               "features, not a fancier algorithm, are what would move the numbers.")