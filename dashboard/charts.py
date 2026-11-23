"""Plotly charts for the dashboard. Each function takes data and returns a figure."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

RED, BLUE, ORANGE, GREY = "#c44e52", "#4c72b0", "#dd8452", "#9ca3af"
RISK_COLORS = {"High": RED, "Medium": ORANGE, "Low": BLUE}


def style(fig, height: int = 380):
    """One consistent look for every chart."""
    fig.update_layout(template="plotly_white", height=height,
                      margin={"l": 10, "r": 10, "t": 55, "b": 10},
                      font={"family": "Inter, Segoe UI, sans-serif", "size": 13},
                      title={"font": {"size": 16}})
    return fig


def churn_rate_bar(df: pd.DataFrame, column: str, title: str):
    """Churn rate (%) per category, colored by value, with the overall rate as a dashed line."""
    table = (df.groupby(column, observed=True)["churn"]
               .agg(customers="size", churned="sum").reset_index())
    table[column] = table[column].astype(str)
    table["churn_rate_pct"] = (100 * table["churned"] / table["customers"]).round(1)
    fig = px.bar(table, x=column, y="churn_rate_pct", text="churn_rate_pct",
                 hover_data=["customers", "churned"], title=title, color="churn_rate_pct",
                 color_continuous_scale="Reds", labels={"churn_rate_pct": "Churn rate (%)"})
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.add_hline(y=100 * df["churn"].mean(), line_dash="dash", line_color=GREY,
                  annotation_text="overall", annotation_position="top right")
    fig.update_layout(coloraxis_showscale=False,
                      yaxis_range=[0, max(table["churn_rate_pct"].max() * 1.25, 5)])
    return style(fig)


def churn_distribution(df: pd.DataFrame):
    counts = df["status"].value_counts().reset_index()
    counts.columns = ["status", "customers"]
    fig = px.pie(counts, names="status", values="customers", hole=0.6, title="Churn distribution",
                 color="status", color_discrete_map={"Active": BLUE, "Churned": RED})
    fig.update_traces(textinfo="percent+value")
    return style(fig)


def risk_donut(active: pd.DataFrame):
    counts = active["risk_level"].value_counts().reindex(["High", "Medium", "Low"]).reset_index()
    counts.columns = ["risk_level", "customers"]
    fig = px.pie(counts, names="risk_level", values="customers", hole=0.6,
                 title="Active customers by risk level", color="risk_level",
                 color_discrete_map=RISK_COLORS)
    fig.update_traces(textinfo="label+value")
    return style(fig)


def revenue_treemap(active: pd.DataFrame):
    """Where is the revenue at risk? Size = revenue at risk, color = average churn probability."""
    risky = active[active["revenue_at_risk"] > 0]
    fig = px.treemap(risky, path=["contract", "internet_service", "risk_level"],
                     values="revenue_at_risk", color="churn_probability",
                     color_continuous_scale="Reds", title="Monthly revenue at risk by segment")
    fig.update_traces(textinfo="label+value")
    return style(fig, 430)


def contract_tenure_heatmap(df: pd.DataFrame):
    """Churn rate (%) for every contract x tenure band combination."""
    pivot = (df.pivot_table(index="contract", columns="tenure_band", values="churn",
                            aggfunc="mean", observed=True) * 100).round(1)
    pivot.columns = pivot.columns.astype(str)
    fig = px.imshow(pivot, text_auto=".1f", aspect="auto", color_continuous_scale="Reds",
                    title="Churn rate (%): contract x tenure", labels={"color": "Churn %"})
    return style(fig, 340)


def risk_scatter(active: pd.DataFrame):
    """Each dot is an active customer. Big dots pay more, so they matter more."""
    fig = px.scatter(active, x="tenure_months", y="churn_probability", color="risk_level",
                     size="monthly_charges", opacity=0.55, color_discrete_map=RISK_COLORS,
                     category_orders={"risk_level": ["High", "Medium", "Low"]},
                     hover_data=["customer_id", "contract", "internet_service", "revenue_at_risk"],
                     title="Active customers: tenure vs churn probability (dot size = monthly charge)",
                     labels={"tenure_months": "Tenure (months)", "churn_probability": "Churn probability"})
    return style(fig, 430)


def gain_curve(df: pd.DataFrame):
    """If we contact the top X% by risk score, what share of all churners do we reach?"""
    ranked = df.sort_values("churn_probability", ascending=False).reset_index(drop=True)
    x = 100 * (ranked.index + 1) / len(ranked)
    y = 100 * ranked["churn"].cumsum() / max(ranked["churn"].sum(), 1)
    fig = go.Figure()
    fig.add_scatter(x=x, y=y, mode="lines", name="Model ranking", line={"color": RED, "width": 3})
    fig.add_scatter(x=[0, 100], y=[0, 100], mode="lines", name="Random contact",
                    line={"color": GREY, "dash": "dash"})
    fig.update_layout(title="Gain curve: share of churners reached by contacting the top X%",
                      xaxis_title="% of customers contacted (highest risk first)",
                      yaxis_title="% of churners reached")
    return style(fig, 400)


def revenue_by_contract(df: pd.DataFrame):
    table = df.groupby(["contract", "status"], observed=True)["monthly_charges"].sum().reset_index()
    fig = px.bar(table, x="contract", y="monthly_charges", color="status", barmode="group",
                 title="Monthly revenue by contract and status",
                 labels={"monthly_charges": "Monthly revenue (USD)"},
                 color_discrete_map={"Active": BLUE, "Churned": RED})
    return style(fig)


def tenure_curve(df: pd.DataFrame):
    """Churn rate per tenure month (tenure 0 excluded), with a 6-month rolling average."""
    curve = df[df["tenure_months"] > 0].groupby("tenure_months")["churn"].mean().mul(100).reset_index()
    curve["smoothed"] = curve["churn"].rolling(6, center=True, min_periods=1).mean()
    fig = px.scatter(curve, x="tenure_months", y="churn", opacity=0.4, title="Churn rate by tenure month",
                     labels={"churn": "Churn rate (%)", "tenure_months": "Tenure (months)"})
    fig.add_scatter(x=curve["tenure_months"], y=curve["smoothed"], mode="lines",
                    name="6-month average", line={"color": RED, "width": 3})
    return style(fig)


def probability_gauge(probability: float, medium: float, high: float):
    """Gauge with Low / Medium / High zones taken from the model's own thresholds."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=100 * probability, number={"suffix": "%"},
        title={"text": "Churn probability (model)"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#1f2937", "thickness": 0.25},
               "steps": [{"range": [0, 100 * medium], "color": "#dbe7f5"},
                         {"range": [100 * medium, 100 * high], "color": "#f8dcc4"},
                         {"range": [100 * high, 100], "color": "#f1c4c6"}]}))
    return style(fig, 280)


def shap_global_bar(shap_df: pd.DataFrame):
    importance = shap_df.drop(columns="customer_id").abs().mean().sort_values().tail(12).reset_index()
    importance.columns = ["feature", "mean_abs_shap"]
    fig = px.bar(importance, x="mean_abs_shap", y="feature", orientation="h",
                 color="mean_abs_shap", color_continuous_scale="Blues",
                 title="Global feature importance (mean |SHAP|, log-odds)")
    fig.update_layout(coloraxis_showscale=False, yaxis_title="", xaxis_title="Mean |SHAP|")
    return style(fig, 430)


def shap_waterfall(contributions: pd.Series, descriptions: dict):
    """How this customer's risk is built up, driver by driver (model, log-odds)."""
    top = contributions[contributions.abs() >= 0.05].sort_values()
    fig = go.Figure(go.Waterfall(
        orientation="h", y=[descriptions[f] for f in top.index], x=top.values,
        measure=["relative"] * len(top), connector={"line": {"color": GREY}},
        increasing={"marker": {"color": RED}}, decreasing={"marker": {"color": BLUE}}))
    fig.update_layout(title="Factors behind this prediction (red raises risk, blue lowers it)",
                      xaxis_title="Impact on churn log-odds")
    return style(fig, 120 + 38 * max(len(top), 3))