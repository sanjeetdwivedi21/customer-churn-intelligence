"""Survival analysis and customer value from tenure and churn.

Method: Kaplan-Meier with tenure (months) as the time, churn as the event, and customers who
have not churned treated as censored at their current tenure.

Caveat: the IBM Telco dataset describes churn as "left within the last month", which does not
strictly fit a cohort design, so these curves are a descriptive illustration, not a forecast.
Values are revenue (not profit), undiscounted, and only counted up to the longest tenure in the
data (72 months), because the curves say nothing about later months.
"""
import numpy as np
import pandas as pd


def kaplan_meier(durations, events) -> pd.DataFrame:
    """Kaplan-Meier estimate with Greenwood 95% intervals.

    Returns one row per event time plus time 0: time, at_risk, events, survival, ci_low, ci_high.
    """
    d = np.asarray(durations).astype(int)
    e = np.asarray(events).astype(int)
    rows = [{"time": 0, "at_risk": len(d), "events": 0, "survival": 1.0,
             "ci_low": 1.0, "ci_high": 1.0}]
    survival, greenwood = 1.0, 0.0
    for t in np.unique(d[e == 1]):
        at_risk = int((d >= t).sum())
        died = int(((d == t) & (e == 1)).sum())
        survival *= 1 - died / at_risk
        if at_risk > died:
            greenwood += died / (at_risk * (at_risk - died))
        se = survival * np.sqrt(greenwood)
        rows.append({"time": int(t), "at_risk": at_risk, "events": died, "survival": survival,
                     "ci_low": max(0.0, survival - 1.96 * se),
                     "ci_high": min(1.0, survival + 1.96 * se)})
    return pd.DataFrame(rows)


def by_month(km: pd.DataFrame, t_max: int, column: str = "survival") -> np.ndarray:
    """The step curve sampled at months 0..t_max (the value holds until the next event)."""
    index = np.searchsorted(km["time"].to_numpy(), np.arange(t_max + 1), side="right") - 1
    return km[column].to_numpy()[index]


def expected_months_kept(survival: np.ndarray, tenure, horizon: int):
    """Expected months a customer stays within the next `horizon` months, given their tenure.

    Sum over k = 1..horizon of S(tenure + k) / S(tenure). Months beyond the longest observed
    tenure are not counted. Returns (expected months, months actually counted).
    """
    s = np.asarray(survival, dtype=float)
    t_max = len(s) - 1
    t0 = np.clip(np.asarray(tenure).astype(int), 0, t_max)
    counted = np.minimum(horizon, t_max - t0)
    cumulative = np.concatenate([[0.0], np.cumsum(s[1:])])          # cumulative[t] = S(1) + ... + S(t)
    with np.errstate(divide="ignore", invalid="ignore"):
        months = (cumulative[t0 + counted] - cumulative[t0]) / s[t0]
    return np.where(s[t0] > 0, months, np.nan), counted


def median_survival(km: pd.DataFrame) -> float:
    """First month where survival drops to 0.5 or below, or NaN if it never does."""
    below = km[km["survival"] <= 0.5]
    return float(below["time"].iloc[0]) if len(below) else float("nan")


def group_survival(df: pd.DataFrame, column: str, t_max: int) -> dict:
    """Kaplan-Meier curves for each value of a column. df needs: tenure_months, churn."""
    curves = {}
    for group, part in df.groupby(column, observed=True, sort=True):
        km = kaplan_meier(part["tenure_months"], part["churn"])
        curves[str(group)] = {"km": km, "customers": len(part),
                              "survival": by_month(km, t_max),
                              "ci_low": by_month(km, t_max, "ci_low"),
                              "ci_high": by_month(km, t_max, "ci_high")}
    return curves


def group_summary(df: pd.DataFrame, column: str, horizon: int, t_max: int) -> pd.DataFrame:
    """Retention and customer value for each group.

    Value columns cover customers who have not churned: expected months kept within the next
    `horizon` months (only months up to the longest observed tenure are counted), the expected
    share of those months kept, and revenue scaled to the full horizon so that groups with many
    long-tenure customers are not unfairly penalised. Customers already at the end of the data
    are given their group's average share.
    """
    curves = group_survival(df, column, t_max)
    rows = []
    for group, curve in curves.items():
        part = df[df[column].astype(str) == group]
        active = part[part["churn"] == 0]
        s = curve["survival"]
        months, counted = expected_months_kept(s, active["tenure_months"], horizon)
        new_customer, _ = expected_months_kept(s, [0], horizon)
        # Share of the countable months a customer is expected to stay. Customers whose tenure
        # is already at the end of the data have nothing to count, so they get their group's
        # average share (the same assumption used for months beyond the data).
        countable = (counted > 0) & ~np.isnan(months)
        charges = active["monthly_charges"].to_numpy()
        share = np.divide(months, counted, out=np.full(len(months), np.nan), where=countable)
        group_share = float(np.nanmean(share)) if countable.any() else float("nan")
        share_filled = np.where(countable, share, group_share)
        rows.append({
            "group": group,
            "customers": curve["customers"],
            "churned": int(part["churn"].sum()),
            "retained_12m": float(s[min(12, t_max)]),
            "retained_24m": float(s[min(24, t_max)]),
            "retained_36m": float(s[min(36, t_max)]),
            "median_months": median_survival(curve["km"]),
            "new_customer_months_kept": float(new_customer[0]),
            "active_avg_months_kept": float(np.nanmean(months)) if len(active) else float("nan"),
            "active_avg_revenue": (float(np.nanmean(months * active["monthly_charges"].to_numpy()))
                                   if len(active) else float("nan")),
            "active_truncated_pct": (100 * float((counted < horizon).mean()) if len(active) else 0.0),
            "active_avg_retention": group_share,
            # Revenue scaled to the full horizon: charge x horizon x share of countable months kept.
            # Assumes retention after the last observed month is like retention in counted months.
            "active_avg_revenue_scaled": (float((horizon * share_filled * charges).mean())
                                          if countable.any() else float("nan")),
        })
    return pd.DataFrame(rows)