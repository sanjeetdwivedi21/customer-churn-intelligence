import numpy as np
import pandas as pd
import pytest

from src.config import DB_PATH
from src.survival import (by_month, expected_months_kept, group_summary, group_survival,
                          kaplan_meier, median_survival)

DURATIONS = [1, 2, 3, 3, 4]
EVENTS = [1, 0, 1, 1, 0]          # the customers with durations 2 and 4 are still active (censored)


def test_kaplan_meier_hand_calculation():
    km = kaplan_meier(DURATIONS, EVENTS)
    assert list(km["time"]) == [0, 1, 3]
    # t=1: 5 at risk, 1 churn -> 0.8.  t=3: 3 at risk, 2 churn -> 0.8 * (1 - 2/3)
    assert km["survival"].tolist() == pytest.approx([1.0, 0.8, 0.8 / 3])
    assert km["at_risk"].tolist() == [5, 5, 3]
    # Greenwood: variance sum = 1/(5*4) at t=1, plus 2/(3*1) at t=3
    assert (km["survival"] - km["ci_low"]).iloc[1] == pytest.approx(1.96 * 0.8 * np.sqrt(0.05))
    assert km["ci_low"].between(0, 1).all() and km["ci_high"].between(0, 1).all()


def test_no_events_means_everyone_survives():
    km = kaplan_meier([3, 5, 7], [0, 0, 0])
    assert len(km) == 1 and km.loc[0, "survival"] == 1.0
    assert np.isnan(median_survival(km))


def test_by_month_is_a_step_function():
    km = kaplan_meier(DURATIONS, EVENTS)
    assert by_month(km, 5).tolist() == pytest.approx([1.0, 0.8, 0.8, 0.8 / 3, 0.8 / 3, 0.8 / 3])


def test_median_survival():
    km = kaplan_meier([1, 2, 3, 4], [1, 1, 1, 1])
    assert median_survival(km) == 2          # survival is 0.5 after the second churn


def test_expected_months_kept_hand_calculation():
    s = by_month(kaplan_meier(DURATIONS, EVENTS), 5)
    months, counted = expected_months_kept(s, [0, 1], horizon=3)
    assert months[0] == pytest.approx(0.8 + 0.8 + 0.8 / 3) and counted[0] == 3
    # tenure 1, horizon 2: (S(2) + S(3)) / S(1)
    months, counted = expected_months_kept(s, [1], horizon=2)
    assert months[0] == pytest.approx((0.8 + 0.8 / 3) / 0.8)


def test_months_after_the_longest_tenure_are_not_counted():
    s = by_month(kaplan_meier(DURATIONS, EVENTS), 5)
    months, counted = expected_months_kept(s, [4, 5], horizon=5)
    assert list(counted) == [1, 0]
    assert months[0] == pytest.approx(1.0)          # S(5) / S(4), and nothing is lost
    assert months[1] == 0


def test_survival_that_reached_zero_gives_nan_not_infinity():
    months, _ = expected_months_kept(np.array([1.0, 0.5, 0.0, 0.0]), [2], horizon=1)
    assert np.isnan(months[0])


def test_group_summary_matches_the_curves():
    df = pd.DataFrame({
        "g": ["A"] * 5 + ["B"] * 4,
        "tenure_months": DURATIONS + [2, 3, 4, 5],
        "churn": EVENTS + [1, 0, 0, 0],
        "monthly_charges": [10.0] * 9,
    })
    summary = group_summary(df, "g", horizon=3, t_max=5)
    a = summary[summary["group"] == "A"].iloc[0]
    assert a["customers"] == 5 and a["churned"] == 3
    assert a["retained_12m"] == pytest.approx(0.8 / 3)         # month 12 is beyond the data
    assert a["new_customer_months_kept"] == pytest.approx(0.8 + 0.8 + 0.8 / 3)
    assert a["active_avg_revenue"] == pytest.approx(10 * a["active_avg_months_kept"])
    # Active customers A: tenure 2 (3 months countable, keeps 1.0) and tenure 4 (1 countable, keeps 1.0)
    assert a["active_avg_retention"] == pytest.approx((1 / 3 + 1) / 2)
    assert a["active_avg_revenue_scaled"] == pytest.approx((10 * 3 * (1 / 3) + 10 * 3 * 1) / 2)
    assert a["active_truncated_pct"] == 50.0
    # Group B has an active customer at tenure 5 (the end of the data). It is kept in the average
    # with the group's own share (1.0 here), not dropped.
    b = summary[summary["group"] == "B"].iloc[0]
    assert b["active_avg_retention"] == pytest.approx(1.0)
    assert b["active_avg_revenue_scaled"] == pytest.approx(10 * 3 * 1.0)
    assert b["active_truncated_pct"] == 100.0
    assert set(group_survival(df, "g", 5)) == {"A", "B"}


needs_db = pytest.mark.skipif(not DB_PATH.exists(), reason="Run python -m src.db_setup first")


@needs_db
def test_real_data_curves_are_sensible():
    from src.data_preprocessing import load_customers
    df = load_customers()
    t_max = int(df["tenure_months"].max())
    overall = kaplan_meier(df["tenure_months"], df["churn"])
    assert int(overall["events"].sum()) == int(df["churn"].sum()) == 1869
    assert (overall["survival"].diff().dropna() <= 0).all()          # never goes up

    curves = group_survival(df, "contract", t_max)
    assert (curves["Two year"]["survival"][36] > curves["One year"]["survival"][36]
            > curves["Month-to-month"]["survival"][36])
    summary = group_summary(df, "contract", horizon=12, t_max=t_max)
    assert (summary["active_avg_months_kept"].between(0, 12)).all()