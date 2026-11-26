import math

import numpy as np
import pandas as pd
import pytest

from src.config import DB_PATH, MODELS_DIR, PROCESSED_DIR
from src.subgroups import (GROUPINGS, add_labels, all_group_metrics, gap_summary,
                           group_metrics, wilson_interval)


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "g": ["A"] * 4 + ["B"] * 2,
        "churn": [1, 1, 0, 0, 1, 0],
        "churn_probability": [0.9, 0.3, 0.2, 0.6, 0.8, 0.1],
    })


def test_wilson_interval_known_values():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3)
    assert high == pytest.approx(0.5962, abs=1e-3)
    low, high = wilson_interval(0, 10)
    assert low == 0 and 0 < high < 0.4          # never below 0, still some upper bound
    assert all(math.isnan(v) for v in wilson_interval(0, 0))


def test_group_metrics_hand_calculation(frame):
    result = group_metrics(frame, "g", threshold=0.5, min_customers=1, min_churners=1)
    a = result[result["group"] == "A"].iloc[0]
    b = result[result["group"] == "B"].iloc[0]
    # Group A flagged: customers 0 and 3 -> one real churner caught out of two
    assert a["customers"] == 4 and a["churners"] == 2
    assert a["recall"] == 0.5 and a["precision"] == 0.5 and a["flagged_pct"] == 50
    assert a["avg_predicted_pct"] == pytest.approx(50) and a["calibration_gap_pts"] == pytest.approx(0)
    assert a["roc_auc"] == pytest.approx(0.75)
    # Group B flagged: customer 4 -> the only churner caught
    assert b["recall"] == 1 and b["precision"] == 1
    assert b["calibration_gap_pts"] == pytest.approx(-5)       # predicted 45%, actual 50%


def test_small_groups_are_marked_unreliable_and_have_no_auc(frame):
    result = group_metrics(frame, "g", threshold=0.5)          # default limits (50 and 20)
    assert not result["reliable"].any()
    assert result["roc_auc"].isna().all()


def test_group_order_is_respected(frame):
    result = group_metrics(frame, "g", threshold=0.5, order=["B", "A"], min_customers=1, min_churners=1)
    assert list(result["group"]) == ["B", "A"]


def test_gap_summary_uses_only_reliable_groups(frame):
    metrics = group_metrics(frame, "g", threshold=0.5, min_customers=1, min_churners=1)
    metrics.insert(0, "grouping", "Test")
    summary = gap_summary(metrics)
    row = summary.iloc[0]
    assert row["highest_recall_group"] == "B" and row["lowest_recall_group"] == "A"
    assert row["recall_gap_pts"] == 50.0
    assert row["highest_recall_churn_pct"] == 50.0 and row["lowest_recall_churn_pct"] == 50.0
    assert gap_summary(metrics.assign(reliable=False)).empty


def test_add_labels_covers_every_grouping_column():
    df = pd.DataFrame({"senior_citizen": [1, 0], "has_partner": [1, 0], "has_dependents": [0, 1],
                       "tenure_band": pd.Categorical(["0-12 months", "49+ months"])})
    labelled = add_labels(df)
    assert list(labelled["senior_label"]) == ["Senior", "Not senior"]
    assert list(labelled["partner_label"]) == ["Has partner", "No partner"]
    assert list(labelled["tenure_band_label"]) == ["0-12 months", "49+ months"]
    for column, _ in GROUPINGS.values():
        assert column in labelled.columns or column in ("gender", "contract",
                                                         "internet_service", "payment_method")


needs_model = pytest.mark.skipif(
    not (DB_PATH.exists() and (MODELS_DIR / "final_model.joblib").exists()
         and (PROCESSED_DIR / "scored_customers.csv").exists()),
    reason="Build the project first (db_setup, model_tuning, explainability)")


@needs_model
def test_real_test_set_groups_add_up_to_the_overall_numbers():
    from src.data_preprocessing import load_customers, load_features
    from src.model_card import held_out_metrics
    from src.model_training import load_split
    from src.prediction import load_artifacts

    _, metadata = load_artifacts()
    features = load_features()
    scored = pd.read_csv(PROCESSED_DIR / "scored_customers.csv")
    data = features.merge(scored[["customer_id", "churn_probability"]], on="customer_id",
                          validate="one_to_one")
    _, X_test, _, _ = load_split()
    test_ids = set(load_customers().loc[X_test.index, "customer_id"])
    test = data[data["customer_id"].isin(test_ids)]
    overall = held_out_metrics(test["churn"], test["churn_probability"], metadata["threshold"])

    metrics = all_group_metrics(add_labels(test), metadata["threshold"])
    for name in GROUPINGS:
        part = metrics[metrics["grouping"] == name]
        assert part["customers"].sum() == overall["customers"] == 1409
        assert part["churners"].sum() == int(test["churn"].sum())
        caught = (part["recall"] * part["churners"]).sum()
        assert caught / part["churners"].sum() == pytest.approx(overall["recall"], abs=1e-9)
        assert np.isfinite(part["avg_predicted_pct"]).all()