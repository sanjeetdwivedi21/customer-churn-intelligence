import json

import numpy as np
import pandas as pd
import pytest

from src.config import MODELS_DIR
from src.model_comparison import (COMPLEXITY, OOF_PATH, agreement_tables, annotate,
                                  calibration_table, choose_model, metrics_table, model_names,
                                  net_benefit_table, pr_table, roc_table, selection_table)


@pytest.fixture
def oof() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 500
    churn = rng.integers(0, 2, n)
    signal = churn * 0.35 + rng.random(n) * 0.65
    return pd.DataFrame({
        "row_index": np.arange(n), "churn": churn, "monthly_charges": rng.uniform(20, 110, n),
        "Logistic Regression": signal,
        "XGBoost": np.clip(signal + rng.normal(0, 0.03, n), 0, 1),
    })


def test_model_names_skip_helper_columns(oof):
    assert model_names(oof) == ["Logistic Regression", "XGBoost"]


def test_roc_table_starts_at_origin_ends_at_one_and_never_decreases(oof):
    table = roc_table(oof)
    for _, part in table.groupby("model"):
        assert (part["fpr"].diff().dropna() >= 0).all() and (part["tpr"].diff().dropna() >= 0).all()
        assert part.iloc[0][["fpr", "tpr"]].tolist() == [0, 0]
        assert part.iloc[-1][["fpr", "tpr"]].tolist() == [1, 1]


def test_metrics_pr_and_calibration_tables(oof):
    metrics = metrics_table(oof)
    assert metrics["roc_auc"].between(0.5, 1).all()
    assert set(pr_table(oof)["model"]) == {"Logistic Regression", "XGBoost"}
    cal = calibration_table(oof, bins=10)
    assert cal[["mean_predicted", "actual_rate"]].apply(lambda s: s.between(0, 1)).all().all()
    assert cal.groupby("model").size().le(10).all()


def test_net_benefit_table_has_a_curve_per_model(oof):
    table = net_benefit_table(oof, offer_cost=50, success_rate=0.25, horizon_months=12)
    assert set(table["model"]) == {"Logistic Regression", "XGBoost"}
    assert np.isfinite(table["net_benefit"]).all()


def test_choose_model_prefers_the_simplest_within_tolerance():
    table = pd.DataFrame({"model": ["Logistic Regression", "XGBoost", "Random Forest"],
                          "net_benefit": [100.0, 101.0, 90.0]})
    assert choose_model(table, 0.02) == "Logistic Regression"      # 100 is within 2% of 101
    table.loc[1, "net_benefit"] = 120.0
    assert choose_model(table, 0.02) == "XGBoost"                  # LR is now too far behind
    assert list(annotate(table, 0.02)["within_tolerance"]) == [False, True, False]


def test_choose_model_returns_none_when_nothing_beats_doing_nothing():
    table = pd.DataFrame({"model": ["Logistic Regression", "XGBoost"], "net_benefit": [-5.0, -1.0]})
    assert choose_model(table, 0.02) is None


def test_selection_table_is_sorted_and_complete(oof):
    table = selection_table(oof, offer_cost=50, success_rate=0.25, horizon_months=12)
    assert table["net_benefit"].is_monotonic_decreasing
    assert set(table.columns) >= {"model", "best_threshold", "net_benefit", "recall", "precision"}


def test_agreement_tables(oof):
    spearman, overlap = agreement_tables(oof, top_share=0.10)
    assert np.allclose(np.diag(spearman), 1) and np.allclose(np.diag(overlap), 1)
    assert spearman.loc["Logistic Regression", "XGBoost"] > 0.9    # almost the same ranking
    assert overlap.to_numpy().min() >= 0 and overlap.to_numpy().max() <= 1


needs_oof = pytest.mark.skipif(
    not (OOF_PATH.exists() and (MODELS_DIR / "model_metadata.json").exists()
         and (MODELS_DIR / "tuning_summary.csv").exists()),
    reason="Run python -m src.model_tuning first")


@needs_oof
def test_comparison_reproduces_the_saved_tuning_results():
    """With default assumptions the tab must agree with what model_tuning decided."""
    from src.model_comparison import load_oof
    metadata = json.loads((MODELS_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(MODELS_DIR / "tuning_summary.csv").set_index("model")
    a = metadata["assumptions"]
    table = selection_table(load_oof(), offer_cost=a["offer_cost"], success_rate=a["offer_success_rate"],
                            horizon_months=a["value_horizon_months"]).set_index("model")

    assert set(table.index) == set(summary.index)
    for name in table.index:
        assert table.loc[name, "net_benefit"] == pytest.approx(summary.loc[name, "net_benefit"], abs=1)
        assert table.loc[name, "best_threshold"] == pytest.approx(summary.loc[name, "threshold"])
    assert choose_model(table.reset_index(), metadata["selection_tolerance"]) == metadata["model_name"]


def test_complexity_table_matches_model_tuning():
    pytest.importorskip("scipy")
    from src.model_tuning import COMPLEXITY as TUNING_COMPLEXITY
    assert COMPLEXITY == TUNING_COMPLEXITY