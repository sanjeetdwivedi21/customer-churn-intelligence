import numpy as np
import pandas as pd
import pytest

from src.business_metrics import best_threshold, customer_value, evaluate_flags
from src.campaign import (break_even_probability, expected_value_flags, rule_flags, simulate)
from src.config import (DB_PATH, MODELS_DIR, OFFER_COST, OFFER_SUCCESS_RATE,
                        VALUE_HORIZON_MONTHS)


@pytest.fixture
def pool() -> pd.DataFrame:
    return pd.DataFrame({
        "churn": [1, 0, 1, 0, 0],
        "churn_probability": [0.9, 0.8, 0.4, 0.1, 0.05],
        "monthly_charges": [100, 50, 80, 20, 30],
        "contract": ["Month-to-month", "Month-to-month", "One year", "Two year", "Month-to-month"],
        "tenure_months": [3, 30, 5, 40, 2],
        "payment_method": ["Electronic check", "Electronic check", "Mailed check",
                           "Mailed check", "Electronic check"],
    })


def test_custom_assumptions_hand_calculation():
    # Flag customers 0 and 1. Only customer 0 churned (charge 100).
    result = evaluate_flags([1, 0, 1, 0, 0], [1, 1, 0, 0, 0], [100, 50, 80, 20, 30],
                            offer_cost=10, success_rate=0.5, horizon_months=6)
    assert result["gross_benefit"] == 0.5 * 100 * 6
    assert result["cost"] == 20
    assert result["net_benefit"] == 280
    assert result["expected_saved"] == 0.5


def test_defaults_match_the_config_values():
    args = ([1, 0, 1], [1, 1, 0], [100, 50, 80])
    explicit = evaluate_flags(*args, offer_cost=OFFER_COST, success_rate=OFFER_SUCCESS_RATE,
                              horizon_months=VALUE_HORIZON_MONTHS)
    assert evaluate_flags(*args) == explicit
    assert list(customer_value([10])) == [10 * VALUE_HORIZON_MONTHS]


def test_higher_offer_cost_never_increases_net_benefit():
    args = ([1, 0, 1, 0], [1, 1, 1, 0], [100, 50, 80, 20])
    cheap = evaluate_flags(*args, offer_cost=10)["net_benefit"]
    dear = evaluate_flags(*args, offer_cost=100)["net_benefit"]
    assert dear < cheap


def test_rule_flags(pool):
    assert list(rule_flags(pool)) == [True, False, False, False, True]


def test_expected_value_rule():
    # success 0.5, horizon 10, cost 50 -> contact when probability x charge >= 10
    flags = expected_value_flags([0.9, 0.8, 0.4, 0.1, 0.05], [100, 50, 80, 20, 30],
                                 offer_cost=50, success_rate=0.5, horizon_months=10)
    assert list(flags) == [True, True, True, False, False]


def test_break_even_probability():
    assert break_even_probability(100, 50, 0.5, 10) == pytest.approx(0.1)
    assert break_even_probability(10, 500, 0.25, 12) > 1     # never worth it
    assert break_even_probability(100, 50, 0.0, 12) == float("inf")


def test_simulate_returns_all_strategies(pool):
    table, sweep, best = simulate(pool, threshold=0.5, offer_cost=10, success_rate=0.5,
                                  horizon_months=6)
    assert len(table) == 6
    nobody = table[table["strategy"] == "Contact nobody"].iloc[0]
    everyone = table[table["strategy"] == "Contact everyone"].iloc[0]
    assert nobody["contacted"] == 0 and nobody["net_benefit"] == 0
    assert everyone["contacted"] == 5
    assert everyone["net_benefit"] == round(0.5 * (100 + 80) * 6 - 10 * 5)
    assert 0.05 <= best <= 0.95
    assert np.isfinite(sweep["net_benefit"]).all()


def test_when_offers_are_too_expensive_nobody_is_best():
    y, proba, charges = [1, 0, 1], [0.9, 0.5, 0.6], [100, 50, 80]
    _, sweep = best_threshold(y, proba, charges, offer_cost=100_000)
    assert sweep["net_benefit"].max() <= 0


needs_model = pytest.mark.skipif(
    not (DB_PATH.exists() and (MODELS_DIR / "final_model.joblib").exists()),
    reason="Build the project first (db_setup, model_tuning)")


@needs_model
def test_reproduces_the_phase_5_test_set_results():
    """With default assumptions the simulator must agree with reports/model_evaluation.md."""
    from src.data_preprocessing import load_customers
    from src.model_training import load_split
    from src.prediction import load_artifacts, score_customers

    model, metadata = load_artifacts()
    customers = load_customers()
    _, X_test, _, _ = load_split()
    scored = score_customers(customers, model, metadata)
    test = customers.loc[X_test.index].assign(
        churn_probability=scored.loc[X_test.index, "churn_probability"])

    table, _, _ = simulate(test, metadata["threshold"], OFFER_COST, OFFER_SUCCESS_RATE,
                           VALUE_HORIZON_MONTHS)
    row = lambda label: table[table["strategy"].str.startswith(label)].iloc[0]  # noqa: E731

    assert len(test) == 1409
    assert row("Contact everyone")["net_benefit"] == pytest.approx(11195, abs=1)
    assert row("SQL rule baseline")["contacted"] == 191
    assert row("SQL rule baseline")["net_benefit"] == pytest.approx(14818, abs=1)
    model_row = row("Model at selected threshold")
    assert model_row["contacted"] == pytest.approx(566, abs=3)
    assert model_row["net_benefit"] == pytest.approx(38221, rel=0.01)