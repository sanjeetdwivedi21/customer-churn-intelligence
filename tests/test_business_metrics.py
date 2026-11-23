import numpy as np

from src.business_metrics import best_threshold, customer_value, evaluate_flags
from src.config import OFFER_COST, OFFER_SUCCESS_RATE, VALUE_HORIZON_MONTHS


def test_customer_value():
    assert list(customer_value([10, 20])) == [10 * VALUE_HORIZON_MONTHS, 20 * VALUE_HORIZON_MONTHS]


def test_evaluate_flags_hand_calculation():
    # Flag customers 0 and 1. Only customer 0 is a real churner (monthly charge 100).
    result = evaluate_flags([1, 0, 1, 0], [1, 1, 0, 0], [100, 50, 80, 20])
    expected = OFFER_SUCCESS_RATE * 100 * VALUE_HORIZON_MONTHS - OFFER_COST * 2
    assert result["flagged"] == 2
    assert result["recall"] == 0.5
    assert result["precision"] == 0.5
    assert abs(result["net_benefit"] - expected) < 1e-6


def test_contacting_nobody_costs_nothing():
    result = evaluate_flags([1, 0, 1], [0, 0, 0], [100, 100, 100])
    assert result["net_benefit"] == 0
    assert result["recall"] == 0


def test_best_threshold_separates_a_perfect_model():
    y = [1, 1, 0, 0]
    proba = [0.9, 0.8, 0.1, 0.05]
    threshold, sweep = best_threshold(y, proba, [100, 100, 100, 100])
    row = sweep[sweep["threshold"] == threshold].iloc[0]
    assert 0.1 < threshold <= 0.8
    assert row["recall"] == 1 and row["precision"] == 1
    assert np.isfinite(sweep["net_benefit"]).all()