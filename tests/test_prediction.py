import pytest

from src.prediction import assign_risk, prepare_features

META = {"risk_thresholds": {"medium": 0.27, "high": 0.50}}


def test_assign_risk_boundaries():
    levels = assign_risk([0.10, 0.27, 0.49, 0.50, 0.90], META)
    assert list(levels) == ["Low", "Medium", "Medium", "High", "High"]


def test_prepare_features_works_without_churn_column(sample_customers):
    X = prepare_features(sample_customers.drop(columns=["churn"]))
    assert len(X) == 4
    assert "monthly_charges" in X.columns


def test_prepare_features_requires_customer_id(sample_customers):
    with pytest.raises(ValueError):
        prepare_features(sample_customers.drop(columns=["customer_id"]))