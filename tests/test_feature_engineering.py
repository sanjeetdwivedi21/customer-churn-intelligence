import pytest

from src.feature_engineering import FEATURE_COLUMNS, build_features, build_preprocessor


def test_build_features_returns_expected_columns(sample_customers):
    X, y = build_features(sample_customers)
    assert list(X.columns) == FEATURE_COLUMNS + ["monthly_charges"]
    assert list(y) == [1, 0, 0, 0]


def test_no_identifier_target_or_collinear_columns_in_features(sample_customers):
    X, _ = build_features(sample_customers)
    for forbidden in ("customer_id", "churn", "total_charges"):
        assert forbidden not in X.columns
    # monthly_charges rides along for business metrics only, never for the model
    assert "monthly_charges" not in FEATURE_COLUMNS


def test_no_internet_addons_become_zero(sample_customers):
    X, _ = build_features(sample_customers)
    row = X.iloc[2]
    for column in ("online_security", "online_backup", "device_protection",
                   "tech_support", "streaming_tv", "streaming_movies", "multiple_lines"):
        assert row[column] == 0


def test_yes_addons_become_one(sample_customers):
    X, _ = build_features(sample_customers)
    assert X.iloc[1]["tech_support"] == 1
    assert X.iloc[0]["streaming_tv"] == 1


def test_autopay_flag(sample_customers):
    X, _ = build_features(sample_customers)
    assert list(X["autopay"]) == [0, 1, 0, 1]


def test_missing_column_raises(sample_customers):
    with pytest.raises(ValueError):
        build_features(sample_customers.drop(columns=["contract"]))


def test_input_is_not_modified(sample_customers):
    before = sample_customers.copy(deep=True)
    build_features(sample_customers)
    assert sample_customers.equals(before)


def test_preprocessor_output_is_clean(sample_customers):
    X, _ = build_features(sample_customers)
    out = build_preprocessor().fit_transform(X)
    assert "monthly_charges" not in out.columns      # dropped by the preprocessor
    assert not out.isna().any().any()
    assert any(c.startswith("contract_") for c in out.columns)