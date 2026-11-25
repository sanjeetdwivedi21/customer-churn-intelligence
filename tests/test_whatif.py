import pytest

from src.config import DB_PATH, MODELS_DIR
from src.whatif import single_changes


def labels(row) -> list[str]:
    return [label for label, _ in single_changes(row)]


def test_changes_for_a_high_risk_fiber_customer(sample_customers):
    row = sample_customers.iloc[0]       # month-to-month, fiber, electronic check, no add-ons
    found = labels(row)
    assert "Contract: One year" in found and "Contract: Two year" in found
    assert "Contract: Month-to-month" not in found          # that is the current contract
    assert "Payment: Electronic check" not in found
    assert sum(label.startswith("Payment:") for label in found) == 3
    assert {"Add tech support", "Add online security",
            "Add online backup", "Add device protection"} <= set(found)
    assert "Switch to paper billing" in found                # currently paperless


def test_no_addon_changes_without_internet(sample_customers):
    found = labels(sample_customers.iloc[2])                 # no internet service
    assert not any(label.startswith("Add ") for label in found)


def test_no_addon_changes_when_customer_already_has_them(sample_customers):
    found = labels(sample_customers.iloc[1])                 # all four add-ons already "Yes"
    assert not any(label.startswith("Add ") for label in found)


def test_single_changes_does_not_modify_the_row(sample_customers):
    row = sample_customers.iloc[0]
    before = row.copy()
    single_changes(row)
    assert row.equals(before)


needs_model = pytest.mark.skipif(
    not (DB_PATH.exists() and (MODELS_DIR / "final_model.joblib").exists()),
    reason="Build the project first (db_setup, model_tuning)")


@needs_model
def test_scoring_matches_saved_model_and_contract_direction():
    from src.data_preprocessing import load_customers
    from src.prediction import load_artifacts, score_customers
    from src.whatif import score_options

    model, metadata = load_artifacts()
    customers = load_customers()
    row = customers[customers["contract"] == "Month-to-month"].iloc[0]

    result = score_options(row, [("Two year", {"contract": "Two year"})], model, metadata)
    direct = score_customers(customers.loc[[row.name]], model, metadata)

    # Row 0 is the current setup and must equal normal scoring
    assert result.loc[0, "churn_probability"] == direct["churn_probability"].iloc[0]
    assert result.loc[0, "change_pct_points"] == 0
    # A two-year contract must not look riskier than month-to-month for the same customer
    assert result.loc[1, "churn_probability"] < result.loc[0, "churn_probability"]
    assert result.loc[1, "change_pct_points"] < 0