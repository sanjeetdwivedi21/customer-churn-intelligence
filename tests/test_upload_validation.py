import io

import pandas as pd
import pytest

from src import upload_validation as uv
from src.config import RAW_CSV_PATH
from src.prediction import prepare_features
from src.upload_validation import UploadError, read_upload, validate_and_convert


@pytest.fixture
def raw_upload() -> pd.DataFrame:
    base = {"gender": "Male", "SeniorCitizen": "0", "Partner": "Yes", "Dependents": "No",
            "PhoneService": "Yes", "PaperlessBilling": "Yes", "TotalCharges": "100"}
    return pd.DataFrame([
        dict(base, customerID="U-1", tenure="3", MultipleLines="No", InternetService="Fiber optic",
             OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
             StreamingTV="Yes", StreamingMovies="No", Contract="Month-to-month",
             PaymentMethod="Electronic check", MonthlyCharges="89.5"),
        dict(base, customerID="U-2", tenure="40", PhoneService="No",
             MultipleLines="No phone service", InternetService="No",
             OnlineSecurity="No internet service", OnlineBackup="No internet service",
             DeviceProtection="No internet service", TechSupport="No internet service",
             StreamingTV="No internet service", StreamingMovies="No internet service",
             Contract="Two year", PaymentMethod="Credit card (automatic)", MonthlyCharges="20"),
    ])


def test_valid_upload_is_converted(raw_upload):
    clean = validate_and_convert(raw_upload)
    assert list(clean["customer_id"]) == ["U-1", "U-2"]
    assert list(clean["has_partner"]) == [1, 1]
    assert list(clean["phone_service"]) == [1, 0]
    assert list(clean["tenure_months"]) == [3, 40]
    assert clean["monthly_charges"].dtype == float
    assert clean.loc[0, "tech_support"] == "No"          # add-ons stay as text
    assert "churn" not in clean.columns


def test_converted_frame_works_with_feature_builder(raw_upload):
    X = prepare_features(validate_and_convert(raw_upload))
    assert len(X) == 2
    assert list(X["autopay"]) == [0, 1]


def test_missing_required_column(raw_upload):
    with pytest.raises(UploadError, match="Missing required columns: Contract"):
        validate_and_convert(raw_upload.drop(columns=["Contract"]))


def test_unexpected_value_names_the_column(raw_upload):
    raw_upload.loc[0, "Contract"] = "Weekly"
    with pytest.raises(UploadError, match="Contract"):
        validate_and_convert(raw_upload)


def test_duplicate_and_empty_ids(raw_upload):
    raw_upload.loc[1, "customerID"] = "U-1"
    with pytest.raises(UploadError, match="repeat an earlier ID"):
        validate_and_convert(raw_upload)
    raw_upload.loc[1, "customerID"] = ""
    with pytest.raises(UploadError, match="empty ID"):
        validate_and_convert(raw_upload)


@pytest.mark.parametrize("column, value", [
    ("tenure", "-1"), ("tenure", "2.5"), ("tenure", "abc"),
    ("MonthlyCharges", "abc"), ("MonthlyCharges", "-5"), ("MonthlyCharges", ""),
])
def test_bad_numbers_are_rejected(raw_upload, column, value):
    raw_upload.loc[0, column] = value
    with pytest.raises(UploadError, match=column):
        validate_and_convert(raw_upload)


def test_churn_column_is_optional_and_converted(raw_upload):
    raw_upload["Churn"] = ["Yes", "No"]
    assert list(validate_and_convert(raw_upload)["churn"]) == [1, 0]


def test_invalid_churn_is_rejected(raw_upload):
    raw_upload["Churn"] = ["Maybe", "No"]
    with pytest.raises(UploadError, match="Churn"):
        validate_and_convert(raw_upload)


def test_whitespace_extra_columns_and_float_senior_flag(raw_upload):
    raw_upload.loc[0, "Contract"] = "  Month-to-month "
    raw_upload["SeniorCitizen"] = ["1.0", "0"]
    raw_upload["SomethingElse"] = "x"
    clean = validate_and_convert(raw_upload)
    assert clean.loc[0, "contract"] == "Month-to-month"
    assert list(clean["senior_citizen"]) == [1, 0]


def test_all_problems_are_reported_together(raw_upload):
    raw_upload.loc[0, "Contract"] = "Weekly"
    raw_upload.loc[1, "tenure"] = "-3"
    with pytest.raises(UploadError) as info:
        validate_and_convert(raw_upload)
    assert "Contract" in str(info.value) and "tenure" in str(info.value)


def test_empty_and_oversized_files(raw_upload, monkeypatch):
    with pytest.raises(UploadError, match="no rows"):
        validate_and_convert(raw_upload.iloc[0:0])
    monkeypatch.setattr(uv, "MAX_ROWS", 1)
    with pytest.raises(UploadError, match="limit"):
        validate_and_convert(raw_upload)


def test_read_upload_keeps_text_and_rejects_empty_file():
    frame = read_upload(io.BytesIO(b"customerID,tenure\nA,007\n"))
    assert frame.loc[0, "tenure"] == "007"
    with pytest.raises(UploadError, match="empty"):
        read_upload(io.BytesIO(b""))


@pytest.mark.skipif(not RAW_CSV_PATH.exists(), reason="dataset not present")
def test_sample_template_is_a_valid_upload():
    clean = validate_and_convert(read_upload(io.StringIO(uv.sample_template())))
    assert len(clean) == 5
