"""Validate and convert an uploaded customer CSV.

Expected format: the same columns as the IBM Telco Customer Churn dataset.
Nothing is guessed or silently fixed: bad values stop the upload with a clear message.
"""
import pandas as pd

from src.config import RAW_CSV_PATH

MAX_ROWS = 10_000
MAX_ERRORS_SHOWN = 8

YES_NO = {"Yes", "No"}
ADDON = {"Yes", "No", "No internet service"}

# Raw column -> allowed values (same values the cleaned database table accepts)
ALLOWED_VALUES = {
    "SeniorCitizen": {"0", "1"},
    "Partner": YES_NO,
    "Dependents": YES_NO,
    "PhoneService": YES_NO,
    "MultipleLines": {"Yes", "No", "No phone service"},
    "InternetService": {"DSL", "Fiber optic", "No"},
    "OnlineSecurity": ADDON,
    "OnlineBackup": ADDON,
    "DeviceProtection": ADDON,
    "TechSupport": ADDON,
    "StreamingTV": ADDON,
    "StreamingMovies": ADDON,
    "Contract": {"Month-to-month", "One year", "Two year"},
    "PaperlessBilling": YES_NO,
    "PaymentMethod": {"Electronic check", "Mailed check",
                      "Bank transfer (automatic)", "Credit card (automatic)"},
}
REQUIRED_COLUMNS = ["customerID", "tenure", "MonthlyCharges", *ALLOWED_VALUES]
# Optional: "Churn" (Yes/No) enables a performance check. "gender" and "TotalCharges"
# are not used by the model and are ignored.


class UploadError(ValueError):
    """The uploaded file cannot be used. The message says why."""


def read_upload(file) -> pd.DataFrame:
    """Read a CSV as text, so nothing is converted before validation."""
    try:
        return pd.read_csv(file, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError as error:
        raise UploadError("The file is empty.") from error
    except (pd.errors.ParserError, UnicodeDecodeError) as error:
        raise UploadError("The file could not be read as a CSV "
                          "(check the separator and encoding).") from error


def _count_text(count: int) -> str:
    return f"{count} row" + ("" if count == 1 else "s")


def validate_and_convert(raw: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned frame in the same format as the database 'customers' table.

    Raises UploadError listing what is wrong. Extra columns are ignored.
    """
    if raw is None or raw.empty:
        raise UploadError("The file has no rows.")
    if len(raw) > MAX_ROWS:
        raise UploadError(f"The file has {len(raw):,} rows. The limit is {MAX_ROWS:,}.")

    df = raw.copy()
    df.columns = [str(column).strip() for column in df.columns]
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise UploadError("Missing required columns: " + ", ".join(missing))

    df = df.astype(str).apply(lambda column: column.str.strip())
    df["SeniorCitizen"] = df["SeniorCitizen"].replace({"0.0": "0", "1.0": "1"})

    errors = []
    for column, allowed in ALLOWED_VALUES.items():
        bad = ~df[column].isin(allowed)
        if bad.any():
            examples = sorted(df.loc[bad, column].unique())[:3]
            errors.append(f"{column}: {_count_text(int(bad.sum()))} with unexpected values "
                          f"{examples}. Allowed: {sorted(allowed)}")

    empty_ids = df["customerID"] == ""
    if empty_ids.any():
        errors.append(f"customerID: {_count_text(int(empty_ids.sum()))} with an empty ID")
    duplicated = df["customerID"].duplicated() & ~empty_ids
    if duplicated.any():
        errors.append(f"customerID: {_count_text(int(duplicated.sum()))} repeat an earlier ID "
                      "(IDs must be unique)")

    tenure = pd.to_numeric(df["tenure"], errors="coerce")
    bad_tenure = ~tenure.between(0, 1200) | (tenure % 1 != 0)
    if bad_tenure.any():
        errors.append(f"tenure: {_count_text(int(bad_tenure.sum()))} with invalid values "
                      "(must be a whole number of months, 0 or more)")

    charges = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
    bad_charges = ~charges.between(0, 100_000)
    if bad_charges.any():
        errors.append(f"MonthlyCharges: {_count_text(int(bad_charges.sum()))} with invalid values "
                      "(must be an amount, 0 or more)")

    if "Churn" in df.columns:
        bad_churn = ~df["Churn"].isin(YES_NO)
        if bad_churn.any():
            errors.append(f"Churn: {_count_text(int(bad_churn.sum()))} with values other than Yes or No. "
                          "Fix them or remove the column.")

    if errors:
        shown = errors[:MAX_ERRORS_SHOWN]
        if len(errors) > MAX_ERRORS_SHOWN:
            shown.append(f"(+{len(errors) - MAX_ERRORS_SHOWN} more issues)")
        raise UploadError("\n".join(shown))

    clean = pd.DataFrame({
        "customer_id": df["customerID"],
        "senior_citizen": df["SeniorCitizen"].astype(int),
        "has_partner": (df["Partner"] == "Yes").astype(int),
        "has_dependents": (df["Dependents"] == "Yes").astype(int),
        "tenure_months": tenure.astype(int),
        "phone_service": (df["PhoneService"] == "Yes").astype(int),
        "multiple_lines": df["MultipleLines"],
        "internet_service": df["InternetService"],
        "online_security": df["OnlineSecurity"],
        "online_backup": df["OnlineBackup"],
        "device_protection": df["DeviceProtection"],
        "tech_support": df["TechSupport"],
        "streaming_tv": df["StreamingTV"],
        "streaming_movies": df["StreamingMovies"],
        "contract": df["Contract"],
        "paperless_billing": (df["PaperlessBilling"] == "Yes").astype(int),
        "payment_method": df["PaymentMethod"],
        "monthly_charges": charges.astype(float),
    })
    if "Churn" in df.columns:
        clean["churn"] = (df["Churn"] == "Yes").astype(int)
    return clean.reset_index(drop=True)


def sample_template(rows: int = 5) -> str:
    """A few rows of the original dataset (without Churn) as a CSV showing the format."""
    sample = pd.read_csv(RAW_CSV_PATH, dtype=str, keep_default_na=False).head(rows)
    return sample.drop(columns=["Churn"], errors="ignore").to_csv(index=False)
