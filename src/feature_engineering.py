"""Feature engineering: turn the cleaned customers table into model-ready inputs.

Nothing here learns from the data (no means, no scaling), so it cannot leak.
Anything that learns (scaling, encoding) lives in the preprocessor, inside the pipeline.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "churn"
ID_COLUMN = "customer_id"

ADDON_COLUMNS = [
    "online_security", "online_backup", "device_protection",
    "tech_support", "streaming_tv", "streaming_movies",
]
NUMERIC_FEATURES = ["tenure_months"]
BINARY_FEATURES = [
    "senior_citizen", "has_partner", "has_dependents", "phone_service",
    "paperless_billing", "multiple_lines", *ADDON_COLUMNS, "autopay",
]
CATEGORICAL_FEATURES = ["contract", "internet_service", "payment_method"]
FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y). X has the model features plus 'monthly_charges' for business metrics.

    The preprocessor drops monthly_charges, so the model never sees it.
    """
    required = (set(FEATURE_COLUMNS) - {"autopay"}) | {TARGET, "monthly_charges"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input data: {sorted(missing)}")

    data = df.copy()

    for column in ADDON_COLUMNS + ["multiple_lines"]:
        data[column] = (data[column] == "Yes").astype(int)

    data["autopay"] = data["payment_method"].str.contains("automatic", case=False).astype(int)

    return data[FEATURE_COLUMNS + ["monthly_charges"]], data[TARGET]


def build_preprocessor() -> ColumnTransformer:
    """Scale tenure, pass binary flags through, one-hot encode categories.

    Any column not listed here (such as monthly_charges) is dropped, so the model never sees it.
    Output is a DataFrame with readable column names, which SHAP needs in Phase 6.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")