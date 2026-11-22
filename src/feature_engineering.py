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
NUMERIC_FEATURES = ["tenure_months", "monthly_charges", "total_charges"]
BINARY_FEATURES = [
    "senior_citizen", "has_partner", "has_dependents", "phone_service",
    "paperless_billing", "multiple_lines", *ADDON_COLUMNS, "autopay",
]
CATEGORICAL_FEATURES = ["contract", "internet_service", "payment_method"]
FEATURE_COLUMNS = NUMERIC_FEATURES + BINARY_FEATURES + CATEGORICAL_FEATURES


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) from the cleaned customers table. The index of df is preserved."""
    required = (set(FEATURE_COLUMNS) - {"autopay"}) | {TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in input data: {sorted(missing)}")

    data = df.copy()

    # Blank total_charges only occurs for tenure 0 (nothing billed yet), so 0 is correct.
    new_customer = data["tenure_months"] == 0
    data.loc[new_customer & data["total_charges"].isna(), "total_charges"] = 0.0
    if data["total_charges"].isna().any():
        raise ValueError("total_charges has missing values for customers with tenure > 0")

    # Yes / No / "No internet service" -> 1 / 0 (the no-service case is already
    # described by the internet_service column)
    for column in ADDON_COLUMNS + ["multiple_lines"]:
        data[column] = (data[column] == "Yes").astype(int)

    data["autopay"] = data["payment_method"].str.contains("automatic", case=False).astype(int)

    return data[FEATURE_COLUMNS], data[TARGET]


def build_preprocessor() -> ColumnTransformer:
    """Scale numbers, pass binary flags through, one-hot encode categories.

    Output is a DataFrame with readable column names, which SHAP needs in Phase 6.
    """
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")