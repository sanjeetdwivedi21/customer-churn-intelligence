"""Load the cleaned churn data from SQLite into pandas.

Check it from the project root:  python -m src.data_preprocessing
"""
import sqlite3
from contextlib import closing

import pandas as pd

from src.config import DB_PATH

# Fixed category order so charts show bands in a sensible order, not alphabetically
TENURE_BAND_ORDER = ["0-12 months", "13-24 months", "25-48 months", "49+ months"]
CHARGE_BAND_ORDER = ["<$35", "$35-70", "$70-90", "$90+"]


def _read_query(query: str) -> pd.DataFrame:
    """Run a SQL query against the churn database and return a DataFrame."""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Build it first: python -m src.db_setup"
        )
    with closing(sqlite3.connect(DB_PATH)) as conn:
        return pd.read_sql_query(query, conn)


def load_customers() -> pd.DataFrame:
    """Return the cleaned customers table (21 columns)."""
    return _read_query("SELECT * FROM customers")


def load_features() -> pd.DataFrame:
    """Return customers plus derived columns (tenure_band, charge_band, service counts)."""
    df = _read_query("SELECT * FROM vw_customer_features")

    if df.empty:
        raise ValueError("vw_customer_features returned no rows")
    if not set(df["churn"].unique()) <= {0, 1}:
        raise ValueError("churn column must contain only 0 and 1")

    df["tenure_band"] = pd.Categorical(df["tenure_band"], TENURE_BAND_ORDER, ordered=True)
    df["charge_band"] = pd.Categorical(df["charge_band"], CHARGE_BAND_ORDER, ordered=True)
    return df


if __name__ == "__main__":
    data = load_features()
    print("Shape:", data.shape)
    print("Churn rate: {:.2%}".format(data["churn"].mean()))
    print(data.dtypes)