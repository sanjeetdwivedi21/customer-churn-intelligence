"""Phase 2: build and validate the SQLite database from the raw CSV.

Run from the project root:  python -m src.db_setup
Safe to re-run: it rebuilds everything from scratch.
"""
import csv
import logging
import sqlite3
import sys
from pathlib import Path

from src.config import DB_PATH, RAW_CSV_PATH, SQL_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Columns we expect in the raw file. Checking them before building SQL strings
# protects against loading the wrong file.
EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "Churn",
]


def run_sql_file(conn: sqlite3.Connection, path: Path) -> None:
    """Execute every statement in a .sql file."""
    conn.executescript(path.read_text(encoding="utf-8"))
    logger.info("Executed %s", path.name)


def load_raw_csv(conn: sqlite3.Connection, csv_path: Path) -> int:
    """Insert CSV rows into raw_customers exactly as text. Returns the row count."""
    # utf-8-sig silently removes a byte-order mark if the file has one
    with open(csv_path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames != EXPECTED_COLUMNS:
            raise ValueError(
                f"Unexpected columns in {csv_path.name}.\n"
                f"Expected: {EXPECTED_COLUMNS}\nFound:    {reader.fieldnames}"
            )
        rows = [tuple(row[col] for col in EXPECTED_COLUMNS) for row in reader]

    placeholders = ", ".join("?" for _ in EXPECTED_COLUMNS)
    conn.executemany(
        f"INSERT INTO raw_customers ({', '.join(EXPECTED_COLUMNS)}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    logger.info("Loaded %d rows into raw_customers", len(rows))
    return len(rows)


def scalar(conn: sqlite3.Connection, query: str):
    """Run a query that returns one value."""
    return conn.execute(query).fetchone()[0]


def validate(conn: sqlite3.Connection, raw_rows: int) -> None:
    """Fail loudly if the clean table does not match the raw data."""
    clean_rows = scalar(conn, "SELECT COUNT(*) FROM customers")
    if clean_rows != raw_rows:
        raise RuntimeError(f"Row mismatch: raw={raw_rows}, clean={clean_rows}")

    churned = scalar(conn, "SELECT SUM(churn) FROM customers")
    null_totals = scalar(conn, "SELECT COUNT(*) FROM customers WHERE total_charges IS NULL")
    tenure_min, tenure_max = conn.execute(
        "SELECT MIN(tenure_months), MAX(tenure_months) FROM customers"
    ).fetchone()
    charge_min, charge_max = conn.execute(
        "SELECT MIN(monthly_charges), MAX(monthly_charges) FROM customers"
    ).fetchone()
    null_with_tenure = scalar(
        conn, "SELECT COUNT(*) FROM customers WHERE total_charges IS NULL AND tenure_months > 0"
    )

    logger.info("customers rows: %d | churned: %d | churn rate: %.2f%%",
                clean_rows, churned, 100 * churned / clean_rows)
    logger.info("NULL total_charges: %d (rows with tenure > 0 among them: %d)",
                null_totals, null_with_tenure)
    logger.info("tenure range: %d to %d months", tenure_min, tenure_max)
    logger.info("monthly_charges range: %.2f to %.2f", charge_min, charge_max)

    if null_with_tenure != 0:
        raise RuntimeError("Found NULL total_charges for customers with tenure > 0")


def main() -> int:
    if not RAW_CSV_PATH.exists():
        logger.error("Dataset not found at %s", RAW_CSV_PATH)
        return 1

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        run_sql_file(conn, SQL_DIR / "01_create_tables.sql")
        raw_rows = load_raw_csv(conn, RAW_CSV_PATH)
        run_sql_file(conn, SQL_DIR / "02_data_cleaning.sql")
        validate(conn, raw_rows)
        logger.info("Build complete: %s", DB_PATH)
        return 0
    except (sqlite3.Error, ValueError, RuntimeError) as error:
        logger.error("Build failed: %s", error)
        return 1
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())