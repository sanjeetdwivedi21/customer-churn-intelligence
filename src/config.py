"""Central configuration: paths and reproducibility settings."""
from pathlib import Path

# Project root = parent of the "src" folder, so paths work from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROJECT_ROOT / "db" / "churn.db"
SQL_DIR = PROJECT_ROOT / "sql"

RANDOM_SEED = 42  # used from Phase 4 onward for reproducible splits and models