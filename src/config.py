"""Central configuration: paths and reproducibility settings."""
from pathlib import Path

# Project root = parent of the "src" folder, so paths work from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROJECT_ROOT / "db" / "churn.db"
SQL_DIR = PROJECT_ROOT / "sql"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RANDOM_SEED = 42   # used for splits and models, so results are reproducible
TEST_SIZE = 0.2    # 20% of customers are held out and untouched until Phase 5