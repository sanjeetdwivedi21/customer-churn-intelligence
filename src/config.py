"""Central configuration: paths, reproducibility and business assumptions."""
from pathlib import Path

# Project root = parent of the "src" folder, so paths work from any working directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_CSV_PATH = PROJECT_ROOT / "data" / "raw" / "telco_churn.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROJECT_ROOT / "db" / "churn.db"
SQL_DIR = PROJECT_ROOT / "sql"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_SEED = 42   # used for splits and models, so results are reproducible
TEST_SIZE = 0.2    # 20% of customers are held out and used once, in Phase 5 evaluation

# --- Model tuning ---
CV_FOLDS = 5
TUNING_ITERATIONS = 20        # random hyperparameter combinations tried per model
SELECTION_TOLERANCE = 0.02    # prefer a simpler model if within 2% of the best net benefit

# --- Business assumptions (NOT from the dataset: edit these to match a real business) ---
OFFER_COST = 50.0             # USD per retention offer (discount + agent time)
OFFER_SUCCESS_RATE = 0.25     # share of would-be churners who stay after an offer
VALUE_HORIZON_MONTHS = 12     # months of revenue counted for a customer we keep
RISK_HIGH_THRESHOLD = 0.50    # probability at or above this = High risk