"""Phase 1 sanity check: confirm the raw dataset loads and looks as expected."""
from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/telco_churn.csv")

def main() -> None:
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {RAW_PATH.resolve()}")

    df = pd.read_csv(RAW_PATH)
    print("Shape:", df.shape)
    print("\nColumns:", list(df.columns))
    print("\nChurn counts:\n", df["Churn"].value_counts())
    print("Churn rate: {:.1%}".format((df["Churn"] == "Yes").mean()))

    # Known quirk: TotalCharges is text because of blank strings
    print("\nTotalCharges dtype:", df["TotalCharges"].dtype)
    blanks = df["TotalCharges"].astype(str).str.strip().eq("")
    print("Blank TotalCharges rows:", blanks.sum())
    print("Their tenure values:", df.loc[blanks, "tenure"].unique())

    print("\nDuplicate customerIDs:", df["customerID"].duplicated().sum())

if __name__ == "__main__":
    main()