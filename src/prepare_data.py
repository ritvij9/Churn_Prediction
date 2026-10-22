
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "telco_churn_clean.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    # Convert TotalCharges to numeric.
    # Blank values occur for customers with zero tenure.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"].replace(r"^\s*$", pd.NA, regex=True),
        errors="raise",
    )

    zero_tenure = df["tenure"].eq(0)
    df.loc[zero_tenure & df["TotalCharges"].isna(), "TotalCharges"] = 0

    if df["TotalCharges"].isna().any():
        raise ValueError("Unexpected missing values in TotalCharges.")

    if (df["TotalCharges"] < 0).any():
        raise ValueError("Negative TotalCharges values found.")

    df = df.drop(columns=["customerID"])

    df.to_csv(OUTPUT_PATH, index=False)

    print(f"Cleaned dataset saved to: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print(f"Missing TotalCharges: {df['TotalCharges'].isna().sum()}")
    print(f"Zero-tenure customers: {df['tenure'].eq(0).sum()}")
    print(f"Churn rate: {df['Churn'].eq('Yes').mean():.2%}")


if __name__ == "__main__":
    main()
