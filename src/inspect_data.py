
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset shape: {df.shape}")

    print("\nTotalCharges conversion check:")
    total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
    print(f"Original dtype: {df['TotalCharges'].dtype}")
    print(f"Values that cannot be converted: {total_charges.isna().sum()}")

    print("\nChurn distribution:")
    print(df["Churn"].value_counts())
    print(f"Churn rate: {df['Churn'].eq('Yes').mean():.2%}")

    print("\nCategorical column values:")
    for column in df.select_dtypes(include="str").columns:
        print(f"\n{column}: {df[column].nunique()} unique values")
        print(df[column].value_counts(dropna=False).head(10).to_dict())


if __name__ == "__main__":
    main()
