
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"


def main():
    df = pd.read_csv(DATA_PATH)

    total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
    invalid = total_charges.isna()

    print("Records with blank or invalid TotalCharges:")
    print(df.loc[invalid, ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]])

    print("\nTenure summary for these records:")
    print(df.loc[invalid, "tenure"].describe())

    print("\nAre all invalid records zero-tenure customers?")
    print(bool((df.loc[invalid, "tenure"] == 0).all()))


if __name__ == "__main__":
    main()
