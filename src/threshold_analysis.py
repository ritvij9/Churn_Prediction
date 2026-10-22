
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_churn_clean.csv"
OUTPUT_PATH = ROOT / "reports" / "threshold_analysis.csv"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns="Churn")
    y = df["Churn"].map({"No": 0, "Yes": 1})

    categorical = X.select_dtypes(include="str").columns.tolist()
    numerical = X.select_dtypes(exclude="str").columns.tolist()

    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numerical),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=2000)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]

    rows = []
    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        predicted = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "customers_to_contact": int(predicted.sum()),
            "actual_churners_in_test": int(y_test.sum()),
            "true_churners_identified": int(((predicted == 1) & (y_test == 1)).sum()),
            "precision": precision_score(y_test, predicted, zero_division=0),
            "recall": recall_score(y_test, predicted, zero_division=0),
        })

    results = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)
    print(results.to_string(index=False))
    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
