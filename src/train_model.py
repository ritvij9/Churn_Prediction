
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_churn_clean.csv"
REPORT_PATH = ROOT / "reports" / "model_results.txt"


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns="Churn")
    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected target labels.")

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

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": Pipeline([
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=2000)),
        ]),
        "Random forest": Pipeline([
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=300, random_state=42, class_weight="balanced"
            )),
        ]),
    }

    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else pred.astype(float)
        )

        results.append(
            f"\n{'=' * 55}\n{name}\n"
            f"Accuracy: {accuracy_score(y_test, pred):.3f}\n"
            f"ROC-AUC: {roc_auc_score(y_test, prob):.3f}\n"
            f"PR-AUC (average precision): "
            f"{average_precision_score(y_test, prob):.3f}\n"
            f"Confusion matrix:\n{confusion_matrix(y_test, pred)}\n"
            f"Classification report:\n"
            f"{classification_report(y_test, pred, zero_division=0)}"
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(results), encoding="utf-8")
    print("\n".join(results))
    print(f"\nResults saved to: {REPORT_PATH}")


if __name__ == "__main__":
    main()
