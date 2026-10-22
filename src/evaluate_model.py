
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_churn_clean.csv"
REPORT_DIR = ROOT / "reports"


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
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, ax=axes[0], colorbar=False
    )
    RocCurveDisplay.from_predictions(y_test, prob, ax=axes[1])
    PrecisionRecallDisplay.from_predictions(y_test, prob, ax=axes[2])

    axes[0].set_title("Confusion Matrix")
    axes[1].set_title("ROC Curve")
    axes[2].set_title("Precision–Recall Curve")
    fig.tight_layout()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "evaluation_curves.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved evaluation charts: {output}")


if __name__ == "__main__":
    main()
