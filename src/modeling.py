
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "telco_churn_clean.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns="Churn")
    y = df["Churn"].map({"No": 0, "Yes": 1})

    if y.isna().any():
        raise ValueError("Unexpected churn labels.")

    return X, y


def split_data(X, y):
    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )


def build_preprocessor(X):
    categorical = X.select_dtypes(include="str").columns.tolist()
    numerical = X.select_dtypes(exclude="str").columns.tolist()

    return ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numerical),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def build_logistic_regression(X):
    return Pipeline([
        ("preprocessor", build_preprocessor(X)),
        ("model", LogisticRegression(max_iter=2000)),
    ])
