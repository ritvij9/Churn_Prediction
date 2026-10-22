
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from src.modeling import (
    DATA_PATH,
    build_logistic_regression,
    load_data,
    split_data,
)


ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = ROOT / "reports" / "model_results.txt"


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": build_logistic_regression(X_train),
        "Random forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
        ),
    }

    # Random Forest needs the same preprocessing as Logistic Regression.
    from sklearn.pipeline import Pipeline
    from src.modeling import build_preprocessor

    models["Random forest"] = Pipeline([
        ("preprocessor", build_preprocessor(X_train)),
        ("model", models["Random forest"]),
    ])

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        prob = model.predict_proba(X_test)[:, 1]

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
