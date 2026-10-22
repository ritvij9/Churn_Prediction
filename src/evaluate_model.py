
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)

from src.modeling import (
    ROOT,
    build_logistic_regression,
    load_data,
    split_data,
)


def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    model = build_logistic_regression(X_train)
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = model.predict(X_test)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    ConfusionMatrixDisplay.from_predictions(
        y_test, predictions, ax=axes[0], colorbar=False
    )
    RocCurveDisplay.from_predictions(y_test, probabilities, ax=axes[1])
    PrecisionRecallDisplay.from_predictions(y_test, probabilities, ax=axes[2])

    axes[0].set_title("Confusion Matrix")
    axes[1].set_title("ROC Curve")
    axes[2].set_title("Precision–Recall Curve")
    fig.tight_layout()

    report_dir = ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "evaluation_curves.png"
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved evaluation charts: {output}")


if __name__ == "__main__":
    main()
