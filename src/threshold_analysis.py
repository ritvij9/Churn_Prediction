
import pandas as pd
from sklearn.metrics import precision_score, recall_score

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

    rows = []
    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        predictions = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "customers_to_contact": int(predictions.sum()),
            "actual_churners_in_test": int(y_test.sum()),
            "true_churners_identified": int(
                ((predictions == 1) & (y_test == 1)).sum()
            ),
            "precision": precision_score(
                y_test, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_test, predictions, zero_division=0
            ),
        })

    results = pd.DataFrame(rows)
    output = ROOT / "reports" / "threshold_analysis.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved to: {output}")


if __name__ == "__main__":
    main()
