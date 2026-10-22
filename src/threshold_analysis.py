
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from src.modeling import (
    ROOT,
    build_logistic_regression,
    load_data,
    split_data_three_way,
)


THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]


def evaluate_thresholds(y_true, probabilities):
    rows = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "customers_to_contact": int(predictions.sum()),
            "actual_churners": int(y_true.sum()),
            "true_churners_identified": int(
                ((predictions == 1) & (y_true == 1)).sum()
            ),
            "precision": precision_score(
                y_true, predictions, zero_division=0
            ),
            "recall": recall_score(
                y_true, predictions, zero_division=0
            ),
        })

    return pd.DataFrame(rows)


def main():
    X, y = load_data()

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_data_three_way(X, y)

    # Fit the model using training data only.
    model = build_logistic_regression(X_train)
    model.fit(X_train, y_train)

    # Compare thresholds using validation data only.
    validation_probabilities = model.predict_proba(X_validation)[:, 1]
    validation_results = evaluate_thresholds(
        y_validation, validation_probabilities
    )

    # Select the threshold with the highest validation F1-score.
    validation_results["f1_score"] = (
        2
        * validation_results["precision"]
        * validation_results["recall"]
        / (
            validation_results["precision"]
            + validation_results["recall"]
        ).replace(0, float("nan"))
    )

    best_row = validation_results.loc[
        validation_results["f1_score"].idxmax()
    ]
    selected_threshold = float(best_row["threshold"])

    # Evaluate the selected threshold once on the held-out test data.
    test_probabilities = model.predict_proba(X_test)[:, 1]
    test_predictions = (
        test_probabilities >= selected_threshold
    ).astype(int)

    test_results = pd.DataFrame([{
        "selected_threshold": selected_threshold,
        "customers_to_contact": int(test_predictions.sum()),
        "actual_churners": int(y_test.sum()),
        "true_churners_identified": int(
            ((test_predictions == 1) & (y_test == 1)).sum()
        ),
        "precision": precision_score(
            y_test, test_predictions, zero_division=0
        ),
        "recall": recall_score(
            y_test, test_predictions, zero_division=0
        ),
    }])

    output_dir = ROOT / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_output = output_dir / "threshold_validation.csv"
    test_output = output_dir / "threshold_test_results.csv"

    validation_results.to_csv(validation_output, index=False)
    test_results.to_csv(test_output, index=False)

    print("\nThreshold comparison on validation data:")
    print(validation_results.to_string(index=False))

    print(f"\nSelected threshold by validation F1-score: {selected_threshold:.1f}")

    print("\nFinal evaluation on test data:")
    print(test_results.to_string(index=False))

    print(f"\nValidation results saved to: {validation_output}")
    print(f"Test results saved to: {test_output}")


if __name__ == "__main__":
    main()
