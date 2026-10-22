
import unittest
from pathlib import Path

import pandas as pd

from src.modeling import (
    build_logistic_regression,
    load_data,
    split_data,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn_clean.csv"


class TestChurnData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA_PATH)

    def test_expected_shape(self):
        self.assertEqual(self.df.shape, (7043, 20))

    def test_dataset_has_no_missing_values(self):
        self.assertFalse(self.df.isna().any().any())

    def test_target_values(self):
        self.assertEqual(set(self.df["Churn"].unique()), {"Yes", "No"})

    def test_total_charges_are_numeric_and_complete(self):
        self.assertTrue(
            pd.api.types.is_numeric_dtype(self.df["TotalCharges"])
        )
        self.assertFalse(self.df["TotalCharges"].isna().any())

    def test_zero_tenure_charges(self):
        zero_tenure = self.df["tenure"] == 0
        self.assertTrue(
            (self.df.loc[zero_tenure, "TotalCharges"] == 0).all()
        )

    def test_split_is_reproducible(self):
        X, y = load_data()

        X_train, X_test, y_train, y_test = split_data(X, y)
        X_train_2, X_test_2, y_train_2, y_test_2 = split_data(X, y)

        self.assertEqual(len(X_train), 5634)
        self.assertEqual(len(X_test), 1409)
        self.assertEqual(X_train.index.tolist(), X_train_2.index.tolist())
        self.assertEqual(X_test.index.tolist(), X_test_2.index.tolist())

        # Check that churn proportions are approximately preserved.
        self.assertAlmostEqual(y_train.mean(), y.mean(), delta=0.01)
        self.assertAlmostEqual(y_test.mean(), y.mean(), delta=0.01)

    def test_model_handles_unseen_categories(self):
        X, y = load_data()
        X_train, X_test, y_train, y_test = split_data(X, y)

        model = build_logistic_regression(X_train)
        model.fit(X_train, y_train)

        sample = X_test.iloc[[0]].copy()
        categorical = sample.select_dtypes(include="str").columns

        if len(categorical):
            sample[categorical[0]] = "UNSEEN_CATEGORY"

        probabilities = model.predict_proba(sample)

        self.assertEqual(probabilities.shape, (1, 2))
        self.assertTrue(
            ((probabilities >= 0) & (probabilities <= 1)).all()
        )
        self.assertAlmostEqual(probabilities.sum(), 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
