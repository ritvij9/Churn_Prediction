
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "telco_churn_clean.csv"


class TestChurnData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = pd.read_csv(DATA_PATH)

    def test_expected_shape(self):
        self.assertEqual(self.df.shape, (7043, 20))

    def test_no_duplicate_customers(self):
        # customerID was intentionally excluded from the modelling dataset.
        self.assertEqual(len(self.df), 7043)

    def test_target_values(self):
        self.assertEqual(set(self.df["Churn"].unique()), {"Yes", "No"})

    def test_total_charges_are_numeric_and_complete(self):
        self.assertTrue(pd.api.types.is_numeric_dtype(self.df["TotalCharges"]))
        self.assertFalse(self.df["TotalCharges"].isna().any())

    def test_zero_tenure_charges(self):
        zero_tenure = self.df["tenure"] == 0
        self.assertTrue((self.df.loc[zero_tenure, "TotalCharges"] == 0).all())


if __name__ == "__main__":
    unittest.main()
