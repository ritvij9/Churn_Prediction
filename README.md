
# Customer Churn Prediction

A reproducible machine-learning project that predicts customer churn using the IBM Telco Customer Churn sample dataset.

## Project Overview

The goal is to identify customers at higher risk of leaving a telecom service and explore how churn predictions could support customer-retention decisions.

## Dataset

- 7,043 customer records
- 20 columns after removing the customer identifier
- Target: `Churn` (`Yes` / `No`)
- Observed churn rate: 26.54%
- Source: IBM Telco Customer Churn sample dataset

The dataset represents a fictional telecom business. Results should not be assumed to generalise to other companies.

## Models

The training script compares:
- Majority-class baseline
- Logistic Regression
- Random Forest

Numerical preprocessing, categorical encoding and imputation are fitted inside scikit-learn pipelines using training data only. The data is split into training and test sets with stratification.

## Evaluation

The project reports accuracy, ROC-AUC, average precision, confusion matrices, precision, recall and F1-score.

Generated outputs:
- `reports/model_results.txt`
- `reports/evaluation_curves.png`

The holdout set is used for evaluation. Model selection, threshold selection and any further tuning should be done without repeatedly optimising against the test set.

## Data Preparation

The original CSV is retained locally. The prepared dataset excludes the customer ID and converts `TotalCharges` to numeric. Eleven blank charge values belong to zero-tenure customers and are set to zero, an assumption documented in the preparation script.

## Setup

Requires Python 3.14 or another compatible Python version.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

From the project root:

```powershell
python src/inspect_data.py
python src/check_data_quality.py
python src/prepare_data.py
python src/train_model.py
python src/evaluate_model.py
python -m unittest discover -s tests -v
```

## Limitations

- This is a public sample dataset, not live production data.
- Predictive associations do not establish causes of churn.
- A model score alone does not prove that retention campaigns will work.
- Business deployment would require cost-sensitive threshold selection, intervention testing, monitoring and validation on relevant business data.

## Repository Structure

```text
data/       Local datasets (not committed)
src/        Data inspection, preparation and modelling scripts
tests/      Automated data checks
reports/    Generated metrics and evaluation charts
```

## Retention Campaign Threshold Analysis

The logistic regression model was evaluated at different probability thresholds on the held-out test set.

| Threshold | Customers flagged | Churners identified | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.2 | 685 | 320 | 46.7% | 85.6% |
| 0.3 | 543 | 282 | 51.9% | 75.4% |
| 0.4 | 440 | 250 | 56.8% | 66.8% |
| 0.5 | 318 | 209 | 65.7% | 55.9% |
| 0.6 | 209 | 150 | 71.8% | 40.1% |
| 0.7 | 93 | 69 | 74.2% | 18.4% |

A lower threshold flags more customers and captures more churners, but also produces more false positives. A higher threshold reduces outreach volume while missing more churners.

These are retrospective results on a held-out test set. The appropriate operational threshold depends on retention-team capacity, intervention cost, customer value, and the effectiveness of retention offers. These results do not establish that contacting flagged customers will prevent churn.
