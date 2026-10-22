
# Customer Churn Prediction

A reproducible machine-learning project that predicts customer churn using the IBM Telco Customer Churn sample dataset. The project covers data quality checks, preprocessing, model comparison, evaluation, and retention-campaign threshold analysis.

## Project Overview

Customer churn occurs when customers stop using a company's service. Identifying customers at higher risk of leaving can help businesses investigate retention opportunities and allocate resources more effectively.

This project builds and evaluates classification models to predict whether a telecom customer will churn.

The objective is to demonstrate an end-to-end data science workflow and translate model predictions into business-relevant insights.

## Business Problem

A telecom company wants to identify customers who are more likely to leave its service.

The analysis addresses four questions:

1. How accurately can customer churn be predicted?
2. Which model performs best across relevant classification metrics?
3. How does changing the prediction threshold affect the number of customers flagged?
4. What trade-offs exist between identifying churners and limiting retention outreach?

The analysis explores potential decision support. It does not establish that contacting predicted churners will prevent them from leaving.

## Dataset

This project uses the public IBM Telco Customer Churn sample dataset.

- Original records: 7,043
- Original columns: 21
- Columns after removing the customer identifier: 20
- Target variable: `Churn`
- Target classes: `Yes` and `No`
- Observed churn rate: 26.54%

The dataset represents a fictional telecom business. The results should not be assumed to generalise to real companies or other industries.

The raw dataset is stored locally and is excluded from version control.

## Data Quality and Preparation

The data preparation workflow includes:

- Inspecting dataset dimensions and categorical values.
- Checking the target distribution.
- Identifying invalid or blank `TotalCharges` values.
- Verifying the circumstances associated with those values.
- Converting `TotalCharges` to a numeric column.
- Removing the `customerID` identifier from the modelling dataset.
- Checking the cleaned dataset for missing values.

Eleven records contain blank `TotalCharges` values. All eleven have zero tenure and a churn label of `No`. The preparation script assigns zero to these charges, an explicit assumption based on the observed records.

The cleaned dataset contains 7,043 rows and 20 columns.

Identical feature rows are not automatically removed: after removing customer identifiers, two distinct customers can have identical recorded attributes.

## Models

Three approaches are compared:

1. **Majority-class baseline** — predicts the most common class for every customer.
2. **Logistic Regression** — provides a classification baseline with interpretable model coefficients.
3. **Random Forest** — captures potentially nonlinear relationships and interactions between features.

### Preprocessing

The modelling pipeline uses:

- Median imputation for numerical features.
- Standard scaling for numerical features in the logistic regression pipeline.
- Most-frequent imputation for categorical features.
- One-hot encoding for categorical features.
- A scikit-learn pipeline to combine preprocessing and model training.

Preprocessing is fitted on the training data to reduce data leakage. The modelling workflow uses stratified splits and a fixed random seed for reproducibility.

## Evaluation

The models are evaluated using:

- Accuracy
- ROC-AUC
- Average precision (PR-AUC)
- Confusion matrices
- Precision
- Recall
- F1-score

### Model Comparison Results

The following results are from the original stratified 80/20 holdout evaluation.

| Model | Accuracy | ROC-AUC | Average Precision |
|---|---:|---:|---:|
| Majority-class baseline | 0.735 | 0.500 | 0.265 |
| Logistic Regression | 0.806 | 0.842 | 0.634 |
| Random Forest | 0.771 | 0.822 | 0.603 |

Logistic Regression has the strongest ROC-AUC and average precision among these three models on that holdout split.

At the default classification threshold, the Random Forest identifies more churners than Logistic Regression, but it also flags more customers who do not churn.

These results illustrate why model evaluation should consider more than accuracy alone. The appropriate model depends on the business objective and the relative costs of false positives and false negatives.

**Important:** the model comparison above uses the original 80/20 split. The updated threshold workflow uses a separate 60/20/20 training, validation, and test split. The results from these two workflows should not be compared as if they came from the same test set.

The reported results are specific to their respective data splits and should not be interpreted as a guarantee of future performance.

## Retention Campaign Threshold Analysis

A classification threshold determines the probability above which a customer is flagged as being at risk of churn.

Lowering the threshold generally flags more customers and identifies more churners, but it also increases the number of false positives. Raising the threshold reduces outreach volume but misses more churners.

### Threshold Selection Method

The updated workflow divides the dataset into three stratified subsets:

- **Training set (60%)** — used to fit the Logistic Regression model.
- **Validation set (20%)** — used to compare candidate thresholds and select one using F1-score.
- **Test set (20%)** — used to evaluate the selected threshold separately.

The candidate thresholds are 0.2, 0.3, 0.4, 0.5, 0.6 and 0.7.

The threshold with the highest validation F1-score was **0.4**.

### Final Test-Set Results

| Metric | Result |
|---|---:|
| Selected threshold | 0.4 |
| Test customers | 1,409 |
| Customers flagged | 442 |
| Actual churners in test set | 374 |
| Churners identified | 248 |
| Precision | 56.1% |
| Recall | 66.3% |

At this threshold, the model flagged 442 customers and correctly identified 248 customers who churned in the test data. It missed 126 of the 374 actual churners.

Precision of 56.1% means that approximately 56% of flagged customers churned in this test set. Recall of 66.3% means that the model identified approximately 66% of all churners in the test set.

### Business Interpretation

- **Lower threshold:** reaches more customers and identifies more eventual churners, but creates a larger outreach workload.
- **Higher threshold:** concentrates outreach on a smaller group, but misses more eventual churners.
- **Operational threshold:** should depend on campaign capacity, intervention costs, customer value, and the effectiveness of retention offers.

F1-score is a technical selection criterion, not necessarily the optimal business objective. A production decision should consider the relative costs of false positives and false negatives, the value of retaining customers, and the effectiveness of retention offers.

These results measure predictive performance only. They do not demonstrate that contacting flagged customers will prevent churn or generate profit.

### Generated Threshold Reports

- `reports/threshold_validation.csv` — validation metrics for each candidate threshold.
- `reports/threshold_test_results.csv` — test-set results at the selected threshold.

## Generated Outputs

The scripts generate local files in the `reports/` directory:

- `model_results.txt` — model metrics and classification reports.
- `evaluation_curves.png` — model evaluation charts.
- `threshold_validation.csv` — validation results for candidate thresholds.
- `threshold_test_results.csv` — final test evaluation at the selected threshold.

Generated reports are excluded from version control.

## Project Structure

```text
Churn_Prediction_Analysis/
├── data/
│   └── Local datasets (not committed)
├── reports/
│   └── Generated results (not committed)
├── src/
│   ├── __init__.py
│   ├── inspect_data.py
│   ├── check_data_quality.py
│   ├── prepare_data.py
│   ├── modeling.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── threshold_analysis.py
├── tests/
│   └── test_data.py
├── .gitignore
├── README.md
├── requirements.txt
└── churn_prediction_analysis.md
```

## Setup

### Requirements

- Python 3.14 or a compatible Python version.
- Git.
- The original dataset downloaded to the expected local path.

### 1. Clone the repository

```powershell
git clone https://github.com/ritvij9/Churn_Prediction.git
cd Churn_Prediction
```

If you already have the repository locally, open its existing project directory instead of cloning it again.

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Add the dataset

Place the original `Telco-Customer-Churn.csv` file in the local `data/` directory.

Expected path:

```text
data/Telco-Customer-Churn.csv
```

The raw and cleaned datasets are not committed to the repository.

## Run the Project

Run these commands from the project root, with the virtual environment activated.

### 1. Inspect the raw data

```powershell
python -m src.inspect_data
```

### 2. Check data quality

```powershell
python -m src.check_data_quality
```

### 3. Prepare the modelling dataset

```powershell
python -m src.prepare_data
```

### 4. Train and compare models

```powershell
python -m src.train_model
```

### 5. Generate evaluation charts

```powershell
python -m src.evaluate_model
```

### 6. Analyse classification thresholds

```powershell
python -m src.threshold_analysis
```

### 7. Run automated tests

```powershell
python -m unittest discover -s tests -v
```

The preparation step must be completed before running the modelling scripts if the cleaned dataset does not already exist.

## Testing

The automated test suite checks:

- Expected dimensions of the cleaned dataset.
- Missing-value handling.
- Valid target labels.
- Numeric and complete `TotalCharges`.
- Zero-tenure charge handling.
- Reproducibility and approximate stratification of the train/test split.
- Model pipeline handling of previously unseen categorical values.

Run all tests with:

```powershell
python -m unittest discover -s tests -v
```

## Limitations and Next Steps

- The data is a public sample dataset, not live production data.
- The observed relationships do not establish causes of churn.
- Predictive performance does not establish that retention interventions will work.
- The threshold is selected by validation F1-score, which may not reflect actual business costs and benefits.
- The test set is reserved for evaluating the selected threshold in the updated threshold workflow.
- The model comparison script still uses its original 80/20 split; a future improvement would align model comparison and threshold evaluation around one consistent train/validation/test design.
- Further work could include probability calibration, cost-sensitive threshold selection, and evaluation on independent business data.
- Real deployment would require monitoring, operational integration, privacy review, and testing whether retention actions produce measurable benefits.

## Conclusion

This project demonstrates a reproducible customer churn classification workflow, from data inspection and preparation through model comparison and threshold analysis.

In the original holdout evaluation, Logistic Regression performed best on ROC-AUC and average precision among the evaluated models. In the updated threshold workflow, the threshold was selected using validation data and evaluated separately on the test set.

The results illustrate how classification decisions affect customer outreach volume and churner identification. The next step toward a stronger business case is to choose thresholds using realistic business costs and test whether acting on model predictions creates measurable retention value.
