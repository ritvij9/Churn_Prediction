# Customer Churn Prediction Model

Predicts which customers are likely to cancel their subscription, using historical usage and billing data.

## Dataset

```python
import pandas as pd
import numpy as np

np.random.seed(42)
n = 1000

plans = ["Basic", "Pro", "Enterprise"]
countries = ["India", "USA", "UK", "Canada", "Germany"]

data = {
    "customer_id": [f"C{str(i).zfill(4)}" for i in range(1, n+1)],
    "plan": np.random.choice(plans, n, p=[0.5, 0.35, 0.15]),
    "monthly_revenue": None,
    "tenure_days": np.random.randint(10, 730, n),
    "support_tickets": np.random.randint(0, 20, n),
    "logins_per_month": np.random.randint(0, 60, n),
    "feature_usage_score": np.round(np.random.uniform(0, 100, n), 1),
    "country": np.random.choice(countries, n),
    "team_size": np.random.randint(1, 50, n),
    "last_payment_failed": np.random.choice([0, 1], n, p=[0.85, 0.15]),
    "discount_applied": np.random.choice([0, 1], n, p=[0.7, 0.3]),
}

df = pd.DataFrame(data)

price_map = {"Basic": 29.99, "Pro": 79.99, "Enterprise": 199.99}
df["monthly_revenue"] = df["plan"].map(price_map)

df.loc[np.random.choice(df.index, 30), "monthly_revenue"] = np.nan
df.loc[np.random.choice(df.index, 20), "logins_per_month"] = np.nan
df.loc[np.random.choice(df.index, 15), "feature_usage_score"] = np.nan

churn_score = (
    (df["support_tickets"] > 10).astype(int) * 2 +
    (df["logins_per_month"].fillna(0) < 5).astype(int) * 2 +
    (df["feature_usage_score"].fillna(0) < 30).astype(int) * 1.5 +
    (df["last_payment_failed"] == 1).astype(int) * 2.5 +
    (df["tenure_days"] < 90).astype(int) * 1 +
    (df["plan"] == "Basic").astype(int) * 0.5
)

churn_prob = churn_score / churn_score.max()
df["churned"] = (churn_prob > 0.45).astype(int)

df.to_csv("churn_ml_data.csv", index=False)
print(df.head())
```

## Model

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

df = pd.read_csv("churn_ml_data.csv")

df["monthly_revenue"] = df["monthly_revenue"].fillna(df.groupby("plan")["monthly_revenue"].transform("median"))
df["logins_per_month"] = df["logins_per_month"].fillna(df["logins_per_month"].median())
df["feature_usage_score"] = df["feature_usage_score"].fillna(df["feature_usage_score"].median())

df["revenue_per_login"] = df["monthly_revenue"] / (df["logins_per_month"] + 1)
df["tickets_per_month"] = df["support_tickets"] / (df["tenure_days"] / 30 + 1)
df["is_low_engagement"] = (df["logins_per_month"] < 5).astype(int)
df["is_new_customer"] = (df["tenure_days"] < 90).astype(int)
df["is_high_support"] = (df["support_tickets"] > 10).astype(int)

le = LabelEncoder()
df["plan_encoded"] = le.fit_transform(df["plan"])
df["country_encoded"] = le.fit_transform(df["country"])

features = [
    "plan_encoded", "monthly_revenue", "tenure_days",
    "support_tickets", "logins_per_month", "feature_usage_score",
    "team_size", "last_payment_failed", "discount_applied",
    "revenue_per_login", "tickets_per_month",
    "is_low_engagement", "is_new_customer", "is_high_support"
]

X = df[features]
y = df["churned"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred)*100:.0f}%")
print(f"Precision: {precision_score(y_test, y_pred)*100:.0f}%")
print(f"Recall:    {recall_score(y_test, y_pred)*100:.0f}%")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_prob):.2f}")

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)
print(importance.head(5))

df["churn_probability"] = model.predict_proba(X)[:, 1]
at_risk = df[df["churn_probability"] > 0.5].sort_values("churn_probability", ascending=False)
print(at_risk[["customer_id", "churn_probability", "plan", "monthly_revenue"]].head(10))
```

## Charts

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

churn_by_plan = df.groupby("plan")["churned"].mean() * 100
axes[0].bar(churn_by_plan.index, churn_by_plan.values, color=["#FF6B6B", "#4ECDC4", "#45B7D1"])
axes[0].set_title("Churn Rate by Plan")

top5 = importance.head(5)
axes[1].barh(top5["feature"], top5["importance"], color="#4ECDC4")
axes[1].set_title("Top 5 Churn Predictors")
axes[1].invert_yaxis()

axes[2].hist(df[df["churned"]==0]["churn_probability"], bins=30, alpha=0.6, label="Active")
axes[2].hist(df[df["churned"]==1]["churn_probability"], bins=30, alpha=0.6, label="Churned")
axes[2].set_title("Churn Probability Distribution")
axes[2].legend()

plt.tight_layout()
plt.savefig("churn_analysis.png", dpi=150)
plt.show()
```
