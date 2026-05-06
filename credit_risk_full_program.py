import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# -----------------------------
# 1. Create output folders
# -----------------------------
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# -----------------------------
# 2. Load dataset
# -----------------------------
print("Loading dataset...")
credit = fetch_openml(name="credit-g", version=1, as_frame=True)
df = credit.frame


# -----------------------------
# 3. Create target variable
# -----------------------------
df["target"] = df["class"].map({"good": 0, "bad": 1})


# -----------------------------
# 4. Prepare features and target
# -----------------------------
X = df.drop(columns=["class", "target"])
y = df["target"]

X_encoded = pd.get_dummies(X, drop_first=True)


# -----------------------------
# 5. Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# 6. Train model
# -----------------------------
print("Training model...")
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=5000)
)

model.fit(X_train, y_train)


# -----------------------------
# 7. Predictions
# -----------------------------
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]


# -----------------------------
# 8. Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
report = classification_report(y_test, y_pred)
matrix = confusion_matrix(y_test, y_pred)

print("\nCredit Risk Model Results")
print("-------------------------")
print(f"Accuracy: {accuracy:.3f}")
print(f"ROC-AUC: {auc:.3f}")
print("\nClassification Report:")
print(report)
print("\nConfusion Matrix:")
print(matrix)


# -----------------------------
# 9. Save metrics
# -----------------------------
metrics = pd.DataFrame({
    "Metric": ["Accuracy", "ROC-AUC"],
    "Score": [accuracy, auc]
})

metrics.to_csv("outputs/model_metrics.csv", index=False)

with open("outputs/classification_report.txt", "w") as f:
    f.write("Credit Risk Model Results\n")
    f.write("-------------------------\n")
    f.write(f"Accuracy: {accuracy:.3f}\n")
    f.write(f"ROC-AUC: {auc:.3f}\n\n")
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(str(matrix))


# -----------------------------
# 10. Save ROC curve
# -----------------------------
RocCurveDisplay.from_estimator(model, X_test, y_test)
plt.title("ROC Curve - Logistic Regression")
plt.savefig("outputs/roc_curve.png", bbox_inches="tight")
plt.close()


# -----------------------------
# 11. Save coefficient table
# -----------------------------
coef = model.named_steps["logisticregression"].coef_[0]

coef_table = pd.DataFrame({
    "Variable": X_encoded.columns,
    "Coefficient": coef
})

coef_table["Abs_Coefficient"] = coef_table["Coefficient"].abs()
coef_table = coef_table.sort_values("Abs_Coefficient", ascending=False)

coef_table.to_csv("outputs/coefficients.csv", index=False)


# -----------------------------
# 12. Save trained model and column names
# -----------------------------
joblib.dump(model, "models/credit_risk_model.pkl")
joblib.dump(X_encoded.columns.tolist(), "models/model_columns.pkl")


# -----------------------------
# 13. Simple borrower prediction example
# -----------------------------
example_borrower = X_encoded.iloc[[0]]
risk_probability = model.predict_proba(example_borrower)[:, 1][0]
risk_prediction = model.predict(example_borrower)[0]

print("\nExample Borrower Prediction")
print("---------------------------")
print(f"Predicted probability of bad credit risk: {risk_probability:.3f}")

if risk_prediction == 1:
    print("Prediction: BAD CREDIT RISK")
else:
    print("Prediction: GOOD CREDIT RISK")


print("\nAutomation complete.")
print("Saved files:")
print("- outputs/model_metrics.csv")
print("- outputs/classification_report.txt")
print("- outputs/roc_curve.png")
print("- outputs/coefficients.csv")
print("- models/credit_risk_model.pkl")
print("- models/model_columns.pkl")