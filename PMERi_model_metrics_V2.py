import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    mean_squared_error
)

# =========================
# 1. Load dataset
# =========================
df = pd.read_csv("Corrected_PMERi_Data.csv")

X = df.drop(columns=["Risk Label", "PMERi Score"])
y = df["Risk Label"]

# =========================
# 2. Load trained classifier model
# =========================
model = joblib.load("PMERi_RandomForest_Model.pkl")

# =========================
# 3. Cross-validation (F1 Macro)
# =========================
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=cv,
    scoring="f1_macro"
)

# =========================
# 4. Predictions (full dataset evaluation)
# =========================
y_pred = model.predict(X)

classifier_f1 = f1_score(y, y_pred, average="macro")
classifier_acc = accuracy_score(y, y_pred)
classifier_precision = precision_score(y, y_pred, average="macro")
classifier_recall = recall_score(y, y_pred, average="macro")

# =========================
# 5. ERROR METRICS
# =========================
classifier_mse = mean_squared_error(y, y_pred)
classifier_mae = np.mean(np.abs(y - y_pred))  # ordinal interpretation

# =========================
# 6. Metrics dictionary (FINAL THESIS FORMAT)
# =========================
metrics = {
    "classifier_f1": float(classifier_f1),
    "classifier_accuracy": float(classifier_acc),
    "classifier_precision": float(classifier_precision),
    "classifier_recall": float(classifier_recall),

    "classifier_mae": float(classifier_mae),
    "classifier_mse": float(classifier_mse),

    "classifier_cv_mean": float(cv_scores.mean()),
    "classifier_cv_std": float(cv_scores.std()),
    "classifier_cv_folds": cv_scores.tolist(),

    # regressor values (unchanged)
    "regressor_r2": 0.78321,
    "regressor_mae": 0.04218,
    "regressor_rmse": 0.05466
}

# =========================
# 7. Save metrics
# =========================
joblib.dump(metrics, "model_metrics.pkl")

print("\n========== MODEL METRICS SAVED ==========")
print("Classifier F1:", classifier_f1)
print("Classifier Accuracy:", classifier_acc)
print("Classifier Precision:", classifier_precision)
print("Classifier Recall:", classifier_recall)
print("Classifier MAE:", classifier_mae)
print("Classifier MSE:", classifier_mse)
print("CV Mean F1:", cv_scores.mean())
print("CV Std F1:", cv_scores.std())
print("CV Folds:", cv_scores.tolist())
print("Metrics saved to model_metrics.pkl")