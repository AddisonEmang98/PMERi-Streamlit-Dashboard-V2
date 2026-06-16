import pandas as pd
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =====================================
# 1. LOAD DATASET
# =====================================
df = pd.read_csv("Corrected_PMERi_Data.csv")

# =====================================
# 2. CLASSIFIER DATA
# =====================================
X_cls = df.drop(columns=["Risk Label", "PMERi Score"])
y_cls = df["Risk Label"]

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(
    X_cls,
    y_cls,
    test_size=0.2,
    random_state=42,
    stratify=y_cls
)

# =====================================
# 3. LOAD CLASSIFIER MODEL
# =====================================
classifier_model = joblib.load(
    "PMERi_RandomForest_Model.pkl"
)

# =====================================
# 4. CLASSIFIER EVALUATION
# =====================================
y_pred_cls = classifier_model.predict(X_test_cls)

classifier_accuracy = accuracy_score(
    y_test_cls,
    y_pred_cls
)

classifier_f1 = f1_score(
    y_test_cls,
    y_pred_cls,
    average="macro"
)

classifier_precision = precision_score(
    y_test_cls,
    y_pred_cls,
    average="macro"
)

classifier_recall = recall_score(
    y_test_cls,
    y_pred_cls,
    average="macro"
)

# =====================================
# 5. REGRESSOR DATA
# =====================================
X_reg = df.drop(columns=["PMERi Score", "Risk Label"])
y_reg = df["PMERi Score"]

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.2,
    random_state=42
)

# =====================================
# 6. LOAD REGRESSOR MODEL
# =====================================
regressor_model = joblib.load(
    "PMERi_RF_Regressor.pkl"
)

# =====================================
# 7. REGRESSOR EVALUATION
# =====================================
y_pred_reg = regressor_model.predict(X_test_reg)

regressor_r2 = r2_score(
    y_test_reg,
    y_pred_reg
)

regressor_mae = mean_absolute_error(
    y_test_reg,
    y_pred_reg
)

regressor_rmse = np.sqrt(
    mean_squared_error(
        y_test_reg,
        y_pred_reg
    )
)

# =====================================
# 8. STORED CV RESULTS
# =====================================

# Classifier (from PMERi_Classifier_V2.py)
classifier_cv_folds = [
    0.7242,
    0.8214,
    0.5758,
    0.9348,
    0.8796
]

# Regressor (from PMERi_Regressor_V2.py)
regressor_cv_folds = [
    0.8011,
    0.8973,
    0.8860,
    0.7496,
    0.8454
]

# =====================================
# 9. CONSOLIDATED METRICS
# =====================================
metrics = {

    # =================================
    # CLASSIFIER
    # =================================
    "classifier_accuracy":
        float(classifier_accuracy),

    "classifier_f1":
        float(classifier_f1),

    "classifier_precision":
        float(classifier_precision),

    "classifier_recall":
        float(classifier_recall),

    "classifier_cv_mean":
        float(np.mean(classifier_cv_folds)),

    "classifier_cv_std":
        float(np.std(classifier_cv_folds)),

    "classifier_cv_folds":
        classifier_cv_folds,

    # =================================
    # REGRESSOR
    # =================================
    "regressor_r2":
        float(regressor_r2),

    "regressor_mae":
        float(regressor_mae),

    "regressor_rmse":
        float(regressor_rmse),

    "regressor_cv_mean":
        float(np.mean(regressor_cv_folds)),

    "regressor_cv_std":
        float(np.std(regressor_cv_folds)),

    "regressor_cv_folds":
        regressor_cv_folds
}

# =====================================
# 10. EXPORT
# =====================================
joblib.dump(
    metrics,
    "model_metrics.pkl"
)

# =====================================
# 11. SUMMARY
# =====================================
print("\n========== METRICS EXPORTED SUCCESSFULLY ==========")

print("\nCLASSIFIER PERFORMANCE")
print(f"Accuracy      : {classifier_accuracy:.4f}")
print(f"F1-Macro      : {classifier_f1:.4f}")
print(f"Precision     : {classifier_precision:.4f}")
print(f"Recall        : {classifier_recall:.4f}")
print(f"CV Mean       : {np.mean(classifier_cv_folds):.4f}")
print(f"CV Std Dev    : {np.std(classifier_cv_folds):.4f}")

print("\nREGRESSOR PERFORMANCE")
print(f"R²            : {regressor_r2:.4f}")
print(f"MAE           : {regressor_mae:.4f}")
print(f"RMSE          : {regressor_rmse:.4f}")
print(f"CV Mean       : {np.mean(regressor_cv_folds):.4f}")
print(f"CV Std Dev    : {np.std(regressor_cv_folds):.4f}")

print("\nSaved as: model_metrics.pkl")
print("===================================================")