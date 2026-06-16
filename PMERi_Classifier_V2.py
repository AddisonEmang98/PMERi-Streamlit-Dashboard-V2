import pandas as pd
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# -------------------
# 1. Load data
# -------------------
df = pd.read_csv("Corrected_PMERi_Data.csv")

# -------------------
# 2. Split features/target
# -------------------
X = df.drop(columns=["Risk Label", "PMERi Score"])
y = df["Risk Label"]

print("\nFeatures Used:")
print(X.columns.tolist())

# -------------------
# 3. Train-test split
# -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------
# 4. Base model
# -------------------
rf = RandomForestClassifier(
    random_state=42
)

# -------------------
# 5. Hyperparameter grid
# -------------------
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

# -------------------
# 6. GridSearchCV
# -------------------
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring="f1_macro",
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

# -------------------
# 7. Best model
# -------------------
best_model = grid_search.best_estimator_

print("\n========== GRID SEARCH RESULTS ==========")
print("Best Parameters:")
print(grid_search.best_params_)

print("\nBest Cross-Validation F1-Macro:")
print(grid_search.best_score_)

# =========================================================
# 7A. FULL 5-FOLD STRATIFIED CROSS VALIDATION (DETAILED)
# =========================================================
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    best_model,
    X,
    y,
    cv=cv,
    scoring="f1_macro",
    n_jobs=-1
)

print("\n========== 5-FOLD CROSS VALIDATION RESULTS ==========")

for i, score in enumerate(cv_scores, start=1):
    print(f"Fold {i}: {score:.4f}")

print("\nMean F1-Macro:")
print(f"{cv_scores.mean():.4f}")

print("Standard Deviation:")
print(f"{cv_scores.std():.4f}")

print("Minimum Score:")
print(f"{cv_scores.min():.4f}")

print("Maximum Score:")
print(f"{cv_scores.max():.4f}")

# -------------------
# 8. Prediction
# -------------------
y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)

print("\nFirst 10 Prediction Probabilities:")
print(y_prob[:10])

# -------------------
# 9. Evaluation
# -------------------
print("\n========== TEST SET RESULTS ==========")

print("\nAccuracy:")
print(accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -------------------
# 10. Feature Importance
# -------------------
feature_importance = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)

feature_importance.to_csv(
    "PMERi_Feature_Importance.csv",
    header=["Importance"]
)

# -------------------
# 11. Save Model
# -------------------
joblib.dump(
    best_model,
    "PMERi_RandomForest_Model.pkl"
)

print("\nModel saved successfully as:")
print("PMERi_RandomForest_Model.pkl")

print("\nFeature importance saved as:")
print("PMERi_Feature_Importance.csv")