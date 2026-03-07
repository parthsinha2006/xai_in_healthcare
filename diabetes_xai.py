# ============================================================
# DIABETES 130-US XAI STABILITY PROJECT (OPTIMIZED VERSION)
# ============================================================

print("Starting Diabetes XAI Stability Project...")

import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv("diabetes_130_us/diabetic_data.csv")

data.drop(["encounter_id", "patient_nbr"], axis=1, inplace=True)

data.replace("?", np.nan, inplace=True)
data.replace("Unknown/Invalid", np.nan, inplace=True)

data["readmitted"] = data["readmitted"].apply(
    lambda x: 0 if x == "NO" else 1
)

y = data["readmitted"]
X = data.drop("readmitted", axis=1)

X.fillna("Missing", inplace=True)

# ============================================================
# 2. ONE-HOT ENCODING
# ============================================================

print("Applying One-Hot Encoding...")

X = pd.get_dummies(X, drop_first=True)

# Convert everything to float32
X = X.astype(np.float32)

print("Shape after encoding:", X.shape)

# ============================================================
# 3. FEATURE SELECTION (CRITICAL FOR SHAP)
# ============================================================

print("Reducing feature space...")

temp_rf = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

temp_rf.fit(X, y)

selector = SelectFromModel(temp_rf, threshold="median", prefit=True)
X_reduced = selector.transform(X)

print("Shape after feature selection:", X_reduced.shape)

# ============================================================
# 4. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_reduced, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 5. TRAIN FINAL RANDOM FOREST
# ============================================================

print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# ============================================================
# 6. SHAP EXPLANATION (SAFE VERSION)
# ============================================================

print("\nGenerating SHAP explanations...")

X_sample = X_test[:10]

explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_sample)

# Select class 1
shap_values_class1 = shap_values[1]

shap.summary_plot(shap_values_class1, X_sample, show=False)
plt.savefig("diabetes_shap_summary.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# 7. STABILITY ANALYSIS
# ============================================================

print("\nRunning Stability Analysis...")

def get_shap_importance(model, X_data):
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_data)
    return np.abs(shap_vals[1]).mean(axis=0)

seeds = [1, 10, 20, 30, 42]
rankings = []

for seed in seeds:
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    importance = get_shap_importance(model, X_sample)
    rankings.append(importance)

rankings = np.array(rankings)

correlations = []

for i in range(len(rankings)-1):
    corr, _ = spearmanr(rankings[i], rankings[i+1])
    correlations.append(corr)

print("Spearman Stability Scores:", correlations)
print("Average Stability:", np.mean(correlations))

# ============================================================
# 8. PERTURBATION TEST
# ============================================================

print("\nTesting Stability Under Noise...")

noise = np.random.normal(0, 0.01, X_sample.shape)
X_noisy = X_sample + noise

shap_original = get_shap_importance(rf, X_sample)
shap_noisy = get_shap_importance(rf, X_noisy)

perturb_corr, _ = spearmanr(shap_original, shap_noisy)

print("Stability under perturbation:", perturb_corr)

print("\nDIABETES PROJECT COMPLETED SUCCESSFULLY")