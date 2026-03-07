# ============================================================
# HEART DISEASE XAI STABILITY PROJECT
# Compatible with NEW SHAP versions
# ============================================================

print("Starting XAI Stability Project...")

import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from scipy.stats import spearmanr

# ============================================================
# 1. LOAD HEART DISEASE DATA
# ============================================================

print("Loading processed Cleveland dataset...")

columns = [
    "age","sex","cp","trestbps","chol","fbs",
    "restecg","thalach","exang","oldpeak",
    "slope","ca","thal","target"
]

data = pd.read_csv(
    "heart_disease/processed.cleveland.data",
    names=columns
)

# Replace '?' with NaN
data.replace("?", np.nan, inplace=True)

# Convert to numeric
data = data.apply(pd.to_numeric)

# Impute missing values
imputer = SimpleImputer(strategy="mean")
data[:] = imputer.fit_transform(data)

# Convert target to binary (0 = no disease, 1 = disease)
data["target"] = data["target"].apply(lambda x: 0 if x == 0 else 1)

X = data.drop("target", axis=1)
y = data["target"]

print("Dataset shape:", X.shape)

# ============================================================
# 2. TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 3. TRAIN RANDOM FOREST
# ============================================================

print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced"
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("\nModel Performance:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

# ============================================================
# 4. SHAP EXPLANATIONS (FIXED FOR BINARY CLASSIFICATION)
# ============================================================

print("\nGenerating SHAP explanations...")

explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_test)

# Select class 1 explanations (Disease)
shap_values_class1 = shap_values[:, :, 1]

plt.figure()
shap.plots.beeswarm(shap_values_class1, show=False)
plt.title("SHAP Global Feature Importance (Disease Class)")
plt.tight_layout()
plt.savefig("heart_shap_summary.png", dpi=300)
plt.show()


# ============================================================
# 5. PERMUTATION IMPORTANCE
# ============================================================

print("\nComputing Permutation Importance...")

perm = permutation_importance(
    rf, X_test, y_test, n_repeats=5, random_state=42
)

perm_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": perm.importances_mean
}).sort_values(by="Importance", ascending=False)

print("\nTop Features (Permutation Importance):")
print(perm_df)

# ============================================================
# 6. LIME EXPLANATION
# ============================================================

print("\nGenerating LIME explanation...")

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train.values,
    feature_names=X.columns,
    class_names=["No Disease","Disease"],
    mode="classification"
)

exp = lime_explainer.explain_instance(
    X_test.values[0],
    rf.predict_proba,
    num_features=8
)

print("\nLIME Explanation:")
print(exp.as_list())

# ============================================================
# 7. STABILITY ANALYSIS (UPDATED FOR NEW SHAP)
# ============================================================

print("\nRunning Stability Analysis...")

def get_shap_importance(model, X_data):
    explainer = shap.Explainer(model, X_train)
    shap_vals = explainer(X_data)
    
    # select class 1
    shap_vals_class1 = shap_vals[:, :, 1]
    
    return np.abs(shap_vals_class1.values).mean(axis=0)

seeds = [1, 10, 20, 30, 42]
rankings = []

for seed in seeds:
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=seed,
        class_weight="balanced"
    )
    model.fit(X_train, y_train)
    importance = get_shap_importance(model, X_test)
    rankings.append(importance)

rankings = np.array(rankings)

correlations = []

for i in range(len(rankings)-1):
    corr, _ = spearmanr(rankings[i], rankings[i+1])
    correlations.append(corr)

print("Spearman Stability Scores:", correlations)
print("Average Stability:", np.mean(correlations))

# ============================================================
# 8. DATA PERTURBATION TEST
# ============================================================

print("\nTesting Stability Under Small Noise...")

noise = np.random.normal(0, 0.01, X_test.shape)
X_noisy = X_test + noise

shap_original = get_shap_importance(rf, X_test)
shap_noisy = get_shap_importance(rf, X_noisy)

perturb_corr, _ = spearmanr(shap_original, shap_noisy)

print("Stability under perturbation:", perturb_corr)

print("\nPROJECT COMPLETED SUCCESSFULLY")