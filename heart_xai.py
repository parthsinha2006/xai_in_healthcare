# ============================================================
# HEART DISEASE XAI STABILITY PROJECT (FIXED + COMPLETE)
# ============================================================

print("Starting Heart Disease XAI Stability Project...")

import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")  # Must be before any other matplotlib import — safe for VSCode terminal
import matplotlib.pyplot as plt
import seaborn as sns
import os

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
    "age", "sex", "cp", "trestbps", "chol", "fbs",
    "restecg", "thalach", "exang", "oldpeak",
    "slope", "ca", "thal", "target"
]

data = pd.read_csv(
    "heart_disease/processed.cleveland.data",
    names=columns
)

data.replace("?", np.nan, inplace=True)
data = data.apply(pd.to_numeric)

imputer = SimpleImputer(strategy="mean")
data_imputed = pd.DataFrame(
    imputer.fit_transform(data),
    columns=data.columns
)

# Binary target: 0 = no disease, 1 = disease
data_imputed["target"] = data_imputed["target"].apply(lambda x: 0 if x == 0 else 1)

X = data_imputed.drop("target", axis=1)
y = data_imputed["target"]

print("Dataset shape:", X.shape)

# ============================================================
# 2. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Reset indices — critical to avoid SHAP shape mismatch
# (imputer resets to 0-based index but train_test_split preserves original)
X_train = X_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

# ============================================================
# 3. TRAIN RANDOM FOREST
# ============================================================

print("Training Random Forest...")

rf = RandomForestClassifier(
    n_estimators=150,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("\nModel Performance:")
print("  Accuracy :", accuracy_score(y_test, y_pred))
print("  Recall   :", recall_score(y_test, y_pred))
print("  F1 Score :", f1_score(y_test, y_pred))

# ============================================================
# 4. SHAP EXPLANATIONS
# ============================================================

print("\nGenerating SHAP explanations...")

shap_explainer = shap.TreeExplainer(rf)
shap_values = shap_explainer.shap_values(X_test)

# Handle both old SHAP (list) and new SHAP (3D array)
if isinstance(shap_values, list):
    shap_vals_class1 = shap_values[1]
    expected_val = shap_explainer.expected_value[1]
else:
    shap_vals_class1 = shap_values[:, :, 1]
    expected_val = shap_explainer.expected_value[1]

print("  SHAP values shape :", shap_vals_class1.shape)
print("  X_test shape      :", X_test.shape)

shap_explanation = shap.Explanation(
    values=shap_vals_class1,
    base_values=np.full(len(X_test), expected_val),
    data=X_test.values,
    feature_names=X_test.columns.tolist()
)

# Beeswarm plot
plt.figure()
shap.plots.beeswarm(shap_explanation, show=False)
plt.title("SHAP Global Feature Importance (Disease Class)")
plt.tight_layout()
plt.savefig("heart_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: heart_shap_beeswarm.png")

# Bar plot
plt.figure()
shap.plots.bar(shap_explanation, show=False)
plt.tight_layout()
plt.savefig("heart_shap_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: heart_shap_bar.png")

# ============================================================
# 5. PERMUTATION IMPORTANCE
# ============================================================

print("\nComputing Permutation Importance...")

perm = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

perm_df = pd.DataFrame({
    "Feature"   : X.columns,
    "Importance": perm.importances_mean,
    "Std"       : perm.importances_std
}).sort_values(by="Importance", ascending=False)

print("\nTop Features (Permutation Importance):")
print(perm_df.to_string(index=False))

plt.figure(figsize=(8, 5))
plt.barh(
    perm_df["Feature"][::-1],
    perm_df["Importance"][::-1],
    xerr=perm_df["Std"][::-1],
    color="#4C72B0"
)
plt.title("Permutation Feature Importance")
plt.xlabel("Mean Accuracy Decrease")
plt.tight_layout()
plt.savefig("heart_permutation_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: heart_permutation_importance.png")

# ============================================================
# 6. LIME EXPLANATION
# ============================================================

print("\nGenerating LIME explanations...")

lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X.columns.tolist(),
    class_names=["No Disease", "Disease"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

lime_feature_importance = []

for i in range(3):
    exp = lime_explainer.explain_instance(
        data_row=X_test.values[i],
        predict_fn=rf.predict_proba,
        num_features=X_test.shape[1],
        num_samples=1000
    )

    print(f"\n  LIME Explanation — Instance {i}:")
    for feat, weight in exp.as_list():
        print(f"    {feat:40s}: {weight:+.4f}")

    html_path = f"heart_lime_instance_{i}.html"
    exp.save_to_file(html_path)
    print(f"  Saved: {html_path}")

    fig = exp.as_pyplot_figure()
    fig.tight_layout()
    fig.savefig(f"heart_lime_instance_{i}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: heart_lime_instance_{i}.png")

    weights_dict = dict(exp.as_list())
    lime_feature_importance.append(weights_dict)

# ============================================================
# 7. SHAP vs LIME COMPARISON
# ============================================================

print("\nComparing SHAP vs LIME top features...")

shap_mean = np.abs(shap_vals_class1).mean(axis=0)
shap_top_idx = np.argsort(shap_mean)[::-1]
shap_top_features = [X_test.columns[i] for i in shap_top_idx]
shap_top_values = shap_mean[shap_top_idx]

lime_agg = {}
for instance_weights in lime_feature_importance:
    for feat, val in instance_weights.items():
        lime_agg[feat] = lime_agg.get(feat, 0) + abs(val)

lime_sorted = sorted(lime_agg.items(), key=lambda x: x[1], reverse=True)
lime_top_features = [f[0] for f in lime_sorted]
lime_top_values   = [f[1] / 3 for f in lime_sorted]

n = min(10, len(shap_top_features), len(lime_top_features))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].barh(shap_top_features[:n][::-1], shap_top_values[:n][::-1], color="#4C72B0")
axes[0].set_title("SHAP — Top Features (Mean |SHAP|)")
axes[0].set_xlabel("Mean |SHAP value|")

axes[1].barh(lime_top_features[:n][::-1], lime_top_values[:n][::-1], color="#DD8452")
axes[1].set_title("LIME — Top Features (Avg |Weight|, 3 instances)")
axes[1].set_xlabel("Average |LIME weight|")

plt.tight_layout()
plt.savefig("heart_shap_vs_lime.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: heart_shap_vs_lime.png")

# ============================================================
# 8. STABILITY ANALYSIS — SHAP ACROSS SEEDS
# ============================================================

print("\nRunning SHAP Stability Analysis across seeds...")

def get_shap_importance(model, X_data):
    """Returns mean absolute SHAP values for class 1."""
    exp = shap.TreeExplainer(model)
    vals = exp.shap_values(X_data)
    if isinstance(vals, list):
        return np.abs(vals[1]).mean(axis=0)
    else:
        return np.abs(vals[:, :, 1]).mean(axis=0)

seeds = [1, 10, 20, 30, 42]
shap_rankings = []

for seed in seeds:
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    importance = get_shap_importance(model, X_test)
    shap_rankings.append(importance)
    print(f"  Seed {seed} done.")

shap_rankings = np.array(shap_rankings)

correlations = []
for i in range(len(shap_rankings) - 1):
    corr, _ = spearmanr(shap_rankings[i], shap_rankings[i + 1])
    correlations.append(corr)

print("  Spearman Stability Scores:", [round(c, 4) for c in correlations])
print("  Average SHAP Stability   :", round(np.mean(correlations), 4))

plt.figure(figsize=(7, 4))
plt.plot(
    [f"Seed {seeds[i]}→{seeds[i+1]}" for i in range(len(correlations))],
    correlations,
    marker="o", color="#4C72B0", linewidth=2, markersize=8
)
plt.ylim(0, 1.05)
plt.axhline(np.mean(correlations), color="red", linestyle="--",
            label=f"Mean = {np.mean(correlations):.4f}")
plt.title("SHAP Stability — Spearman Correlation Across Seeds")
plt.ylabel("Spearman Correlation")
plt.xlabel("Seed Pair")
plt.legend()
plt.tight_layout()
plt.savefig("heart_shap_stability.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: heart_shap_stability.png")

# ============================================================
# 9. PERTURBATION TEST
# ============================================================

print("\nTesting SHAP Stability Under Gaussian Noise...")

np.random.seed(42)
noise_levels = [0.001, 0.005, 0.01, 0.05, 0.1]
perturb_scores = []

shap_original = get_shap_importance(rf, X_test)

for noise_std in noise_levels:
    noise = np.random.normal(0, noise_std, X_test.shape)
    X_noisy = pd.DataFrame(
        X_test.values + noise,
        columns=X_test.columns
    )
    shap_noisy = get_shap_importance(rf, X_noisy)
    corr, _ = spearmanr(shap_original, shap_noisy)
    perturb_scores.append(corr)
    print(f"  Noise std={noise_std:.3f}  →  Stability: {corr:.4f}")

plt.figure(figsize=(7, 4))
plt.plot(noise_levels, perturb_scores, marker="s", color="#55A868", linewidth=2, markersize=8)
plt.xscale("log")
plt.ylim(0, 1.05)
plt.title("SHAP Stability Under Input Perturbation")
plt.xlabel("Noise Std Dev (log scale)")
plt.ylabel("Spearman Correlation with Original")
plt.tight_layout()
plt.savefig("heart_shap_perturbation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: heart_shap_perturbation.png")

# ============================================================
# 10. LIME STABILITY ACROSS SEEDS
# ============================================================

print("\nRunning LIME Stability Analysis...")

lime_rankings = []

for seed in seeds:
    model = RandomForestClassifier(
        n_estimators=150,
        random_state=seed,
        class_weight="balanced",
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    exp = lime_explainer.explain_instance(
        data_row=X_test.values[0],
        predict_fn=model.predict_proba,
        num_features=X_test.shape[1],
        num_samples=1000
    )

    weights_dict = dict(exp.as_list())
    importance_vec = np.array([
        abs(weights_dict.get(feat, 0.0)) for feat in X_test.columns
    ])
    lime_rankings.append(importance_vec)
    print(f"  Seed {seed} done.")

lime_rankings = np.array(lime_rankings)

lime_correlations = []
for i in range(len(lime_rankings) - 1):
    corr, _ = spearmanr(lime_rankings[i], lime_rankings[i + 1])
    lime_correlations.append(corr)

print("  LIME Stability Scores:", [round(c, 4) for c in lime_correlations])
print("  Average LIME Stability:", round(np.mean(lime_correlations), 4))

# ============================================================
# 11. FINAL SHAP vs LIME STABILITY COMPARISON
# ============================================================

print("\nFinal Stability Comparison:")
print(f"  SHAP avg stability : {np.mean(correlations):.4f}")
print(f"  LIME avg stability : {np.mean(lime_correlations):.4f}")

methods = ["SHAP", "LIME"]
avg_stabilities = [np.mean(correlations), np.mean(lime_correlations)]

plt.figure(figsize=(5, 4))
bars = plt.bar(methods, avg_stabilities, color=["#4C72B0", "#DD8452"], width=0.4)
plt.ylim(0, 1.1)
plt.title("Average Stability: SHAP vs LIME\n(Spearman Corr across seeds)")
plt.ylabel("Average Spearman Correlation")
for bar, val in zip(bars, avg_stabilities):
    plt.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
             f"{val:.4f}", ha="center", fontsize=12)
plt.tight_layout()
plt.savefig("heart_stability_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: heart_stability_comparison.png")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*55)
print("  HEART DISEASE XAI STABILITY PROJECT — COMPLETED")
print("="*55)
print("\nOutput files generated:")
output_files = [
    "heart_shap_beeswarm.png",
    "heart_shap_bar.png",
    "heart_permutation_importance.png",
    "heart_lime_instance_0.png",
    "heart_lime_instance_1.png",
    "heart_lime_instance_2.png",
    "heart_lime_instance_0.html",
    "heart_lime_instance_1.html",
    "heart_lime_instance_2.html",
    "heart_shap_vs_lime.png",
    "heart_shap_stability.png",
    "heart_shap_perturbation.png",
    "heart_stability_comparison.png",
]
for f in output_files:
    status = "Found." if os.path.exists(f) else "MISSING"
    print(f"  [{status}] {f}")
