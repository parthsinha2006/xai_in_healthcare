# ============================================================
# DIABETES 130-US XAI STABILITY PROJECT (ULTRAFAST + FIXED)
# ============================================================

print("Starting Diabetes XAI Stability Project...")

import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from scipy.stats import spearmanr

# ============================================================
# 1. LOAD & PREP
# ============================================================

print("Loading data...")
data = pd.read_csv("diabetes_130_us/diabetic_data.csv")
data.drop(["encounter_id", "patient_nbr"], axis=1, inplace=True)
data.replace("?", np.nan, inplace=True)
data.replace("Unknown/Invalid", np.nan, inplace=True)
data["readmitted"] = data["readmitted"].apply(lambda x: 0 if x == "NO" else 1)

y = data["readmitted"]
X = data.drop("readmitted", axis=1)
X.fillna("Missing", inplace=True)

# ============================================================
# 2. SUBSAMPLE TO 10K
# ============================================================

print("Subsampling to 10k rows...")
idx = np.random.RandomState(42).choice(len(X), size=10000, replace=False)
X = X.iloc[idx].reset_index(drop=True)
y = y.iloc[idx].reset_index(drop=True)

# ============================================================
# 3. ENCODE
# ============================================================

print("Encoding...")
X = pd.get_dummies(X, drop_first=True).astype(np.float32)
print("Shape after encoding:", X.shape)

# ============================================================
# 4. TRAIN/TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ============================================================
# 5. FEATURE SELECTION (fit on numpy to avoid warning)
# ============================================================

print("Feature selection...")
temp_rf = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1, max_depth=8)
temp_rf.fit(X_train.values, y_train)          # .values → no feature name warning
selector = SelectFromModel(temp_rf, threshold="median", prefit=True)

sel_mask   = selector.get_support()
sel_cols   = X_train.columns[sel_mask]
X_train    = pd.DataFrame(X_train.values[:, sel_mask], columns=sel_cols)
X_test     = pd.DataFrame(X_test.values[:, sel_mask],  columns=sel_cols)
print("Shape after feature selection:", X_train.shape)

# ============================================================
# 6. TRAIN MAIN MODEL
# ============================================================

print("Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=30,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1,
    max_depth=8
)
rf.fit(X_train.values, y_train)

y_pred = rf.predict(X_test.values)
print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"  F1 Score : {f1_score(y_test, y_pred):.4f}")

# ============================================================
# 7. SHAP — with version-safe value extraction
# ============================================================

print("\nGenerating SHAP explanations...")
X_sample = X_test.iloc[:20]

shap_explainer = shap.TreeExplainer(rf)
shap_values    = shap_explainer.shap_values(X_sample.values)

# FIX: handle both old shap (list) and new shap (3D array)
if isinstance(shap_values, list):
    # Old SHAP: [class0_array, class1_array]
    sv_class1      = shap_values[1]
    expected_value = shap_explainer.expected_value[1]
else:
    # New SHAP: single 3D array (n_samples, n_features, n_classes)
    if shap_values.ndim == 3:
        sv_class1  = shap_values[:, :, 1]
        expected_value = shap_explainer.expected_value[1]
    else:
        # Already 2D (some versions return this for binary)
        sv_class1  = shap_values
        ev = shap_explainer.expected_value
        expected_value = ev[1] if hasattr(ev, '__len__') else ev

print(f"  shap_values shape : {np.array(shap_values).shape}")
print(f"  sv_class1 shape   : {sv_class1.shape}")
print(f"  X_sample shape    : {X_sample.shape}")

shap_explanation = shap.Explanation(
    values=sv_class1,
    base_values=np.full(len(X_sample), expected_value),
    data=X_sample.values,
    feature_names=X_sample.columns.tolist()
)

plt.figure()
shap.plots.beeswarm(shap_explanation, show=False)
plt.tight_layout()
plt.savefig("diabetes_shap_beeswarm.png", dpi=100, bbox_inches="tight")
plt.close()
print("  Saved: diabetes_shap_beeswarm.png")

plt.figure()
shap.plots.bar(shap_explanation, show=False)
plt.tight_layout()
plt.savefig("diabetes_shap_bar.png", dpi=100, bbox_inches="tight")
plt.close()
print("  Saved: diabetes_shap_bar.png")

# ============================================================
# 8. LIME
# ============================================================

print("\nGenerating LIME explanations...")
lime_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["No Readmit", "Readmit"],
    mode="classification",
    discretize_continuous=True,
    random_state=42
)

lime_feature_importance = []
for i in range(3):
    exp = lime_explainer.explain_instance(
        data_row=X_sample.iloc[i].values,
        predict_fn=rf.predict_proba,
        num_features=10,
        num_samples=200
    )
    exp.save_to_file(f"diabetes_lime_instance_{i}.html")
    fig = exp.as_pyplot_figure()
    fig.tight_layout()
    fig.savefig(f"diabetes_lime_instance_{i}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: diabetes_lime_instance_{i}.png")
    lime_feature_importance.append(dict(exp.as_list()))

# ============================================================
# 9. SHAP vs LIME COMPARISON
# ============================================================

print("\nComparing SHAP vs LIME...")
shap_mean     = np.abs(sv_class1).mean(axis=0)
shap_top_idx  = np.argsort(shap_mean)[::-1][:10]
shap_top_feat = [X_sample.columns[i] for i in shap_top_idx]
shap_top_vals = shap_mean[shap_top_idx]

lime_agg = {}
for iw in lime_feature_importance:
    for feat, val in iw.items():
        lime_agg[feat] = lime_agg.get(feat, 0) + abs(val)
lime_sorted   = sorted(lime_agg.items(), key=lambda x: x[1], reverse=True)[:10]
lime_top_feat = [f[0] for f in lime_sorted]
lime_top_vals = [f[1] / 3 for f in lime_sorted]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(shap_top_feat[::-1], shap_top_vals[::-1], color="#4C72B0")
axes[0].set_title("SHAP — Top 10 Features")
axes[0].set_xlabel("Mean |SHAP value|")
axes[1].barh(lime_top_feat[::-1], lime_top_vals[::-1], color="#DD8452")
axes[1].set_title("LIME — Top 10 Features")
axes[1].set_xlabel("Avg |LIME weight|")
plt.tight_layout()
plt.savefig("diabetes_shap_vs_lime.png", dpi=100, bbox_inches="tight")
plt.close()
print("  Saved: diabetes_shap_vs_lime.png")

# ============================================================
# 10. HELPER — version-safe SHAP importance
# ============================================================

def get_shap_importance(model, X_data):
    exp = shap.TreeExplainer(model)
    sv  = exp.shap_values(X_data.values if hasattr(X_data, 'values') else X_data)
    if isinstance(sv, list):
        return np.abs(sv[1]).mean(axis=0)
    elif sv.ndim == 3:
        return np.abs(sv[:, :, 1]).mean(axis=0)
    else:
        return np.abs(sv).mean(axis=0)

# ============================================================
# 11. SHAP STABILITY (3 seeds)
# ============================================================

print("\nSHAP Stability Analysis...")
seeds         = [1, 10, 42]
shap_rankings = []

for seed in seeds:
    m = RandomForestClassifier(
        n_estimators=30, random_state=seed,
        class_weight="balanced", n_jobs=-1, max_depth=8
    )
    m.fit(X_train.values, y_train)
    shap_rankings.append(get_shap_importance(m, X_sample))

shap_rankings = np.array(shap_rankings)
correlations  = [spearmanr(shap_rankings[i], shap_rankings[i+1])[0]
                 for i in range(len(shap_rankings)-1)]
print(f"  Scores  : {[round(c,4) for c in correlations]}")
print(f"  Average : {np.mean(correlations):.4f}")

plt.figure(figsize=(7, 4))
plt.plot([f"{seeds[i]}→{seeds[i+1]}" for i in range(len(correlations))],
         correlations, marker="o", color="#4C72B0", linewidth=2, markersize=8)
plt.ylim(0, 1.05)
plt.axhline(np.mean(correlations), color="red", linestyle="--",
            label=f"Mean={np.mean(correlations):.4f}")
plt.title("SHAP Stability — Spearman Corr Across Seeds")
plt.ylabel("Spearman Correlation"); plt.xlabel("Seed Pair")
plt.legend(); plt.tight_layout()
plt.savefig("diabetes_shap_stability.png", dpi=100, bbox_inches="tight")
plt.close()
print("  Saved: diabetes_shap_stability.png")

# ============================================================
# 12. PERTURBATION TEST
# ============================================================

print("\nPerturbation Test...")
np.random.seed(42)
noise_levels   = [0.001, 0.01, 0.1]
shap_original  = get_shap_importance(rf, X_sample)
perturb_scores = []

for ns in noise_levels:
    X_noisy = pd.DataFrame(
        X_sample.values + np.random.normal(0, ns, X_sample.shape),
        columns=X_sample.columns
    )
    corr, _ = spearmanr(shap_original, get_shap_importance(rf, X_noisy))
    perturb_scores.append(corr)
    print(f"  noise={ns:.3f} → {corr:.4f}")

plt.figure(figsize=(7, 4))
plt.plot(noise_levels, perturb_scores, marker="s", color="#55A868", linewidth=2, markersize=8)
plt.xscale("log"); plt.ylim(0, 1.05)
plt.title("SHAP Stability Under Input Perturbation")
plt.xlabel("Noise Std (log)"); plt.ylabel("Spearman Corr")
plt.tight_layout()
plt.savefig("diabetes_shap_perturbation.png", dpi=100, bbox_inches="tight")
plt.close()
print("  Saved: diabetes_shap_perturbation.png")

# ============================================================
# 13. LIME STABILITY
# ============================================================

print("\nLIME Stability Analysis...")
lime_rankings = []
for seed in seeds:
    m = RandomForestClassifier(
        n_estimators=30, random_state=seed,
        class_weight="balanced", n_jobs=-1, max_depth=8
    )
    m.fit(X_train.values, y_train)
    exp = lime_explainer.explain_instance(
        data_row=X_sample.iloc[0].values,
        predict_fn=m.predict_proba,
        num_features=10,
        num_samples=200
    )
    wd  = dict(exp.as_list())
    vec = np.array([abs(wd.get(f, 0.0)) for f in X_sample.columns])
    lime_rankings.append(vec)

lime_rankings     = np.array(lime_rankings)
lime_correlations = [spearmanr(lime_rankings[i], lime_rankings[i+1])[0]
                     for i in range(len(lime_rankings)-1)]
print(f"  Scores  : {[round(c,4) for c in lime_correlations]}")
print(f"  Average : {np.mean(lime_correlations):.4f}")

# ============================================================
# 14. FINAL COMPARISON CHART
# ============================================================

avg_stabilities = [np.mean(correlations), np.mean(lime_correlations)]
print(f"\nFinal: SHAP={avg_stabilities[0]:.4f}  LIME={avg_stabilities[1]:.4f}")

plt.figure(figsize=(5, 4))
bars = plt.bar(["SHAP", "LIME"], avg_stabilities, color=["#4C72B0", "#DD8452"], width=0.4)
plt.ylim(0, 1.1)
plt.title("Avg Stability: SHAP vs LIME\n(Spearman Corr across seeds)")
plt.ylabel("Avg Spearman Correlation")
for bar, val in zip(bars, avg_stabilities):
    plt.text(bar.get_x()+bar.get_width()/2, val+0.02, f"{val:.4f}", ha="center", fontsize=12)
plt.tight_layout()
plt.savefig("diabetes_stability_comparison.png", dpi=100, bbox_inches="tight")
plt.close()
print("  Saved: diabetes_stability_comparison.png")

# ============================================================
# DONE
# ============================================================

print("\n" + "="*55)
print("  COMPLETED")
print("="*55)
output_files = [
    "diabetes_shap_beeswarm.png", "diabetes_shap_bar.png",
    "diabetes_lime_instance_0.png", "diabetes_lime_instance_1.png", "diabetes_lime_instance_2.png",
    "diabetes_lime_instance_0.html", "diabetes_lime_instance_1.html", "diabetes_lime_instance_2.html",
    "diabetes_shap_vs_lime.png", "diabetes_shap_stability.png",
    "diabetes_shap_perturbation.png", "diabetes_stability_comparison.png",
]
for f in output_files:
    print(f"  [{'✓' if os.path.exists(f) else '✗ MISSING'}] {f}")
