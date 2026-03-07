# XAI Stability Evaluation for Healthcare Machine Learning

## Overview

This project evaluates the **stability and reliability of Explainable AI (XAI) methods** applied to healthcare machine learning models. The study compares **SHAP, LIME, and Permutation Importance** explanations for a **Random Forest classifier** trained on the **UCI Cleveland Heart Disease dataset**.

The goal is to analyze how stable model explanations remain under **input perturbations and model retraining**, which is important for deploying AI systems in **clinical decision-making environments**.

---

# Key Objectives

* Train a **Random Forest classifier** for heart disease prediction.
* Generate explanations using **SHAP, LIME, and Permutation Importance**.
* Measure **stability of explanations** under small input perturbations.
* Compare explanation consistency using **Spearman rank correlation**.
* Evaluate whether explanations remain reliable for healthcare use.

---

# Dataset

**UCI Cleveland Heart Disease Dataset**

* Samples: **303 patients**
* Features: **13 clinical attributes**
* Task: **Binary classification (presence of heart disease)**

Example features:

| Feature  | Description                       |
| -------- | --------------------------------- |
| age      | Patient age                       |
| sex      | Gender                            |
| cp       | Chest pain type                   |
| trestbps | Resting blood pressure            |
| chol     | Serum cholesterol                 |
| thalach  | Maximum heart rate achieved       |
| exang    | Exercise induced angina           |
| oldpeak  | ST depression                     |
| slope    | Slope of peak exercise ST segment |
| ca       | Number of major vessels           |
| thal     | Thalassemia                       |

---

# Model

A **Random Forest classifier** was used due to its strong performance on structured healthcare data.

### Model Performance

| Metric   | Value |
| -------- | ----- |
| Accuracy | 0.885 |
| Recall   | 0.964 |
| F1 Score | 0.885 |

Dataset shape: **(303, 13)**

---

# Explainable AI Methods Evaluated

## SHAP (SHapley Additive Explanations)

* Game-theoretic feature attribution method
* Provides **consistent and additive explanations**

## LIME (Local Interpretable Model-agnostic Explanations)

* Local surrogate model for instance-level explanations
* Approximates predictions with linear models

## Permutation Feature Importance

* Measures global feature importance
* Based on accuracy drop after feature shuffling

---

# Top Features (Permutation Importance)

| Feature  | Importance |
| -------- | ---------- |
| cp       | 0.065574   |
| ca       | 0.036066   |
| thalach  | 0.032787   |
| trestbps | 0.022951   |
| exang    | 0.022951   |

Chest pain type (**cp**) was the most influential predictor for heart disease classification.

---

# Stability Analysis

Explanation stability was evaluated using **Spearman Rank Correlation** across multiple trials.

| Trial   | Stability Score |
| ------- | --------------- |
| Trial 1 | 0.9835          |
| Trial 2 | 0.9780          |
| Trial 3 | 0.9835          |
| Trial 4 | 0.9835          |

Average stability score: **0.982**

Perturbation stability: **1.0**

Results show **SHAP explanations are the most stable** under small input noise.

---

# Key Findings

* **SHAP achieved the highest explanation stability**
* **Permutation Importance provided consistent global insights**
* **LIME showed larger variation in feature rankings**
* Stable explanations are critical for **clinical trust and deployment**

---

# Tech Stack

* Python
* Scikit-learn
* SHAP
* LIME
* NumPy
* Pandas
* Matplotlib

---

# Project Structure

```id="project_structure"
XAI-Stability-Healthcare/
│
├── data/
│   └── cleveland_dataset.csv
│
├── notebooks/
│   └── experiment.ipynb
│
├── src/
│   ├── model_training.py
│   ├── shap_analysis.py
│   ├── lime_analysis.py
│   └── stability_test.py
│
├── results/
│   ├── tables
│   └── plots
│
├── research_paper.pdf
│
└── README.md
```

---

# How to Run the Project

### 1 Install dependencies

```id="install"
pip install -r requirements.txt
```

### 2 Run the training pipeline

```id="run"
python main.py
```

This will:

* Train the Random Forest model
* Generate SHAP and LIME explanations
* Compute feature importance
* Perform stability analysis

---

# Research Paper

The full research paper describing the methodology and experimental results is included in this repository.

---

# Future Improvements

* Evaluate additional models (XGBoost, LightGBM)
* Test more healthcare datasets
* Develop automated **explanation reliability benchmarks**
* Integrate clinician feedback evaluation

---

# Author

**Parth Sinha**
B.Tech Artificial Intelligence & Machine Learning
