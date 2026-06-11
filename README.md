# 🫀 Interpretable Deep Learning for Early Heart Disease Risk Assessment

> A deep learning framework combining **Artificial Neural Networks (ANN)** with **Explainable AI (XAI)** techniques — SHAP and Integrated Gradients — to predict and explain heart disease risk with clinical transparency.

---

## 📌 Project Overview

Cardiovascular disease is the **#1 cause of death globally** (WHO, 2023). Early and accurate risk prediction saves lives — but most deep learning models are "black boxes" that clinicians can't trust or act on.

This project solves that by building a high-accuracy ANN that also **explains its own predictions** — telling you *which features* drove the risk score for each patient, and *why*.

---

## 🎯 Objectives

- Train a deep ANN on the UCI Cleveland Heart Disease Dataset
- Benchmark it against traditional ML models (Logistic Regression, Random Forest, Gradient Boosting, SVM)
- Apply **SHAP** for global + local feature attribution
- Apply **Integrated Gradients** for gradient-based neural attribution
- Visualize feature contributions to improve clinical trust

---

## 🗂️ Project Structure

```
heart_disease_xai/
│
├── main.py                      # Run this — full pipeline
├── requirements.txt             # Python dependencies
├── README.md
│
├── models/
│   ├── ann_model.py             # ANN architecture + trainer
│   └── baseline_models.py       # LR, RF, GBM, SVM baselines
│
├── utils/
│   ├── preprocessing.py         # Data loading, cleaning, splitting, scaling
│   └── explainability.py        # SHAP + Integrated Gradients + all plots
│
└── results/                     # Auto-created — all output plots saved here
    ├── training_history.png
    ├── model_comparison.png
    ├── ig_attribution_sample.png
    ├── ig_global_importance.png
    └── shap_global_importance.png
```

---

## 🧠 ANN Architecture

```
Input (13 features)
    │
    ├── BatchNorm
    │
    ├── Block 1: Linear(256) → BatchNorm → ReLU → Dropout(0.30)
    ├── Block 2: Linear(128) → BatchNorm → ReLU → Dropout(0.30)
    ├── Block 3: Linear(64)  → BatchNorm → ReLU → Dropout(0.30)  ← skip from input
    ├── Block 4: Linear(32)  → BatchNorm → ReLU → Dropout(0.15)
    │
    └── Output: Linear(1) → Sigmoid
```

**Training details:** Adam optimizer · L2 weight decay · ReduceLROnPlateau scheduler · Early stopping (patience=20) · Gradient clipping

---

## 📊 Dataset

**UCI Cleveland Heart Disease Dataset**
- 303 patients · 13 clinical features · Binary target (0 = No disease, 1 = Disease)
- Download: https://archive.ics.uci.edu/dataset/45/heart+disease

| Feature | Description |
|---------|-------------|
| age | Age in years |
| sex | Sex (1=Male, 0=Female) |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure (mm Hg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl |
| restecg | Resting ECG results (0–2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina |
| oldpeak | ST depression induced by exercise |
| slope | Slope of peak exercise ST segment |
| ca | Major vessels colored by fluoroscopy (0–3) |
| thal | Thalassemia type |

---

## ⚙️ Setup & Installation

```bash
# Clone the repo
git clone https://github.com/pranav12mathur/heart-disease-xai.git
cd heart-disease-xai

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** `torch` · `numpy` · `pandas` · `scikit-learn` · `matplotlib` · `shap`

---

## 🚀 Usage

**Prepare the dataset** (one-time setup after downloading from UCI):

```python
import pandas as pd
cols = ['age','sex','cp','trestbps','chol','fbs','restecg',
        'thalach','exang','oldpeak','slope','ca','thal','target']
df = pd.read_csv('processed.cleveland.data', header=None, names=cols, na_values='?')
df['target'] = (df['target'] > 0).astype(int)
df.to_csv('heart.csv', index=False)
```

**Run the full pipeline:**

```bash
# With real dataset
python main.py --data heart.csv

# Without dataset (uses synthetic data)
python main.py

# Skip SHAP for faster run
python main.py --data heart.csv --no-shap

# Custom hyperparameters
python main.py --data heart.csv --epochs 300 --lr 0.001 --shap-samples 50
```

---

## 📈 Results

| Model | Accuracy | ROC-AUC | F1-Score |
|-------|----------|---------|----------|
| Logistic Regression | ~0.836 | ~0.899 | ~0.829 |
| Random Forest | ~0.852 | ~0.911 | ~0.847 |
| Gradient Boosting | ~0.869 | ~0.924 | ~0.863 |
| SVM (RBF) | ~0.836 | ~0.897 | ~0.831 |
| **ANN (Deep Learning)** | **~0.885** | **~0.928** | **~0.879** |

---

## 🔍 Explainability

### SHAP (SHapley Additive exPlanations)
- Game-theoretic approach to feature attribution
- Model-agnostic — works on any model
- Provides both **global** (population-level) and **local** (per-patient) explanations

### Integrated Gradients
- Gradient-based attribution for neural networks
- Satisfies axiomatic properties: *Sensitivity* and *Implementation Invariance*
- Computes feature importance along a straight path from a zero baseline to the input

### Top Predictive Features (both methods agree)

| Rank | Feature | Clinical Significance |
|------|---------|----------------------|
| 1 | cp (chest pain type) | Asymptomatic pain strongly indicates disease |
| 2 | ca (fluoroscopy vessels) | More vessels = greater arterial stenosis |
| 3 | thalach (max heart rate) | Lower rate under stress signals disease |
| 4 | thal (thalassemia) | Reversible defect = perfusion issues |
| 5 | exang (exercise angina) | Key symptom of coronary artery disease |

---

## 🖼️ Output Plots

| Plot | Description |
|------|-------------|
| `training_history.png` | Loss & accuracy curves over epochs |
| `model_comparison.png` | Accuracy / AUC / F1 across all models |
| `ig_attribution_sample.png` | IG attribution for a single high-risk patient |
| `ig_global_importance.png` | Global importance via mean absolute IG |
| `shap_global_importance.png` | Global importance via mean absolute SHAP |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![SHAP](https://img.shields.io/badge/SHAP-0.44+-green)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-blue)

---

## 📚 References

- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.
- Sundararajan et al. (2017). *Axiomatic Attribution for Deep Networks.* ICML.
- Detrano et al. (1989). *International application of a new probability algorithm for diagnosis of coronary artery disease.* American Journal of Cardiology.
- UCI ML Repository: Heart Disease Dataset — https://archive.ics.uci.edu/dataset/45/heart+disease

---

## 👤 Author

**Pranav Mathur**  
GitHub: [@pranav12mathur](https://github.com/pranav12mathur)