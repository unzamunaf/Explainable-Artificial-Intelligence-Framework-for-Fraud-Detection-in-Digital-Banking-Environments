# 🔍 Explainable AI Framework for Fraud Detection in Digital Banking

> A hybrid XAI framework combining **XGBoost + LightGBM ensemble** with **SHAP** and **LIME** explanations for transparent, regulatory-compliant fraud detection in digital banking environments.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?style=flat-square)
![LightGBM](https://img.shields.io/badge/LightGBM-4.0-brightgreen?style=flat-square)
![SHAP](https://img.shields.io/badge/SHAP-0.43-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## 📌 Overview

Financial fraud poses critical threats to digital banking as transaction complexity and fraudster sophistication escalate. While machine learning demonstrates powerful fraud detection capabilities, existing systems function as **black boxes** — providing predictions without explanations, creating regulatory compliance risks and eroding institutional trust.

This project proposes an **Explainable Artificial Intelligence (XAI) framework** that:
- Detects fraud with **high precision** using a gradient boosting ensemble
- Explains every flagged transaction in **human-understandable terms**
- Satisfies **GDPR Article 22** and **EU AI Act** transparency mandates
- Generates **auditable per-prediction reports** for compliance teams

---

## 🏆 Results

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| F1-Score | **0.9438** | > 0.90 | ✅ Achieved |
| AUC-ROC | **0.9942** | > 0.97 | ✅ Achieved |
| Accuracy | **97.60%** | — | ✅ |
| Precision | **94.82%** | — | ✅ |
| Recall | **93.94%** | — | ✅ |
| MCC | **0.9286** | — | ✅ |

### Model Comparison

| Model | F1-Score | AUC-ROC |
|-------|----------|---------|
| Logistic Regression | 0.0000 ❌ | 0.5279 |
| Decision Tree | 0.8526 | 0.9547 |
| Random Forest | 0.9008 | 0.9892 |
| XGBoost | 0.9416 | 0.9934 |
| LightGBM | 0.9424 | 0.9944 |
| **Ensemble (XGB+LGBM)** | **0.9438** ⭐ | **0.9942** ⭐ |

---

## 🗂️ Project Structure

```
xai-fraud-detection/
│
├── data/
│   └── fraud_dataset.csv          # Kaggle Credit Card Fraud dataset (sampled)
│
├── notebooks/
│   ├── 01_preprocessing.ipynb     # Data cleaning & feature engineering
│   ├── 02_model_training.ipynb    # Ensemble model training & evaluation
│   └── 03_explainability.ipynb    # SHAP & LIME explanation generation
│
├── src/
│   ├── preprocessing.py           # Label encoding, scaling, feature parsing
│   ├── feature_engineering.py     # 6 domain-specific engineered features
│   ├── model.py                   # XGBoost + LightGBM ensemble
│   ├── explainability.py          # SHAP TreeExplainer + LIME module
│   └── utils.py                   # Helper functions
│
├── outputs/
│   ├── shap_global_importance.png # SHAP bar plot
│   ├── shap_beeswarm.png          # SHAP beeswarm plot
│   ├── lime_explanation.png       # LIME local explanation
│   ├── confusion_matrix.png       # Confusion matrix
│   ├── roc_curves.png             # ROC curves for all models
│   └── pr_curves.png              # Precision-Recall curves
│
├── requirements.txt               # All dependencies
├── README.md                      # This file
└── LICENSE
```

---

## ⚙️ Five-Stage Pipeline

```
Stage 1: Data Collection
         └── 10,000 credit card transactions (Kaggle dataset)

Stage 2: Data Preprocessing
         └── Label encoding · RobustScaler · 80/20 stratified split

Stage 3: Feature Engineering
         └── Geo distance · Night flag · Age tier · Amount bin · Weekend flag · Card last-4

Stage 4: Ensemble Model Design
         └── XGBoost + LightGBM · Soft voting · SMOTE balancing · 5-fold CV

Stage 5: XAI Explainability Module
         └── SHAP Shapley values · LIME local surrogates · Audit log generation
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Google Colab or Jupyter Notebook (recommended)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/xai-fraud-detection.git
cd xai-fraud-detection

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
xgboost==2.0
lightgbm==4.0
scikit-learn==1.3
shap==0.43
lime==0.2
imbalanced-learn
pandas==2.1
numpy==1.25
matplotlib==3.8
seaborn==0.13
```

### Usage

```python
# 1. Run preprocessing
from src.preprocessing import preprocess_data
X_train, X_test, y_train, y_test = preprocess_data('data/fraud_dataset.csv')

# 2. Train ensemble model
from src.model import train_ensemble
model = train_ensemble(X_train, y_train)

# 3. Generate predictions
predictions = model.predict(X_test)

# 4. Generate SHAP explanations
from src.explainability import generate_shap_explanation
generate_shap_explanation(model, X_test)

# 5. Generate LIME explanation for a single transaction
from src.explainability import generate_lime_explanation
generate_lime_explanation(model, X_test, instance_index=0)
```

---

## 🔧 Feature Engineering

Six domain-specific features constructed from raw transaction data:

| Feature | Description | Type |
|---------|-------------|------|
| `geo_distance` | Euclidean distance between cardholder & merchant coordinates | Numerical |
| `amt_bin` | Transaction amount discretized into 4 quartile bins | Categorical |
| `night_txn` | Binary flag for transactions between 00:00–05:00 | Binary |
| `weekend_flag` | Binary flag for Saturday/Sunday transactions | Binary |
| `cc_last4` | Last 4 digits of credit card (integrity signal) | Numerical |
| `age_risk_tier` | Cardholder age segmented into 4 risk tiers | Categorical |

---

## 💡 Explainability

### SHAP — Global Feature Importance
SHAP (SHapley Additive exPlanations) computes feature contributions using cooperative game theory:

```
f(x) = φ₀ + Σᵢ φᵢ
```

Where `φ₀` is the base value and `φᵢ` is the contribution of feature `i`. **Transaction amount (`amt`)** is the dominant predictor with ~35% of total feature importance.

### LIME — Per-Transaction Explanation
LIME (Local Interpretable Model-agnostic Explanations) builds a local linear surrogate model for each flagged transaction, producing human-readable conditions such as:
> *"amt between 0.42 and 4.05 increases fraud probability"*

---

## 📊 Dataset

| Attribute | Details |
|-----------|---------|
| Source | Kaggle Credit Card Fraud Detection |
| Total Records | 10,000 (sampled from 555,719) |
| Fraud Cases | 2,145 (21.45%) |
| Legitimate Cases | 7,855 (78.55%) |
| Features | 20 (after preprocessing & engineering) |
| Missing Values | None |
| Class Imbalance | Handled via SMOTE (training set only) |
| Train / Test Split | 80% / 20% stratified |

---

## ⚖️ Regulatory Compliance

This framework is designed to satisfy:
- **GDPR Article 22** — Right to explanation for automated decisions
- **EU AI Act** — Algorithmic transparency and accountability
- **PCI-DSS** — Payment card industry data security standards

Every flagged transaction generates an auditable explanation report stored in compliance logs.

---

## 📝 Citation

If you use this work, please cite:

```
@article{munaf2024xai,
  title     = {An Explainable Artificial Intelligence Framework for Fraud Detection in Digital Banking Environments},
  author    = {Munaf, Unza},
  year      = {2024},
  school    = {FAST NUCES, Islamabad, Pakistan},
  course    = {Business Research and Data Mining}
}
```

---

## 👩‍💻 Author

**Unza Munaf**
Department of Finance and Technology
FAST NUCES, Islamabad, Pakistan
📧 i225502@isb.nu.edu.pk

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*⭐ If you found this project useful, please consider giving it a star!*
