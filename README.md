# 🏦 AI Fraud Detection System — Public Sector Bank
### Machine Learning Proof of Concept

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/bank-fraud-detection/blob/main/Bank_Fraud_Detection_PoC.ipynb)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)

---

## 📋 Overview

This project is a **Proof of Concept (PoC)** for an AI-powered Fraud Detection System designed for Indian public sector banks. It demonstrates a complete end-to-end Machine Learning pipeline — from synthetic data generation to real-time prediction and explainability — all runnable on **Google Colab** with zero setup.

### 🎯 Problem Statement
Fraudulent banking transactions cause significant financial losses to both banks and customers. This PoC demonstrates how Machine Learning can detect fraud with high accuracy in real time, enabling banks to:
- **Block** fraudulent transactions before settlement
- **Alert** customers and bank officers instantly
- **Explain** why a transaction was flagged (regulatory compliance)

---

## 🌟 Key Features

| Feature | Description |
|---------|-------------|
| 🗄️ **Synthetic Data** | 100,000 realistic Indian bank transactions (NEFT, RTGS, IMPS, UPI, ATM) |
| 🤖 **3 ML Models** | Logistic Regression, Random Forest, XGBoost |
| ⚖️ **Class Imbalance** | SMOTE oversampling for realistic 5% fraud rate |
| 📊 **Full Evaluation** | AUC-ROC, Precision-Recall, Confusion Matrix, Threshold Analysis |
| 🔎 **Explainability** | SHAP values — know *why* each transaction was flagged |
| 🚀 **Prediction API** | Single-transaction prediction with risk level + recommended action |
| 💼 **Business Impact** | Estimated financial savings analysis |
| 💾 **Model Export** | Saved as `.pkl` for deployment |

---

## 📁 Project Structure

```
bank-fraud-detection/
│
├── 📓 Bank_Fraud_Detection_PoC.ipynb   # Main Colab notebook (run this!)
├── 📄 README.md                         # This file
├── 📄 requirements.txt                  # Python dependencies
├── 📄 LICENSE                           # MIT License
│
├── data/
│   └── 📄 data_schema.md               # Transaction data schema reference
│
└── outputs/                             # Generated after running notebook
    ├── fraud_detection_model.pkl        # Trained XGBoost model
    ├── model_metadata.json              # Model version & metrics
    ├── eda_analysis.png                 # EDA visualizations
    ├── correlation_matrix.png           # Feature correlations
    ├── model_performance.png            # ROC & PR curves
    ├── confusion_matrices.png           # All models side-by-side
    ├── threshold_analysis.png           # Optimal threshold plot
    ├── shap_importance.png              # Feature importance (SHAP)
    ├── shap_summary.png                 # SHAP beeswarm plot
    └── shap_waterfall.png               # Single prediction explanation
```

---

## 🚀 Quick Start — Run on Google Colab

### Option 1: One-Click (Recommended)
1. Click the **"Open in Colab"** badge at the top of this README
2. Select **Runtime → Run all** (`Ctrl+F9`)
3. All dependencies install automatically ✅

### Option 2: Manual Upload
1. Download `Bank_Fraud_Detection_PoC.ipynb`
2. Go to [colab.research.google.com](https://colab.research.google.com)
3. Click **File → Upload notebook**
4. Select the downloaded file
5. Click **Runtime → Run all**

> ⏱️ **Estimated runtime:** 5–10 minutes on Colab free tier (T4 GPU not required)

---

## 🧠 ML Pipeline

```
Raw Transaction Data
        ↓
Feature Engineering (20 features)
        ↓
Train/Test Split (80/20, stratified)
        ↓
SMOTE (handle class imbalance)
        ↓
┌─────────────────────────────────┐
│  Model 1: Logistic Regression   │ ← Interpretable baseline
│  Model 2: Random Forest         │ ← Ensemble, robust
│  Model 3: XGBoost (⭐ Best)     │ ← Highest AUC-ROC
└─────────────────────────────────┘
        ↓
Threshold Optimization
        ↓
SHAP Explainability
        ↓
Real-Time Prediction Interface
```

---

## 📊 Fraud Risk Signals

The model detects these key fraud indicators:

| Signal | Weight | Description |
|--------|--------|-------------|
| 🌙 Night-time transaction | High | Transactions between 10 PM – 6 AM |
| 👤 New payee | High | First-time money transfer to this account |
| 📱 Device change | High | Transaction from a new/unknown device |
| 🌐 IP mismatch | Critical | Login IP doesn't match registered location |
| ⚡ High velocity | High | Many transactions in last 24 hours |
| 💸 Amount anomaly | Medium | Amount far higher than 30-day average |
| 🏦 Account age | Medium | New accounts flagged more aggressively |
| 🗺️ Location distance | Medium | Far from home/registered branch |
| 🔐 Failed logins | Medium | Multiple failed login attempts |

---

## 🎯 Model Performance (on synthetic data)

| Model | AUC-ROC | Avg Precision |
|-------|---------|---------------|
| Logistic Regression | ~0.92 | ~0.65 |
| Random Forest | ~0.98 | ~0.90 |
| **XGBoost** ⭐ | **~0.99** | **~0.95** |

---

## 🔮 Sample Prediction Output

```python
result = predict_fraud({
    'amount': 185000,
    'transaction_type': 'IMPS',
    'hour': 3,          # 3 AM
    'is_new_payee': 1,
    'device_change': 1,
    'ip_mismatch': 1,
    ...
})
```

```
═══════════════════════════════════════════════════════
  FRAUD DETECTION RESULT
═══════════════════════════════════════════════════════
  transaction_id      : TXN_TEST_001
  amount              : ₹1,85,000.00
  fraud_probability   : 0.9423
  fraud_probability_pct: 94.2%
  is_fraud            : True
  prediction          : 🚨 FRAUDULENT
  risk_level          : CRITICAL
  recommended_action  : BLOCK & ESCALATE
  key_risk_factors    : ['Night-time transaction', 'New payee',
                         'Device changed', 'IP mismatch',
                         'High transaction velocity']
═══════════════════════════════════════════════════════
```

---

## ⚙️ Requirements

```txt
xgboost>=1.7.0
shap>=0.42.0
imbalanced-learn>=0.10.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
plotly>=5.13.0
```

All installed automatically inside the notebook.

---

## 🏛️ Production Deployment Roadmap

For real-world bank deployment, the following steps are recommended:

1. **Data Pipeline**
   - Connect to Core Banking System (Finacle / TCS BaNCS) via REST API
   - Replace synthetic data with 12–24 months historical transaction data
   - Ensure PII masking / data anonymization per IT Act 2000

2. **Model Deployment**
   - Containerize with Docker
   - Deploy as REST microservice (FastAPI / Flask)
   - Target latency: < 100ms per transaction decision

3. **Real-Time Streaming**
   - Apache Kafka for transaction event stream
   - Apache Flink / Spark Streaming for real-time scoring

4. **Alert System**
   - SMS alert to customer (via NPCI / bank SMS gateway)
   - Officer dashboard for flagged transaction review
   - Auto-block API integration with payment gateway

5. **Compliance & Governance**
   - Follow RBI Master Direction on Digital Payment Security
   - Maintain audit trail of all model decisions (SHAP logs)
   - Model Explainability report for regulators
   - DPDP Act 2023 compliance for customer data

6. **Monitoring & Retraining**
   - Track model drift with Evidently AI
   - Monthly retraining with latest fraud patterns
   - A/B testing for model version rollout

---

## 📌 Disclaimer

> This project uses **synthetically generated data** for demonstration purposes only. No real customer data or actual bank transactions are used. In production, comply with all applicable RBI guidelines, IT Act 2000, and DPDP Act 2023 before deploying AI systems in banking.

---

## 📜 License

MIT License — free to use, modify, and distribute with attribution.

---

## 🤝 Contributing

Pull requests welcome! For major changes, open an issue first to discuss what you'd like to change.

---

*Built as a Proof of Concept for AI-powered Fraud Detection in Indian Public Sector Banking*
