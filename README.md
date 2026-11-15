# Credit Card Fraud Detection

---

## Abstract

The rapid growth of digital financial transactions has made credit card fraud detection a critical challenge for banks and customers alike. Fraudulent transactions are extremely rare — accounting for less than **0.2%** of all records — which creates a severe **class imbalance** problem. Traditional ML models often fail in this scenario, biasing toward the majority legitimate class and missing fraud cases.

This project implements a structured ML pipeline on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud), applying:

- **EDA** to understand feature distributions
- **SMOTE** and undersampling to handle class imbalance
- **Logistic Regression**, **Random Forest**, and **XGBoost** classifiers
- Evaluation using **Recall, Precision, F1-Score, and AUC-ROC**

---

## Results Summary

| Model | Precision | Recall | F1-Score | AUC-ROC |
|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.06 | **0.92** | 0.11 | — |
| **Random Forest** ✅ | **0.87** | 0.83 | **0.85** | — |
| XGBoost | 0.69 | 0.87 | — | — |

> **Random Forest** provides the best practical trade-off between catching fraud (Recall) and minimising false alarms (Precision).

---

## Project Structure

```
credit-card-fraud-detection/
│
├── data/
│   └── .gitkeep              ← place creditcard.csv here (see below)
│
├── notebooks/
│   └── credit_card_fraud_detection.ipynb   ← full step-by-step notebook
│
├── src/
│   ├── __init__.py
│   ├── eda.py                ← EDA plots (class dist, correlation, boxplots)
│   ├── preprocess.py         ← scaling, splitting, SMOTE
│   ├── train.py              ← LR / RF / XGBoost trainers
│   └── evaluate.py           ← metrics, confusion matrices, ROC curves
│
├── results/
│   ├── figures/              ← auto-generated plots (PNG)
│   └── model_comparison.csv  ← auto-generated metrics table
│
├── docs/
│   └── report.docx           ← original project report
│
├── main.py                   ← end-to-end CLI entry point
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup & Usage

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Create a virtual environment 
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the dataset
Download `creditcard.csv` from Kaggle:  
https://www.kaggle.com/mlg-ulb/creditcardfraud  

Place it in the `data/` folder:
```
data/creditcard.csv
```

### 5. Run the full pipeline
```bash
python main.py
```

Optional flags:
```bash
python main.py --data data/creditcard.csv   # custom dataset path
python main.py --no-smote                   # disable SMOTE
python main.py --skip-eda                   # skip EDA plots (faster CI run)
python main.py --test-size 0.25 --seed 0    # custom split & seed
```

### 6. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/credit_card_fraud_detection.ipynb
```

---

## Machine Learning Pipeline

```
Raw Dataset (creditcard.csv)
        │
        ▼
┌─────────────────────────┐
│  1. EDA                 │  class distribution, amount dist,
│                         │  correlation matrix, outlier boxplots
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  2. Preprocessing       │  StandardScaler on Amount,
│                         │  drop Time, stratified 80/20 split
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  3. SMOTE               │  applied only to training set
│  (imbalance correction) │  0.17% fraud → balanced 50/50
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  4. Model Training      │  Logistic Regression (baseline)
│                         │  Random Forest (ensemble)
│                         │  XGBoost (gradient boosting)
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│  5. Evaluation          │  Recall · Precision · F1 · AUC-ROC
│                         │  confusion matrices, ROC curves
└─────────────────────────┘
```

---

## Dataset

| Attribute | Value |
|---|---|
| Source | [Kaggle – MLG-ULB](https://www.kaggle.com/mlg-ulb/creditcardfraud) |
| Transactions | 284,807 |
| Features | 30 (V1–V28 via PCA, Amount, Time) |
| Fraud cases | 492 (0.172%) |
| Period | September 2013, European cardholders |

> **Note:** The dataset is **not included** in this repository due to its size and Kaggle's terms. Download it from the link above.

---

## References

1. Machine Learning Group – ULB, "Credit Card Fraud Detection Dataset," *Kaggle*, Sept. 2013.
2. N. V. Chawla et al., "SMOTE: Synthetic Minority Over-sampling Technique," *JAIR*, vol. 16, 2002.
3. L. Breiman, "Random Forests," *Machine Learning*, vol. 45, no. 1, 2001.
4. T. Chen & C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *KDD*, 2016.
5. F. Pedregosa et al., "Scikit-learn: Machine Learning in Python," *JMLR*, vol. 12, 2011.

---
