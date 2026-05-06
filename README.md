# 🔍 Credit Card Fraud Detection — End-to-End ML Pipeline

![Python](https://img.shields.io/badge/Python-3.10-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-orange) ![XGBoost](https://img.shields.io/badge/XGBoost-1.7-green) ![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

## 📋 Project Summary

Built an end-to-end binary classification pipeline to detect fraudulent credit card transactions using a highly imbalanced real-world dataset (0.17% fraud rate). The pipeline includes rigorous statistical feature analysis, outlier removal exclusively from the fraud class, mild SMOTE oversampling, and a data-driven three-tier decision framework. The final model achieves **88.78% Recall** and **82.08% Precision** — catching nearly 9 out of every 10 fraudsters while maintaining high confidence in each flag.

---

## 💼 Business Context

Credit card fraud causes billions in annual losses globally. The challenge is not just catching fraud — it's doing so without disrupting legitimate customers. Two costly errors exist:

- **False Negatives** — missed fraud → direct financial loss, repeated fraudster activity, chargeback costs, reputational damage
- **False Positives** — blocked legitimate transactions → customer friction, lost revenue, potential churn

This model powers a **three-tier real-time decision system:**
- Transactions above the auto-block threshold are **blocked immediately**
- Transactions in the mid-range are **routed to human analysts** for manual review
- Transactions below the auto-approve threshold are **approved in real time**

---

## 📊 Dataset Overview

| Field | Detail |
|---|---|
| **Source** | Kaggle — Credit Card Fraud Detection |
| **Size** | 284,807 transactions |
| **Fraud Cases** | 492 (0.17% of total) |
| **Features** | V1-V28 (PCA-transformed) + Amount + Time |
| **Target Variable** | Class (0 = Legitimate, 1 = Fraud) |
| **Class Distribution** | 99.83% Legitimate / 0.17% Fraud |

> **Note:** Features V1-V28 are PCA-transformed to protect cardholder privacy. Traditional feature interpretation is not possible — feature importance is assessed using Random Forest's Mean Decrease in Impurity.

---

## 🔬 Methodology

### 1. Exploratory Data Analysis
- **T-Tests** on all features vs Class — all 28 features statistically significant (p < 0.05)
- Features ranked by T-Statistic magnitude — top predictors: V17, V14, V12, V10, V16
- **KDE plots** and **Boxplots** for top 5 features — confirmed clean separation between fraud and legitimate distributions
- **Amount analysis** — fraud clusters at significantly lower amounts (card-testing behavior)
- **Hour feature engineering** — fraud spikes significantly between 0-6am

### 2. Data Preprocessing
- Engineered `Hour` from raw `Time` feature — captures time-of-day fraud patterns
- Dropped raw `Time` — seconds elapsed has no business meaning
- **Outlier removal from fraud class only** — V10, V12, V14 using IQR × 1.5 — less than 5% of fraud cases removed
- **80/20 stratified train/test split** with `random_state=42`
- **RobustScaler** applied to Amount and Hour — chosen for resistance to extreme outliers
- Two imbalance handling approaches: **Class Weights** and **Mild SMOTE** (`sampling_strategy=0.1`)

### 3. Modeling
- Trained **3 models × 2 imbalance approaches = 6 configurations**
- Models: Logistic Regression, Random Forest, XGBoost
- Primary evaluation metrics: **ROC-AUC** and **F2-Score** (Recall-weighted)

### 4. Threshold Optimization
- **F2-optimal threshold (0.38)** — maximizes fraud detection for operational use
- **Auto-block threshold (0.74)** — 95%+ Precision for automatic blocking
- Three-tier decision framework built from data — not arbitrary cutoffs

---

## 🔍 Key EDA Findings

- **V14 and V17** show the strongest separation between fraud and legitimate transactions — fraudulent transactions cluster in significantly negative values while legitimate transactions cluster around zero
- **Fraud spikes between 0-6am** — a period of minimal customer oversight and reduced monitoring — suggesting deliberate timing by fraudsters
- **Fraudulent transactions cluster at lower amounts** — consistent with two strategies: card-testing (small transactions to verify card validity) and deliberate low-value theft to avoid detection systems
- **Extreme outliers in V10, V12, V14** within the fraud class — suggesting highly unpredictable fraud behavior patterns requiring careful preprocessing
- **0.17% fraud rate** means a naive model predicting everything as legitimate achieves 99.83% accuracy — making accuracy completely meaningless as an evaluation metric

---

## 🤖 Model Performance

### All Models Comparison

| Model | Approach | ROC-AUC | Recall | Precision | F2-Score |
|---|---|---|---|---|---|
| Logistic Regression | SMOTE | 0.9682 | 0.8878 | 0.3346 | 0.6672 |
| Logistic Regression | Class Weights | 0.9719 | 0.9082 | 0.0513 | 0.2093 |
| **Random Forest** | **SMOTE** | **0.9732** | **0.8469** | **0.8830** | **0.8539** |
| Random Forest | Class Weights | 0.9529 | 0.7449 | 0.9605 | 0.7799 |
| XGBoost | SMOTE | 0.9725 | 0.8469 | 0.6640 | 0.8027 |
| XGBoost | Class Weights | 0.9720 | 0.8469 | 0.4392 | 0.7143 |

### Final Model Results
**Model:** Random Forest + SMOTE + Threshold Optimization

```
ROC-AUC:    0.9732
Recall:     0.8878   ← Catches 89% of real fraudsters  
Precision:  0.8208   ← 82% of flags are genuine fraud
F2-Score:   0.8735
```

### Three-Tier Decision Framework

| Tier | Threshold | Precision | Recall | Action |
|---|---|---|---|---|
| **Auto-Block** | > 0.74 | 96% | 73% | Block transaction immediately |
| **Manual Review** | 0.38 — 0.74 | — | — | Route to fraud analyst team |
| **Auto-Approve** | < 0.38 | — | — | Approve in real time |

---

## 🏆 Top Predictive Features

| Rank | Feature | Direction | Meaning |
|---|---|---|---|
| 1 | V14 | Negative → Higher Fraud Risk | Strongest behavioral fraud signal |
| 2 | V17 | Negative → Higher Fraud Risk | Second strongest discriminator |
| 3 | V10 | Negative → Higher Fraud Risk | Consistent across EDA and model |
| 4 | V12 | Negative → Higher Fraud Risk | Clear KDE separation confirmed |
| 5 | V11 | Positive → Higher Fraud Risk | Non-linear signal captured by RF |

> V2 emerged as important to the model despite not ranking in the top 5 by T-Statistic — suggesting it carries non-linear fraud signal that T-Tests cannot detect.

---

## 💡 Business Recommendations

1. **Night-Hours Enhanced Monitoring** — Implement two-factor authentication and real-time SMS alerts for transactions occurring between 0-6am. This targets the period where fraud activity is significantly elevated while legitimate transaction volume decreases.

2. **Sequential Transaction Detection** — Develop a rule that flags repeated small-amount transactions from the same card within a short time window. This directly targets card-testing behavior — where fraudsters use micro-transactions to verify card validity before larger theft.

3. **V14 and V17 Priority Scoring** — Prioritize these features as primary inputs to the real-time scoring system. Transactions where these PCA components fall into significantly negative values should be automatically assigned high-risk scores — confirmed by both T-Tests and Random Forest importance.

4. **Three-Tier Real-Time System** — Deploy the data-driven decision framework: auto-block above 0.74 (96% Precision), route to human analysts between 0.38-0.74, auto-approve below 0.38. This balances fraud detection against customer experience and operational cost.

5. **Proactive Customer Communication** — Establish an immediate SMS or push notification protocol for blocked transactions. With 18% of flagged transactions potentially legitimate, rapid communication reduces customer friction and prevents unnecessary card cancellations.

---

## 🛠️ Tech Stack

```
Python          3.10
pandas          2.0
numpy           1.24
scikit-learn    1.3
imbalanced-learn 0.11  (SMOTE)
xgboost         1.7
matplotlib      3.7
seaborn         0.12
scipy           1.11
```

---

## ▶️ How To Run

```bash
# 1. Clone the repository
git clone https://github.com/Mustafa-Mirghani/Fraud-Detection.git

# 2. Navigate to project directory
cd Fraud-Detection

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Jupyter Notebook
jupyter notebook notebooks/fraud_detection_final.ipynb

# 5. Run all cells
```

> **Note:** Dataset available on [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). Download and place in `data/` folder before running.

---

## 📈 Future Improvements


- [ ] Test ensemble stacking — combining Random Forest stability with XGBoost power
- [ ] Continuous learning pipeline — retrain periodically on fresh transaction data
- [ ] Explore deep learning approaches with larger datasets

---

## 👤 Author

**Mustafa Mirghani**

🐙 [GitHub](https://github.com/Mustafa-Mirghani)

---

