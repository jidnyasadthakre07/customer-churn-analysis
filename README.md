# 📉 Customer Churn Analysis & Prediction
### Telecom Customer Retention Intelligence | End-to-End Data Analytics Project


## 📌 Project Overview

Customer churn is one of the most critical business problems in the telecom industry — acquiring a new customer costs **5x more** than retaining an existing one. This project builds a **complete end-to-end analytics solution** that identifies at-risk customers, uncovers the key drivers of churn, and delivers actionable business recommendations through an interactive Power BI dashboard.

This project simulates a real-world **Data Analyst role at a Big 4 consulting firm**, covering the full pipeline from raw data to business insight delivery.

---

## 🎯 Business Problem

> *"A telecom company is experiencing a 26.5% customer churn rate. The business needs to understand WHY customers are leaving and WHO is most likely to leave next — so the retention team can take proactive action."*

**Stakeholder Ask:**
- Identify the top factors driving customer churn
- Segment customers by churn risk level
- Predict which customers are likely to churn
- Recommend data-driven retention strategies

---

## 🗂️ Dataset

| Property | Details |
|---|---|
| **Source** | [Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| **Records** | 7,043 customers |
| **Features** | 21 columns (demographics, services, billing, contract) |
| **Target Column** | `Churn` (Yes / No) |
| **Class Imbalance** | 73.5% No Churn / 26.5% Churn |

### Key Columns Used

| Column | Type | Description |
|---|---|---|
| `tenure` | Numeric | Months the customer has been with the company |
| `Contract` | Categorical | Month-to-month / One year / Two year |
| `MonthlyCharges` | Numeric | Monthly billing amount ($18–$119) |
| `TotalCharges` | Numeric | Total amount billed (had data type issue — fixed) |
| `InternetService` | Categorical | DSL / Fiber Optic / No |
| `PaymentMethod` | Categorical | Electronic check / Credit card / etc. |
| `Churn` | Binary Target | Yes = churned, No = retained |

---

## 🛠️ Tools & Technologies

| Tool | Purpose |
|---|---|
| **Python** (pandas, numpy) | Data cleaning & EDA |
| **Scikit-learn** | Machine learning model |
| **Matplotlib / Seaborn** | Data visualizations |
| **Power BI** | Interactive business dashboard (4 pages) |
| **Excel** | Initial data exploration & validation |
| **Jupyter Notebook** | Development environment |
| **GitHub** | Version control & project showcase |

---

## 📁 Project Structure

```
customer-churn-analysis/
│
├── 📂 data/
│   ├── telco_churn.csv                  ← Raw dataset
│   ├── telco_churn_cleaned.csv          ← Cleaned dataset
│   └── churn_predictions.csv           ← ML model output for Power BI
│
├── 📂 notebooks/
│   ├── data_cleaning.ipynb           ← Data cleaning & fixing issues
│   ├── eda.ipynb                     ← Exploratory data analysis
│   └── model_building.ipynb         ← Random Forest model
│
├── 📂 dashboard/
│   └── Customer_Churn_Analysis.pbix    ← Power BI dashboard (4 pages)
│
├── 📂 visuals/
│   ├── churn_distribution.png
│   ├── churn_by_contract.png
│   ├── tenure_vs_churn.png
│   ├── charges_vs_churn.png
│   └── feature_importance.png
│
└── README.md
```

---

## 🔍 Step 1 — Data Cleaning

**Issues Found & Fixed:**

| Issue | Column | Fix Applied |
|---|---|---|
| Wrong data type (text instead of number) | `TotalCharges` | Converted using `pd.to_numeric()` |
| 11 blank values after type conversion | `TotalCharges` | Filled with median value |
| Irrelevant unique identifier | `customerID` | Dropped from analysis |
| Target column in text format | `Churn` (Yes/No) | Encoded to binary (1/0) |

```python
# Key cleaning steps
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
```

**Result:** Clean dataset with 7,043 rows and zero missing values ✅

---

## 📊 Step 2 — Exploratory Data Analysis (EDA)

### Key Business Insights Discovered

---

#### 🔑 Insight 1 — Contract Type is the Strongest Churn Driver

| Contract Type | Churn Rate |
|---|---|
| Month-to-month | **42.7%** 🔴 |
| One year | 11.3% 🟡 |
| Two year | 2.8% 🟢 |

> **Business Implication:** Month-to-month customers are **15x more likely** to churn than two-year contract customers. Converting even 20% of month-to-month customers to annual contracts could significantly reduce churn.

---

#### 🔑 Insight 2 — New Customers Leave Early

| Customer Group | Average Tenure |
|---|---|
| Churned Customers | **18.0 months** |
| Retained Customers | **37.6 months** |

> **Business Implication:** The first 12 months are the highest-risk period. Onboarding programs and early loyalty incentives should be prioritized for new customers.

---

#### 🔑 Insight 3 — Fiber Optic Customers Have High Churn

| Internet Service | Churn Rate |
|---|---|
| Fiber Optic | **41.9%** 🔴 |
| DSL | 19.0% 🟡 |
| No Internet | 7.4% 🟢 |

> **Business Implication:** Fiber optic customers are likely churning due to pricing or service quality dissatisfaction. A pricing or service quality review is recommended.

---

#### 🔑 Insight 4 — Payment Method Reveals Risk

| Payment Method | Churn Rate |
|---|---|
| Electronic Check | **45.3%** 🔴 |
| Mailed Check | 19.1% |
| Bank Transfer (Auto) | 16.7% |
| Credit Card (Auto) | 15.2% 🟢 |

> **Business Implication:** Electronic check users churn at 3x the rate of auto-payment users. Incentivizing customers to switch to automatic payments could reduce churn significantly.

---

#### 🔑 Insight 5 — Higher Charges = Higher Churn Risk

| Customer Group | Avg Monthly Charges |
|---|---|
| Churned | **$74.44** |
| Retained | **$61.27** |

> **Business Implication:** Churned customers pay $13 more per month on average — suggesting price sensitivity. Targeted discount offers for high-paying at-risk customers could improve retention.

---

#### 🔑 Insight 6 — Senior Citizens Are Vulnerable

| Segment | Churn Rate |
|---|---|
| Senior Citizens | **41.7%** |
| Non-Senior Citizens | 23.6% |

> **Business Implication:** Senior citizens churn at nearly double the rate. Dedicated support programs and simplified plan offerings may help retain this segment.

---

## 🤖 Step 3 — Machine Learning Model

### Model: Random Forest Classifier

**Why Random Forest?**
- Handles both numeric and categorical features well
- Resistant to overfitting compared to single decision trees
- Provides feature importance scores — critical for business explanation
- Works well with imbalanced datasets like this one (26.5% churn)

### Model Performance

| Metric | Score |
|---|---|
| **ROC-AUC Score** | **0.87** |
| Accuracy | 80% |
| Precision (Churn) | 67% |
| Recall (Churn) | 55% |

> **Why ROC-AUC over Accuracy?** The dataset is imbalanced (73.5% No / 26.5% Yes). A model that predicts "No Churn" for everyone would be 73.5% accurate — but completely useless. ROC-AUC measures the model's ability to distinguish churners from non-churners, making it the correct metric here.

### Top 10 Churn Drivers (Feature Importance)

| Rank | Feature | Business Meaning |
|---|---|---|
| 1 | `tenure` | How long they've been a customer |
| 2 | `MonthlyCharges` | How much they pay per month |
| 3 | `TotalCharges` | Cumulative billing amount |
| 4 | `Contract_Two year` | Long-term contract = lower churn |
| 5 | `InternetService_Fiber optic` | Fiber customers at higher risk |
| 6 | `PaymentMethod_Electronic check` | Electronic check = high risk |
| 7 | `Contract_One year` | Annual contract = moderate protection |
| 8 | `OnlineSecurity_Yes` | Security add-on reduces churn |
| 9 | `TechSupport_Yes` | Support users less likely to leave |
| 10 | `PaperlessBilling_Yes` | Digital billing users churn more |

```python
# Model building summary
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("ROC-AUC Score:", roc_auc_score(y_test, y_pred))  # 0.87
```

---

## 📈 Step 4 — Power BI Dashboard

The dashboard contains **4 pages**, each answering a specific business question:

| Page | Title | Key Visuals |
|---|---|---|
| Page 1 | Customer Demographics Analysis | KPI cards, Churn by Gender, Senior Citizen analysis, Tenure groups |
| Page 2 | Service Subscription Analysis | Churn by Internet Service, Add-on service impact, Contract filters |
| Page 3 | Contract & Billing Insights | Churn by Contract type, Payment method analysis, Revenue breakdown |
| Page 4 | Churn Prediction & Key Drivers | Feature importance, Revenue by Churn, Multi-variable churn rate |

**DAX Measure Created:**
```dax
Churn Rate Numeric = DIVIDE(
    COUNTROWS(FILTER('cca data set', 'cca data set'[Churn] = "Yes")),
    COUNTROWS('cca data set')
)
```

**Custom Column Created:**
```dax
Tenure Group = 
SWITCH(TRUE(),
    'cca data set'[tenure] <= 12, "0-12 Months (New)",
    'cca data set'[tenure] <= 24, "13-24 Months",
    'cca data set'[tenure] <= 48, "25-48 Months",
    "49-72 Months (Loyal)"
)
```

---

## 💡 Business Recommendations

Based on the analysis, here are 5 data-driven recommendations for the retention team:

1. **🎯 Target Month-to-Month Customers** — Offer loyalty discounts or perks to convert them to annual contracts. Even a 20% conversion could reduce overall churn by ~8%.

2. **🆕 Strengthen New Customer Onboarding** — Customers in their first 12 months are highest risk. Implement a 90-day check-in program with dedicated support.

3. **💳 Incentivize Auto-Payment Enrollment** — Electronic check users churn at 45.3%. Offering a small monthly discount for switching to auto-pay could reduce this significantly.

4. **📡 Investigate Fiber Optic Pricing** — With 41.9% churn, fiber optic customers are clearly dissatisfied. A pricing audit or service quality review is recommended.

5. **👴 Create Senior Citizen Retention Program** — With 41.7% churn in this segment, simplified plans and dedicated phone support could improve retention.

---

## 🚀 How to Run This Project

### 1. Clone the Repository
```bash
git clone https://github.com/jidnyasadthakre07/customer-churn-analysis.git
cd customer-churn-analysis
```

### 2. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Run the Notebooks in Order
```bash
jupyter notebook
```
Open and run:
- `notebooks/01_data_cleaning.ipynb`
- `notebooks/02_eda.ipynb`
- `notebooks/03_model_building.ipynb`

### 4. View the Dashboard
Open `dashboard/Customer_Churn_Analysis.pbix` in **Power BI Desktop**

---

## 📚 Key Learnings

- Real-world datasets always have data quality issues — `TotalCharges` being stored as text is a classic example
- Class imbalance is common in churn datasets — always use ROC-AUC, not just accuracy
- Business context matters more than model complexity — a clear insight beats a complex model
- Feature importance from ML models validates what EDA already suggested
- Dashboards should tell a story, not just display numbers

---

## 👤 Author

**Jidnyasa Thakre**

📧 jidnyasathakre3@gmail.com

🔗 [LinkedIn](https://www.linkedin.com/in/jidnyasathakre/)

🐙 [GitHub](https://github.com/jidnyasadthakre07)

---
