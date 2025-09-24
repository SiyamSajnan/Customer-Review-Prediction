# 🛒 Customer Review Prediction

## 📖 Overview
This project focuses on predicting **customer review sentiment** (positive vs negative) using machine learning.  
By analyzing an e-commerce dataset that includes orders, payments, items, and reviews, the goal is to understand the key factors driving customer satisfaction and provide actionable insights for businesses.  

---

## 🎯 Business Problem
Customer reviews are critical indicators of product and service quality.  
Manually analyzing thousands of reviews is resource-intensive, so the project aims to build an **automated prediction system** to help business stakeholders:
- Identify dissatisfied customers early.
- Improve product/service strategies.
- Enhance customer experience.

---

## 🗂️ Dataset
The dataset was constructed by joining multiple relational tables:
- **Orders**
- **Customers**
- **Order Items**
- **Order Payments**
- **Order Reviews**

Key preprocessing steps included:
- Handling **missing values** and dropping irrelevant columns.
- **Outlier removal** (e.g., price > 300, unrealistic fulfillment times).
- **Feature engineering**: fulfillment time, estimated delivery time, etc.
- **Class imbalance correction** using downsampling.
- Type conversions with **Label Encoding** for categorical features.

---

## 🔍 Exploratory Data Analysis (EDA)
EDA revealed key patterns:
- Longer fulfillment times were associated with poorer reviews.
- Payment installments showed weak correlation with review outcomes.
- Significant **class imbalance** (positive reviews ≈ 4x negative).

📊 Visualizations:  
- Boxplots (price, fulfillment time, installments by review).  
- Distribution of categorical features (payment type, customer city).  
- Correlations between numeric features.

---

## 🤖 Machine Learning Models
Five models were implemented and compared:
- Logistic Regression  
- K-Nearest Neighbors (KNN)  
- Naïve Bayes  
- XGBoost  
- Random Forest  

**Evaluation Metrics:** Accuracy, Precision, Recall, F1-Score  
**Cross-validation:** 10-fold CV  
**Train-test split:** 80/20  

---

## ⚙️ Model Performance
| Model                | Accuracy | Std Dev |
|-----------------------|----------|---------|
| Logistic Regression   | 63.1%    | 0.64%   |
| KNN                  | 60.6%    | 0.85%   |
| Naïve Bayes          | 54.3%    | 0.77%   |
| XGBoost              | 66.9%    | 0.76%   |
| Random Forest        | **68.9%**| 0.52%   |

**Feature importance** highlighted:  
- Fulfillment Time  
- Payment Value  
- Price  
- Freight Value  
- Customer City  

---

## 🔧 Hyperparameter Tuning
Used **GridSearchCV** to optimize models:
- Logistic Regression → `C=10`  
- KNN → `n_neighbors=5`  
- Naïve Bayes → `alpha=0.1`  
- XGBoost → `lr=0.1, max_depth=7, n_estimators=200`  
- Random Forest → `n_estimators=100`

---

## 🧑‍🤝‍🧑 Ensemble Learning
A **Voting Classifier** was built using Logistic Regression, XGBoost, and Random Forest.  

**Best Result:**  
- Soft Voting without feature selection → **69.3% accuracy**

---

## 📊 Final Results
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| 0 (Negative) | 0.73 | 0.58 | 0.65 |
| 1 (Positive) | 0.65 | 0.78 | 0.71 |

**Overall Accuracy:** 68%  
The ensemble model provided the best balance across metrics.

---

## 🛠️ Tech Stack
- **Languages:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, Matplotlib, Seaborn  
- **Tools:** Jupyter Notebook  

---

## 📑 Documentation
- Clear data preparation pipeline.  
- EDA visualizations.  
- Model comparison and tuning details.  
- Confusion matrix and evaluation metrics.  

---

## 🚀 Future Improvements
- Experiment with **SMOTE** or advanced resampling for class imbalance.  
- Implement **deep learning models** (LSTM for text-based review data).  
- Deploy as an **interactive dashboard** in Streamlit or Power BI.

---

## 📌 Author
👤 Siyam Sajnan Chowdhury  
- MSc in Computer Science (AI)  
- MSc in Data Analytics & Design Thinking for Business

---
