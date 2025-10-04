# 💳 Credit Card Fraud Detection using Machine Learning

This project aims to **detect fraudulent credit card transactions** using **machine learning techniques**.  
The dataset used is highly imbalanced and comes from **Kaggle’s [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**.  
The project demonstrates data preprocessing, feature scaling, model training, and performance evaluation.

---

## 📂 Dataset
- **Source:** [Kaggle - Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data)
- **Description:**  
  This dataset contains transactions made by European cardholders in September 2013.  
  It presents **284,807 transactions**, among which **492 are fraudulent** (only **0.172%** of all transactions).  
  Each transaction has **30 features**, including:
  - 28 anonymized features (V1–V28) obtained via PCA
  - `Time` and `Amount` columns
  - `Class` column — target variable (1 = fraud, 0 = normal)

---

## 🧠 Objective
To build a machine learning model capable of identifying fraudulent transactions accurately while handling the **imbalanced nature** of the dataset.

---

## ⚙️ Technologies Used
- Python 🐍  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib, Seaborn  
- Jupyter Notebook

---

## 🧩 Steps Involved
1. **Data Preprocessing**
   - Handle missing values (if any)
   - Standardize features (`StandardScaler`)
   - Handle data imbalance (using undersampling/oversampling techniques)
2. **Model Building**
   - Algorithms used: `RandomForestClassifier`, `LogisticRegression`, etc.
3. **Evaluation Metrics**
   - Confusion Matrix  
   - Classification Report  
   - ROC Curve & AUC Score
4. **Visualization**
   - Fraud vs Non-Fraud distribution  
   - Correlation heatmap  

---

## 📊 Results
The Random Forest model achieved:
- **Accuracy:** ~99.9%  
- **Recall (Fraud class):** ~87%  
- **AUC Score:** ~0.97  

*(Note: Performance may vary slightly depending on random state and sampling method.)*

---

## 🚀 How to Run the Project
1. **Clone this repository**
   ```bash
   git clone https://github.com/<your-username>/credit-card-fraud-detection.git
   cd credit-card-fraud-detection

