# 🏦 CreditWise Loan Approval Prediction System

An intelligent loan approval system powered by Machine Learning for **SecureTrust Bank**.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://creditwise-loan-approval-prediction-system.streamlit.app)

---

## 🚀 Live Demo
👉 [Open Streamlit App](https://creditwise-loan-approval-prediction-system.streamlit.app)

---

## 📋 Problem Statement
SecureTrust Bank processes hundreds of loan applications daily. The previous manual process was:
- ⏱️ Time-consuming
- ⚖️ Biased and inconsistent
- ❌ Rejecting good customers / approving risky ones

This ML system automates the decision with **86.5% accuracy**.

---

## 🤖 Model Details

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 86.5%  |
| Precision | 78.3%  |
| Recall    | 77.1%  |
| F1 Score  | 77.7%  |

**Algorithm:** Gaussian Naive Bayes (best precision among tested models)  
**Models tested:** Logistic Regression, KNN, Naive Bayes

---

## 📁 Project Structure

```
CreditWise-Loan-Approval-Prediction-System/
│
├── app.py                    # Streamlit web app
├── train_model.py            # Model training script
├── loan_approval_data.csv    # Dataset (1000 applicants)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
│
└── models/                   # Saved model artifacts
    ├── nb_model.pkl          # Trained Naive Bayes model
    ├── scaler.pkl            # StandardScaler
    ├── ohe.pkl               # OneHotEncoder
    ├── le_edu.pkl            # LabelEncoder (Education)
    ├── le_target.pkl         # LabelEncoder (Target)
    └── feature_names.pkl     # Feature column names
```

---

## 🗂️ Dataset Features

| Feature             | Description                          |
|---------------------|--------------------------------------|
| Applicant_Income    | Monthly income of applicant          |
| Coapplicant_Income  | Monthly income of co-applicant       |
| Credit_Score        | Credit bureau score                  |
| DTI_Ratio           | Debt-to-Income ratio                 |
| Savings             | Savings balance                      |
| Collateral_Value    | Value of collateral                  |
| Loan_Amount         | Requested loan amount                |
| Loan_Term           | Duration in months                   |
| Employment_Status   | Salaried / Self-Employed / Business  |
| Education_Level     | Graduate / Postgraduate / Undergraduate |
| ...and more         |                                      |

---

## 🛠️ Local Setup

```bash
# 1. Clone the repo
git clone https://github.com/Jeel-Pipaliya/CreditWise-Loan-Approval-Prediction-System.git
cd CreditWise-Loan-Approval-Prediction-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (generates models/ folder)
python train_model.py

# 4. Run the app
streamlit run app.py
```

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → Select your repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** ✅

---

## 👨‍💻 Author
**Jeel Pipaliya**  
AIML Student 
📧 pipaliyajeelj@gmail.com
