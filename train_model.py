"""
CreditWise Loan System - Model Training Script
Trains a Naive Bayes model (best precision) with feature engineering
and saves all artifacts needed for Streamlit deployment.
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

# ── 1. Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv("loan_approval_data.csv")
print(f"Dataset shape: {df.shape}")

# ── 2. Handle Missing Values ──────────────────────────────────────────────────
categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols   = df.select_dtypes(include=["float64", "int64"]).columns
numerical_cols   = [c for c in numerical_cols if c not in ["Applicant_ID"]]

num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])

cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])

# ── 3. Drop ID ────────────────────────────────────────────────────────────────
df = df.drop("Applicant_ID", axis=1)

# ── 4. Encode Target ──────────────────────────────────────────────────────────
le_target = LabelEncoder()
df["Loan_Approved"] = le_target.fit_transform(df["Loan_Approved"])  # Yes=1, No=0
print("Target classes:", le_target.classes_)

# ── 5. Encode Education Level ─────────────────────────────────────────────────
le_edu = LabelEncoder()
df["Education_Level"] = le_edu.fit_transform(df["Education_Level"])

# ── 6. One-Hot Encode Categoricals ───────────────────────────────────────────
ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
            "Property_Area", "Gender", "Employer_Category"]

ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
encoded = ohe.fit_transform(df[ohe_cols])
encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

# ── 7. Feature Engineering ────────────────────────────────────────────────────
df["DTI_Ratio_sq"]    = df["DTI_Ratio"] ** 2
df["Credit_Score_sq"] = df["Credit_Score"] ** 2

# ── 8. Train / Test Split ─────────────────────────────────────────────────────
X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
y = df["Loan_Approved"]
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ── 9. Scale ──────────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── 10. Train Naive Bayes ─────────────────────────────────────────────────────
nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)
y_pred = nb_model.predict(X_test_scaled)

print("\n✅ Naive Bayes (Best Model)")
print(f"  Precision : {precision_score(y_test, y_pred):.4f}")
print(f"  Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"  F1 Score  : {f1_score(y_test, y_pred):.4f}")
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"  Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# ── 11. Save Artifacts ────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

joblib.dump(nb_model,       "models/nb_model.pkl")
joblib.dump(scaler,         "models/scaler.pkl")
joblib.dump(ohe,            "models/ohe.pkl")
joblib.dump(le_edu,         "models/le_edu.pkl")
joblib.dump(le_target,      "models/le_target.pkl")
joblib.dump(feature_names,  "models/feature_names.pkl")

print("\n✅ All model artifacts saved to models/")
