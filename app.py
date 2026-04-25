"""
CreditWise Loan Approval Prediction System
Streamlit App — SecureTrust Bank
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditWise — Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { margin: 0; font-size: 2.4rem; }
    .main-header p  { margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem; }

    .result-approved {
        background: linear-gradient(135deg, #1b5e20, #2e7d32);
        color: white; padding: 2rem; border-radius: 12px;
        text-align: center; font-size: 1.6rem; font-weight: bold;
        box-shadow: 0 4px 20px rgba(46,125,50,0.4);
    }
    .result-rejected {
        background: linear-gradient(135deg, #b71c1c, #c62828);
        color: white; padding: 2rem; border-radius: 12px;
        text-align: center; font-size: 1.6rem; font-weight: bold;
        box-shadow: 0 4px 20px rgba(198,40,40,0.4);
    }
    .metric-card {
        background: #f8faff; border: 1px solid #dce8ff;
        border-radius: 10px; padding: 1rem 1.2rem;
        text-align: center;
    }
    .stButton>button {
        width: 100%; background: #1e3a5f; color: white;
        border: none; border-radius: 8px; padding: 0.75rem;
        font-size: 1.1rem; font-weight: 600; cursor: pointer;
        transition: background 0.2s;
    }
    .stButton>button:hover { background: #2d6a9f; }
    .section-title {
        font-size: 1.05rem; font-weight: 700; color: #1e3a5f;
        border-left: 4px solid #2d6a9f; padding-left: 0.6rem;
        margin: 1.2rem 0 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    base = os.path.dirname(__file__)
    model_dir = os.path.join(base, "models")
    return {
        "model":         joblib.load(os.path.join(model_dir, "nb_model.pkl")),
        "scaler":        joblib.load(os.path.join(model_dir, "scaler.pkl")),
        "ohe":           joblib.load(os.path.join(model_dir, "ohe.pkl")),
        "le_edu":        joblib.load(os.path.join(model_dir, "le_edu.pkl")),
        "le_target":     joblib.load(os.path.join(model_dir, "le_target.pkl")),
        "feature_names": joblib.load(os.path.join(model_dir, "feature_names.pkl")),
    }

artifacts = load_artifacts()

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🏦 CreditWise Loan Approval System</h1>
    <p>SecureTrust Bank — Intelligent ML-Powered Loan Decision Engine</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — Model Info ──────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/bank.png", width=80)
    st.markdown("### 📊 Model Performance")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy",  "86.5%")
    col2.metric("Precision", "78.3%")
    col1.metric("Recall",    "77.1%")
    col2.metric("F1 Score",  "77.7%")
    st.markdown("---")
    st.markdown("**Algorithm:** Gaussian Naive Bayes")
    st.markdown("**Dataset:** 1,000 applicants")
    st.markdown("**Features:** 20 input variables")
    st.markdown("---")
    st.markdown("### 🔑 Key Predictors")
    st.markdown("- 💳 Credit Score")
    st.markdown("- 💰 Applicant Income")
    st.markdown("- 📉 DTI Ratio")
    st.markdown("- 💵 Savings Balance")
    st.markdown("- 🏠 Collateral Value")

# ── Input Form ────────────────────────────────────────────────────────────────
st.markdown("## 📝 Applicant Information")
st.markdown("Fill in all fields below and click **Predict** to get an instant decision.")

with st.form("loan_form"):
    # ── Personal Information ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    age            = c1.number_input("Age",             min_value=18, max_value=75, value=30)
    gender         = c2.selectbox("Gender",             ["Male", "Female"])
    marital_status = c3.selectbox("Marital Status",     ["Married", "Single"])
    dependents     = c4.number_input("Dependents",      min_value=0, max_value=10, value=0)

    education_level   = c1.selectbox("Education Level",   ["Graduate", "Not Graduate"])
    employment_status = c2.selectbox("Employment Status", ["Salaried", "Self-Employed", "Business"])
    employer_category = c3.selectbox("Employer Category", ["Govt", "Private", "Self"])
    property_area     = c4.selectbox("Property Area",     ["Urban", "Semi-Urban", "Rural"])

    # ── Financial Information ─────────────────────────────────────────────────
    st.markdown('<div class="section-title">💰 Financial Information</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    applicant_income    = c1.number_input("Applicant Income (₹)",    min_value=0, value=50000, step=1000)
    coapplicant_income  = c2.number_input("Co-Applicant Income (₹)", min_value=0, value=0,     step=1000)
    savings             = c3.number_input("Savings Balance (₹)",     min_value=0, value=100000, step=5000)
    collateral_value    = c4.number_input("Collateral Value (₹)",    min_value=0, value=500000, step=10000)

    st.markdown('<div class="section-title">📊 Credit Profile</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    credit_score   = c1.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=700,
        help="Typical bureau score range is 300 to 900.",
    )
    dti_ratio      = c2.number_input(
        "DTI Ratio (%)",
        min_value=0.0,
        max_value=100.0,
        value=30.0,
        step=0.5,
        help="Debt-to-Income ratio. Lower is generally better.",
    )
    existing_loans = c3.number_input("Existing Loans", min_value=0, max_value=10, value=0)

    # ── Loan Details ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏦 Loan Details</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    loan_amount  = c1.number_input("Loan Amount (₹)",    min_value=10000, value=500000, step=10000)
    loan_term    = c2.number_input("Loan Term (months)", min_value=6,     value=120,    step=6)
    loan_purpose = c3.selectbox("Loan Purpose",          ["Home", "Education", "Personal", "Business"])

    submitted = st.form_submit_button("🔍 Predict Loan Approval")

# ── Prediction ────────────────────────────────────────────────────────────────
if submitted:
    try:
        # ── Build raw row ─────────────────────────────────────────────────────
        raw = pd.DataFrame([{
            "Applicant_Income":   applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Age":                age,
            "Dependents":         dependents,
            "Credit_Score":       credit_score,
            "Existing_Loans":     existing_loans,
            "DTI_Ratio":          dti_ratio,
            "Savings":            savings,
            "Collateral_Value":   collateral_value,
            "Loan_Amount":        loan_amount,
            "Loan_Term":          loan_term,
            "Education_Level":    education_level,
            "Employment_Status":  employment_status,
            "Marital_Status":     marital_status,
            "Loan_Purpose":       loan_purpose,
            "Property_Area":      property_area,
            "Gender":             gender,
            "Employer_Category":  employer_category,
        }])

        # ── Encode Education ──────────────────────────────────────────────────
        le_edu = artifacts["le_edu"]
        try:
            raw["Education_Level"] = le_edu.transform(raw["Education_Level"])
        except ValueError:
            raw["Education_Level"] = 0

        # ── One-Hot Encode ────────────────────────────────────────────────────
        ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                    "Property_Area", "Gender", "Employer_Category"]
        ohe = artifacts["ohe"]
        encoded = ohe.transform(raw[ohe_cols])
        encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols))
        raw = pd.concat([raw.drop(columns=ohe_cols).reset_index(drop=True),
                         encoded_df.reset_index(drop=True)], axis=1)

        # ── Feature Engineering ───────────────────────────────────────────────
        raw["DTI_Ratio_sq"]    = raw["DTI_Ratio"] ** 2
        raw["Credit_Score_sq"] = raw["Credit_Score"] ** 2
        raw = raw.drop(columns=["Credit_Score", "DTI_Ratio"], errors="ignore")

        # ── Align Columns ─────────────────────────────────────────────────────
        feature_names = artifacts["feature_names"]
        for col in feature_names:
            if col not in raw.columns:
                raw[col] = 0
        raw = raw[feature_names]

        # ── Scale & Predict ───────────────────────────────────────────────────
        scaled     = artifacts["scaler"].transform(raw)
        prediction = artifacts["model"].predict(scaled)[0]
        proba      = artifacts["model"].predict_proba(scaled)[0]
        label      = artifacts["le_target"].inverse_transform([prediction])[0]

        approved_prob = proba[list(artifacts["le_target"].classes_).index("Yes")] * 100
        rejected_prob = 100 - approved_prob

        # ── Display Result ────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("## 📋 Prediction Result")

        col_res, col_prob = st.columns([1, 1])
        with col_res:
            if label == "Yes":
                st.markdown(f"""
                <div class="result-approved">
                    ✅ LOAN APPROVED<br>
                    <span style="font-size:1rem;font-weight:400">
                        Applicant qualifies based on ML analysis
                    </span>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-rejected">
                    ❌ LOAN REJECTED<br>
                    <span style="font-size:1rem;font-weight:400">
                        Applicant does not meet approval criteria
                    </span>
                </div>""", unsafe_allow_html=True)

        with col_prob:
            st.markdown("#### Confidence Scores")
            st.progress(int(approved_prob), text=f"✅ Approval Probability: {approved_prob:.1f}%")
            st.progress(int(rejected_prob), text=f"❌ Rejection Probability: {rejected_prob:.1f}%")

        # ── Application Summary ───────────────────────────────────────────────
        st.markdown("### 📊 Application Summary")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💳 Credit Score",      credit_score)
        c2.metric("💰 Monthly Income",    f"₹{applicant_income:,}")
        c3.metric("📉 DTI Ratio",         f"{dti_ratio}%")
        c4.metric("🏦 Loan Amount",       f"₹{loan_amount:,}")
        c1.metric("💵 Savings",           f"₹{savings:,}")
        c2.metric("🏠 Collateral",        f"₹{collateral_value:,}")
        c3.metric("📋 Existing Loans",    existing_loans)
        c4.metric("⏳ Loan Term",         f"{loan_term} months")

        st.info("⚠️ This prediction is generated by an ML model and should be reviewed by a loan officer before final disbursement.")

    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.exception(e)
