import pandas as pd
import streamlit as st
import joblib

# Page setup
st.set_page_config(page_title="Credit Risk Predictor", layout="centered")

st.title("Credit Risk Prediction App")
st.write("This app predicts whether a borrower is likely to be a good or bad credit risk.")

# Load saved model and columns
model = joblib.load("models/credit_risk_model.pkl")
model_columns = joblib.load("models/model_columns.pkl")

st.header("Enter borrower information")

# User inputs
duration = st.number_input("Loan duration in months", min_value=1, max_value=100, value=24)
credit_amount = st.number_input("Credit amount", min_value=100, max_value=50000, value=3000)
installment_commitment = st.number_input("Installment commitment", min_value=1, max_value=4, value=2)
residence_since = st.number_input("Residence since", min_value=1, max_value=4, value=2)
age = st.number_input("Age", min_value=18, max_value=100, value=35)
existing_credits = st.number_input("Existing credits", min_value=1, max_value=10, value=1)
num_dependents = st.number_input("Number of dependents", min_value=1, max_value=10, value=1)

checking_status = st.selectbox(
    "Checking status",
    ["<0", "0<=X<200", ">=200", "no checking"]
)

credit_history = st.selectbox(
    "Credit history",
    [
        "no credits/all paid",
        "all paid",
        "existing paid",
        "delayed previously",
        "critical/other existing credit"
    ]
)

purpose = st.selectbox(
    "Loan purpose",
    [
        "new car", "used car", "furniture/equipment", "radio/tv",
        "domestic appliance", "repairs", "education", "vacation",
        "retraining", "business", "other"
    ]
)

savings_status = st.selectbox(
    "Savings status",
    ["<100", "100<=X<500", "500<=X<1000", ">=1000", "no known savings"]
)

employment = st.selectbox(
    "Employment length",
    ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"]
)

personal_status = st.selectbox(
    "Personal status",
    ["male div/sep", "female div/dep/mar", "male single", "male mar/wid"]
)

other_parties = st.selectbox(
    "Other parties",
    ["none", "co applicant", "guarantor"]
)

property_magnitude = st.selectbox(
    "Property magnitude",
    ["real estate", "life insurance", "car", "no known property"]
)

other_payment_plans = st.selectbox(
    "Other payment plans",
    ["bank", "stores", "none"]
)

housing = st.selectbox(
    "Housing",
    ["rent", "own", "for free"]
)

job = st.selectbox(
    "Job",
    ["unemp/unskilled non res", "unskilled resident", "skilled", "high qualif/self emp/mgmt"]
)

own_telephone = st.selectbox(
    "Own telephone",
    ["none", "yes"]
)

foreign_worker = st.selectbox(
    "Foreign worker",
    ["yes", "no"]
)

# Create borrower dataframe
borrower = pd.DataFrame([{
    "checking_status": checking_status,
    "duration": duration,
    "credit_history": credit_history,
    "purpose": purpose,
    "credit_amount": credit_amount,
    "savings_status": savings_status,
    "employment": employment,
    "installment_commitment": installment_commitment,
    "personal_status": personal_status,
    "other_parties": other_parties,
    "residence_since": residence_since,
    "property_magnitude": property_magnitude,
    "age": age,
    "other_payment_plans": other_payment_plans,
    "housing": housing,
    "existing_credits": existing_credits,
    "job": job,
    "num_dependents": num_dependents,
    "own_telephone": own_telephone,
    "foreign_worker": foreign_worker
}])

# Encode borrower like training data
borrower_encoded = pd.get_dummies(borrower, drop_first=True)

# Match model columns
borrower_encoded = borrower_encoded.reindex(columns=model_columns, fill_value=0)

# Predict
if st.button("Predict Credit Risk"):
    probability = model.predict_proba(borrower_encoded)[:, 1][0]
    prediction = model.predict(borrower_encoded)[0]

    st.subheader("Prediction Result")
    st.write(f"Predicted probability of bad credit risk: **{probability:.2%}**")

    if prediction == 1:
        st.error("Prediction: BAD CREDIT RISK")
    else:
        st.success("Prediction: GOOD CREDIT RISK")