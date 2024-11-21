import streamlit as st
import pandas as pd
import numpy as np
from lime import lime_tabular
import sklearn
import joblib
import json

print(sklearn.__version__)
model = joblib.load("C:/Users/91822/Desktop/college/ML lab/Project_2/loan_approval_model.pkl")
with open('metadata.json', 'r') as f:
    meta = json.load(f)
col = meta['scale_col']
columns = meta['columns']
noisy_X_train = joblib.load("C:/Users/91822/Desktop/college/ML lab/Project_2/noisy_X_train.pkl")
ct = joblib.load("C:/Users/91822/Desktop/college/ML lab/Project_2/column_transformer.joblib")
ss = joblib.load("C:/Users/91822/Desktop/college/ML lab/Project_2/standard_scaler.joblib")
gender_le=joblib.load("C:/Users/91822/Desktop/college/ML lab/Project_2/gender_encoder.joblib")
married_le=joblib.load("C:/Users/91822/Desktop/college/ML lab/Project_2/gender_encoder.joblib")
education_le=joblib.load("C:/Users/91822/Desktop/college/ML lab/Project_2/education_encoder.joblib")
self_emp_le=joblib.load("C:/Users/91822/Desktop/college/ML lab/Project_2/self_employed_encoder.joblib")
credit_le=joblib.load("C:/Users/91822/Desktop/college/ML lab/Project_2/credit_history_encoder.joblib")
loan_le=joblib.load("C:/Users/91822/Desktop/college/ML lab/Project_2/loan_status_encoder.joblib")
explainer = lime_tabular.LimeTabularExplainer(
        noisy_X_train.values,
        feature_names=list(noisy_X_train.columns),
        class_names=['No loan', 'Loan will be approved'],  
        mode='classification'
    )

st.title("Loan Approval Prediction")

with st.form("loan_risk_form"):
    ApplicantIncome = st.number_input("Applicant Income", min_value=0, value=300)
    CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0, value=0)
    Loanamount = st.number_input("Loan Amount", min_value=0, value=100)
    Loanperiod = st.number_input("Loan Period (in days)", min_value=0, value=360)
    dependents = st.selectbox("Dependents", options=['0', '1', '2', '3+'])
    property_area = st.selectbox("Property Area", options=['Rural', 'Semiurban', 'Urban'])
    gender = st.selectbox("Gender", options=['Male', 'Female'])
    married = st.selectbox("Marital Status", options=['Married', 'Single'], format_func=lambda x: "Yes" if x == 'Married' else "No")
    Education = st.selectbox("Education", options=['Graduate', 'Not Graduate'])
    Self_Employed = st.selectbox("Self Employed", options=['Yes', 'No'])
    Credit_History = st.selectbox("Credit History", options=[1, 0], format_func=lambda x: 1 if x == 1 else 0)  
    submitted = st.form_submit_button("Predict Risk")
    y='Y'

if submitted:
    pred_data=pd.DataFrame([[dependents,property_area,gender,married,Education,Self_Employed,ApplicantIncome,CoapplicantIncome,Loanamount,Loanperiod,Credit_History,y]],columns=['Dependents','Property_Area','Gender','Married', 'Education', 'Self_Employed', 'ApplicantIncome','CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term','Credit_History','Loan_Status'])

    pred_data['Gender']=gender_le.fit_transform(pred_data['Gender'])
    pred_data['Married']=married_le.fit_transform(pred_data['Married'])
    pred_data['Education']=education_le.fit_transform(pred_data['Education'])
    pred_data['Self_Employed']=self_emp_le.fit_transform(pred_data['Self_Employed'])
    pred_data['Credit_History']=credit_le.fit_transform(pred_data['Credit_History'])
    pred_data['Loan_Status']=loan_le.fit_transform(pred_data['Loan_Status'])
    
    temp=ct.transform(pred_data)
    pred_data=pd.DataFrame(temp,columns=columns)
    pred_data[col]=ss.transform(pred_data[col])

    x=pred_data.drop('Loan_Status',axis=1)
    y = model.predict(x)[0]

    # Get prediction probability
    prediction_proba = model.predict_proba(x)[0]

    st.write(f"Predicted Class: {'No approval' if y == 0 else 'Will be approved'}")
    st.write("Prediction Probabilities:", prediction_proba)

    # Explain prediction with LIME
    instance = x.values.reshape(-1)
    explanation = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=len(pred_data.columns)
    )

    st.write("Local Feature Importance (LIME):")
    for feature_name, weight in explanation.as_list():
        st.write(f"{feature_name}: {weight:.4f}")

    fig = explanation.as_pyplot_figure()
    st.pyplot(fig)
