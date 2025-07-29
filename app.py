import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the models and scaler
@st.cache_resource
def load_models():
    rf_model = joblib.load('models/random_forest_model.pkl')
    lr_model = joblib.load('models/logistic_regression_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    feature_names = joblib.load('models/feature_names.pkl')
    return rf_model, lr_model, scaler, feature_names

rf_model, lr_model, scaler, feature_names = load_models()

# Set page title
st.title('Loan Approval Prediction')
st.write('Enter the following information to predict loan approval status')

# Create input form
col1, col2 = st.columns(2)

with col1:
    no_of_dependents = st.number_input('Number of Dependents', min_value=0, max_value=10, value=0)
    education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
    self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
    loan_term = st.number_input('Loan Term', min_value=1, max_value=360, value=12)
    cibil_score = st.number_input('CIBIL Score', min_value=300, max_value=900, value=700)

with col2:
    income_annum = st.number_input('Annual Income', min_value=0, value=100000)
    loan_amount = st.number_input('Loan Amount', min_value=0, value=50000)
    residential_assets_value = st.number_input('Residential Assets Value', min_value=0, value=0)
    commercial_assets_value = st.number_input('Commercial Assets Value', min_value=0, value=0)
    luxury_assets_value = st.number_input('Luxury Assets Value', min_value=0, value=0)
    bank_asset_value = st.number_input('Bank Asset Value', min_value=0, value=0)

# Create prediction button
if st.button('Predict Loan Approval'):
    # Prepare input data
    input_data = {
        'no_of_dependents': no_of_dependents,
        'income_annum': income_annum,
        'loan_amount': loan_amount,
        'loan_term': loan_term,
        'cibil_score': cibil_score,
        'residential_assets_value': residential_assets_value,
        'commercial_assets_value': commercial_assets_value,
        'luxury_assets_value': luxury_assets_value,
        'bank_asset_value': bank_asset_value,
        'education_ Graduate': 1 if education == 'Graduate' else 0,
        'education_ Not Graduate': 1 if education == 'Not Graduate' else 0,
        'self_employed_ No': 1 if self_employed == 'No' else 0,
        'self_employed_ Yes': 1 if self_employed == 'Yes' else 0
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_names]  # Ensure correct column order
    
    # Scale the input
    input_scaled = scaler.transform(input_df)
    
    # Make predictions
    rf_pred = rf_model.predict_proba(input_scaled)[0]
    lr_pred = lr_model.predict_proba(input_scaled)[0]
    
    # Display results
    st.write('### Prediction Results')
    col1, col2 = st.columns(2)
    
    with col1:
        st.write('Random Forest Model:')
        st.write(f'Approval Probability: {rf_pred[1]:.2%}')
        st.write(f'Rejection Probability: {rf_pred[0]:.2%}')
        
    with col2:
        st.write('Logistic Regression Model:')
        st.write(f'Approval Probability: {lr_pred[1]:.2%}')
        st.write(f'Rejection Probability: {lr_pred[0]:.2%}')
    
    # Final prediction based on Random Forest (more accurate model)
    final_prediction = 'Approved' if rf_pred[1] > 0.5 else 'Rejected'
    st.write('### Final Prediction')
    st.write(f'Loan Application Status: **{final_prediction}**')
    
    # Display important note
    st.info('Note: This is a prediction based on historical data and should be used as a reference only. Final loan approval decisions should be made by qualified financial professionals.')
