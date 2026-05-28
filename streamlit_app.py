
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model, preprocessor, and feature names
@st.cache_resource
def load_resources():
    model = joblib.load('random_forest_model.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
    feature_names = joblib.load('feature_names.joblib')
    return model, preprocessor, feature_names

model, preprocessor, feature_names = load_resources()

# Web App Title
st.title('Telecom Customer Churn Prediction')
st.write('Predict whether a customer will churn based on their services and account information.')

# Input Features
st.sidebar.header('Customer Information')

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    SeniorCitizen = st.sidebar.selectbox('Senior Citizen', ('No', 'Yes'))
    Partner = st.sidebar.selectbox('Partner', ('No', 'Yes'))
    Dependents = st.sidebar.selectbox('Dependents', ('No', 'Yes'))
    tenure = st.sidebar.slider('Tenure (months)', 0, 72, 12)
    PhoneService = st.sidebar.selectbox('Phone Service', ('No', 'Yes'))
    MultipleLines = st.sidebar.selectbox('Multiple Lines', ('No phone service', 'No', 'Yes'))
    InternetService = st.sidebar.selectbox('Internet Service', ('DSL', 'Fiber optic', 'No'))
    OnlineSecurity = st.sidebar.selectbox('Online Security', ('No', 'Yes', 'No internet service'))
    OnlineBackup = st.sidebar.selectbox('Online Backup', ('No', 'Yes', 'No internet service'))
    DeviceProtection = st.sidebar.selectbox('Device Protection', ('No', 'Yes', 'No internet service'))
    TechSupport = st.sidebar.selectbox('Tech Support', ('No', 'Yes', 'No internet service'))
    StreamingTV = st.sidebar.selectbox('Streaming TV', ('No', 'Yes', 'No internet service'))
    StreamingMovies = st.sidebar.selectbox('Streaming Movies', ('No', 'Yes', 'No internet service'))
    Contract = st.sidebar.selectbox('Contract', ('Month-to-month', 'One year', 'Two year'))
    PaperlessBilling = st.sidebar.selectbox('Paperless Billing', ('No', 'Yes'))
    PaymentMethod = st.sidebar.selectbox('Payment Method', ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'))
    MonthlyCharges = st.sidebar.slider('Monthly Charges', 18.0, 120.0, 50.0)
    TotalCharges = st.sidebar.slider('Total Charges', 0.0, 9000.0, 500.0)

    # Feature Engineering (must match notebook's feature engineering)
    AvgMonthlyCharge = TotalCharges / (tenure + 1) if tenure > 0 else MonthlyCharges

    # Convert Yes/No to binary for TotalServices, PremiumServices, StreamingServices
    # Standardize 'No internet service' to 'No' and 'No phone service' to 'No'
    OnlineSecurity_std = 'No' if OnlineSecurity == 'No internet service' else OnlineSecurity
    OnlineBackup_std = 'No' if OnlineBackup == 'No internet service' else OnlineBackup
    DeviceProtection_std = 'No' if DeviceProtection == 'No internet service' else DeviceProtection
    TechSupport_std = 'No' if TechSupport == 'No internet service' else TechSupport
    StreamingTV_std = 'No' if StreamingTV == 'No internet service' else StreamingTV
    StreamingMovies_std = 'No' if StreamingMovies == 'No internet service' else StreamingMovies
    MultipleLines_std = 'No' if MultipleLines == 'No phone service' else MultipleLines

    TotalServices = sum([
        1 if PhoneService == 'Yes' else 0,
        1 if MultipleLines_std == 'Yes' else 0,
        1 if OnlineSecurity_std == 'Yes' else 0,
        1 if OnlineBackup_std == 'Yes' else 0,
        1 if DeviceProtection_std == 'Yes' else 0,
        1 if TechSupport_std == 'Yes' else 0,
        1 if StreamingTV_std == 'Yes' else 0,
        1 if StreamingMovies_std == 'Yes' else 0
    ])

    PremiumServices = sum([
        1 if OnlineSecurity_std == 'Yes' else 0,
        1 if OnlineBackup_std == 'Yes' else 0,
        1 if DeviceProtection_std == 'Yes' else 0,
        1 if TechSupport_std == 'Yes' else 0
    ])

    StreamingServices = sum([
        1 if StreamingTV_std == 'Yes' else 0,
        1 if StreamingMovies_std == 'Yes' else 0
    ])

    def tenure_category(tenure):
        if tenure <= 12:
            return 'New'
        elif tenure <= 24:
            return 'Short-term'
        elif tenure <= 48:
            return 'Medium-term'
        else:
            return 'Long-term'
    TenureCategory = tenure_category(tenure)

    def monthly_charge_category(charge):
        if charge <= 40:
            return 'Low'
        elif charge <= 70:
            return 'Medium'
        else:
            return 'High'
    MonthlyChargeCategory = monthly_charge_category(MonthlyCharges)


    data = {'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'tenure': tenure,
            'PhoneService': PhoneService,
            'MultipleLines': MultipleLines_std,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity_std,
            'OnlineBackup': OnlineBackup_std,
            'DeviceProtection': DeviceProtection_std,
            'TechSupport': TechSupport_std,
            'StreamingTV': StreamingTV_std,
            'StreamingMovies': StreamingMovies_std,
            'Contract': Contract,
            'PaperlessBilling': PaperlessBilling,
            'PaymentMethod': PaymentMethod,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges,
            'AvgMonthlyCharge': AvgMonthlyCharge,
            'TotalServices': TotalServices,
            'PremiumServices': PremiumServices,
            'StreamingServices': StreamingServices,
            'TenureCategory': TenureCategory,
            'MonthlyChargeCategory': MonthlyChargeCategory}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Preprocess the input data
# The preprocessor expects columns in the order it was trained on.
# We need to ensure the input_df has all original feature columns + engineered features.
# Align columns based on the feature_names from training

# Create a dummy dataframe with all expected columns for the preprocessor
# This is crucial for OneHotEncoder to work correctly with unseen categories
# (though we used handle_unknown='ignore', it's best to have consistent columns)

# Extract original column names before one-hot encoding from the preprocessor
# This is a bit complex as ColumnTransformer returns an array. Need to reconstruct column names.

# Step 1: Create a mock DataFrame with all expected columns to fit preprocessor if needed
# or to ensure the input_df has the right structure.
# For now, we will create a DataFrame with the columns in `feature_names` that match original `X`.

# The `feature_names` stored are the final names after preprocessing. We need the raw feature names.
# For preprocessing in the app, we need the raw categorical and numerical column names.

# Let's assume original_categorical_cols and original_numerical_cols are available from the notebook training context
# Since they are not directly saved, we manually define them based on our EDA and preprocessing steps.

original_numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlyCharge', 'TotalServices', 'PremiumServices', 'StreamingServices']
original_categorical_cols = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TenureCategory', 'MonthlyChargeCategory']

# Ensure the input_df has the same columns as the training data used for the preprocessor
# This means dropping engineered features if they were not part of the original X before preprocessing
# and then ensuring order and types match.

# Re-order and select columns for preprocessing
input_df_processed = input_df[original_numerical_cols + original_categorical_cols]

# Apply preprocessing
# The preprocessor was fitted on X_train_raw, so it expects those columns.
# It's crucial that `input_df_processed` has the same columns in the same order as `X_train_raw` had.
# A robust way is to train the preprocessor on `X` (full dataset before split, without churn) and then save it.
# For now, assuming the column order is consistent.

processed_features = preprocessor.transform(input_df_processed)

# Display input features and prediction
st.subheader('Prediction')

# Make prediction
prediction = model.predict(processed_features)
prediction_proba = model.predict_proba(processed_features)

churn_status = 'Churn' if prediction[0] == 1 else 'No Churn'
st.write(f'The customer is likely to: **{churn_status}**')
st.write(f'Churn Probability: **{prediction_proba[0][1]:.2f}**')
st.write(f'No Churn Probability: **{prediction_proba[0][0]:.2f}**')

if prediction[0] == 1:
    st.warning('This customer is predicted to CHURN!')
else:
    st.success('This customer is predicted to NOT CHURN.')

