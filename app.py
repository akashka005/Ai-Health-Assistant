import streamlit as st
import numpy as np
import joblib

# Load trained models
log_model = joblib.load('log_model.pkl')
rf_model = joblib.load('rf_model.pkl')

st.title('AI-Powered Personal Health Assistant')

# Input fields for user data
age = st.slider('Age', 1, 100)
sex = st.selectbox('Sex', ['Male', 'Female'])
cp = st.selectbox('Chest Pain Type', [0, 1, 2, 3])
trestbps = st.slider('Resting Blood Pressure (mm Hg)', 80, 200)
chol = st.slider('Cholesterol (mg/dL)', 100, 400)
thalach = st.slider('Maximum Heart Rate Achieved', 60, 220)
oldpeak = st.slider('ST Depression', 0.0, 6.0)

# Prepare input for prediction
input_data = np.array([[age, 1 if sex == 'Male' else 0, cp, trestbps, chol, thalach, oldpeak]])

# Predict using Logistic Regression model
if st.button('Predict with Logistic Regression'):
    prediction = log_model.predict(input_data)
    st.write('Prediction: Heart Disease' if prediction[0] == 1 else 'Prediction: No Heart Disease')

# Predict using Random Forest model
if st.button('Predict with Random Forest'):
    prediction_rf = rf_model.predict(input_data)
    st.write('Prediction: Heart Disease' if prediction_rf[0] == 1 else 'Prediction: No Heart Disease')
