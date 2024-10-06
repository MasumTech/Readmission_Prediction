import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the trained Gradient Boosting model
try:
    with open('Gradient_Boosting_diabetes_model.pkl', 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file not found. Make sure the Gradient_Boosting_diabetes_model.pkl file is in the same directory as this script.")

# Custom CSS for UI styling
st.markdown("""
    <style>
        .title-box {
            background-color: #008080;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        .result-box {
            background-color: #20B2AA;
            color: white;
            padding: 10px;
            border-radius: 5px;
        }
        .input-box {
            margin-bottom: 15px;
            border: 1px solid #008080;
            border-radius: 5px;
            padding: 10px;
        }
    </style>
    """, unsafe_allow_html=True)

# Display Title
st.markdown("""
    <div class="title-box">
        <h1>Diabetes Readmission Prediction Tool</h1>
    </div>
    """, unsafe_allow_html=True)

# Input section for user data
st.write("### Input Patient Information:")

# Collect user input with manual validation
discharge_disposition = st.selectbox('Discharge Disposition:', ['Home care', 'Transfer', 'Outpatient', 'Expired'])
age = st.slider('Age:', min_value=15, max_value=115, value=45)
time_in_hospital = st.slider('Duration of Hospital Stay (days):', min_value=1, max_value=30, value=7)
number_of_diagnoses = st.number_input('Number of Diagnoses:', min_value=1, max_value=50, value=5)
num_procedures = st.number_input('Number of Procedures:', min_value=0, max_value=50, value=1)
race = st.selectbox('Race:', ['Caucasian', 'African', 'American', 'Other'])
circulatory_diagnosis = st.selectbox('Circulatory Diagnosis:', ['Yes', 'No'])
number_inpatient_log1p = st.number_input('Number of inpatient events:', value=0, help="Range: 0-50")
admission_type = st.selectbox('Admission Source:', ['Referral', 'Transfer', 'Undefined', 'Newborn'])
num_medications = st.number_input('Number of Medications:', value=1, help="Range: 1-100")

# Input validation
valid_input = True

# Validate age
if not (15 <= age <= 115):
    st.warning('Age should be between 15 and 115.')
    valid_input = False

# Make prediction
if valid_input and st.button("Predict Readmission Probability"):
    # Prepare the user data for the model
    input_data = {
        'discharge_disposition_id': 1 if discharge_disposition == 'Home care' else 2 if discharge_disposition == 'Transfer' else 3 if discharge_disposition == 'Outpatient' else 4,
        'age': age,
        'time_in_hospital': time_in_hospital,
        'number_of_diagnoses': number_of_diagnoses,
        'num_procedures': num_procedures,
        'race': 1 if race == 'Caucasian' else 2 if race == 'AfricanAmerican' else 3,
        'circulatory_diagnosis': 1 if circulatory_diagnosis == 'Yes' else 0,
        'number_inpatient_log1p': number_inpatient_log1p,
        'admission_type_id': 1 if admission_type == 'Referral' else 2 if admission_type == 'Transfer' else 3 if admission_type == 'Undefined' else 4,
        'num_medications': num_medications
    }

    # Convert to DataFrame to match training format
    input_df = pd.DataFrame([input_data])
    
    # Use get_dummies to match the training data
    input_df = pd.get_dummies(input_df)

    # Reindex to ensure it has the same columns as the training set
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    # Prediction using Gradient Boosting model
    prediction_proba = model.predict_proba(input_df)
    prob_of_readmission = prediction_proba[0][1]  # Probability of being readmitted

    # Display prediction result
    st.markdown(f"""
        <div class="result-box">
            <h2>Predicted Readmission Probability:</h2>
            <h3>{prob_of_readmission * 100:.2f}%</h3>
        </div>
        """, unsafe_allow_html=True)

# Reset button functionality
if st.button("Reset"):
    st.session_state.clear()  # Clear session state
    st.stop()  # Stops the script here, which can effectively reset the display
