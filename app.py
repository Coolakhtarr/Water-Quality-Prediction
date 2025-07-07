# Import all the necessary libraries
import pandas as pd
import numpy as np
import joblib
import pickle
import streamlit as st
import os
import gdown
# Download the model if not present
file_id = "18RJzu35vyuMgpcAE590u1IaDvHY3-SWq"
output = "pollution_model.pkl"
if not os.path.exists(output):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)
# trigger rebuild
# Load the model and structure
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")
# Streamlit page configuration
st.set_page_config(page_title="Water Pollutants Predictor", layout="centered")

# Page header
st.markdown("""
    <div style="background-color:#0d6efd;padding:1rem 2rem;border-radius:8px">
        <h2 style="color:white;text-align:center;">Water Pollutants Predictor</h2>
        <p style="color:white;text-align:center;">Enter the station and year to predict key water pollutants</p>
    </div>
""", unsafe_allow_html=True)

# Input section
st.markdown("### Enter Input Details")
col1, col2 = st.columns(2)

with col1:
    year_input = st.number_input("Year", min_value=2000, max_value=2100, value=2022)

with col2:
    station_id = st.text_input("Station ID", value='1')

# Predict button
if st.button("Predict"):
    if not station_id:
        st.warning("Please enter a valid Station ID.")
    else:
        input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O₂ (Oxygen)', 'NO₃ (Nitrate)', 'NO₂ (Nitrite)', 'SO₄ (Sulfate)', 'PO₄ (Phosphate)', 'Cl (Chloride)']

        # Display results
        st.markdown("### Prediction Results")
        for pollutant, val in zip(pollutants, predicted_pollutants):
            st.write(f"**{pollutant}:** {val:.2f} mg/L")

# Let's create an User interface
st.title("Water Pollutants Predictor")
st.write("Predict the water pollutants based on Year and Station ID")

# User inputs
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("Enter Station ID", value='1')

# To encode and then predict
if st.button('Predict'):
    if not station_id:
        st.warning('Please enter the station ID')
    else:
        # Prepare the input
        input_df = pd.DataFrame({'year': [year_input], 'id':[station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Align with model cols
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Predict
        predicted_pollutants = model.predict(input_encoded)[0]
        pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

        st.subheader(f"Predicted pollutant levels for the station '{station_id}' in {year_input}:")
        predicted_values = {}
        for p, val in zip(pollutants, predicted_pollutants):
            st.write(f'{p}:{val:.2f}')
