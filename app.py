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

# Predict
predicted_pollutants = model.predict(input_encoded)[0]
pollutants = ['O2 (Oxygen)', 'NO3 (Nitrate)', 'NO2 (Nitrite)', 'SO4 (Sulfate)', 'PO4 (Phosphate)', 'CL (Chloride)']

# Create a DataFrame for output
output_df = pd.DataFrame({
    "Pollutant": pollutants,
    "Predicted Level (mg/L)": [round(val, 2) for val in predicted_pollutants]
})

# Display the output in a table with style
st.markdown(f"### 💧 Predicted Pollutant Levels for Station `{station_id}` in {year_input}")
st.dataframe(
    output_df.style.set_properties(**{
        'background-color': '#e6f2ff',
        'color': '#003366',
        'border-color': 'white',
        'text-align': 'center'
    }).set_table_styles([{
        'selector': 'th',
        'props': [('background-color', '#003366'), ('color', 'white'), ('text-align', 'center')]
    }]),
    use_container_width=True
)


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
