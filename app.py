import streamlit as st
import pandas as pd
import joblib

# Load saved model, encoder, and feature columns
rf_model = joblib.load('models/rf_model.pkl')
size_encoder = joblib.load('models/size_encoder.pkl')
feature_columns = joblib.load('models/feature_columns.pkl')

def predict_dog_size(traits_dict):
    input_df = pd.DataFrame([traits_dict])
    # Add missing columns with 0
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    # Reorder columns to match training
    input_df = input_df[feature_columns]
    prediction_encoded = rf_model.predict(input_df)[0]
    prediction_label = size_encoder.inverse_transform([prediction_encoded])[0]
    return prediction_label

st.title("🐶 Dog Size Predictor")

# Input sliders
lifespan = st.slider('Lifespan (years)', 5, 20, 10)
barking = st.slider('Barking (1-5)', 1, 5, 3)
shedding = st.slider('Shedding (1-5)', 1, 5, 3)
energy_level = st.slider('Energy Level (1-5)', 1, 5, 3)
trainability = st.slider('Trainability (1-5)', 1, 5, 3)
protectiveness = st.slider('Protectiveness (1-5)', 1, 5, 3)
friendliness = st.slider('Friendliness (1-5)', 1, 5, 3)

# Breed groups (one-hot) — set to 0 by default
breed_groups = [
    'breed_group_hound',
    'breed_group_non-sporting',
    'breed_group_sporting',
    'breed_group_toy',
    'breed_group_working'
]

if st.button("Predict Dog Size"):
    traits = {
        'lifespan': lifespan,
        'barking': barking,
        'shedding': shedding,
        'energy_level': energy_level,
        'trainability': trainability,
        'protectiveness': protectiveness,
        'friendliness': friendliness
    }

    # Add breed groups with 0
    for bg in breed_groups:
        traits[bg] = 0

    size = predict_dog_size(traits)
    st.success(f"The predicted dog size is: {size}")
