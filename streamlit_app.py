import streamlit as st
import joblib
import numpy as np

# Load the trained model directly
model = joblib.load('house_price_model_final.pkl')

st.title("🏠 House Price Prediction App")

# User input fields
area = st.number_input("Area (sq. ft)", min_value=100, max_value=10000, value=1200)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
floors = st.number_input("Floors", min_value=1, max_value=5, value=1)
year_built = st.number_input("Year Built", min_value=1900, max_value=2025, value=2010)
garage = st.selectbox("Garage", ["No", "Yes"])
location = st.selectbox("Location", ["Downtown", "Suburbs", "Countryside"])
condition = st.selectbox("Condition", ["Poor", "Average", "Good", "Excellent"])

# Encode categorical fields (same as in training)
garage_map = {'No': 0, 'Yes': 1}
location_map = {'Downtown': 0, 'Suburbs': 1, 'Countryside': 2}
condition_map = {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3}

if st.button("Predict Price"):
    features = np.array([[area, bedrooms, bathrooms, floors, year_built,
                          garage_map[garage], location_map[location], condition_map[condition]]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted House Price: ${prediction:,.2f}")
