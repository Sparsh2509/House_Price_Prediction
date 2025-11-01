import streamlit as st
import requests

st.title("House Price Prediction via API")

# Input fields
area = st.number_input("Area", 100, 1000)
bedrooms = st.number_input("Bedrooms", 1, 10)
bathrooms = st.number_input("Bathrooms", 1, 10)
floors = st.number_input("Floors", 0, 5)
year_built = st.number_input("Year Built", 1900, 2025)
garage = st.selectbox("Garage", ["yes", "no"])
location = st.selectbox("Location", ["country side", "down town", "city center", "suburb"])
condition = st.selectbox("Condition", ["excellent", "good", "fair", "poor"])

# Predict button
if st.button("Predict Price"):
    payload = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floors": floors,
        "year_built": year_built,
        "garage": garage.lower(),  # ensure lowercase
        "location": location.lower(),
        "condition": condition.lower()
    }

    # Call FastAPI
    url = "https://house-price-env.eba-nimenbn7.ap-south-1.elasticbeanstalk.com/predict"
    response = requests.post(url, json=payload)

    if response.status_code == 200:
        if "predicted_price" in response.json():
            st.success(f"Predicted Price: {response.json()['predicted_price']}")
        else:
            st.error(f"Error: {response.json()}")
    else:
        st.error(f"Error: {response.json()}")
