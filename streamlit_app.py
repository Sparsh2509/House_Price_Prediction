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
        "garage": garage,
        "location": location,
        "condition": condition
    }

    try:
        url = "http://house-price-env.eba-nimenbn7.ap-south-1.elasticbeanstalk.com/predict"
        response = requests.post(url, json=payload, timeout=10)  # set timeout

        # Check response
        if response.status_code == 200:
            data = response.json()
            if "predicted_price" in data:
                st.success(f"Predicted Price: {data['predicted_price']}")
            elif "error" in data:
                st.error(f"API Error: {data['error']} \nDetails: {data.get('details', '')}")
            else:
                st.error(f"Unexpected API response: {data}")
        else:
            st.error(f"HTTP Error {response.status_code}: {response.text}")

    except requests.exceptions.ConnectTimeout:
        st.error("Connection Timeout: The API server did not respond.")
    except requests.exceptions.RequestException as e:
        st.error(f"Request failed: {str(e)}")
