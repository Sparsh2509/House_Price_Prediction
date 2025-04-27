from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Load trained model
model = joblib.load('house_price_model_final.joblib')

# Define input data structure
class HouseData(BaseModel):
    Area: float
    Bedrooms: int
    Bathrooms: int
    Floors: int
    YearBuilt: int
    Garage: str         # 'Yes' or 'No'
    Location: str       # 'Downtown', 'Suburban', 'Urban', 'Rural'
    Condition: str      # 'Excellent', 'Good', 'Average', 'Bad'

# Helper function to encode categorical features
def encode_features(data: HouseData):
    # Garage encoding
    garage_encoded = 1 if data.Garage.lower() == "yes" else 0

    # Location encoding
    location_mapping = {
        "downtown": 0,
        "suburban": 1,
        "urban": 2,
        "rural": 3
    }
    location_encoded = location_mapping.get(data.Location.lower(), 1)  # Default: Suburban

    # Condition encoding
    condition_mapping = {
        "excellent": 0,
        "good": 1,
        "average": 2,
        "bad": 3
    }
    condition_encoded = condition_mapping.get(data.Condition.lower(), 2)  # Default: Average

    # Return features in correct order
    return [
        data.Area,
        data.Bedrooms,
        data.Bathrooms,
        data.Floors,
        data.YearBuilt,
        garage_encoded,
        location_encoded,
        condition_encoded
    ]

# Define prediction endpoint
@app.post("/predict")
def predict_price(data: HouseData):
    # Prepare the input data
    input_features = encode_features(data)

    # Create DataFrame for model
    input_df = pd.DataFrame([input_features], columns=[
        'Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Garage', 'Location', 'Condition'
    ])

    # Make prediction
    prediction = model.predict(input_df)[0]

    return {
        "Predicted House Price": round(prediction, 2)
    }
