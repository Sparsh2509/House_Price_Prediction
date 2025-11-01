from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
from datetime import datetime
import joblib
import pandas as pd
import traceback

# Load trained model and label encoders
model = joblib.load('house_price_model_final.pkl')

# Initialize FastAPI
app = FastAPI()

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "Welcome to House Price Prediction API"}

# Input schema with validations
class HouseFeatures(BaseModel):
    area: float = Field(..., ge=100, le=1000, description="Area must be between 100 and 1000")
    bedrooms: int = Field(..., gt=0, description="Bedrooms must be greater than 0")
    bathrooms: int = Field(..., gt=0, description="Bathrooms must be greater than 0")
    floors: int = Field(..., ge=0, description="Floors must be 0 or greater")
    year_built: int = Field(..., description="Year built must not exceed current year")
    garage: str = Field(..., description="Garage must be 'yes' or 'no'")
    location: str = Field(..., description="Location must be one of 'country side', 'down town', 'city center', 'suburb'")
    condition: str = Field(..., description="Condition must be one of 'excellent', 'good', 'fair', 'poor'")

    @validator('year_built')
    def year_not_in_future(cls, v):
        current_year = datetime.now().year
        if v > current_year:
            raise ValueError(f"Year built cannot exceed {current_year}")
        return v

    @validator('garage')
    def garage_must_be_yes_or_no(cls, v):
        if v.lower() not in ['yes', 'no']:
            raise ValueError("Garage must be 'yes' or 'no'")
        return v.lower()

    @validator('location')
    def location_must_be_valid(cls, v):
        allowed_locations = ['country side', 'down town', 'city center', 'suburb']
        if v.lower() not in allowed_locations:
            raise ValueError(f"Location must be one of {allowed_locations}")
        return v.lower()

    @validator('condition')
    def condition_must_be_valid(cls, v):
        allowed_conditions = ['excellent', 'good', 'fair', 'poor']
        if v.lower() not in allowed_conditions:
            raise ValueError(f"Condition must be one of {allowed_conditions}")
        return v.lower()

# Define mapping dictionaries
garage_mapping = {'no': 0, 'yes': 1}
location_mapping = {'country side': 0, 'down town': 1, 'city center': 2, 'suburb': 3}
condition_mapping = {'excellent': 0, 'good': 1, 'fair': 2, 'poor': 3}

# Prediction endpoint
@app.post("/predict")
def predict_price(features: HouseFeatures):
    try:
        # Map categorical fields
        garage_encoded = garage_mapping[features.garage]
        location_encoded = location_mapping[features.location]
        condition_encoded = condition_mapping[features.condition]

        # Prepare input DataFrame in the exact order and names as training
        input_df = pd.DataFrame([[
            features.area,
            features.bedrooms,
            features.bathrooms,
            features.floors,
            features.year_built,
            garage_encoded,
            location_encoded,
            condition_encoded
        ]], columns=['Area','Bedrooms','Bathrooms','Floors','YearBuilt','Garage','Location','Condition'])

        # Predict
        prediction = model.predict(input_df)

        return {"predicted_price": round(prediction[0], 2)}

    except Exception as e:
        print("Prediction error:", e)
        traceback.print_exc()
        return {"error": "Prediction failed", "details": str(e)}
