# main.py
from fastapi import FastAPI
from pydantic import BaseModel, Field, validator
from datetime import datetime
import joblib
import pandas as pd
import logging

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ----------------- Load Model -----------------
model = joblib.load("house_price_model_final.pkl")

# ----------------- FastAPI App -----------------
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Prediction API!"}

# ----------------- Input Schema -----------------
class HouseFeatures(BaseModel):
    area: float = Field(..., ge=100, le=1000)
    bedrooms: int = Field(..., gt=0)
    bathrooms: int = Field(..., gt=0)
    floors: int = Field(..., ge=0)
    year_built: int
    garage: str
    location: str
    condition: str

    @validator("year_built")
    def year_not_future(cls, v):
        current_year = datetime.now().year
        if v > current_year:
            raise ValueError(f"Year built cannot exceed {current_year}")
        return v

    @validator("garage")
    def garage_valid(cls, v):
        if v.lower() not in ["yes", "no"]:
            raise ValueError("Garage must be 'yes' or 'no'")
        return v.lower()

    @validator("location")
    def location_valid(cls, v):
        allowed = ["country side", "down town", "city center", "suburb"]
        if v.lower() not in allowed:
            raise ValueError(f"Location must be one of {allowed}")
        return v.lower()

    @validator("condition")
    def condition_valid(cls, v):
        allowed = ["excellent", "good", "fair", "poor"]
        if v.lower() not in allowed:
            raise ValueError(f"Condition must be one of {allowed}")
        return v.lower()

# ----------------- Mappings -----------------
garage_mapping = {"no": 0, "yes": 1}
location_mapping = {"country side": 0, "down town": 1, "city center": 2, "suburb": 3}
condition_mapping = {"excellent": 0, "good": 1, "fair": 2, "poor": 3}

# ----------------- Prediction Endpoint -----------------
@app.post("/predict")
def predict_price(features: HouseFeatures):
    logger.info(f"Received features: {features.dict()}")
    
    # Encode categorical variables
    data = pd.DataFrame([[
        features.area,
        features.bedrooms,
        features.bathrooms,
        features.floors,
        features.year_built,
        garage_mapping[features.garage],
        location_mapping[features.location],
        condition_mapping[features.condition]
    ]], columns=["area","bedrooms","bathrooms","floors","year_built","garage","location","condition"])

    try:
        prediction = model.predict(data)
        predicted_price = round(prediction[0], 2)
        logger.info(f"Predicted price: {predicted_price}")
        return {"predicted_price": predicted_price}
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        return {"error": "Prediction failed"}
