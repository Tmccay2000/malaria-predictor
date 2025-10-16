# malaria_api.py
from fastapi import FastAPI, HTTPException, Request, Header
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import datetime
import logging

# ------------------------------
# CONFIGURATION
# ------------------------------
API_KEY = "YOUR_SECURE_API_KEY"  # Change before deployment
LOG_FILE = "api_requests.log"

# Logging configuration
logging.basicConfig(filename=LOG_FILE, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------
# Load model once
# ------------------------------
model = joblib.load("final_rf_model.pkl")

app = FastAPI(title="🦟 Malaria Prediction API", version="2.0")

# ------------------------------
# Define input structure
# ------------------------------
class InputData(BaseModel):
    RH2M_mean: float = Field(..., description="Average humidity level")
    RH2M_lag_1: float
    RH2M_lag_2: float
    RH2M_lag_3: float
    Malaria_lag_1: float
    Malaria_lag_2: float
    Malaria_lag_3: float
    Season_Wet: int = Field(..., ge=0, le=1, description="1 for Wet, 0 for Dry")

# ------------------------------
# Authentication check
# ------------------------------
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        return HTTPException(status_code=401, detail="Invalid or missing API key")
    response = await call_next(request)
    return response

# ------------------------------
# Root Endpoint
# ------------------------------
@app.get("/")
def read_root():
    return {"message": "Malaria Prediction API is running!", "version": "2.0"}

# ------------------------------
# Prediction Endpoint
# ------------------------------
@app.post("/predict")
def predict(data: InputData):
    try:
        features = np.array([[data.RH2M_mean, data.RH2M_lag_1, data.RH2M_lag_2, data.RH2M_lag_3,
                              data.Malaria_lag_1, data.Malaria_lag_2, data.Malaria_lag_3, data.Season_Wet]])
        
        prediction = model.predict(features)[0]

        # Log request
        log_entry = {
            "timestamp": datetime.datetime.now(),
            "input": data.dict(),
            "prediction": float(prediction)
        }
        logging.info(log_entry)
        
        # Save request to history file
        df = pd.DataFrame([log_entry])
        df.to_csv("prediction_history.csv", mode='a', header=False, index=False)

        return {"predicted_malaria_incidence": round(prediction, 2)}

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")
