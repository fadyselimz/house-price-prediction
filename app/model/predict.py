from joblib import load
import numpy as np
from app.config import Model_Path, Scaler_Path

model  = load(Model_Path)
scaler = load(Scaler_Path)

def predict_house_price(data):
    features = np.array([[
        data.MedInc, data.HouseAge, data.AveRooms, data.AveBedrms,
        data.Population, data.AveOccup, data.Latitude, data.Longitude
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]

    return {
        "predicted_price": round(float(prediction), 4),
        "predicted_price_usd": f"${round(float(prediction) * 100000):,}"
    }