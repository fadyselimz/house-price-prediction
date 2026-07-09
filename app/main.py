from fastapi import FastAPI, HTTPException
from app.schemas.house import HouseFeatures, PredictionResponse
from app.model.predict import predict_house_price
from app.config import APP_NAME, APP_VERSION, MODEL_NAME

app = FastAPI(title=APP_NAME, version=APP_VERSION)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/model-info")
def model_info():
    return {
        "model": MODEL_NAME,
        "version": APP_VERSION,
        "features": ["MedInc", "HouseAge", "AveRooms", "AveBedrms", 
                     "Population", "AveOccup", "Latitude", "Longitude"]
    }

@app.post("/predict", response_model=PredictionResponse)
def predict(data: HouseFeatures):
    try:
        return predict_house_price(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))