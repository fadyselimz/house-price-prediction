from pydantic import BaseModel

class HouseFeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBedrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float

class PredictionResponse(BaseModel):
    predicted_price: float
    predicted_price_usd: str