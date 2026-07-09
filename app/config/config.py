import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

Model_Path  = os.path.join(BASE_DIR, "model", "house_price_model.joblib")
Scaler_Path = os.path.join(BASE_DIR, "model", "house_price_scaler.joblib")

APP_NAME     = "House Price Prediction"
APP_VERSION  = "1.0.0"
MODEL_NAME   = "Random Forest Regressor"

def ensure_model_paths():
    os.makedirs(os.path.dirname(Model_Path), exist_ok=True)
    os.makedirs(os.path.dirname(Scaler_Path), exist_ok=True)