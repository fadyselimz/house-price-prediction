import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from joblib import dump, load

import mlflow
import mlflow.sklearn
from  mlflow.client import MlflowClient
from mlflow.models import infer_signature

sys.path[:0] = [str(Path(__file__).resolve().parents[2])]
from app.config.config import Model_Path, Scaler_Path, ensure_model_paths
ensure_model_paths()


mlflow.set_experiment("house-price-prediction")
client = MlflowClient()


dataset = fetch_california_housing(as_frame=True)
X = pd.DataFrame(data=dataset.data, columns=dataset.feature_names)
Y = pd.Series(data=dataset.target, name=dataset.target_names[0])

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

Scaler = StandardScaler()
X_train_scaled = Scaler.fit_transform(X_train)
X_test_scaled  = Scaler.transform(X_test)

with mlflow.start_run(run_name="random-forest-regressor") as run:

    mlflow.set_tag("mlflow.note.content",
        "Random Forest Regressor model for California Housing Price prediction.\n"
        "Dataset: California Housing (20,640 samples, 8 features).\n"
        "Target: Median house value in units of 100,000.\n"
        "Preprocessing: StandardScaler applied to all 8 features.\n"
        "Split: 80% train / 20% test."
    )

    mlflow.set_tags({
        "model_type":   "RandomForestRegressor",
        "dataset":      "California Housing",
        "problem_type": "regression",
        "developer":    "Fady",
        "scaler":       "StandardScaler",
        "train_size":   "0.8",
        "test_size":    "0.2"
    })

model = RandomForestRegressor(n_estimators=100, random_state=42)

mlflow.log_params({
      "n_estimators": model.n_estimators,
        "random_state": model.random_state,
        "max_depth":    str(model.max_depth),
        "max_features": model.max_features,
        "min_samples_split": model.min_samples_split,
        "min_samples_leaf":  model.min_samples_leaf
})

model.fit(X_train_scaled, Y_train)
Y_pred = model.predict(X_test_scaled)

mlflow.log_metrics({
        "mae":  round(mean_absolute_error(Y_test, Y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(Y_test, Y_pred)), 4),
        "r2":   round(r2_score(Y_test, Y_pred), 4)
    })

X_train_df = pd.DataFrame(X_train_scaled, columns=X.columns)
signature  = infer_signature(X_train_df, model.predict(X_train_scaled))

mlflow.sklearn.log_model(
        sk_model=model,
        name="model",
        signature=signature,
        registered_model_name="HousePriceModel",
        pip_requirements=[
            "scikit-learn",
            "pandas",
            "numpy"
        ],
        metadata={
            "description": "Random Forest Regressor for house price prediction",
            "input_features": list(X.columns),
            "output": "Median house value in units of $100,000"
        }
    )

dump(model, Model_Path)
dump(Scaler, Scaler_Path)

print(f"Run ID: {run.info.run_id}")
print(f"MAE:  {mean_absolute_error(Y_test, Y_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(Y_test, Y_pred)):.4f}")
print(f"R²:   {r2_score(Y_test, Y_pred):.4f}")
print("Model and scaler saved.")
 
model_name = "HousePriceModel"
latest = client.get_latest_versions(model_name)[0]

client.update_registered_model(
    name=model_name,
    description=(
        "Random Forest Regressor predicting California house prices. "
        "Trained on California Housing dataset (20,640 samples, 8 features). "
        "Input: 8 scaled housing features. "
        "Output: Median house value in units of $100,000."
    )
)

client.set_registered_model_alias(
    name=model_name,
    alias="production",
    version=latest.version
)

print(f"Model '{model_name}' v{latest.version} → alias 'production' set in Registry")