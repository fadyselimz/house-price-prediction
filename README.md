# 🏠 House Price Prediction API

A production-ready Machine Learning REST API that predicts California house prices based on 8 property features. Built with FastAPI, tracked with MLflow, and containerized with Docker.

---

## 📌 Features

- Predicts house prices using a Random Forest Regressor
- REST API built with FastAPI
- Full MLflow experiment tracking (params, metrics, model registry)
- Model promoted to `production` alias in MLflow Registry
- Docker support for consistent deployment
- Input validation using Pydantic
- StandardScaler preprocessing pipeline

---

## 🛠️ Tech Stack

- **Python 3.10**
- **FastAPI** — REST API framework
- **scikit-learn** — Random Forest Regressor + StandardScaler
- **MLflow** — experiment tracking and model registry
- **Pandas / NumPy** — data processing
- **Uvicorn** — ASGI server
- **Docker** — containerized deployment

---

## 📁 Project Structure

```
house-price-prediction/
├── app/
│   ├── main.py                    # FastAPI routes
│   ├── __init__.py
│   ├── config/
│   │   └── config.py              # Paths and app configuration
│   ├── ml/
│   │   ├── predict.py             # Model loading and inference
│   │   ├── train.py               # Training script with MLflow
│   │   └── __init__.py
│   └── schemas/
│       ├── house.py               # Pydantic request/response models
│       └── __init__.py
├── notebook/
│   └── housePrediction.ipynb      # Exploratory analysis
├── Dockerfile
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Setup

### 1. Clone the repository

```bash
git clone https://github.com/fadyselimz/house-price-prediction-api.git
cd house-price-prediction-api
```

### 2. Create and activate a virtual environment

**Windows**
```bash
py -m venv venv
venv\Scripts\Activate.ps1
```

**Linux / macOS**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

The model files are not included in the repository. Generate them by running:

```bash
py app/ml/train.py
```

This will:
- Train a Random Forest Regressor on the California Housing dataset
- Log all params, metrics, and the model to MLflow
- Register the model in the MLflow Model Registry as `HousePriceModel`
- Save `house_price_model.joblib` and `house_price_scaler.joblib` for the API

---

## ▶️ Running the API

```bash
uvicorn app.main:app --reload
```

Available at:
- API: `http://127.0.0.1:8000`
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 🐳 Running with Docker

```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

Open `http://localhost:8000/docs`

---

## 📊 MLflow Tracking

View experiment history, metrics, and the model registry:

```bash
mlflow ui
```

Open `http://localhost:5000`

Tracked per run:
- **Params**: `n_estimators`, `max_depth`, `max_features`, `min_samples_split`, `min_samples_leaf`, `random_state`
- **Metrics**: `mae`, `rmse`, `r2`
- **Model**: registered as `HousePriceModel` with alias `production`

---

## 📥 API Endpoints

| Method | Endpoint      | Description                        |
|--------|---------------|------------------------------------|
| GET    | `/health`     | Health check                       |
| GET    | `/model-info` | Returns model name and version     |
| POST   | `/predict`    | Returns predicted house price      |

### Example request to `/predict`

```json
{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.984,
  "AveBedrms": 1.024,
  "Population": 322.0,
  "AveOccup": 2.555,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```

### Example response

```json
{
  "predicted_price": 4.526,
  "predicted_price_usd": "$452,600"
}
```

---

## 🧠 Model Details

- **Algorithm**: Random Forest Regressor (scikit-learn)
- **Dataset**: California Housing (20,640 samples, 8 features)
- **Target**: Median house value in units of $100,000
- **Preprocessing**: StandardScaler applied to all 8 features
- **Split**: 80% train / 20% test
- **Metrics**: MAE ~0.33, RMSE ~0.51, R² ~0.81

---

## 🧪 Testing

Test via Swagger UI at `http://localhost:8000/docs` or using curl:

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "MedInc": 8.3252,
  "HouseAge": 41.0,
  "AveRooms": 6.984,
  "AveBedrms": 1.024,
  "Population": 322.0,
  "AveOccup": 2.555,
  "Latitude": 37.88,
  "Longitude": -122.23
}'
```

---

## 📈 Future Improvements

- Add `mlflow.evaluate()` once MLflow 3.x API stabilizes
- CI/CD pipeline with GitHub Actions
- Cloud deployment (Render / AWS / Azure)
- Model monitoring and drift detection
- Automated testing

---

## 👨‍💻 Author

**Fady Selim** — CS student building toward a career in ML/AI engineering

- GitHub: https://github.com/fadyselimz

---

## 📄 License

This project is licensed under the MIT License.
