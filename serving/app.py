from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path

from serving.schemas import FeatureVectorInput, PredictionResponse
from ml_crypto.inference.online import OnlinePredictor
from ml_crypto.config import get_config

app = FastAPI(
    title="Quant ML Prediction API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
predictor = None

@app.on_event("startup")
def load_model():
    global predictor
    config = get_config("config/config.yaml")
    prod_model_path = config.artifacts.models_dir / "production_model.pkl"
    try:
        predictor = OnlinePredictor(prod_model_path)
    except FileNotFoundError:
        print("[WARNING] production model not found, API active in condition uninitialized model")

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": predictor is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(features: FeatureVectorInput):
    if predictor is None:
        raise HTTPException(status_code=503, detail="model server not initialized, model not active :(")
    
    feature_dict = features.dict()
    result = predictor.predict_sample(feature_dict)
    return result

ui_path = Path("ui")
if ui_path.exists():
    app.mount("/static", StaticFiles(directory="ui"), name="static")

    @app.get("/ui")
    def serve_ui():
        return FileResponse("ui/index.html")