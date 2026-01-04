import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from pathlib import Path

MODEL_PATH = Path(r"./Models\Simple_Models\RandomForestClassifier.joblib")

app = FastAPI(
    title=" Best Dataset Prediction API",
    version="1.0"
)

pipeline = None  


# --------------------------------------------------
# Load model ON STARTUP (NOT import time)
# --------------------------------------------------
@app.on_event("startup")
def load_model():
    global pipeline
    try:
        pipeline = joblib.load(MODEL_PATH)
        print("✅ Model loaded successfully")
    except Exception as e:
        print("❌ Failed to load model:", e)
        pipeline = None


# --------------------------------------------------
# Schemas
# --------------------------------------------------
from pydantic import BaseModel
from typing import Optional

class DatasetInput(BaseModel):
    id: int
    ref: str
    subtitle: str
    creatorname: str
    creatorurl: str
    totalbytes: int
    url: str
    lastupdated: str
    downloadcount: int
    ownername: str
    ownerref: str
    title: str
    viewcount: int
    tags: str


class PredictionOutput(BaseModel):
    predicted_cluster: int
    cluster_probability: float



# --------------------------------------------------
# Health
# --------------------------------------------------
@app.get("/")
def root():
    return {"status": "running"}

@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------
# Predict
# --------------------------------------------------
import logging
import pandas as pd
from fastapi import HTTPException

# Logging konfiguratsiyasi
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@app.post("/predict", response_model=PredictionOutput)
def predict(data: DatasetInput):

    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Input → DataFrame
    df = pd.DataFrame([data.model_dump()])

    # Predict probabilities (multiclass)
    proba_all = pipeline.predict_proba(df)[0]
    logger.debug("DEBUG predict_proba: %s", proba_all)

    # Eng yuqori ehtimollikdagi cluster
    predicted_cluster = int(proba_all.argmax())
    cluster_probability = float(proba_all[predicted_cluster])

    return PredictionOutput(
        predicted_cluster=predicted_cluster,
        cluster_probability=round(cluster_probability, 4)
    )
