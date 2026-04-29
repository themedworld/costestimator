import os
import joblib
import pandas as pd
from fastapi import FastAPI, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from jose import jwt, JWTError

load_dotenv()

# =========================
# MODELE RANDOM FOREST
# =========================
MODEL_PATH = os.getenv("MODEL_PATH", "./best_project_cost_model.pkl")
ENCODERS_PATH = os.getenv("ENCODERS_PATH", "./encoders.pkl")

JWT_SECRET = os.getenv("JWT_SECRET", "secret")
JWT_ALGORITHM = "HS256"

print("📁 MODEL:", MODEL_PATH)

# load model
try:
    model = joblib.load(MODEL_PATH)
    print("✅ RandomForest chargé")
except Exception as e:
    print("❌ Model error:", e)
    model = None

# load encoders
try:
    encoders = joblib.load(ENCODERS_PATH)
    print("✅ Encoders chargés")
except Exception as e:
    print("❌ Encoders error:", e)
    encoders = None


# =========================
# FASTAPI
# =========================
app = FastAPI(title="RandomForest Cost API")
security = HTTPBearer(auto_error=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# INPUT MODEL
# =========================
class ProjectRequest(BaseModel):
    programmingLanguages: str
    framework: str
    database: str
    serverDetails: str
    architecture: str
    apiIntegration: str
    securityRequirements: str
    devOpsRequirements: str
    estimatedDurationDays: int
    priority: str
    businessImpact: str
    teamSize: int
    complexity: str
    mainModules: str


# =========================
# ENCODING FUNCTION
# =========================
def encode_input(df):
    for col in df.columns:
        if col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))
    return df


# =========================
# PREDICT
# =========================
@app.post("/predict-cost")
def predict_cost(project: ProjectRequest):

    if model is None:
        return {"error": "Model not loaded"}

    feature_cols = [
        "programmingLanguages","framework","database","serverDetails",
        "architecture","apiIntegration","securityRequirements",
        "devOpsRequirements","estimatedDurationDays","priority",
        "businessImpact","teamSize","complexity","mainModules"
    ]

    # dataframe
    df = pd.DataFrame([project.dict()])[feature_cols]

    # encode
    if encoders is not None:
        df = encode_input(df)

    # prediction
    pred = model.predict(df)[0]

    # sécurité anti négatif
    pred = max(0, float(pred))

    return {
        "estimated_cost": round(pred, 2),
        "currency": "USD"
    }


# =========================
# HEALTH
# =========================
@app.get("/health")
def health():
    return {
        "model": "loaded" if model else "missing",
        "encoders": "loaded" if encoders else "missing"
    }
