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

# ✅ Charger depuis .env
MODEL_PATH = os.getenv("MODEL_PATH", "./pipeline_xgb_projectIt_20260412_231221.joblib")
JWT_SECRET = os.getenv("JWT_SECRET", "guUJXj47cvyNNTm8ERCz5ii3HKmqL8raixgK3vVmQw")
JWT_ALGORITHM = "HS256"

print(f"📁 MODEL_PATH: {MODEL_PATH}")
print(f"🔐 JWT_SECRET: {JWT_SECRET[:10]}...")

# ✅ Charger le modèle
try:
    pipeline = joblib.load(MODEL_PATH)
    print(f"✅ Modèle chargé avec succès")
except Exception as e:
    print(f"⚠️ Erreur chargement modèle: {e}")
    pipeline = None

app = FastAPI(title="Project Cost Estimator API", version="1.0.0")
security = HTTPBearer(auto_error=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Schéma de validation
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

    class Config:
        json_schema_extra = {
            "example": {
                "programmingLanguages": "Python",
                "framework": "FastAPI",
                "database": "PostgreSQL",
                "serverDetails": "AWS EC2",
                "architecture": "Microservices",
                "apiIntegration": "REST",
                "securityRequirements": "OAuth2",
                "devOpsRequirements": "Kubernetes",
                "estimatedDurationDays": 95,
                "priority": "High",
                "businessImpact": "Critical",
                "teamSize": 7,
                "complexity": "High",
                "mainModules": "Traffic-Control"
            }
        }

def verify_jwt_optional(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Vérifie JWT (optionnel)"""
    if not credentials:
        return {"sub": "anonymous"}
    try:
        payload = jwt.decode(credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError:
        return {"sub": "anonymous"}
    except Exception:
        return {"sub": "anonymous"}

@app.post("/predict-cost")
def predict_cost(project: ProjectRequest, token: dict = Depends(verify_jwt_optional)):
    """Estime le coût du projet"""
    try:
        user = token.get("sub", "anonymous")
        print(f"🔐 Utilisateur: {user}")
        
        if pipeline is None:
            print("❌ Modèle non chargé")
            return {
                "estimated_cost": 0,
                "currency": "USD",
                "error": "Model not loaded",
                "requested_by": user
            }

        # ✅ Colonnes exactes du modèle
        feature_cols = [
            "programmingLanguages", "framework", "database", "serverDetails",
            "architecture", "apiIntegration", "securityRequirements",
            "devOpsRequirements", "estimatedDurationDays", "priority",
            "businessImpact", "teamSize", "complexity", "mainModules"
        ]
        
        # ✅ Créer DataFrame
        data_dict = project.dict()
        df = pd.DataFrame([data_dict], columns=feature_cols)
        
        print(f"📥 Entrée: {data_dict}")
        print(f"📊 Colonnes: {df.columns.tolist()}")
        
        try:
            # ✅ Prédiction
            pred = pipeline.predict(df)
            estimated_cost = float(pred[0])
            
            print(f"🎯 Coût estimé: {estimated_cost} USD")
            
            return {
                "estimated_cost": round(estimated_cost, 2),
                "currency": "USD",
                "confidence": "high",
                "requested_by": user
            }
            
        except Exception as pred_error:
            print(f"❌ Erreur prédiction: {pred_error}")
            return {
                "estimated_cost": 0,
                "currency": "USD",
                "error": f"Prediction error: {str(pred_error)}",
                "requested_by": user
            }
        
    except Exception as e:
        print(f"❌ Erreur globale: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {
            "estimated_cost": 0,
            "currency": "USD",
            "error": str(e),
            "requested_by": "anonymous"
        }

@app.get("/health")
def health(token: dict = Depends(verify_jwt_optional)):
    """État de l'API"""
    return {
        "status": "online",
        "model": "loaded" if pipeline is not None else "missing",
        "api_version": "1.0.0",
        "authenticated_user": token.get("sub", "anonymous")
    }

@app.get("/")
def root():
    """Documentation"""
    return {
        "title": "Project Cost Estimator API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict-cost": "Estimer le coût du projet",
            "GET /health": "État de l'API",
            "GET /docs": "Documentation Swagger"
        }
    }
@app.post("/debug")
def debug_request(project: ProjectRequest):
    """Debug - voir les données reçues"""
    data_dict = project.dict()
    print("=== DEBUG ===")
    print(f"Types reçus:")
    for key, value in data_dict.items():
        print(f"  {key}: {value} (type: {type(value).__name__})")
    print("=============")
    return {"received": data_dict}
@app.options("/predict-cost")
def options():
    """CORS preflight"""
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001, reload=True)