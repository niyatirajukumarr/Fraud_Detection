from fastapi import FastAPI
import joblib
import numpy as np
from models.risk_update import update_risk

early_model = joblib.load("models/early_model.pkl")
delayed_model = joblib.load("models/delayed_model.pkl")

app = FastAPI()

@app.post("/predict/early")
def predict_early(txn: dict):
    features = np.array([list(txn.values())])
    risk = early_model.predict_proba(features)[0][1]

    return {
        "stage": "early",
        "risk_score": round(float(risk), 3),
        "decision": "FLAG" if risk > 0.7 else "ALLOW"
    }

@app.post("/predict/update")
def predict_update(data: dict):
    early_risk = data["early_risk"]
    delayed_features = np.array([data["features"]])

    delayed_risk = delayed_model.predict_proba(delayed_features)[0][1]
    final_risk = update_risk(early_risk, delayed_risk)

    return {
        "stage": "updated",
        "early_risk": early_risk,
        "delayed_risk": round(float(delayed_risk), 3),
        "final_risk": final_risk,
        "decision": "FLAG" if final_risk > 0.7 else "ALLOW"
    }
