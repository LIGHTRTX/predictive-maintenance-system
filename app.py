from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load("failure_model.pkl")
scaler = joblib.load("scaler.pkl")

RATED_CURRENT = 5.0

class SensorInput(BaseModel):
    air_temp: float
    process_temp: float
    speed: float
    torque: float
    tool_wear: float


def overload_ratio(current, rated_current=RATED_CURRENT):
    return current / rated_current


def generate_signal(over_ratio):
    t = np.linspace(0, 1, 1000)
    base = 0.3*np.sin(2*np.pi*30*t) + 0.05*np.random.randn(1000)
    if over_ratio > 1.5:
        base += 0.4*np.sin(2*np.pi*200*t)
    return base


def compute_rms(signal):
    return np.sqrt(np.mean(signal**2))


def unified_risk(failure_prob, over_ratio, rms):
    normalized_overload = min(over_ratio/2, 1)
    normalized_rms = min(rms/0.5, 1)
    return 0.4*failure_prob + 0.3*normalized_overload + 0.3*normalized_rms


def classify(risk_score):
    if risk_score < 0.4:
        return "Normal"
    elif risk_score < 0.7:
        return "Warning"
    else:
        return "Critical"


@app.post("/predict")
def predict(data: SensorInput):

    X = np.array([[data.air_temp,
                   data.process_temp,
                   data.speed,
                   data.torque,
                   data.tool_wear]])

    X_scaled = scaler.transform(X)
    failure_prob = model.predict_proba(X_scaled)[0][1]

    over_ratio = overload_ratio(data.torque)

    signal = generate_signal(over_ratio)
    rms = compute_rms(signal)

    risk_score = unified_risk(failure_prob, over_ratio, rms)
    status = classify(risk_score)

    return {
        "failure_probability": float(failure_prob),
        "overload_ratio": float(over_ratio),
        "rms_vibration": float(rms),
        "risk_score": float(risk_score),
        "status": status
    }
