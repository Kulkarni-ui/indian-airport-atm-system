"""
models/delay_predictor.py
Scikit-learn delay prediction model with feature engineering for Indian aviation.
Trains on synthetic data; swap in real DGCA historical data for production.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_PATH = Path(__file__).parent / "delay_model.pkl"

SEVERITY_MAP  = {"CLEAR": 0, "LOW_VIS": 1, "MODERATE": 2, "SEVERE": 3, "EXTREME": 4}
AIRCRAFT_WAKE = {"ATR72": 0, "A319": 1, "A320": 1, "A321": 1, "B738": 1, "B77W": 2, "B787": 2}


def _generate_training_data(n: int = 5000) -> pd.DataFrame:
    """Generate synthetic but realistic training data for Indian domestic routes."""
    rng = np.random.default_rng(42)
    records = []
    airlines      = ["AIC", "IGO", "SEJ", "GOW", "VTI", "BLU"]
    aircraft_types = list(AIRCRAFT_WAKE.keys())
    airports      = ["VIDP", "VABB", "VOMM", "VOBL", "VECO"]

    for _ in range(n):
        aircraft   = rng.choice(aircraft_types)
        dep_sev    = rng.choice(list(SEVERITY_MAP.keys()), p=[0.5, 0.15, 0.15, 0.12, 0.08])
        arr_sev    = rng.choice(list(SEVERITY_MAP.keys()), p=[0.5, 0.15, 0.15, 0.12, 0.08])
        wind_speed = rng.uniform(0, 35)
        vis_km     = rng.uniform(0.1, 15)
        hour_utc   = rng.integers(0, 24)
        congestion = rng.uniform(0, 1)
        historical_otp = rng.uniform(0.55, 0.97)
        is_peak    = 1 if hour_utc in range(2, 8) or hour_utc in range(10, 16) else 0  # IST peaks

        # Delay model (ground truth for synthetic data)
        delay = (
            SEVERITY_MAP[arr_sev] * 18 +
            SEVERITY_MAP[dep_sev] * 10 +
            max(0, wind_speed - 20) * 2 +
            max(0, 3 - vis_km) * 12 +
            congestion * 35 +
            (1 - historical_otp) * 60 +
            is_peak * rng.uniform(0, 20) +
            rng.normal(0, 8)
        )
        delay = max(0, delay)

        records.append({
            "aircraft_wake_cat":    AIRCRAFT_WAKE[aircraft],
            "dep_weather_severity": SEVERITY_MAP[dep_sev],
            "arr_weather_severity": SEVERITY_MAP[arr_sev],
            "wind_speed_kts":       wind_speed,
            "visibility_km":        vis_km,
            "hour_utc":             hour_utc,
            "is_peak_hour":         is_peak,
            "congestion_score":     congestion,
            "historical_otp":       historical_otp,
            "delay_minutes":        delay,
        })

    return pd.DataFrame(records)


def train_model() -> dict:
    """Train and persist the delay prediction model. Returns evaluation metrics."""
    df = _generate_training_data()
    features = [c for c in df.columns if c != "delay_minutes"]
    X, y = df[features].values, df["delay_minutes"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.05,
            subsample=0.8, random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)

    preds  = pipeline.predict(X_test)
    mae    = mean_absolute_error(y_test, preds)
    within_15 = np.mean(np.abs(preds - y_test) < 15) * 100

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"pipeline": pipeline, "features": features}, f)

    return {
        "mae_minutes":       round(mae, 2),
        "within_15min_pct":  round(within_15, 1),
        "training_samples":  len(X_train),
    }


def load_model() -> Optional[dict]:
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


def predict(
    dep_weather_severity: str,
    arr_weather_severity: str,
    wind_speed_kts: float,
    visibility_km: float,
    hour_utc: int,
    congestion_score: float,
    historical_otp: float,
    aircraft_type: str = "A320",
) -> dict:
    """
    Predict delay using the trained ML model.

    Returns:
        dict with predicted_delay_minutes and confidence interval.
    """
    bundle = load_model()
    if bundle is None:
        metrics = train_model()
        bundle  = load_model()

    pipeline = bundle["pipeline"]
    features = bundle["features"]

    wake_cat = AIRCRAFT_WAKE.get(aircraft_type, 1)
    is_peak  = 1 if hour_utc in range(2, 8) or hour_utc in range(10, 16) else 0

    feature_vec = np.array([[
        wake_cat,
        SEVERITY_MAP.get(dep_weather_severity, 0),
        SEVERITY_MAP.get(arr_weather_severity, 0),
        wind_speed_kts,
        visibility_km,
        hour_utc,
        is_peak,
        congestion_score,
        historical_otp,
    ]])

    predicted = float(pipeline.predict(feature_vec)[0])
    predicted = max(0, predicted)

    # Bootstrap confidence interval using leaf variance
    individual_preds = [
        est.predict(pipeline.named_steps["scaler"].transform(feature_vec))[0]
        for est in pipeline.named_steps["model"].estimators_.flatten()
    ]
    std = float(np.std(individual_preds))
    confidence = max(0.5, min(0.95, 1.0 - (std / (predicted + 1))))

    return {
        "predicted_delay_minutes": round(predicted, 1),
        "confidence":              round(confidence, 2),
        "lower_bound":             max(0, round(predicted - 1.5 * std, 1)),
        "upper_bound":             round(predicted + 1.5 * std, 1),
    }


if __name__ == "__main__":
    print("Training delay prediction model...")
    metrics = train_model()
    print(json.dumps(metrics, indent=2))

    test = predict("CLEAR", "MODERATE", 15.0, 4.0, 14, 0.7, 0.80, "A320")
    print("Sample prediction:", json.dumps(test, indent=2))
