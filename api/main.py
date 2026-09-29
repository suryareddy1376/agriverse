from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import torch
import joblib
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.transformer import SpatioTemporalTransformer

app = FastAPI(title="AgriVerse API", description="Digital Twin for Crop Yield Prediction")

# Load model and scaler globally
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = None
scaler = None
model_version = "v1.0"
data_version = "v1.0"
seq_len = 6

def load_model():
    global model, scaler
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'transformer')
    model_path = os.path.join(model_dir, 'best_model.pth')
    scaler_path = os.path.join(model_dir, 'scaler.joblib')
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        # Using 5 as num_features (weather_vars=5 + static_vars=4 but we used 9)
        # Let's check exactly how many features were passed
        F = scaler.mean_.shape[0]
        model = SpatioTemporalTransformer(num_features=F, d_model=64, num_heads=2, num_layers=2, dropout=0.1, max_seq_len=seq_len, num_regions=1)
        model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
        model.to(device)
        model.eval()
        return True
    return False

@app.on_event("startup")
async def startup_event():
    success = load_model()
    if not success:
        print("Warning: Model not found. Please train the model first.")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/data/sources")
def get_sources():
    prov_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'provenance.yaml')
    if os.path.exists(prov_path):
        with open(prov_path, 'r') as f:
            return {"sources": f.read()}
    return {"sources": "Provenance file not found."}

@app.get("/regions")
def get_regions():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'agriVerse_tamilnadu_rice.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        return {"states": df['state'].unique().tolist(), "districts": df['district'].unique().tolist()}
    return {"states": [], "districts": []}

class PredictRequest(BaseModel):
    state: str
    district: str
    crop: str
    season: str
    year: int

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'agriVerse_tamilnadu_rice.csv')
    df = pd.read_csv(data_path)
    
    # Get historical record as base
    record = df[(df['district'] == req.district) & (df['year'] == req.year) & (df['season'] == req.season)]
    
    if len(record) == 0:
        raise HTTPException(status_code=404, detail="Data not found for the requested region and time.")
        
    record = record.iloc[0]
    
    # Construct sequence
    weather_vars = ['temperature_mean', 'rainfall', 'humidity', 'solar_radiation', 'ndvi']
    static_vars = ['soil_ph', 'soil_organic_carbon', 'soil_clay', 'area_sqkm']
    weather_vars = [v for v in weather_vars if f"{v}_m1" in df.columns]
    static_vars = [v for v in static_vars if v in df.columns]
    
    F = len(weather_vars) + len(static_vars)
    X_seq = np.zeros((1, seq_len, 1, F))
    
    for i in range(seq_len):
        for j, v in enumerate(weather_vars):
            val = record.get(f"{v}_m{i+1}", np.nan)
            X_seq[0, i, 0, j] = val if not pd.isna(val) else 0.0
        for j, v in enumerate(static_vars):
            val = record.get(v, np.nan)
            X_seq[0, i, 0, len(weather_vars) + j] = val if not pd.isna(val) else 0.0
            
    # Inference
    X_flat = X_seq.reshape(-1, F)
    X_scaled = scaler.transform(X_flat).reshape(1, seq_len, 1, F)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred, quantiles = model(X_tensor)
        
    pred_val = float(pred.item())
    lower = float(quantiles[0, 0, 0].item())
    upper = float(quantiles[0, 0, 1].item())
    
    return {
        "state": req.state,
        "district": req.district,
        "crop": req.crop,
        "season": req.season,
        "prediction_tonnes_per_ha": max(0, pred_val),
        "lower_bound": max(0, lower),
        "upper_bound": max(0, upper),
        "model_version": model_version,
        "data_version": data_version
    }
    
class SimulateRequest(BaseModel):
    state: str
    district: str
    crop: str
    season: str
    year: int
    rainfall_modifier_pct: float = 0.0
    temp_modifier_abs: float = 0.0
    ndvi_modifier_pct: float = 0.0

@app.post("/simulate")
def simulate(req: SimulateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'agriVerse_tamilnadu_rice.csv')
    df = pd.read_csv(data_path)
    record = df[(df['district'] == req.district) & (df['year'] == req.year) & (df['season'] == req.season)]
    
    if len(record) == 0:
        raise HTTPException(status_code=404, detail="Data not found for simulation base.")
    record = record.iloc[0]
    
    weather_vars = ['temperature_mean', 'rainfall', 'humidity', 'solar_radiation', 'ndvi']
    static_vars = ['soil_ph', 'soil_organic_carbon', 'soil_clay', 'area_sqkm']
    weather_vars = [v for v in weather_vars if f"{v}_m1" in df.columns]
    static_vars = [v for v in static_vars if v in df.columns]
    
    F = len(weather_vars) + len(static_vars)
    X_seq_base = np.zeros((1, seq_len, 1, F))
    X_seq_sim = np.zeros((1, seq_len, 1, F))
    
    for i in range(seq_len):
        for j, v in enumerate(weather_vars):
            val = record.get(f"{v}_m{i+1}", np.nan)
            val = val if not pd.isna(val) else 0.0
            
            X_seq_base[0, i, 0, j] = val
            
            # Apply modifiers
            sim_val = val
            if v == 'rainfall':
                sim_val = val * (1 + req.rainfall_modifier_pct / 100.0)
            elif v == 'temperature_mean':
                sim_val = val + req.temp_modifier_abs
            elif v == 'ndvi':
                sim_val = min(1.0, max(0.0, val * (1 + req.ndvi_modifier_pct / 100.0)))
                
            X_seq_sim[0, i, 0, j] = sim_val
            
        for j, v in enumerate(static_vars):
            val = record.get(v, np.nan)
            val = val if not pd.isna(val) else 0.0
            X_seq_base[0, i, 0, len(weather_vars) + j] = val
            X_seq_sim[0, i, 0, len(weather_vars) + j] = val
            
    # Scale and predict
    X_base_scaled = scaler.transform(X_seq_base.reshape(-1, F)).reshape(1, seq_len, 1, F)
    X_sim_scaled = scaler.transform(X_seq_sim.reshape(-1, F)).reshape(1, seq_len, 1, F)
    
    with torch.no_grad():
        pred_base, _ = model(torch.tensor(X_base_scaled, dtype=torch.float32).to(device))
        pred_sim, _ = model(torch.tensor(X_sim_scaled, dtype=torch.float32).to(device))
        
    base_val = max(0, float(pred_base.item()))
    sim_val = max(0, float(pred_sim.item()))
    
    diff = sim_val - base_val
    pct_diff = (diff / base_val * 100) if base_val > 0 else 0
    
    return {
        "baseline_prediction": base_val,
        "scenario_prediction": sim_val,
        "absolute_difference": diff,
        "percentage_difference": pct_diff
    }
