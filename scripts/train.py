import os
import json
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib

def calculate_metrics(y_true, y_pred, model_name, split):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # MAPE can be undefined if y_true is 0, add small epsilon
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    metrics = {
        "model": model_name,
        "split": split,
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "mape": float(mape)
    }
    return metrics

def train_baselines():
    data_path = os.path.join("data", "processed", "agriVerse_tamilnadu_rice.csv")
    if not os.path.exists(data_path):
        print("Dataset not found. Run build_real_dataset.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Define features
    weather_feats = [col for col in df.columns if any(p in col for p in ['temperature', 'rainfall', 'humidity', 'solar_radiation'])]
    soil_feats = [col for col in df.columns if col.startswith('soil_')]
    ndvi_feats = [col for col in df.columns if col.startswith('ndvi_')]
    
    features = weather_feats + soil_feats + ndvi_feats
    
    # Check what features exist
    existing_features = [f for f in features if f in df.columns]
    
    target = 'yield_tonnes_per_ha'
    
    # Sort by year to ensure correct chronological splitting
    df = df.sort_values(by=['year', 'season'])
    
    # Split boundaries
    train_end = 2008
    val_end = 2011
    
    train_df = df[df['year'] <= train_end]
    val_df = df[(df['year'] > train_end) & (df['year'] <= val_end)]
    test_df = df[df['year'] > val_end]
    
    X_train, y_train = train_df[existing_features], train_df[target]
    X_val, y_val = val_df[existing_features], val_df[target]
    X_test, y_test = test_df[existing_features], test_df[target]
    
    # Handle NaN by simple median imputation for MVP
    X_train = X_train.fillna(X_train.median())
    X_val = X_val.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())
    
    models = {
        "LinearRegression": LinearRegression(),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
        "XGBoost": xgb.XGBRegressor(n_estimators=100, random_state=42, objective='reg:squarederror')
    }
    
    all_metrics = []
    
    out_dir = os.path.join("models", "baselines")
    os.makedirs(out_dir, exist_ok=True)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        val_metrics = calculate_metrics(y_val, y_val_pred, name, "validation")
        test_metrics = calculate_metrics(y_test, y_test_pred, name, "test")
        
        all_metrics.extend([val_metrics, test_metrics])
        
        print(f"{name} Test R2: {test_metrics['r2']:.3f}, Test RMSE: {test_metrics['rmse']:.3f}")
        
        model_path = os.path.join(out_dir, f"{name.lower()}.joblib")
        joblib.dump(model, model_path)
        
    metrics_df = pd.DataFrame(all_metrics)
    metrics_path = os.path.join("artifacts", "baseline_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics to {metrics_path}")
    
    # Save test predictions for visualization
    test_preds = test_df.copy()
    for name, model in models.items():
        test_preds[f"{name}_pred"] = model.predict(X_test)
        
    preds_path = os.path.join("artifacts", "baseline_test_predictions.csv")
    test_preds.to_csv(preds_path, index=False)
    
if __name__ == "__main__":
    train_baselines()
