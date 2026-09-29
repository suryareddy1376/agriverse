import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
import joblib

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.transformer import SpatioTemporalTransformer

class AgriDataset(Dataset):
    def __init__(self, X, y, quantiles=False):
        # X: (batch, seq_len, num_regions, features)
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def pinball_loss(y_true, y_pred, quantiles):
    """Quantile regression loss"""
    losses = []
    for i, q in enumerate(quantiles):
        err = y_true - y_pred[:, :, i]
        loss = torch.max((q - 1) * err, q * err)
        losses.append(loss)
    return torch.mean(torch.stack(losses))

def train_transformer():
    data_path = os.path.join("data", "processed", "agriVerse_tamilnadu_rice.csv")
    if not os.path.exists(data_path):
        print("Dataset not found. Run build_real_dataset.py first.")
        return
        
    df = pd.read_csv(data_path)
    df = df.sort_values(by=['year', 'season'])
    
    # We will build sequences of weather and ndvi
    seq_len = 6
    weather_vars = ['temperature_mean', 'rainfall', 'humidity', 'solar_radiation', 'ndvi']
    static_vars = ['soil_ph', 'soil_organic_carbon', 'soil_clay', 'area_sqkm']
    
    # Check existing variables
    weather_vars = [v for v in weather_vars if f"{v}_m1" in df.columns]
    static_vars = [v for v in static_vars if v in df.columns]
    
    # Fill NAs
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
            
    # For MVP, we'll treat each row as a sample of (seq_len, 1 region, features)
    # A true graph model would pass all regions simultaneously.
    
    X_seq = np.zeros((len(df), seq_len, 1, len(weather_vars) + len(static_vars)))
    y = df['yield_tonnes_per_ha'].values
    
    for i in range(seq_len):
        for j, v in enumerate(weather_vars):
            X_seq[:, i, 0, j] = df[f"{v}_m{i+1}"].values
            
        # Static vars repeat across time
        for j, v in enumerate(static_vars):
            X_seq[:, i, 0, len(weather_vars) + j] = df[v].values
            
    # Split
    train_end = 2008
    val_end = 2011
    
    train_idx = df['year'] <= train_end
    val_idx = (df['year'] > train_end) & (df['year'] <= val_end)
    test_idx = df['year'] > val_end
    
    X_train, y_train = X_seq[train_idx], y[train_idx]
    X_val, y_val = X_seq[val_idx], y[val_idx]
    X_test, y_test = X_seq[test_idx], y[test_idx]
    
    # Scale features
    scaler = StandardScaler()
    B_tr, T, R, F = X_train.shape
    X_train_flat = X_train.reshape(-1, F)
    scaler.fit(X_train_flat)
    
    X_train = scaler.transform(X_train_flat).reshape(B_tr, T, R, F)
    X_val = scaler.transform(X_val.reshape(-1, F)).reshape(X_val.shape[0], T, R, F)
    X_test = scaler.transform(X_test.reshape(-1, F)).reshape(X_test.shape[0], T, R, F)
    
    # Scale target? Not necessary if using MSE, but good practice. We'll leave it unscaled for easy interpretability.
    
    train_loader = DataLoader(AgriDataset(X_train, y_train), batch_size=16, shuffle=True)
    val_loader = DataLoader(AgriDataset(X_val, y_val), batch_size=16)
    test_loader = DataLoader(AgriDataset(X_test, y_test), batch_size=16)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    model = SpatioTemporalTransformer(num_features=F, d_model=64, num_heads=2, num_layers=2, dropout=0.1, max_seq_len=T, num_regions=1).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    epochs = 50
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1) # (B, 1) since R=1
            
            optimizer.zero_grad()
            pred, quantiles = model(batch_x)
            
            loss_mse = criterion(pred, batch_y)
            loss_q = pinball_loss(batch_y.unsqueeze(-1), quantiles, [0.1, 0.9])
            
            loss = loss_mse + 0.5 * loss_q
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1)
                pred, quantiles = model(batch_x)
                val_loss += criterion(pred, batch_y).item()
                
        val_loss /= len(val_loader)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            out_dir = os.path.join("models", "transformer")
            os.makedirs(out_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(out_dir, "best_model.pth"))
            
    # Evaluation
    model.load_state_dict(torch.load(os.path.join(out_dir, "best_model.pth"), weights_only=True))
    model.eval()
    
    test_preds, test_actuals, lower_bounds, upper_bounds = [], [], [], []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            pred, quantiles = model(batch_x)
            test_preds.extend(pred.squeeze(1).cpu().numpy())
            test_actuals.extend(batch_y.cpu().numpy())
            lower_bounds.extend(quantiles[:, 0, 0].cpu().numpy())
            upper_bounds.extend(quantiles[:, 0, 1].cpu().numpy())
            
    test_preds = np.array(test_preds)
    test_actuals = np.array(test_actuals)
    
    mae = np.mean(np.abs(test_preds - test_actuals))
    rmse = np.sqrt(np.mean((test_preds - test_actuals)**2))
    
    print(f"\nTransformer Test RMSE: {rmse:.3f}, MAE: {mae:.3f}")
    
    # Save artifacts
    scaler_path = os.path.join(out_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    
    results = pd.DataFrame({
        'actual': test_actuals,
        'predicted': test_preds,
        'lower_bound': lower_bounds,
        'upper_bound': upper_bounds
    })
    
    results_path = os.path.join("artifacts", "transformer_test_predictions.csv")
    results.to_csv(results_path, index=False)
    print(f"Saved predictions and intervals to {results_path}")

if __name__ == "__main__":
    train_transformer()
