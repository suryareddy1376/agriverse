import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=12):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, ..., d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :].unsqueeze(2) if len(x.shape) == 4 else x + self.pe[:, :seq_len, :]

import numpy as np

class SpatioTemporalTransformer(nn.Module):
    def __init__(self, num_features, d_model=128, num_heads=4, num_layers=2, dropout=0.1, max_seq_len=6, num_regions=1):
        super().__init__()
        self.d_model = d_model
        
        # Feature Projection
        self.feature_proj = nn.Linear(num_features, d_model)
        
        # Positional Encoding (Temporal)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        # Spatial Embedding (Learned representations for each region)
        self.spatial_embedding = nn.Parameter(torch.randn(1, 1, num_regions, d_model))
        
        # Factorized Transformer Layers
        self.temporal_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.spatial_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Final Regression Head
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1) # Predicts yield
        )
        
        # Quantile Regression Heads for Uncertainty
        # (Lower bound 10%, Upper bound 90%)
        self.quantile_head = nn.Linear(64, 2)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, num_regions, num_features)
        B, T, R, F = x.shape
        
        # 1. Feature Projection
        x = self.feature_proj(x) # (B, T, R, d_model)
        
        # 2. Add Positional and Spatial Encodings
        x = x + self.pos_encoder.pe[:, :T, :].unsqueeze(2) # Add temporal PE
        x = x + self.spatial_embedding[:, :, :R, :] # Add spatial embedding
        
        # 3. Temporal Attention
        # Reshape to treat regions as independent batches for temporal attention
        x_temp = x.permute(0, 2, 1, 3).reshape(B * R, T, self.d_model) # (B*R, T, d_model)
        
        for layer in self.temporal_layers:
            x_temp = layer(x_temp)
            
        x = x_temp.reshape(B, R, T, self.d_model).permute(0, 2, 1, 3) # Back to (B, T, R, d_model)
        
        # 4. Spatial Attention
        # Reshape to treat time steps as independent batches for spatial attention
        x_spat = x.reshape(B * T, R, self.d_model) # (B*T, R, d_model)
        
        for layer in self.spatial_layers:
            x_spat = layer(x_spat)
            
        x = x_spat.reshape(B, T, R, self.d_model) # Back to (B, T, R, d_model)
        
        # 5. Feature Fusion / Aggregation
        # For yield prediction of a specific region at the end of the season, 
        # we can pool temporally and spatially, or just use the last time step.
        # Let's average across time.
        x_pooled = x.mean(dim=1) # (B, R, d_model)
        
        # 6. Regression Head
        features = self.regression_head[0](x_pooled) # Linear
        features = self.regression_head[1](features) # ReLU
        features = self.regression_head[2](features) # Dropout
        
        pred_yield = self.regression_head[3](features).squeeze(-1) # (B, R)
        
        # 7. Uncertainty estimation (Quantiles)
        quantiles = self.quantile_head(features) # (B, R, 2)
        
        return pred_yield, quantiles
