import os
import pandas as pd
import numpy as np
from datetime import datetime

def generate_proxy_ndvi():
    """
    Sentinel-2 was launched in 2015, but our primary historical yield dataset covers 1997-2013.
    Extracting 16 years of Landsat/MODIS imagery locally without GEE exceeds laptop-first MVP limits.
    As per project constraints, we use an optional SYNTHETIC_DEMO dataset as a fallback for software testing.
    This generates realistic, weather-correlated NDVI values.
    """
    print("Warning: Sentinel-2 data is not available for the 1997-2013 historical period.")
    print("Generating SYNTHETIC_DEMO NDVI dataset based on weather correlations for software testing only.")
    
    weather_path = os.path.join("data", "interim", "weather_monthly_tamil_nadu.csv")
    if not os.path.exists(weather_path):
        print("Please run download_weather.py first.")
        return
        
    weather = pd.read_csv(weather_path)
    
    # Generate synthetic NDVI that correlates loosely with precipitation and temperature
    # NDVI = base + f(precip) - f(temp_extreme) + noise
    np.random.seed(42)
    
    ndvi_list = []
    
    for idx, row in weather.iterrows():
        # A simple synthetic plant growth model
        # Base NDVI for bare soil / sparse veg in TN
        base_ndvi = 0.25
        
        # Precipitation response (delayed greening approximated by current month rain)
        rain_effect = min(row['precipitation'] / 400.0, 1.0) * 0.45 
        
        # Temperature stress (too hot reduces NDVI)
        temp_stress = max(0, (row['temperature_mean'] - 30) * 0.02)
        
        ndvi_mean = base_ndvi + rain_effect - temp_stress
        
        # Add spatial/temporal noise
        ndvi_mean += np.random.normal(0, 0.05)
        
        # Bound between 0 and 1
        ndvi_mean = max(0.05, min(0.95, ndvi_mean))
        
        ndvi_max = min(1.0, ndvi_mean + np.random.uniform(0.05, 0.15))
        ndvi_min = max(0.0, ndvi_mean - np.random.uniform(0.05, 0.15))
        
        ndvi_list.append({
            'district': row['district'],
            'year': row['year'],
            'month': row['month'],
            'ndvi_mean': ndvi_mean,
            'ndvi_max': ndvi_max,
            'ndvi_min': ndvi_min,
            'ndvi_std': np.random.uniform(0.02, 0.08),
            'ndvi_valid_observation_count': np.random.randint(2, 6),
            'is_synthetic_demo': True
        })
        
    ndvi_df = pd.DataFrame(ndvi_list)
    
    out_dir = os.path.join("data", "raw", "sentinel2")
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "sentinel2_ndvi_synthetic_demo.csv")
    ndvi_df.to_csv(raw_path, index=False)
    
    interim_dir = os.path.join("data", "interim")
    os.makedirs(interim_dir, exist_ok=True)
    processed_path = os.path.join(interim_dir, "ndvi_tamil_nadu.csv")
    ndvi_df.to_csv(processed_path, index=False)
    
    print(f"Saved SYNTHETIC_DEMO NDVI to {processed_path}")
    
    # Save provenance
    prov_dir = os.path.join("data")
    prov_path = os.path.join(prov_dir, "provenance.yaml")
    
    prov_entry = """
sentinel2_ndvi:
  name: "SYNTHETIC_DEMO Sentinel-2 proxy"
  provider: "Generated locally"
  url: "N/A"
  download_date: "2026-09-29"
  variables: ["ndvi_mean", "ndvi_max", "ndvi_min", "ndvi_std", "ndvi_valid_observation_count", "is_synthetic_demo"]
  processing_steps: 
    - Generated synthetically from weather variables due to 1997-2013 temporal mismatch with Sentinel-2 launch.
    - EXPLICITLY LABELLED AS SYNTHETIC DEMO.
"""
    with open(prov_path, "a") as f:
        f.write(prov_entry)

if __name__ == "__main__":
    generate_proxy_ndvi()
