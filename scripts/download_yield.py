import os
import pandas as pd
import numpy as np

def download_and_process_yield():
    print("Downloading India Crop Production dataset...")
    url = "https://raw.githubusercontent.com/SourajitaDewasi/Crop-Yield-Prediction-in-India-using-ML/master/crop_production.csv"
    
    try:
        df = pd.read_csv(url)
        print(f"Downloaded {len(df)} records.")
    except Exception as e:
        print(f"Failed to download dataset automatically: {e}")
        print("Please manually download crop_production.csv and place it in data/raw/yield/")
        return
        
    out_dir = os.path.join("data", "raw", "yield")
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "crop_production.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved raw data to {raw_path}")
    
    # Filter for Tamil Nadu and Rice
    print("Filtering for Tamil Nadu and Rice...")
    tn_rice = df[(df['State_Name'].str.strip().str.upper() == 'TAMIL NADU') & 
                 (df['Crop'].str.strip().str.upper() == 'RICE')].copy()
    
    print(f"Found {len(tn_rice)} records for Tamil Nadu Rice.")
    
    # Process
    tn_rice.rename(columns={
        'State_Name': 'state',
        'District_Name': 'district',
        'Crop_Year': 'year',
        'Season': 'season',
        'Crop': 'crop',
        'Area': 'area_harvested',
        'Production': 'production'
    }, inplace=True)
    
    # Clean up strings
    tn_rice['state'] = tn_rice['state'].str.strip().str.upper()
    tn_rice['district'] = tn_rice['district'].str.strip().str.upper()
    tn_rice['crop'] = tn_rice['crop'].str.strip().str.lower()
    tn_rice['season'] = tn_rice['season'].str.strip()
    
    # Calculate yield (tonnes/hectare) since the dataset only provides Area and Production.
    # Note: Explicitly documented as calculated per project requirements.
    tn_rice['production'] = pd.to_numeric(tn_rice['production'], errors='coerce')
    tn_rice['area_harvested'] = pd.to_numeric(tn_rice['area_harvested'], errors='coerce')
    
    # Remove NaN or 0 area to avoid division by zero
    tn_rice = tn_rice.dropna(subset=['production', 'area_harvested'])
    tn_rice = tn_rice[tn_rice['area_harvested'] > 0]
    
    tn_rice['yield_tonnes_per_ha'] = tn_rice['production'] / tn_rice['area_harvested']
    
    interim_dir = os.path.join("data", "interim")
    os.makedirs(interim_dir, exist_ok=True)
    processed_path = os.path.join(interim_dir, "yield_tamil_nadu_rice.csv")
    
    tn_rice.to_csv(processed_path, index=False)
    print(f"Saved processed yield data to {processed_path}")
    
    # Save provenance
    prov_dir = os.path.join("data")
    prov_path = os.path.join(prov_dir, "provenance.yaml")
    
    prov_entry = """
yield_tamil_nadu_rice:
  name: "India Crop Production Dataset (1997-2015)"
  provider: "Ministry of Agriculture and Farmers Welfare (via Kaggle/GitHub)"
  url: "https://raw.githubusercontent.com/SourajitaDewasi/Crop-Yield-Prediction-in-India-using-ML/master/crop_production.csv"
  download_date: "2026-09-29"
  variables: ["state", "district", "year", "season", "crop", "area", "production"]
  processing_steps: 
    - Filtered for state=Tamil Nadu, crop=Rice
    - Yield calculated as production / area
"""
    with open(prov_path, "a") as f:
        f.write(prov_entry)

if __name__ == "__main__":
    download_and_process_yield()
