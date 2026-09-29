import os
import time
import requests
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

def download_soilgrids():
    print("Loading district centroids...")
    districts_path = os.path.join("data", "interim", "tamil_nadu_districts.geojson")
    
    if not os.path.exists(districts_path):
        print("Please run download_geography.py first.")
        return
        
    gdf = gpd.read_file(districts_path)
    
    out_dir = os.path.join("data", "raw", "soil")
    os.makedirs(out_dir, exist_ok=True)
    
    properties = ["phh2o", "soc", "clay"]
    depth = "15-30cm" # Representative depth for crop root zone
    value_type = "mean"
    
    prop_query = "&".join([f"property={p}" for p in properties])
    
    all_soil = []
    
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Downloading SoilGrids"):
        district = row['district']
        lat = row['centroid_lat']
        lon = row['centroid_lon']
        
        url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon:.4f}&lat={lat:.4f}&{prop_query}&depth={depth}&value={value_type}"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    
                    record = {'district': district}
                    for prop_layer in data.get('properties', {}).get('layers', []):
                        prop_name = prop_layer['name']
                        try:
                            # SoilGrids returns values mapped with a conversion factor (e.g., pH * 10)
                            # We get the raw value and the unit measure
                            val = prop_layer['depths'][0]['values'][value_type]
                            unit = prop_layer['unit_measure']['target_units']
                            factor = prop_layer['unit_measure']['d_factor']
                            
                            if val is not None:
                                record[f"soil_{prop_name}"] = val / factor
                            else:
                                record[f"soil_{prop_name}"] = pd.NA
                        except Exception as e:
                            record[f"soil_{prop_name}"] = pd.NA
                            
                    all_soil.append(record)
                    break
                else:
                    print(f"Error for {district}: {response.status_code}")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception for {district} on attempt {attempt+1}: {e}")
                time.sleep(2)
                
        time.sleep(1) # rate limit for ISRIC
        
    if all_soil:
        df = pd.DataFrame(all_soil)
        
        raw_path = os.path.join(out_dir, "soilgrids_raw.csv")
        df.to_csv(raw_path, index=False)
        print(f"Saved raw soil to {raw_path}")
        
        # Rename slightly for consistency
        df.rename(columns={
            'soil_phh2o': 'soil_ph',
            'soil_soc': 'soil_organic_carbon',
            'soil_bdod': 'bulk_density'
        }, inplace=True)
        
        interim_dir = os.path.join("data", "interim")
        os.makedirs(interim_dir, exist_ok=True)
        processed_path = os.path.join(interim_dir, "soil_tamil_nadu.csv")
        df.to_csv(processed_path, index=False)
        print(f"Saved processed soil to {processed_path}")
        
        # Save provenance
        prov_dir = os.path.join("data")
        prov_path = os.path.join(prov_dir, "provenance.yaml")
        
        prov_entry = f"""
soil_tamil_nadu:
  name: "SoilGrids 2.0"
  provider: "ISRIC - World Soil Information"
  url: "https://rest.isric.org/"
  download_date: "2026-09-29"
  coverage: "Tamil Nadu district centroids, depth 15-30cm"
  variables: {properties}
  processing_steps: 
    - Downloaded point data via REST API
    - Applied division factors to get target units
"""
        with open(prov_path, "a") as f:
            f.write(prov_entry)
            
if __name__ == "__main__":
    download_soilgrids()
