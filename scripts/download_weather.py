import os
import time
import requests
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

def download_nasa_power_weather():
    print("Loading district centroids...")
    districts_path = os.path.join("data", "interim", "tamil_nadu_districts.geojson")
    
    if not os.path.exists(districts_path):
        print("Please run download_geography.py first.")
        return
        
    gdf = gpd.read_file(districts_path)
    
    # Check yield data to know what years we need
    yield_path = os.path.join("data", "interim", "yield_tamil_nadu_rice.csv")
    if os.path.exists(yield_path):
        yield_df = pd.read_csv(yield_path)
        start_year = yield_df['year'].min()
        end_year = yield_df['year'].max()
    else:
        start_year = 1997
        end_year = 2015
        
    print(f"Required weather data period: {start_year} to {end_year}")
    
    # NASA POWER parameters
    # T2M: Temperature at 2 Meters (C)
    # T2M_MIN: Minimum Temperature at 2 Meters (C)
    # T2M_MAX: Maximum Temperature at 2 Meters (C)
    # PRECTOTCORR: Precipitation Corrected (mm/day)
    # RH2M: Relative Humidity at 2 Meters (%)
    # ALLSKY_SFC_SW_DWN: All Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day) -> Solar Radiation
    # WS2M: Wind Speed at 2 Meters (m/s)
    
    params = "T2M,T2M_MIN,T2M_MAX,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN,WS2M"
    start_date = f"{start_year}0101"
    end_date = f"{end_year}1231"
    
    out_dir = os.path.join("data", "raw", "weather")
    os.makedirs(out_dir, exist_ok=True)
    
    all_weather = []
    
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Downloading NASA POWER weather"):
        district = row['district']
        lat = row['centroid_lat']
        lon = row['centroid_lon']
        
        url = f"https://power.larc.nasa.gov/api/temporal/daily/point?parameters={params}&community=AG&longitude={lon:.4f}&latitude={lat:.4f}&start={start_date}&end={end_date}&format=JSON"
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                if response.status_code == 200:
                    data = response.json()
                    
                    # Convert properties to dataframe
                    df = pd.DataFrame(data['properties']['parameter'])
                    df.index.name = 'date'
                    df.reset_index(inplace=True)
                    
                    # -999 is missing in NASA POWER
                    df.replace(-999.0, pd.NA, inplace=True)
                    
                    df['district'] = district
                    all_weather.append(df)
                    break
                else:
                    print(f"Error for {district}: {response.status_code}")
                    time.sleep(2)
            except Exception as e:
                print(f"Exception for {district} on attempt {attempt+1}: {e}")
                time.sleep(2)
        
        time.sleep(0.5) # rate limit
        
    if all_weather:
        combined = pd.concat(all_weather, ignore_index=True)
        # Convert date string to datetime
        combined['date'] = pd.to_datetime(combined['date'], format="%Y%m%d")
        
        raw_path = os.path.join(out_dir, "nasa_power_daily.csv")
        combined.to_csv(raw_path, index=False)
        print(f"Saved raw weather to {raw_path}")
        
        # Monthly aggregation
        combined['year'] = combined['date'].dt.year
        combined['month'] = combined['date'].dt.month
        
        monthly = combined.groupby(['district', 'year', 'month']).agg({
            'T2M': 'mean',
            'T2M_MIN': 'mean',
            'T2M_MAX': 'mean',
            'PRECTOTCORR': 'sum',
            'RH2M': 'mean',
            'ALLSKY_SFC_SW_DWN': 'mean',
            'WS2M': 'mean'
        }).reset_index()
        
        monthly.rename(columns={
            'T2M': 'temperature_mean',
            'T2M_MIN': 'temperature_min',
            'T2M_MAX': 'temperature_max',
            'PRECTOTCORR': 'precipitation',
            'RH2M': 'relative_humidity',
            'ALLSKY_SFC_SW_DWN': 'solar_radiation',
            'WS2M': 'wind_speed'
        }, inplace=True)
        
        interim_dir = os.path.join("data", "interim")
        os.makedirs(interim_dir, exist_ok=True)
        processed_path = os.path.join(interim_dir, "weather_monthly_tamil_nadu.csv")
        monthly.to_csv(processed_path, index=False)
        print(f"Saved monthly weather to {processed_path}")
        
        # Save provenance
        prov_dir = os.path.join("data")
        prov_path = os.path.join(prov_dir, "provenance.yaml")
        
        prov_entry = f"""
weather_tamil_nadu:
  name: "NASA POWER Daily Meteorology"
  provider: "NASA Langley Research Center"
  url: "https://power.larc.nasa.gov/"
  download_date: "2026-09-29"
  coverage: "Tamil Nadu district centroids, {start_year}-{end_year}"
  variables: ["T2M", "T2M_MIN", "T2M_MAX", "PRECTOTCORR", "RH2M", "ALLSKY_SFC_SW_DWN", "WS2M"]
  processing_steps: 
    - Downloaded daily point data for district centroids
    - Missing values (-999) handled
    - Aggregated to monthly mean/sum
"""
        with open(prov_path, "a") as f:
            f.write(prov_entry)
            
if __name__ == "__main__":
    download_nasa_power_weather()
