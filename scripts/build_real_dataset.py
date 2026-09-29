import os
import pandas as pd
import numpy as np
import geopandas as gpd

def create_district_mapping():
    mapping_dir = os.path.join("data", "mappings")
    os.makedirs(mapping_dir, exist_ok=True)
    mapping_path = os.path.join(mapping_dir, "district_mapping.csv")
    
    # Map from Yield names to Geo names (which are used for Weather and Soil as canonical)
    data = [
        {'source_name': 'KANCHIPURAM', 'canonical_name': 'KANCHEEPURAM', 'state': 'TAMIL NADU', 'valid_from': None, 'valid_to': None, 'mapping_reason': 'Spelling variation'},
        {'source_name': 'THE NILGIRIS', 'canonical_name': 'NILGIRIS', 'state': 'TAMIL NADU', 'valid_from': None, 'valid_to': None, 'mapping_reason': 'Prefix difference'},
        {'source_name': 'TUTICORIN', 'canonical_name': 'THOOTHUKUDI', 'state': 'TAMIL NADU', 'valid_from': None, 'valid_to': None, 'mapping_reason': 'Historical name vs Official name'},
        {'source_name': 'TIRUCHIRAPPALLI', 'canonical_name': 'TIRUCHCHIRAPPALLI', 'state': 'TAMIL NADU', 'valid_from': None, 'valid_to': None, 'mapping_reason': 'Spelling variation'},
        {'source_name': 'TIRUNELVELI', 'canonical_name': 'TIRUNELVELI KATTABO', 'state': 'TAMIL NADU', 'valid_from': None, 'valid_to': None, 'mapping_reason': 'Legacy name in GeoJSON'}
    ]
    
    mapping_df = pd.DataFrame(data)
    mapping_df.to_csv(mapping_path, index=False)
    print(f"Created district mapping at {mapping_path}")
    return mapping_df

def temporal_alignment(weather, ndvi, yield_df, crop_calendar):
    """
    For a given crop season, extract the relevant monthly features.
    Rice in Tamil Nadu has multiple seasons (Kharif, Rabi, Summer).
    We align the weather to the months leading up to harvest.
    """
    # Create wide format features
    feature_rows = []
    
    for idx, row in yield_df.iterrows():
        dist = row['district']
        yr = row['year']
        season = row['season'].strip()
        
        # Simplified crop calendar for MVP based on season string
        # Kharif: Jun-Oct
        # Rabi: Nov-Mar
        # Whole Year: Jan-Dec
        # We will take the 5 months of weather before typical harvest.
        
        if 'Kharif' in season:
            start_month, end_month = 6, 10
            feature_year = yr
        elif 'Rabi' in season:
            # Rabi spans across years, but yield year is typically the harvest year.
            # E.g., Nov yr-1 to Mar yr. We'll approximate to Nov-Mar of yr.
            start_month, end_month = 11, 3
            feature_year = yr
        else:
            # Fallback to general growing season Jun-Nov
            start_month, end_month = 6, 11
            feature_year = yr
            
        # Get weather for this district and year
        w_dist = weather[(weather['district'] == dist)]
        n_dist = ndvi[(ndvi['district'] == dist)]
        
        # For simplicity in MVP, we take the average of the weather features for the feature_year
        # A full sequence model would take the 6 months as separate columns (or sequence)
        w_season = w_dist[w_dist['year'] == feature_year]
        n_season = n_dist[n_dist['year'] == feature_year]
        
        if len(w_season) == 0:
            continue
            
        # Create sequence columns (m1 to m6 for the growing season)
        # We will just pad to 6 months
        months_to_extract = [(start_month + i - 1) % 12 + 1 for i in range(6)]
        
        feat_dict = {'location_id': f"{dist}_{yr}_{season}", **row.to_dict()}
        
        # We will extract 6 months of sequences
        valid = True
        for i, m in enumerate(months_to_extract):
            # If month < start_month and it's Rabi, we might need yr-1. Handling simplified for MVP.
            m_data = w_season[w_season['month'] == m]
            nm_data = n_season[n_season['month'] == m]
            
            if len(m_data) > 0:
                m_row = m_data.iloc[0]
                feat_dict[f'temperature_mean_m{i+1}'] = m_row['temperature_mean']
                feat_dict[f'rainfall_m{i+1}'] = m_row['precipitation']
                feat_dict[f'humidity_m{i+1}'] = m_row['relative_humidity']
                feat_dict[f'solar_radiation_m{i+1}'] = m_row['solar_radiation']
            else:
                valid = False
                
            if len(nm_data) > 0:
                feat_dict[f'ndvi_m{i+1}'] = nm_data.iloc[0]['ndvi_mean']
            else:
                feat_dict[f'ndvi_m{i+1}'] = pd.NA
                
        if valid:
            feature_rows.append(feat_dict)
            
    return pd.DataFrame(feature_rows)

def build_dataset():
    print("Building unified real dataset...")
    
    # 1. Load data
    yield_df = pd.read_csv("data/interim/yield_tamil_nadu_rice.csv")
    weather_df = pd.read_csv("data/interim/weather_monthly_tamil_nadu.csv")
    
    soil_path = "data/interim/soil_tamil_nadu.csv"
    if os.path.exists(soil_path):
        soil_df = pd.read_csv(soil_path)
    else:
        print("Soil data not ready yet. Skipping soil.")
        soil_df = pd.DataFrame(columns=['district'])
        
    ndvi_df = pd.read_csv("data/interim/ndvi_tamil_nadu.csv")
    geo_df = gpd.read_file("data/interim/tamil_nadu_districts.geojson")
    
    # 2. Entity Resolution
    mapping_df = create_district_mapping()
    mapping_dict = dict(zip(mapping_df.source_name, mapping_df.canonical_name))
    
    yield_df['canonical_district'] = yield_df['district'].map(lambda x: mapping_dict.get(x, x))
    
    # 3. Filter Yield to only include districts we have geography for
    geo_districts = set(geo_df['district'].unique())
    yield_df = yield_df[yield_df['canonical_district'].isin(geo_districts)]
    
    # Rename back so merge works smoothly
    yield_df.drop(columns=['district'], inplace=True)
    yield_df.rename(columns={'canonical_district': 'district'}, inplace=True)
    
    # 4. Temporal Alignment
    unified_df = temporal_alignment(weather_df, ndvi_df, yield_df, crop_calendar=None)
    
    # 5. Spatial Join (Static Features)
    if not soil_df.empty:
        unified_df = pd.merge(unified_df, soil_df, on='district', how='left')
        
    geo_features = geo_df[['district', 'area_sqkm', 'centroid_lat', 'centroid_lon']]
    unified_df = pd.merge(unified_df, geo_features, on='district', how='left')
    
    # Validation checks
    print("\nData Quality Check:")
    print(f"Total records: {len(unified_df)}")
    print(f"Missing yield: {unified_df['yield_tonnes_per_ha'].isna().sum()}")
    if not soil_df.empty:
        print(f"Missing soil pH: {unified_df['soil_ph'].isna().sum()}")
    
    # Drop records with missing target
    unified_df = unified_df.dropna(subset=['yield_tonnes_per_ha'])
    
    # Data Quality Report
    dq_report = unified_df.isnull().sum().reset_index()
    dq_report.columns = ['Feature', 'Missing_Values']
    dq_report['Missing_Percentage'] = (dq_report['Missing_Values'] / len(unified_df)) * 100
    
    art_dir = os.path.join("artifacts")
    os.makedirs(art_dir, exist_ok=True)
    dq_report.to_csv(os.path.join(art_dir, "data_quality_summary.csv"), index=False)
    
    # Create HTML report
    with open(os.path.join(art_dir, "data_quality_report.html"), "w") as f:
        f.write("<html><body><h1>Data Quality Report</h1>")
        f.write(dq_report.to_html(index=False))
        f.write("</body></html>")
        
    out_dir = os.path.join("data", "processed")
    os.makedirs(out_dir, exist_ok=True)
    
    csv_path = os.path.join(out_dir, "agriVerse_tamilnadu_rice.csv")
    parquet_path = os.path.join(out_dir, "agriVerse_tamilnadu_rice.parquet")
    
    unified_df.to_csv(csv_path, index=False)
    unified_df.to_parquet(parquet_path, index=False)
    
    print(f"\nSaved final unified dataset to {csv_path}")
    
    print("\nCoverage Report:")
    print(f"Districts: {unified_df['district'].nunique()}")
    print(f"Years: {unified_df['year'].nunique()}")
    print(f"Rice observations: {len(unified_df)}")

if __name__ == "__main__":
    build_dataset()
