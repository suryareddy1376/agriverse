import os
import json
import urllib.request
import geopandas as gpd

def download_tamil_nadu_geography():
    print("Downloading India district GeoJSON...")
    url = "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read())
        
    print(f"Fetched {len(data['features'])} total districts.")
    
    tn_features = [f for f in data['features'] if f['properties'].get('NAME_1') == 'Tamil Nadu']
    print(f"Filtered {len(tn_features)} Tamil Nadu districts.")
    
    # Save the raw filtered GeoJSON
    out_dir = os.path.join("data", "raw", "geography")
    os.makedirs(out_dir, exist_ok=True)
    
    raw_path = os.path.join(out_dir, "tamil_nadu_districts_raw.geojson")
    with open(raw_path, 'w') as f:
        json.dump({"type": "FeatureCollection", "features": tn_features}, f)
        
    print(f"Saved raw GeoJSON to {raw_path}")
    
    # Load with GeoPandas to standardize and get centroids
    gdf = gpd.read_file(raw_path)
    # Simplify names
    gdf['district'] = gdf['NAME_2'].str.upper()
    gdf['state'] = "TAMIL NADU"
    
    # Project to a metric CRS to calculate area and accurate centroids, e.g. EPSG:32644 (UTM 44N) or similar.
    # We will use EPSG:4326 for storing, but let's calculate centroids accurately first
    gdf_metric = gdf.to_crs(epsg=32644)
    gdf['area_sqkm'] = gdf_metric.geometry.area / 10**6
    gdf['centroid_lon'] = gdf.geometry.centroid.x
    gdf['centroid_lat'] = gdf.geometry.centroid.y
    
    # Keep essential columns
    final_gdf = gdf[['state', 'district', 'area_sqkm', 'centroid_lon', 'centroid_lat', 'geometry']]
    
    processed_path = os.path.join("data", "interim", "tamil_nadu_districts.geojson")
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    final_gdf.to_file(processed_path, driver="GeoJSON")
    print(f"Saved processed geography to {processed_path}")
    
    # Save provenance
    prov_dir = os.path.join("data")
    prov_path = os.path.join(prov_dir, "provenance.yaml")
    
    prov_entry = """
geography_tamil_nadu:
  name: "Tamil Nadu District Boundaries (2011)"
  provider: "geohacker/india (GitHub)"
  url: "https://raw.githubusercontent.com/geohacker/india/master/district/india_district.geojson"
  download_date: "2026-09-29"
  variables: ["geometry", "district name"]
"""
    with open(prov_path, "a") as f:
        f.write(prov_entry)
        
if __name__ == "__main__":
    download_tamil_nadu_geography()
