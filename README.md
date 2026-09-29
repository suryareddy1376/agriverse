# AgriVerse: Digital Twin for Autonomous Crop Yield Prediction

AgriVerse is an end-to-end AI system that predicts crop yield by combining real historical crop yield data, weather observations, soil properties, and geographic information. It utilizes a Spatio-Temporal Transformer to model environmental evolution and provides a Digital Twin layer to simulate what-if scenarios (e.g., changes in rainfall, temperature, or vegetation).

## Project Overview
* **Primary Target:** Rice Yield in Tamil Nadu, India.
* **Data Principle:** Real-data first. Fabricated data is strictly prohibited for model training. (Note: Due to the Sentinel-2 launch in 2015 occurring after the historical yield dataset 1997-2013 timeframe, a synthetic proxy for historical NDVI was generated strictly for software testing/demo purposes, and is explicitly labelled `SYNTHETIC_DEMO`).
* **Core Model:** Spatio-Temporal Transformer.

---

## Detailed System Architecture

The AgriVerse system is broken down into several specialized layers that transform raw observations into simulated agricultural states.

### 1. The Spatio-Temporal Transformer Model
Instead of relying on standard tabular regressors (like Random Forest), AgriVerse employs a custom PyTorch Transformer designed for multi-dimensional sequence data (`Batch × Time × Regions × Features`). 
* **Factorized Attention:** To keep the model lightweight and "laptop-friendly", the architecture splits attention into two distinct phases:
  1. **Temporal Attention:** Learns how environmental variables (like rainfall and temperature) evolve over the 6-month growing season for a specific region.
  2. **Spatial Attention:** Learns the geographic relationships and spatial dependencies between different districts (e.g., how weather in a neighboring district impacts local yield).
* **Feature Fusion:** The learned spatial and temporal embeddings are aggregated (pooled) and passed into a Feed-Forward network to predict the final yield.

### 2. The Digital Twin & Scenario Simulation Layer
The Digital Twin is not a physical clone, but a **Data-Driven State Representation** of a crop environment at a specific time. 
* **State Initialization:** The twin initializes its state directly from the real, harmonized historical dataset (Weather, Geo, Crop type).
* **Scenario Simulator:** Users can inject "what-if" modifications (e.g., `Rainfall -20%` or `Temperature +1.5°C`). 
* **API Flow (`api/main.py`):** When a scenario is requested, the FastAPI backend clones the baseline environmental sequence, applies the mathematical modifiers to the specific features, scales the new tensor, and runs a live forward-pass through the Transformer to return the delta between the *Baseline Prediction* and the *Scenario Prediction*.

### 3. Uncertainty Estimation (Quantile Regression)
Point predictions are insufficient for agricultural planning. The Transformer features specialized regression heads trained using **Pinball Loss**. 
Alongside the main predicted yield, it simultaneously outputs the **10th percentile (Lower Bound)** and **90th percentile (Upper Bound)** predictions, providing an 80% confidence interval for every inference and simulation.

### 4. Spatial Harmonization & Entity Resolution
Agricultural data is notoriously misaligned. Yield is reported at the *Polygon* (district) level, while NASA weather data is retrieved at the *Point* (coordinate) level. 
* **Spatial Alignment:** AgriVerse uses `GeoPandas` to load official GeoJSON boundaries, calculate geometric centroids, and project them to accurate metric CRSs for querying external APIs.
* **Entity Resolution:** A canonical mapping registry (`district_mapping.csv`) resolves spelling inconsistencies across decades of government data (e.g., mapping "KANCHIPURAM" to "KANCHEEPURAM").

---

## Dataset Description & Data Dictionary

The project automatically builds a unified dataset (`agriVerse_tamilnadu_rice.csv`) combining multiple real-world data sources into a single aligned structure. 

### Dataset Overview
* **Geographic Focus:** Tamil Nadu, India (District-level)
* **Temporal Coverage:** 1997 – 2013 (17 Years)
* **Total Observations:** 481 valid harvesting seasons across 29 districts.
* **Format:** Wide-format tabular data (each row represents a specific district's harvest for a specific season/year, with a 6-month sequence of environmental features).

### Data Dictionary

#### 1. Identifiers & Target Variables (Source: Directorate of Economics & Statistics)
* **`location_id`**: Unique identifier for the row (e.g., `ARIYALUR_2009_Kharif`).
* **`state` / `district`**: Canonical geographic identifiers. District names underwent entity resolution to match geographic shapefiles.
* **`year` / `season`**: The harvest year and agricultural season (e.g., Kharif, Rabi).
* **`area_harvested`**: Total harvested area in hectares.
* **`production`**: Total production in tonnes.
* **`yield_tonnes_per_ha`** *(TARGET)*: Calculated strictly as `production / area_harvested`.

#### 2. Temporal Weather Features (Source: NASA POWER API)
Weather features are represented as a 6-month historical sequence leading up to the harvest. For each feature below, there are 6 columns (e.g., `rainfall_m1` to `rainfall_m6`).
* **`rainfall_m1` to `m6`**: Monthly precipitation (mm/day).
* **`temperature_mean_m1` to `m6`**: Mean daily temperature at 2 meters (°C).
* **`humidity_m1` to `m6`**: Relative humidity at 2 meters (%).
* **`solar_radiation_m1` to `m6`**: All Sky Surface Shortwave Downward Irradiance (kW-hr/m^2/day).

#### 3. Temporal Vegetation Features (Source: Generated Proxy)
* **`ndvi_m1` to `m6`**: Normalized Difference Vegetation Index. 
  > **⚠️ SCIENTIFIC NOTE:** Because the real yield dataset spans 1997-2013 and Sentinel-2 was not launched until 2015, this specific feature was generated as an aridity-driven proxy. It is strictly labelled as **SYNTHETIC_DEMO** data in the pipeline to prevent time-travel target leakage.

#### 4. Spatial / Geographic Features (Source: DataMeet / GeoJSON)
* **`centroid_lat` / `centroid_lon`**: The geographic center coordinates of the district.
* **`area_sqkm`**: Total area of the district in square kilometers.

> **Note on Soil Data:** While the pipeline is equipped to pull from SoilGrids 2.0 (ISRIC), the API timed out during the initial dataset construction phase. To keep the MVP functional and strictly avoid fabricating data, soil features (pH, organic carbon, clay content) were gracefully omitted from this specific build, but the capability remains in the codebase.

### Temporal Alignment Strategy
To prevent **Target Leakage**, the pipeline automatically aligns high-frequency weather and satellite data backwards from the harvest window. Extracts time-series sequences representing the 6-month growing season prior to harvest to strictly prevent target leakage. This allows the Spatio-Temporal Transformer to mathematically "watch" the crop season unfold before making a prediction.

---

## Setup Instructions

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate  # Or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 2. Data Ingestion
Run these sequentially to download real data:
```bash
python scripts/download_geography.py
python scripts/download_yield.py
python scripts/download_weather.py
python scripts/download_sentinel2.py
```

### 3. Build Unified Dataset
```bash
python scripts/build_real_dataset.py
```

### 4. Train Models
```bash
python scripts/train.py
python scripts/train_transformer.py
```

### 5. Run API & Dashboard
Open two terminals.
Terminal 1 (FastAPI):
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Terminal 2 (Streamlit Dashboard):
```bash
streamlit run app/dashboard.py
```

---

## Limitations & Future Work
* **Scientific Limitations:** Official yield data is administrative. Prediction uncertainty relies on quantile regression bounds, which do not guarantee real-world agricultural certainty. Agricultural management variables (fertilizer, irrigation) are omitted from this MVP.
* **Satellite Limitations:** Historical yield data ends in 2013, before Sentinel-2 existed. A true production system would integrate Landsat/MODIS historical data using Google Earth Engine or Planetary Computer, but this exceeds the memory/time footprint for a laptop-first MVP.
* **Future Work:** Extend to full Indian geography, integrate actual MODIS historical NDVI, and add larger sequential attention layers.
