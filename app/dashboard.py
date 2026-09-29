import streamlit as st
import pandas as pd
import numpy as np
import os
import requests
import plotly.express as px
import plotly.graph_objects as go
import json

st.set_page_config(page_title="AgriVerse Digital Twin", layout="wide", initial_sidebar_state="expanded")

# --- Constants & State ---
API_URL = "http://localhost:8000"

st.sidebar.title("AgriVerse")
st.sidebar.markdown("Digital Twin for Autonomous Crop Yield Prediction")

page = st.sidebar.radio("Navigation", [
    "Overview", 
    "Real Data Explorer", 
    "Digital Twin & Simulation", 
    "Model Evaluation"
])

@st.cache_data
def load_unified_data():
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed', 'agriVerse_tamilnadu_rice.csv')
    return pd.read_csv(data_path)

@st.cache_data
def load_metrics():
    metrics_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'baseline_metrics.csv')
    return pd.read_csv(metrics_path)

@st.cache_data
def load_transformer_preds():
    preds_path = os.path.join(os.path.dirname(__file__), '..', 'artifacts', 'transformer_test_predictions.csv')
    return pd.read_csv(preds_path)

if page == "Overview":
    st.title("AgriVerse Digital Twin")
    st.markdown("""
    ### Spatio-Temporal Crop Yield Prediction
    
    This system predicts rice yield in Tamil Nadu using historical environmental observations and a custom **Spatio-Temporal Transformer**.
    
    * **Target**: Rice yield (tonnes/ha) from official records (1997-2013).
    * **Weather**: Real NASA POWER daily observations.
    * **Satellite**: SYNTHETIC_DEMO proxy (since Sentinel-2 launched in 2015, outside the target window).
    * **Geography**: Open district boundaries.
    
    #### DATA MODE: REAL 
    *(Except for historical NDVI which uses a synthetic proxy due to time-travel constraints. This is explicitly noted.)*
    """)
    
    st.info("Navigate to the **Digital Twin & Simulation** tab to modify environmental variables and see the predicted yield change.")
    
elif page == "Real Data Explorer":
    st.title("Real Data Explorer")
    df = load_unified_data()
    
    districts = sorted(df['district'].unique())
    selected_dist = st.selectbox("Select District", districts)
    
    dist_df = df[df['district'] == selected_dist].sort_values(by='year')
    
    st.subheader(f"Historical Yield for {selected_dist}")
    fig = px.line(dist_df, x='year', y='yield_tonnes_per_ha', markers=True, title=f"Rice Yield Over Time - {selected_dist}")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Environmental Inputs (Mean across season)")
    # Average the sequence features for simple display
    for var in ['temperature_mean', 'rainfall']:
        cols = [f"{var}_m{i}" for i in range(1, 7) if f"{var}_m{i}" in df.columns]
        if cols:
            dist_df[f'{var}_avg'] = dist_df[cols].mean(axis=1)
            
    fig2 = px.line(dist_df, x='year', y=['temperature_mean_avg', 'rainfall_avg'], title="Seasonal Weather")
    st.plotly_chart(fig2, use_container_width=True)

elif page == "Digital Twin & Simulation":
    st.title("Scenario Simulation")
    
    df = load_unified_data()
    
    col1, col2 = st.columns(2)
    with col1:
        dist = st.selectbox("Select District", sorted(df['district'].unique()))
    with col2:
        dist_df = df[df['district'] == dist]
        year = st.selectbox("Select Baseline Year", sorted(dist_df['year'].unique(), reverse=True))
        
    season = dist_df[dist_df['year'] == year]['season'].values[0]
    
    st.markdown("---")
    st.subheader("Simulate Environmental Changes")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        rain_mod = st.slider("Rainfall Modification (%)", -50.0, 50.0, 0.0, 5.0)
    with c2:
        temp_mod = st.slider("Temperature Change (°C)", -3.0, 3.0, 0.0, 0.5)
    with c3:
        ndvi_mod = st.slider("Vegetation Index (NDVI) Change (%)", -30.0, 30.0, 0.0, 5.0)
        
    if st.button("Run Simulation"):
        try:
            # Baseline Predict
            res_base = requests.post(f"{API_URL}/predict", json={
                "state": "TAMIL NADU",
                "district": dist,
                "crop": "rice",
                "season": season,
                "year": int(year)
            }).json()
            
            # Scenario Predict
            res_sim = requests.post(f"{API_URL}/simulate", json={
                "state": "TAMIL NADU",
                "district": dist,
                "crop": "rice",
                "season": season,
                "year": int(year),
                "rainfall_modifier_pct": rain_mod,
                "temp_modifier_abs": temp_mod,
                "ndvi_modifier_pct": ndvi_mod
            }).json()
            
            st.markdown("### Simulation Results")
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Baseline Predicted Yield", f"{res_base.get('prediction_tonnes_per_ha', 0):.2f} t/ha")
            rc2.metric("Scenario Predicted Yield", f"{res_sim.get('scenario_prediction', 0):.2f} t/ha", delta=f"{res_sim.get('absolute_difference', 0):.2f} t/ha")
            rc3.metric("Change", f"{res_sim.get('percentage_difference', 0):.1f} %", delta_color="normal")
            
            st.info(f"Uncertainty Bound (Baseline): {res_base.get('lower_bound', 0):.2f} - {res_base.get('upper_bound', 0):.2f} t/ha")
            
        except Exception as e:
            st.error(f"Make sure FastAPI is running. Error: {e}")
            
elif page == "Model Evaluation":
    st.title("Model Comparison")
    
    st.markdown("Metrics calculated on REAL chronological holdout dataset (Test Split: 2012-2013).")
    
    try:
        metrics = load_metrics()
        test_metrics = metrics[metrics['split'] == 'test'].set_index('model')
        
        st.dataframe(test_metrics[['rmse', 'r2', 'mae']].style.highlight_min(subset=['rmse'], color='lightgreen'))
        
        st.subheader("Transformer Backtesting (Test Set)")
        trans_preds = load_transformer_preds()
        
        fig = px.scatter(trans_preds, x='actual', y='predicted', title="Actual vs Predicted (Spatio-Temporal Transformer)")
        fig.add_shape(type="line", x0=0, y0=0, x1=trans_preds['actual'].max(), y1=trans_preds['actual'].max(), line=dict(color="red", dash="dash"))
        st.plotly_chart(fig)
        
    except FileNotFoundError:
        st.warning("Metrics files not found. Run train.py and train_transformer.py first.")
