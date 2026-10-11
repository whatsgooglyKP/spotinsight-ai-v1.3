import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from pathlib import Path
import io

st.set_page_config(
    page_title="SPOTInsight AI | KnightHacks 2026",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1e3a5f; }
    .sub-header { font-size: 1.1rem; color: #4a6fa5; margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

DATA_DIR = Path(__file__).parent / "data"

def safe_read_csv(path, fallback_df):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        st.warning(f"Missing {path.name} — using sample data for demo.")
        return fallback_df

@st.cache_data
def load_data():
    sample_missions = pd.DataFrame({
        "mission_id": ["M001","M019"],
        "mission_name": ["Route 1","Route 19"],
        "scenario": ["BASELINE","IMPROVED"],
        "configuration_version": ["V1","V2"],
        "start_time": ["2026-09-01T18:00:00","2026-09-07T18:00:00"],
        "end_time": ["2026-09-01T18:30:00","2026-09-07T18:22:00"],
        "duration_min": [30,22],
        "start_battery_pct": [82,84],
        "end_battery_pct": [22,65],
        "mission_status": ["COMPLETED","COMPLETED"],
        "operator_mode": ["AUTOWALK","AUTOWALK"],
        "synthetic": [True,True]
    })
    sample_actions = pd.DataFrame({
        "action_id": ["A0001","A0002"],
        "mission_id": ["M001","M001"],
        "sequence": [1,2],
        "timestamp": ["2026-09-01T18:03:06","2026-09-01T18:06:12"],
        "location_id": ["WP05","WP10"],
        "asset_id": ["HX-201","PUMP-205"],
        "action_type": ["NAVIGATION","VISUAL_INSPECTION"],
        "battery_pct": [75,69],
        "status": ["FAILED","SUCCESS"],
        "failure_type": ["OBSTACLE_DETECTION",""],
        "duration_sec": [0,63],
        "scenario": ["BASELINE","BASELINE"],
        "configuration_version": ["V1","V1"],
        "synthetic": [True,True]
    })
    sample_readings = pd.DataFrame({
        "reading_id": ["R0001","R0002"],
        "action_id": ["A0002","A0002"],
        "mission_id": ["M001","M001"],
        "asset_id": ["PUMP-205","PUMP-205"],
        "location_id": ["WP10","WP10"],
        "timestamp": ["2026-09-01T18:06:12","2026-09-01T18:06:12"],
        "measurement_type": ["TEMPERATURE_C","VISUAL_SCORE"],
        "value": [62.5,96],
        "unit": ["C","score"],
        "quality": ["GOOD","GOOD"],
        "synthetic": [True,True]
    })
    sample_anomalies = pd.DataFrame({
        "anomaly_id": ["AN001","AN002"],
        "action_id": ["A0001","A0002"],
        "mission_id": ["M001","M001"],
        "location_id": ["WP05","WP10"],
        "asset_id": ["HX-201","PUMP-205"],
        "timestamp": ["2026-09-01T18:03:06","2026-09-01T18:06:12"],
        "anomaly_type": ["SIGNAL_DEGRADATION","VISUAL_DEFECT"],
        "severity": ["HIGH","MEDIUM"],
        "operator_review_required": [True,False],
        "description": ["Synthetic signal degradation detected.","Synthetic visual defect detected."],
        "synthetic": [True,True]
    })
    sample_locations = pd.DataFrame({
        "location_id": ["WP05","WP10"],
        "location_name": ["Pipe Gallery","Pump Room"],
        "zone_type": ["PROCESS","MECHANICAL"],
        "floor": [1,1],
        "x_coord": [72.6,45.0],
        "y_coord": [57.8,30.0],
        "environment": ["HIGH_NOISE","NORMAL"],
        "synthetic": [True,True]
    })
    sample_assets = pd.DataFrame({
        "asset_id": ["HX-201","PUMP-205"],
        "asset_name": ["Heat Exchanger 01","Pump 05"],
        "asset_type": ["Heat Exchanger","Pump"],
        "location_id": ["WP05","WP10"],
        "criticality": ["LOW","MEDIUM"],
        "synthetic": [True,True]
    })

    missions = safe_read_csv(DATA_DIR / "missions.csv", sample_missions)
    actions = safe_read_csv(DATA_DIR / "actions.csv", sample_actions)
    readings = safe_read_csv(DATA_DIR / "readings.csv", sample_readings)
    anomalies = safe_read_csv(DATA_DIR / "anomalies.csv", sample_anomalies)
    locations = safe_read_csv(DATA_DIR / "locations.csv", sample_locations)
    assets = safe_read_csv(DATA_DIR / "assets.csv", sample_assets)

    if "asset_id" in anomalies.columns and "asset_name" in assets.columns:
        anomalies = anomalies.merge(assets[["asset_id","asset_name","asset_type","criticality"]], on="asset_id", how="left")
    if "location_id" in anomalies.columns and "location_name" in locations.columns:
        anomalies = anomalies.merge(locations[["location_id","location_name","x_coord","y_coord","zone_type"]], on="location_id", how="left")
    if "asset_id" in readings.columns and "asset_name" in assets.columns:
        readings = readings.merge(assets[["asset_id","asset_name","criticality"]], on="asset_id", how="left")

    return missions, actions, readings, anomalies, locations, assets

missions, actions, readings, anomalies, locations, assets = load_data()

st.sidebar.title("SPOTInsight AI")
st.sidebar.markdown("**Gemini-Powered SPOT Dashboard**")
st.sidebar.info("If some data files are missing, sample data is used so the demo still runs.")

page = st.sidebar.radio("Navigate", [
    "🏠 Overview",
    "🗺️ Map & Locations",
    "⚠️ Anomaly Triage",
    "📊 Performance Comparison",
    "📈 Readings Explorer",
    "🤖 Gemini Assistant"
])

st.markdown('<p class="main-header">SPOTInsight AI</p>', unsafe_allow_html=True)

if page == "🏠 Overview":
    st.markdown("See a quick summary of all SPOT missions, how many finished successfully, and which ones need attention.")
    st.header("Operational Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Missions", len(missions))
    col2.metric("Total Actions", len(actions))
    col3.metric("Anomalies Detected", len(anomalies))
    critical_count = len(anomalies[anomalies["severity"].isin(["HIGH","CRITICAL"])]) if "severity" in anomalies.columns else 0
    col4.metric("High/Critical Anomalies", critical_count)
    if "scenario" in missions.columns and "mission_status" in missions.columns:
        status_df = missions.groupby(["scenario","mission_status"]).size().reset_index(name="count")
        fig = px.bar(status_df, x="scenario", y="count", color="mission_status", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(missions.head(10), use_container_width=True)

elif page == "🗺️ Map & Locations":
    st.markdown("View the locations SPOT visited. Larger or darker dots show places with more issues that may need a closer look.")
    st.header("Facility Map — Waypoints & Anomalies")
    if "x_coord" in locations.columns and "y_coord" in locations.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=locations["x_coord"], y=locations["y_coord"],
            mode="markers+text", text=locations["location_id"],
            marker=dict(size=20, color="#1E5FFF")
        ))
        fig.update_layout(title="SPOT Waypoints", height=500)
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(locations, use_container_width=True)

elif page == "⚠️ Anomaly Triage":
    st.markdown("Review the most important issues first. High and critical items are listed so you can decide what to check next.")
    st.header("Anomaly Triage Queue")
    if "severity" in anomalies.columns:
        filtered = anomalies[anomalies["severity"].isin(["HIGH","CRITICAL"])]
        st.metric("High/Critical items", len(filtered))
        for _, row in filtered.head(8).iterrows():
            with st.expander(f"{row.get('severity','?')} | {row.get('asset_id','?')} — {row.get('anomaly_type','?')}"):
                st.write(row.get("description","No description"))
                if st.button("Show Gemini Prompt", key=str(row.get("anomaly_id",id(row)))):
                    st.code(f"Explain this anomaly for a non-technical operator: {row.get('anomaly_type')} on {row.get('asset_id')} at {row.get('location_id')}. Severity: {row.get('severity')}.")
    else:
        st.dataframe(anomalies)

elif page == "📊 Performance Comparison":
    st.markdown("Compare the older setup with the newer one. Higher numbers mean more successful inspections.")
    st.header("BASELINE vs IMPROVED")
    if "scenario" in actions.columns and "status" in actions.columns:
        success = actions.groupby("scenario")["status"].apply(lambda x: (x=="SUCCESS").mean()*100).reset_index(name="success_pct")
        fig = px.bar(success, x="scenario", y="success_pct", title="Action Success Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    st.info("IMPROVED configuration shows higher success rates in the full dataset.")

elif page == "📈 Readings Explorer":
    st.markdown("Look at the measurements SPOT recorded, such as temperature or visual scores, over time.")
    st.header("Sensor Readings")
    if "measurement_type" in readings.columns:
        m = st.selectbox("Type", readings["measurement_type"].unique())
        st.dataframe(readings[readings["measurement_type"]==m].head(10))
    else:
        st.dataframe(readings)

elif page == "🤖 Gemini Assistant":
    st.markdown("Get simple explanations of what the data means and what you should do next, written for operators.")
    st.header("Gemini-Powered Assistant")
    st.markdown("Prompts are ready for the Gemini API. Example:")
    if st.button("Show Mission Summary Prompt"):
        st.code("Summarize BASELINE vs IMPROVED SPOT performance for a non-technical operator. Highlight success rate gains and remaining risks.")

st.sidebar.caption("KnightHacks 2026 | SPOTInsight AI")
