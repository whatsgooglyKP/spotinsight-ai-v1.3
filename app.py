import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="SPOTInsight AI", page_icon="🤖", layout="wide")

# Embedded minimal data so the app always runs
missions = pd.DataFrame({
    "mission_id": ["M001","M004","M019","M027"],
    "scenario": ["BASELINE","BASELINE","IMPROVED","IMPROVED"],
    "start_time": ["2026-09-01T18:00:00","2026-09-02T18:00:00","2026-09-07T18:00:00","2026-09-10T10:00:00"],
    "duration_min": [30,25,22,19],
    "mission_status": ["COMPLETED","FAILED","COMPLETED","COMPLETED"],
    "end_battery_pct": [22,30,65,72]
})
actions = pd.DataFrame({
    "action_id": ["A0001","A0002","A0003"],
    "mission_id": ["M001","M001","M019"],
    "scenario": ["BASELINE","BASELINE","IMPROVED"],
    "status": ["FAILED","SUCCESS","SUCCESS"],
    "location_id": ["WP05","WP10","WP10"],
    "asset_id": ["HX-201","PUMP-205","PUMP-205"]
})
anomalies = pd.DataFrame({
    "anomaly_id": ["AN001","AN002","AN003"],
    "severity": ["HIGH","CRITICAL","MEDIUM"],
    "anomaly_type": ["SIGNAL_DEGRADATION","ABNORMAL_TEMPERATURE","VISUAL_DEFECT"],
    "asset_id": ["HX-201","HX-202","PUMP-205"],
    "location_id": ["WP05","WP05","WP10"],
    "description": ["Signal quality dropped below threshold.","Temperature exceeded safe range.","Minor visual defect noted."]
})
locations = pd.DataFrame({
    "location_id": ["WP05","WP10","WP09"],
    "location_name": ["Pipe Gallery","Pump Room","Turbine Hall South"],
    "x_coord": [72.6,45.0,36.3],
    "y_coord": [57.8,30.0,94.9]
})

st.sidebar.title("SPOTInsight AI")
st.sidebar.caption("Operational Intelligence for SPOT Inspections")
page = st.sidebar.radio("Go to", [
    "What Happened?",
    "Where & Why It Matters",
    "What Needs Attention",
    "Is Performance Improving?",
    "What to Investigate Next"
])

st.title("SPOTInsight AI")
st.caption("Helping operators understand SPOT inspection data without becoming data experts.")

if page == "What Happened?":
    st.header("What Happened?")
    st.markdown("A clear summary of recent SPOT missions and whether they finished successfully.")
    c1,c2,c3 = st.columns(3)
    c1.metric("Missions", len(missions))
    c2.metric("Completed", (missions.mission_status=="COMPLETED").sum())
    c3.metric("Failed", (missions.mission_status=="FAILED").sum())
    status = missions.groupby(["scenario","mission_status"]).size().reset_index(name="count")
    fig = px.bar(status, x="scenario", y="count", color="mission_status", barmode="group")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(missions, use_container_width=True)

elif page == "Where & Why It Matters":
    st.header("Where Did It Happen?")
    st.markdown("Waypoints are the stops on SPOT’s inspection route. Coordinates are synthetic and only for visualization.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=locations.x_coord, y=locations.y_coord, mode="markers+text",
        text=locations.location_id, textposition="top center",
        marker=dict(size=22, color="#1E5FFF")
    ))
    fig.update_layout(
        xaxis_title="Synthetic X Position",
        yaxis_title="Synthetic Y Position",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("These positions are fictional and do not represent a real facility.")
    st.dataframe(locations, use_container_width=True)

elif page == "What Needs Attention":
    st.header("What Requires Attention?")
    st.markdown("Issues ranked by severity so operators know what to check first.")
    for _, row in anomalies.iterrows():
        with st.expander(f"{row.severity} — {row.anomaly_type} on {row.asset_id} at {row.location_id}"):
            st.write(row.description)

elif page == "Is Performance Improving?":
    st.header("Is Performance Improving?")
    st.markdown("Comparing the original setup (BASELINE) with the updated setup (IMPROVED).")
    success = actions.groupby("scenario")["status"].apply(lambda x: (x=="SUCCESS").mean()*100).reset_index(name="pct")
    fig = px.bar(success, x="scenario", y="pct", text="pct", title="Action Success Rate (%)")
    st.plotly_chart(fig, use_container_width=True)
    st.success("The IMPROVED configuration shows higher success rates, meaning more reliable inspections.")

elif page == "What to Investigate Next":
    st.header("What Should an Operator Investigate Next?")
    st.markdown("Highest-priority items based on severity.")
    st.dataframe(anomalies[anomalies.severity.isin(["HIGH","CRITICAL"])], use_container_width=True)

st.sidebar.caption("KnightHacks 2026 SPOT Challenge")
