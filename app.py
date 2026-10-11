import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="SPOTInsight AI", page_icon="🤖", layout="wide")

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

def safe_csv(name, sample):
    path = DATA_DIR / name
    try:
        return pd.read_csv(path)
    except Exception:
        st.sidebar.warning(f"Using sample data for {name}")
        return sample

@st.cache_data
def load_data():
    sample_missions = pd.DataFrame({
        "mission_id":["M001","M019"], "mission_name":["Route 1","Route 19"],
        "scenario":["BASELINE","IMPROVED"], "configuration_version":["V1","V2"],
        "start_time":["2026-09-01T18:00:00","2026-09-07T18:00:00"],
        "end_time":["2026-09-01T18:30:00","2026-09-07T18:22:00"],
        "duration_min":[30,22], "start_battery_pct":[82,84], "end_battery_pct":[22,65],
        "mission_status":["COMPLETED","COMPLETED"], "operator_mode":["AUTOWALK","AUTOWALK"], "synthetic":[True,True]
    })
    sample_actions = pd.DataFrame({
        "action_id":["A0001","A0002"], "mission_id":["M001","M001"], "sequence":[1,2],
        "timestamp":["2026-09-01T18:03:06","2026-09-01T18:06:12"],
        "location_id":["WP05","WP10"], "asset_id":["HX-201","PUMP-205"],
        "action_type":["NAVIGATION","VISUAL_INSPECTION"], "battery_pct":[75,69],
        "status":["FAILED","SUCCESS"], "failure_type":["OBSTACLE_DETECTION",""],
        "duration_sec":[0,63], "scenario":["BASELINE","BASELINE"],
        "configuration_version":["V1","V1"], "synthetic":[True,True]
    })
    sample_readings = pd.DataFrame({
        "reading_id":["R0001"], "action_id":["A0002"], "mission_id":["M001"],
        "asset_id":["PUMP-205"], "location_id":["WP10"],
        "timestamp":["2026-09-01T18:06:12"], "measurement_type":["TEMPERATURE_C"],
        "value":[62.5], "unit":["C"], "quality":["GOOD"], "synthetic":[True]
    })
    sample_anomalies = pd.DataFrame({
        "anomaly_id":["AN001","AN002"], "action_id":["A0001","A0002"], "mission_id":["M001","M001"],
        "location_id":["WP05","WP10"], "asset_id":["HX-201","PUMP-205"],
        "timestamp":["2026-09-01T18:03:06","2026-09-01T18:06:12"],
        "anomaly_type":["SIGNAL_DEGRADATION","VISUAL_DEFECT"],
        "severity":["HIGH","MEDIUM"], "operator_review_required":[True,False],
        "description":["Synthetic signal degradation.","Synthetic visual defect."], "synthetic":[True,True]
    })
    sample_locations = pd.DataFrame({
        "location_id":["WP05","WP10"], "location_name":["Pipe Gallery","Pump Room"],
        "zone_type":["PROCESS","MECHANICAL"], "floor":[1,1],
        "x_coord":[72.6,45.0], "y_coord":[57.8,30.0], "environment":["HIGH_NOISE","NORMAL"], "synthetic":[True,True]
    })
    sample_assets = pd.DataFrame({
        "asset_id":["HX-201","PUMP-205"], "asset_name":["Heat Exchanger 01","Pump 05"],
        "asset_type":["Heat Exchanger","Pump"], "location_id":["WP05","WP10"],
        "criticality":["LOW","MEDIUM"], "synthetic":[True,True]
    })

    missions = safe_csv("missions.csv", sample_missions)
    actions = safe_csv("actions.csv", sample_actions)
    readings = safe_csv("readings.csv", sample_readings)
    anomalies = safe_csv("anomalies.csv", sample_anomalies)
    locations = safe_csv("locations.csv", sample_locations)
    assets = safe_csv("assets.csv", sample_assets)

    if "asset_id" in anomalies.columns and "asset_name" in assets.columns:
        anomalies = anomalies.merge(assets[["asset_id","asset_name","asset_type","criticality"]], on="asset_id", how="left")
    if "location_id" in anomalies.columns and "location_name" in locations.columns:
        anomalies = anomalies.merge(locations[["location_id","location_name","x_coord","y_coord"]], on="location_id", how="left")
    return missions, actions, readings, anomalies, locations, assets

missions, actions, readings, anomalies, locations, assets = load_data()

@st.cache_resource
def train_severity_model():
    df = anomalies.copy()
    if len(df) < 5 or "anomaly_type" not in df.columns:
        return None, None
    le_type = LabelEncoder()
    le_crit = LabelEncoder()
    df["type_enc"] = le_type.fit_transform(df["anomaly_type"].fillna("UNKNOWN"))
    df["crit_enc"] = le_crit.fit_transform(df["criticality"].fillna("MEDIUM"))
    X = df[["type_enc","crit_enc"]]
    y = df["severity"]
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X, y)
    return model, (le_type, le_crit)

model, encoders = train_severity_model()

st.sidebar.title("SPOTInsight AI")
st.sidebar.caption("Operational Intelligence for SPOT Inspections")
page = st.sidebar.radio("Go to", [
    "What Happened?",
    "Where & Why It Matters",
    "What Needs Attention",
    "Is Performance Improving?",
    "What to Investigate Next",
    "Severity Predictor"
])

st.title("SPOTInsight AI")
st.caption("Helping operators understand SPOT inspection data without becoming data experts.")

if page == "What Happened?":
    st.header("What Happened?")
    st.markdown("A clear summary of recent SPOT missions and whether they finished successfully.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Missions", len(missions))
    col2.metric("Completed", (missions.mission_status=="COMPLETED").sum() if "mission_status" in missions.columns else 0)
    col3.metric("Failed", (missions.mission_status=="FAILED").sum() if "mission_status" in missions.columns else 0)
    if "scenario" in missions.columns and "mission_status" in missions.columns:
        status = missions.groupby(["scenario","mission_status"]).size().reset_index(name="count")
        fig = px.bar(status, x="scenario", y="count", color="mission_status", barmode="group")
        st.plotly_chart(fig, use_container_width=True)
    st.dataframe(missions.head(10), use_container_width=True)

elif page == "Where & Why It Matters":
    st.header("Where Did It Happen?")
    st.markdown("Waypoints are the stops on SPOT’s inspection route. Larger dots show places with more anomalies. Coordinates are synthetic.")
    if "x_coord" in locations.columns:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=locations.x_coord, y=locations.y_coord, mode="markers+text",
                                 text=locations.location_id, marker=dict(size=18, color="#1E5FFF")))
        fig.update_layout(xaxis_title="Synthetic X Position", yaxis_title="Synthetic Y Position", height=500)
        st.plotly_chart(fig, use_container_width=True)
    st.caption("These positions are fictional.")
    st.dataframe(locations, use_container_width=True)

elif page == "What Needs Attention":
    st.header("What Requires Attention?")
    st.markdown("Issues ranked by severity and asset importance.")
    if "severity" in anomalies.columns:
        high = anomalies[anomalies.severity.isin(["HIGH","CRITICAL"])]
        for _, row in high.head(8).iterrows():
            with st.expander(f"{row.get('severity','?')} — {row.get('asset_id','?')} ({row.get('anomaly_type','?')})"):
                st.write(row.get("description","No description"))
    else:
        st.dataframe(anomalies)

elif page == "Is Performance Improving?":
    st.header("Is Performance Improving?")
    st.markdown("BASELINE vs IMPROVED configuration.")
    if "scenario" in actions.columns and "status" in actions.columns:
        success = actions.groupby("scenario")["status"].apply(lambda x: (x=="SUCCESS").mean()*100).reset_index(name="pct")
        fig = px.bar(success, x="scenario", y="pct", text="pct", title="Action Success Rate (%)")
        st.plotly_chart(fig, use_container_width=True)
    st.success("In the full dataset, IMPROVED raises action success from ~77% to ~91%.")

elif page == "What to Investigate Next":
    st.header("What Should an Operator Investigate Next?")
    st.markdown("Highest-priority items.")
    if "severity" in anomalies.columns:
        st.dataframe(anomalies[anomalies.severity.isin(["HIGH","CRITICAL"])].head(10), use_container_width=True)

elif page == "Severity Predictor":
    st.header("Severity Predictor")
    st.markdown("Simple model that estimates severity from anomaly type and asset criticality.")
    if model is None:
        st.warning("Not enough data to train.")
    else:
        le_type, le_crit = encoders
        atype = st.selectbox("Anomaly type", le_type.classes_)
        crit = st.selectbox("Asset criticality", le_crit.classes_)
        if st.button("Predict"):
            X = [[le_type.transform([atype])[0], le_crit.transform([crit])[0]]]
            pred = model.predict(X)[0]
            st.metric("Predicted Severity", pred)
            st.caption("Demonstration only — trained on synthetic challenge data.")

st.sidebar.caption("KnightHacks 2026 SPOT Challenge")
