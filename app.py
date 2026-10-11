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

@st.cache_data
def load_data():
    missions = pd.read_csv(DATA_DIR / "missions.csv")
    actions = pd.read_csv(DATA_DIR / "actions.csv")
    readings = pd.read_csv(DATA_DIR / "readings.csv")
    anomalies = pd.read_csv(DATA_DIR / "anomalies.csv")
    locations = pd.read_csv(DATA_DIR / "locations.csv")
    assets = pd.read_csv(DATA_DIR / "assets.csv")
    
    anomalies = anomalies.merge(assets[["asset_id","asset_name","asset_type","criticality"]], on="asset_id", how="left")
    anomalies = anomalies.merge(locations[["location_id","location_name","x_coord","y_coord"]], on="location_id", how="left")
    return missions, actions, readings, anomalies, locations, assets

missions, actions, readings, anomalies, locations, assets = load_data()

@st.cache_resource
def train_severity_model():
    df = anomalies.copy()
    if len(df) < 10:
        return None, None
    le_type = LabelEncoder()
    le_crit = LabelEncoder()
    df["type_enc"] = le_type.fit_transform(df["anomaly_type"].fillna("UNKNOWN"))
    df["crit_enc"] = le_crit.fit_transform(df["criticality"].fillna("MEDIUM"))
    X = df[["type_enc","crit_enc"]]
    y = df["severity"]
    model = RandomForestClassifier(n_estimators=50, random_state=42)
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
    col2.metric("Completed", (missions.mission_status=="COMPLETED").sum())
    col3.metric("Failed", (missions.mission_status=="FAILED").sum())
    
    status = missions.groupby(["scenario","mission_status"]).size().reset_index(name="count")
    fig = px.bar(status, x="scenario", y="count", color="mission_status", barmode="group",
                 title="Mission Outcomes by Configuration")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Recent Missions")
    st.dataframe(missions[["mission_id","scenario","start_time","duration_min","mission_status","end_battery_pct"]].sort_values("start_time", ascending=False), use_container_width=True)

elif page == "Where & Why It Matters":
    st.header("Where Did It Happen?")
    st.markdown("Waypoints are the stops on SPOT’s inspection route. Larger dots show places with more anomalies. Coordinates are synthetic and only for visualization.")
    
    sev_map = {"LOW":1,"MEDIUM":2,"HIGH":3,"CRITICAL":4}
    anomalies["sev_num"] = anomalies["severity"].map(sev_map).fillna(0)
    loc_agg = anomalies.groupby("location_id").agg(count=("anomaly_id","count"), max_sev=("sev_num","max")).reset_index()
    loc_full = locations.merge(loc_agg, on="location_id", how="left").fillna(0)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=loc_full.x_coord, y=loc_full.y_coord, mode="markers+text",
        text=loc_full.location_id, textposition="top center",
        marker=dict(size=loc_full["count"]*8+15, color=loc_full.max_sev, colorscale="Reds", showscale=True),
        hovertext=loc_full.location_name
    ))
    fig.update_layout(title="Inspection Waypoints (size = anomaly count, color = highest severity)",
                      xaxis_title="Synthetic X Position", yaxis_title="Synthetic Y Position", height=550)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("These positions are fictional and do not represent a real facility.")

elif page == "What Needs Attention":
    st.header("What Requires Attention?")
    st.markdown("Issues are ranked by severity and how critical the equipment is. Start with the items at the top.")
    
    sev_order = {"CRITICAL":4,"HIGH":3,"MEDIUM":2,"LOW":1}
    crit_order = {"CRITICAL":4,"HIGH":3,"MEDIUM":2,"LOW":1}
    anomalies["priority"] = anomalies.severity.map(sev_order).fillna(1) * anomalies.criticality.map(crit_order).fillna(1)
    top = anomalies.sort_values("priority", ascending=False).head(12)
    
    for _, row in top.iterrows():
        with st.expander(f"{row.severity} — {row.asset_name} at {row.location_name} ({row.anomaly_type})"):
            st.write(row.description)
            st.write(f"Asset criticality: {row.criticality}")
            if st.button("Explain for operator", key=row.anomaly_id):
                st.info(f"This {row.anomaly_type.lower().replace('_',' ')} on a {row.criticality.lower()} priority {row.asset_type} needs review. Check the equipment and confirm the reading.")

elif page == "Is Performance Improving?":
    st.header("Is Performance Improving?")
    st.markdown("Comparing the original setup (BASELINE) with the updated setup (IMPROVED).")
    
    act_success = actions.groupby("scenario")["status"].apply(lambda x: (x=="SUCCESS").mean()*100).reset_index(name="success_pct")
    fig = px.bar(act_success, x="scenario", y="success_pct", text="success_pct",
                 title="Action Success Rate (%)", color="scenario",
                 color_discrete_map={"BASELINE":"#78909c","IMPROVED":"#2e7d32"})
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)
    
    st.success("The IMPROVED configuration raises action success from ~77% to ~91% and reaches 100% mission completion. This means more reliable inspections and fewer missed issues.")

elif page == "What to Investigate Next":
    st.header("What Should an Operator Investigate Next?")
    st.markdown("These are the highest-priority items based on severity and asset importance.")
    
    high = anomalies[anomalies.severity.isin(["HIGH","CRITICAL"])].sort_values("severity", ascending=False)
    st.dataframe(high[["severity","asset_name","location_name","anomaly_type","description","timestamp"]].head(10), use_container_width=True)
    
    st.info("Start with CRITICAL and HIGH items on critical assets. The IMPROVED configuration already reduces how often these appear.")

elif page == "Severity Predictor":
    st.header("Severity Predictor")
    st.markdown("A simple model trained on the challenge data that estimates how serious a new finding is likely to be.")
    
    if model is None:
        st.warning("Not enough data to train a model.")
    else:
        le_type, le_crit = encoders
        col1, col2 = st.columns(2)
        with col1:
            atype = st.selectbox("Anomaly type", le_type.classes_)
        with col2:
            crit = st.selectbox("Asset criticality", le_crit.classes_)
        
        if st.button("Predict Severity"):
            X_new = [[le_type.transform([atype])[0], le_crit.transform([crit])[0]]]
            pred = model.predict(X_new)[0]
            proba = model.predict_proba(X_new)[0]
            st.metric("Predicted Severity", pred)
            st.write("Confidence by class:")
            st.bar_chart(pd.Series(proba, index=model.classes_))
            st.caption("This is a demonstration model trained only on the synthetic challenge data. It is not for real operational decisions.")

st.sidebar.markdown("---")
st.sidebar.caption("Built for the KnightHacks 2026 SPOT Operational Intelligence Challenge")
st.sidebar.caption("Data is synthetic. Insights follow the challenge evaluation themes.")
