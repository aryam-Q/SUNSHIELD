import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="SUNSHIELD Anomaly Dashboard", layout="wide")
st.title("SUNSHIELD — PV Anomaly Detection Dashboard")

@st.cache_data
def load_data():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(BASE_DIR, "anomaly_results_w48.csv")
    df = pd.read_csv(csv_path)
    df["DATE_TIME"] = pd.to_datetime(df["DATE_TIME"])
    return df

df = load_data()

st.sidebar.header("Filters")
plant = st.sidebar.selectbox("Plant", sorted(df["PLANT_ID"].unique()))
inverters = sorted(df[df["PLANT_ID"] == plant]["SOURCE_KEY"].unique())
inverter = st.sidebar.selectbox("Inverter", inverters)

subset = df[(df["PLANT_ID"] == plant) & (df["SOURCE_KEY"] == inverter)].sort_values("DATE_TIME")

col1, col2, col3 = st.columns(3)
col1.metric("Total readings", len(subset))
col2.metric("Anomalies detected", int(subset["is_anomaly"].sum()))
col3.metric("Anomaly rate", f"{100*subset['is_anomaly'].mean():.1f}%")

fig = go.Figure()
fig.add_trace(go.Scatter(x=subset["DATE_TIME"], y=subset["AC_POWER_actual_scaled"], mode="lines", name="Actual AC_POWER"))
fig.add_trace(go.Scatter(x=subset["DATE_TIME"], y=subset["AC_POWER_predicted_scaled"], mode="lines", name="Predicted AC_POWER"))
anomalies = subset[subset["is_anomaly"]]
fig.add_trace(go.Scatter(x=anomalies["DATE_TIME"], y=anomalies["AC_POWER_actual_scaled"], mode="markers", name="Anomaly", marker=dict(color="red", size=8)))
fig.update_layout(title=f"Plant {plant} — Inverter {inverter}", xaxis_title="Time", yaxis_title="AC_POWER (scaled)")
st.plotly_chart(fig, use_container_width=True)

st.subheader("Detected Anomalies")
st.dataframe(anomalies[["DATE_TIME", "AC_POWER_actual_scaled", "AC_POWER_predicted_scaled", "residual"]])
