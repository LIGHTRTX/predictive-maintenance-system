import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

from src.vibration import generate_signal, compute_rms
from src.risk_engine import overload_ratio, unified_risk, classify
from src.database import init_db, insert_log

st.set_page_config(page_title="AI Predictive Maintenance", layout="wide")

model = joblib.load("models/failure_model.pkl")
scaler = joblib.load("models/scaler.pkl")

conn = init_db()

st.title("AI Predictive Maintenance System")

col1, col2, col3, col4, col5 = st.columns(5)

air_temp = col1.slider("Air Temperature [K]", 295.0, 320.0, 300.0)
process_temp = col2.slider("Process Temperature [K]", 305.0, 340.0, 310.0)
rpm = col3.slider("Rotational Speed [rpm]", 1000.0, 2000.0, 1500.0)
torque = col4.slider("Torque [Nm]", 10.0, 70.0, 40.0)
tool_wear = col5.slider("Tool Wear [min]", 0.0, 250.0, 100.0)

input_data = np.array([[air_temp, process_temp, rpm, torque, tool_wear]])
scaled = scaler.transform(input_data)
failure_prob = model.predict_proba(scaled)[0][1]

current = torque * 0.1
over_ratio = overload_ratio(current)

signal = generate_signal(over_ratio)
rms_value = compute_rms(signal)

risk_score = unified_risk(failure_prob, over_ratio, rms_value)
status, color = classify(risk_score)

insert_log(conn, {
    "air_temp": air_temp,
    "process_temp": process_temp,
    "rpm": rpm,
    "torque": torque,
    "current": current,
    "failure_prob": failure_prob,
    "overload_ratio": over_ratio,
    "rms": rms_value,
    "risk_score": risk_score,
    "status": status
})

st.markdown("---")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Failure Probability", f"{failure_prob*100:.2f}%")
m2.metric("Overload Ratio", f"{over_ratio:.2f}")
m3.metric("Vibration RMS", f"{rms_value:.3f}")
m4.metric("Risk Score", f"{risk_score:.2f}")

st.markdown(f"<h2 style='color:{color};'>{status}</h2>", unsafe_allow_html=True)

st.markdown("---")

fig = go.Figure()
fig.add_trace(go.Scatter(y=signal))
fig.update_layout(template="plotly_dark", height=300)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

df_logs = pd.read_sql_query("SELECT * FROM logs ORDER BY id DESC LIMIT 20", conn)
st.dataframe(df_logs)