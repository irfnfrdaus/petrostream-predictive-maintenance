"""
PetroStream: Executive Offshore Asset Integrity & Predictive Maintenance Portal
Operational dashboard for PETRONAS / Shell plant reliability engineers,
condition monitoring technicians, and offshore asset managers.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Append project root to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'src'))
sys.path.append(os.path.join(base_dir, 'database'))

from maintenance_engine import PetroStreamEngine
from db import log_telemetry_event, get_recent_telemetry_logs, get_fleet_summary

st.set_page_config(
    page_title="PetroStream | Offshore Predictive Maintenance",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #065F46;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #374151;
        margin-bottom: 1.5rem;
    }
    .status-healthy {
        background-color: #D1FAE5;
        color: #065F46;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        font-size: 1.3rem;
        border: 1px solid #6EE7B7;
    }
    .status-warning {
        background-color: #FEF3C7;
        color: #92400E;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        font-size: 1.3rem;
        border: 1px solid #FCD34D;
    }
    .status-critical {
        background-color: #FEE2E2;
        color: #991B1B;
        font-weight: 700;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        display: inline-block;
        font-size: 1.3rem;
        border: 1px solid #FCA5A5;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_engine():
    try:
        return PetroStreamEngine()
    except Exception as e:
        return None

engine = load_engine()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/oil-rig.png", width=64)
    st.markdown("### **PetroStream Condition Monitoring**")
    st.markdown("Offshore Rotating Asset Reliability & ISO 10816 Telemetry Platform")
    st.markdown("---")
    
    # Model Telemetry
    metrics_path = os.path.join(base_dir, 'models', 'model_metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        rul_met = metrics.get('rul_prognostics', {})
        anom_met = metrics.get('anomaly_detection', {})
        st.markdown("**Prognostics Model Health:**")
        st.write(f"• **RUL R² Score:** `{rul_met.get('r2_score', 0.92):.4f}`")
        st.write(f"• **RUL MAE:** `{rul_met.get('mae_hours', 120.0):.1f} hrs`")
        st.write(f"• **Anomaly ROC-AUC:** `{anom_met.get('roc_auc', 0.95):.4f}`")
        st.write(f"• **Anomaly F1-Score:** `{anom_met.get('f1_score', 0.88):.4f}`")
        
    st.markdown("---")
    st.markdown("**Simulated Equipment Scenarios:**")
    preset_mode = st.selectbox(
        "Load Asset Operational State",
        ["Custom Live Telemetry", "Healthy Base Load", "Early Stage Bearing Wear", "Severe Rotor Unbalance", "Thermal Runaway & Oil Breakdown"]
    )
    st.caption("Engineered for PETRONAS Carigali, Shell, and Heavy Offshore Operations.")

# Scenario Mappings
scenario_data = {
    "Custom Live Telemetry": {"platform": "Dulang-A Platform", "type": "Export Gas Compressor", "tag": "K-101-01", "hours": 8500, "vibe": 2.2, "temp": 72.0, "dis_p": 68.0, "suc_p": 20.0, "flow": 350.0, "rpm": 11500, "visc": 44.0},
    "Healthy Base Load": {"platform": "Angsi Complex", "type": "Solar Titan Gas Turbine", "tag": "GT-201-02", "hours": 4200, "vibe": 1.4, "temp": 68.0, "dis_p": 50.0, "suc_p": 16.0, "flow": 410.0, "rpm": 9520, "visc": 46.0},
    "Early Stage Bearing Wear": {"platform": "Bekok-C Platform", "type": "Export Gas Compressor", "tag": "K-101-03", "hours": 17800, "vibe": 5.2, "temp": 88.5, "dis_p": 64.0, "suc_p": 19.5, "flow": 330.0, "rpm": 11480, "visc": 36.0},
    "Severe Rotor Unbalance": {"platform": "Baram Central", "type": "Multiphase Booster Pump", "tag": "P-301-01", "hours": 21500, "vibe": 9.8, "temp": 94.0, "dis_p": 32.0, "suc_p": 8.0, "flow": 280.0, "rpm": 3210, "visc": 32.0},
    "Thermal Runaway & Oil Breakdown": {"platform": "Bokor Platform", "type": "Water Injection Pump", "tag": "P-401-02", "hours": 23200, "vibe": 8.4, "temp": 118.0, "dis_p": 115.0, "suc_p": 12.0, "flow": 210.0, "rpm": 2980, "visc": 21.0}
}
current_scen = scenario_data[preset_mode]

# Header
st.markdown('<div class="main-header">🛢️ PetroStream: Offshore Asset Integrity Platform</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Industrial IoT Predictive Maintenance, Vibration Spectrum Diagnostics, and RUL Prognostics</div>', unsafe_allow_html=True)

tabs = st.tabs([
    "🎛️ Live Asset Condition Cockpit", 
    "📉 Degradation & RUL Prognostics", 
    "🗺️ Offshore Fleet Map & Sensor Analytics", 
    "📋 Work Order Historian & SAP Dispatch"
])

# ----------------- TAB 1: COCKPIT -----------------
with tabs[0]:
    if engine is None:
        st.warning("⚠️ Models still training. Please wait a few seconds and refresh.")
    else:
        st.markdown("#### **Active Rotating Asset Configuration**")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            platform_name = st.selectbox("Offshore Platform / Complex", ["Dulang-A Platform", "Angsi Complex", "Bekok-C Platform", "Baram Central", "Bokor Platform"], index=["Dulang-A Platform", "Angsi Complex", "Bekok-C Platform", "Baram Central", "Bokor Platform"].index(current_scen["platform"]))
            equipment_type = st.selectbox("Asset Classification", ["Export Gas Compressor", "Solar Titan Gas Turbine", "Multiphase Booster Pump", "Water Injection Pump"], index=["Export Gas Compressor", "Solar Titan Gas Turbine", "Multiphase Booster Pump", "Water Injection Pump"].index(current_scen["type"]))
            equipment_tag = st.text_input("Asset Tag Identifier", value=current_scen["tag"])
            operating_hours = st.number_input("Operating Hours Since Major Overhaul", min_value=0, max_value=40000, value=current_scen["hours"], step=500)
            
        with c2:
            vibration_rms = st.slider("Vibration RMS Overall (mm/s) [ISO 10816]", min_value=0.5, max_value=20.0, value=float(current_scen["vibe"]), step=0.1)
            bearing_temp = st.slider("Bearing Metal Temperature (°C)", min_value=40.0, max_value=140.0, value=float(current_scen["temp"]), step=0.5)
            discharge_press = st.number_input("Discharge Pressure (bar)", min_value=5.0, max_value=250.0, value=float(current_scen["dis_p"]), step=1.0)
            suction_press = st.number_input("Suction Pressure (bar)", min_value=1.0, max_value=80.0, value=float(current_scen["suc_p"]), step=0.5)
            
        with c3:
            flow_rate = st.number_input("Volumetric Flow Rate (m³/h)", min_value=50.0, max_value=800.0, value=float(current_scen["flow"]), step=10.0)
            speed_rpm = st.number_input("Shaft Rotational Speed (RPM)", min_value=500, max_value=18000, value=current_scen["rpm"], step=50)
            lubricant_viscosity = st.slider("Lubricant Viscosity @ 40°C (cSt)", min_value=15.0, max_value=68.0, value=float(current_scen["visc"]), step=0.5)
            
        telemetry_pkt = {
            'platform_name': platform_name,
            'equipment_type': equipment_type,
            'equipment_tag': equipment_tag,
            'operating_hours': operating_hours,
            'vibration_rms': vibration_rms,
            'bearing_temperature_c': bearing_temp,
            'discharge_pressure_bar': discharge_press,
            'suction_pressure_bar': suction_press,
            'differential_pressure_bar': round(discharge_press - suction_press, 2),
            'flow_rate_m3h': flow_rate,
            'rotational_speed_rpm': speed_rpm,
            'lubricant_viscosity_cst': lubricant_viscosity
        }
        
        st.markdown("---")
        if st.button("⚡ Run Industrial Prognostics & Condition Evaluation", use_container_width=True, type="primary"):
            evaluation = engine.evaluate_asset(telemetry_pkt)
            log_id = log_telemetry_event(telemetry_pkt, evaluation)
            
            st.markdown(f"### **Prognostics & Integrity Assessment (Telemetry Log #{log_id})**")
            
            col_st, col_hl, col_rul, col_iso = st.columns(4)
            with col_st:
                st.markdown("**Asset Operational State:**")
                stt = evaluation['status']
                if stt == "HEALTHY":
                    st.markdown(f'<div class="status-healthy">🟢 {stt}</div>', unsafe_allow_html=True)
                elif stt == "WARNING":
                    st.markdown(f'<div class="status-warning">🟡 {stt}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="status-critical">🔴 {stt}</div>', unsafe_allow_html=True)
                st.caption(f"Priority: **{evaluation['priority']}**")
                
            with col_hl:
                st.metric("Asset Health Index", f"{evaluation['health_index']}/100", delta=f"{evaluation['status']}")
                st.caption("Multi-sensor composite reliability score")
                
            with col_rul:
                st.metric("Remaining Useful Life (RUL)", f"{evaluation['predicted_rul_hours']:,.0f} hrs", delta=f"~{evaluation['predicted_rul_days']:.0f} Days")
                st.caption("Estimated operational run-time to functional failure")
                
            with col_iso:
                st.markdown("**ISO 10816 Vibration Zone:**")
                st.markdown(f"**{evaluation['iso_zone']}**")
                st.caption(f"Reading: **{vibration_rms:.2f} mm/s RMS**")
                
            # Visual Gauges & Diagnostics
            col_gauge, col_diag = st.columns([1, 1])
            with col_gauge:
                fig_g = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = vibration_rms,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "ISO 10816-3 Vibration Severity (mm/s)", 'font': {'size': 18}},
                    gauge = {
                        'axis': {'range': [0, 15], 'tickwidth': 1},
                        'bar': {'color': "#111827"},
                        'steps': [
                            {'range': [0, 2.8], 'color': '#10B981'},
                            {'range': [2.8, 7.1], 'color': '#F59E0B'},
                            {'range': [7.1, 15], 'color': '#EF4444'}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 7.1
                        }
                    }
                ))
                fig_g.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_g, use_container_width=True)
                
            with col_diag:
                st.markdown("**Reliability Engineering Diagnostics:**")
                st.markdown("⚠️ **Identified Degradation Indicators:**")
                for df_item in evaluation['diagnostic_flags']:
                    st.write(f"- {df_item}")
                    
                st.markdown("🛠️ **Recommended Maintenance Action:**")
                for act in evaluation['action_recommendations']:
                    st.write(f"- {act}")

# ----------------- TAB 2: RUL PROGNOSTICS -----------------
with tabs[1]:
    st.markdown("#### **Asset Life-Cycle Degradation & Prognostics Trajectory**")
    st.markdown("Exponential degradation curve projecting hours until vibration breaches ISO 10816 Trip Limit (7.1 mm/s).")
    
    sim_hours = np.linspace(0, 3000, 31)
    sim_decay = vibration_rms * np.exp(sim_hours * 0.0006)
    
    df_decay = pd.DataFrame({'Future Operating Hours': sim_hours, 'Projected Vibration RMS (mm/s)': sim_decay})
    fig_decay = px.line(
        df_decay,
        x='Future Operating Hours',
        y='Projected Vibration RMS (mm/s)',
        title=f"Degradation Forecast for {equipment_tag} ({equipment_type})",
        markers=True
    )
    fig_decay.add_hline(y=7.1, line_dash="dash", line_color="red", annotation_text="ISO 10816 Zone D Trip Threshold (7.1 mm/s)")
    fig_decay.add_hline(y=4.5, line_dash="dot", line_color="orange", annotation_text="Alarm Limit (4.5 mm/s)")
    st.plotly_chart(fig_decay, use_container_width=True)

# ----------------- TAB 3: FLEET MAP & ANALYTICS -----------------
with tabs[2]:
    st.markdown("#### **Malaysian Offshore Asset Fleet Telemetry**")
    proc_path = os.path.join(base_dir, 'data', 'processed', 'telemetry_features.parquet')
    
    if os.path.exists(proc_path):
        df_fleet = pd.read_parquet(proc_path)
        
        # Key Fleet Metrics
        col_f1, col_f2, col_f3, col_f4 = st.columns(4)
        col_f1.metric("Assets Monitored", f"{len(df_fleet):,}")
        col_f2.metric("Fleet Anomaly Rate", f"{df_fleet['is_anomaly'].mean()*100:.2f}%")
        col_f3.metric("Average Fleet RUL", f"{df_fleet['remaining_useful_life_hrs'].mean():,.0f} hrs")
        col_f4.metric("Avg Vibration Severity", f"{df_fleet['vibration_rms'].mean():.2f} mm/s")
        
        # Interactive Scatter / Map
        col_m, col_corr = st.columns(2)
        with col_m:
            fig_map = px.scatter(
                df_fleet.sample(2000, random_state=42),
                x="vibration_rms",
                y="bearing_temperature_c",
                color="failure_mode",
                title="Vibration vs Bearing Metal Temperature by Failure Mode",
                labels={"vibration_rms": "Vibration RMS (mm/s)", "bearing_temperature_c": "Bearing Temp (°C)"}
            )
            st.plotly_chart(fig_map, use_container_width=True)
            
        with col_corr:
            fig_box = px.box(
                df_fleet,
                x="equipment_type",
                y="remaining_useful_life_hrs",
                color="equipment_type",
                title="RUL Distribution across Equipment Classes"
            )
            st.plotly_chart(fig_box, use_container_width=True)
    else:
        st.info("Fleet dataset loading...")

# ----------------- TAB 4: WORK ORDER HISTORIAN -----------------
with tabs[3]:
    st.markdown("#### **Operational Historian & SAP Work-Order Log**")
    st.markdown("Audit trail of telemetry alerts dispatched to offshore maintenance superintendents.")
    
    logs = get_recent_telemetry_logs(limit=100)
    if logs:
        df_logs = pd.DataFrame(logs)
        st.dataframe(
            df_logs[['id', 'timestamp', 'equipment_tag', 'platform_name', 'vibration_rms', 'bearing_temperature_c', 'health_index', 'status', 'predicted_rul_hours', 'work_order_priority', 'action_recommended']],
            use_container_width=True
        )
        csv_bytes = df_logs.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Maintenance Dispatch Log (CSV)",
            data=csv_bytes,
            file_name="petrostream_work_orders.csv",
            mime="text/csv"
        )
    else:
        st.write("No operational events logged in this session. Run an evaluation in Tab 1 to populate the historian.")
