"""
PetroStream: Industrial Maintenance Decision & Prognostics Engine
Implements ISO 10816-3 mechanical vibration standards, real-time health indexing,
and automated SAP/Maximo work-order dispatch logic for plant reliability teams.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

from feature_engineering import engineer_telemetry_features

class PetroStreamEngine:
    def __init__(self, models_dir: str = None):
        if models_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            models_dir = os.path.join(base_dir, 'models')
            
        rul_path = os.path.join(models_dir, 'rul_model.joblib')
        anom_path = os.path.join(models_dir, 'anomaly_detector.joblib')
        
        if not os.path.exists(rul_path) or not os.path.exists(anom_path):
            raise FileNotFoundError("Models not trained yet. Run train_model.py first.")
            
        self.rul_model = joblib.load(rul_path)
        anom_dict = joblib.load(anom_path)
        self.anom_preproc = anom_dict['preprocessor']
        self.anom_model = anom_dict['model']
        
    def evaluate_asset(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates real-time sensor packet against ISO 10816 standards,
        Isolation Forest anomaly scoring, and RUL prognostics.
        """
        df_raw = pd.DataFrame([telemetry])
        df_feat = engineer_telemetry_features(df_raw)
        
        # 1. Prognostics: Remaining Useful Life (RUL) Prediction
        drop_cols = ['equipment_tag', 'basin', 'latitude', 'longitude']
        cols_present = [c for c in drop_cols if c in df_feat.columns]
        X_eval = df_feat.drop(columns=cols_present)
        
        predicted_rul = float(self.rul_model.predict(X_eval)[0])
        predicted_rul = max(0.0, round(predicted_rul, 1))
        
        # 2. Anomaly Detection
        X_trans = self.anom_preproc.transform(X_eval)
        raw_pred = self.anom_model.predict(X_trans)[0]
        anom_score = float(-self.anom_model.score_samples(X_trans)[0])
        is_anomalous = bool(raw_pred == -1 or anom_score > 0.60)
        
        # 3. ISO 10816-3 Vibration Severity Zone
        vibe = float(telemetry.get('vibration_rms', 2.0))
        if vibe <= 1.8:
            iso_zone = "Zone A (Newly Commissioned / Superior)"
            iso_color = "#10B981"
        elif vibe <= 4.5:
            iso_zone = "Zone B (Unrestricted Continuous Operation)"
            iso_color = "#3B82F6"
        elif vibe <= 7.1:
            iso_zone = "Zone C (Unsatisfactory - Restricted Operation)"
            iso_color = "#F59E0B"
        else:
            iso_zone = "Zone D (Unacceptable - Trip / Immediate Shutdown)"
            iso_color = "#EF4444"
            
        # 4. Asset Health Index (0 to 100)
        temp = float(telemetry.get('bearing_temperature_c', 70.0))
        hours = float(telemetry.get('operating_hours', 5000.0))
        visc = float(telemetry.get('lubricant_viscosity_cst', 42.0))
        
        vibe_penalty = max(0, (vibe - 2.0) * 8.0)
        temp_penalty = max(0, (temp - 75.0) * 1.5)
        visc_penalty = max(0, (40.0 - visc) * 2.0)
        hours_penalty = min(25, (hours / 24000.0) * 25.0)
        
        health_index = max(2.0, min(100.0, 100.0 - (vibe_penalty + temp_penalty + visc_penalty + hours_penalty)))
        health_index = round(health_index, 1)
        
        # 5. Work Order & Maintenance Dispatch Logic
        diagnostic_flags = []
        action_recommendations = []
        
        if vibe > 7.1:
            diagnostic_flags.append(f"Severe vibration ({vibe:.2f} mm/s) in ISO Zone D.")
            action_recommendations.append("Immediate trip required. Inspect dynamic rotor balance and coupling alignment.")
        elif vibe > 4.5:
            diagnostic_flags.append(f"Elevated vibration ({vibe:.2f} mm/s) in ISO Zone C.")
            action_recommendations.append("Schedule vibration spectral analysis (1X/2X harmonics) within 48 operating hours.")
            
        if temp > 95.0:
            diagnostic_flags.append(f"High bearing metal temperature ({temp:.1f}°C) approaching thermal trip limit.")
            action_recommendations.append("Verify lube oil cooler inlet/outlet delta-P and check for bearing race scoring.")
            
        if visc < 30.0:
            diagnostic_flags.append(f"Low lubricant viscosity ({visc:.1f} cSt) indicates oil degradation or fuel dilution.")
            action_recommendations.append("Collect oil sample for particle count (ISO 4406) and schedule lube flush.")
            
        if predicted_rul < 250.0:
            diagnostic_flags.append(f"Critical RUL horizon (< 250 running hours remaining).")
            action_recommendations.append("Prepare replacement mechanical seal and bearing assembly for upcoming offshore turnaround.")
            
        # Overall Status
        if health_index >= 75 and vibe <= 4.5 and temp <= 85.0:
            status = "HEALTHY"
            status_color = "#10B981"
            priority = "P4 - Routine Predictive Monitoring"
        elif health_index >= 45 and vibe <= 7.1 and temp <= 95.0:
            status = "WARNING"
            status_color = "#F59E0B"
            priority = "P2 - Plan Maintenance Inspection"
        else:
            status = "CRITICAL"
            status_color = "#EF4444"
            priority = "P1 - Immediate Intervention / Trip"
            
        return {
            'health_index': health_index,
            'status': status,
            'status_color': status_color,
            'priority': priority,
            'iso_zone': iso_zone,
            'iso_color': iso_color,
            'predicted_rul_hours': predicted_rul,
            'predicted_rul_days': round(predicted_rul / 24.0, 1),
            'is_anomalous': is_anomalous,
            'anomaly_score': round(anom_score, 4),
            'diagnostic_flags': diagnostic_flags if diagnostic_flags else ["All parameters within normal operating envelope."],
            'action_recommendations': action_recommendations if action_recommendations else ["Continue standard vibration and condition monitoring."]
        }
