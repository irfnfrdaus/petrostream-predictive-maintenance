"""
PetroStream: Industrial Feature Engineering & Sensor Transformation
Computes physical stress indices, thermal gradients, and ISO 10816 vibration ratios.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_FEATURES = ['platform_name', 'equipment_type']

NUMERICAL_FEATURES = [
    'operating_hours',
    'vibration_rms',
    'bearing_temperature_c',
    'discharge_pressure_bar',
    'suction_pressure_bar',
    'differential_pressure_bar',
    'flow_rate_m3h',
    'rotational_speed_rpm',
    'lubricant_viscosity_cst',
    'vibration_severity_ratio',
    'pressure_ratio',
    'thermal_gradient',
    'hydraulic_power_kw',
    'mechanical_stress_index'
]

def engineer_telemetry_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes thermodynamic and mechanical degradation ratios:
    1. Vibration Severity Ratio (ISO 10816 baseline = 2.8 mm/s)
    2. Pressure Ratio (Discharge / Suction)
    3. Thermal Gradient (Bearing heat accumulation over operating life)
    4. Hydraulic Power (kW)
    5. Mechanical Stress Index (Vibration * Temperature cross-product)
    """
    data = df.copy()
    
    # 1. ISO 10816-3 Vibration Severity Ratio (normalized against 2.8 mm/s permissible boundary)
    data['vibration_severity_ratio'] = data['vibration_rms'] / 2.80
    
    # 2. Pressure Compression Ratio
    data['pressure_ratio'] = data['discharge_pressure_bar'] / (data['suction_pressure_bar'] + 1e-4)
    
    # 3. Thermal Gradient (°C per 1,000 running hours above ambient 25°C)
    data['thermal_gradient'] = (data['bearing_temperature_c'] - 25.0) / ((data['operating_hours'] / 1000.0) + 1.0)
    
    # 4. Thermodynamic Hydraulic Power (kW)
    data['hydraulic_power_kw'] = (data['flow_rate_m3h'] * data['differential_pressure_bar'] * 100.0) / 3600.0
    
    # 5. Composite Mechanical Stress Index
    data['mechanical_stress_index'] = (data['vibration_rms'] * data['bearing_temperature_c']) / 100.0
    
    return data

def build_telemetry_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
