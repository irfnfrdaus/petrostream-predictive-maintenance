"""
PetroStream: Industrial Oil & Gas Asset Telemetry Generator
Simulates realistic IoT sensor streams for critical rotating assets (gas turbines, 
centrifugal compressors, multiphase pumps) across Malaysian offshore platforms.
Calibrated according to ISO 10816-3 mechanical vibration standards.
"""

import os
import numpy as np
import pandas as pd

PLATFORMS = [
    {"name": "Dulang-A Platform", "basin": "Malay Basin (Terengganu)", "lat": 5.467, "lon": 104.750},
    {"name": "Angsi Complex", "basin": "Offshore Peninsular", "lat": 5.340, "lon": 105.280},
    {"name": "Bekok-C Platform", "basin": "PM304 Block", "lat": 5.410, "lon": 105.210},
    {"name": "Baram Central", "basin": "Baram Delta (Sarawak)", "lat": 4.600, "lon": 113.970},
    {"name": "Bokor Platform", "basin": "Offshore Miri (Sarawak)", "lat": 4.550, "lon": 113.620}
]

EQUIPMENT_TYPES = [
    {"type": "Export Gas Compressor", "tag_prefix": "K-101", "rated_rpm": 11500, "base_press": 65.0},
    {"type": "Solar Titan Gas Turbine", "tag_prefix": "GT-201", "rated_rpm": 9500, "base_press": 48.0},
    {"type": "Multiphase Booster Pump", "tag_prefix": "P-301", "rated_rpm": 3200, "base_press": 28.0},
    {"type": "Water Injection Pump", "tag_prefix": "P-401", "rated_rpm": 2950, "base_press": 110.0}
]

FAILURE_MODES = [
    "NORMAL",
    "BEARING_INNER_RACE_WEAR",
    "ROTOR_UNBALANCE",
    "SEAL_DEGRADATION",
    "IMPELLER_CAVITATION"
]

def generate_telemetry_dataset(n_samples: int = 15000, random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    
    # 1. Asset & Platform Assignment
    platforms_idx = np.random.choice(len(PLATFORMS), size=n_samples)
    platform_names = [PLATFORMS[i]["name"] for i in platforms_idx]
    basins = [PLATFORMS[i]["basin"] for i in platforms_idx]
    lats = [PLATFORMS[i]["lat"] + np.random.uniform(-0.01, 0.01) for i in platforms_idx]
    lons = [PLATFORMS[i]["lon"] + np.random.uniform(-0.01, 0.01) for i in platforms_idx]
    
    equip_idx = np.random.choice(len(EQUIPMENT_TYPES), size=n_samples)
    equipment_names = [EQUIPMENT_TYPES[i]["type"] for i in equip_idx]
    equipment_tags = [f"{EQUIPMENT_TYPES[i]['tag_prefix']}-{np.random.randint(1, 4):02d}" for i in equip_idx]
    rated_rpms = np.array([EQUIPMENT_TYPES[i]["rated_rpm"] for i in equip_idx])
    base_pressures = np.array([EQUIPMENT_TYPES[i]["base_press"] for i in equip_idx])
    
    # 2. Operational Hours (Cumulative operating hours since last overhaul)
    operating_hours = np.random.uniform(500, 24000, size=n_samples)
    
    # 3. Latent Health & Degradation State (0.0 = functional failure, 1.0 = brand new)
    # Natural wear increases non-linearly after 15,000 hours
    age_factor = np.clip(operating_hours / 22000.0, 0, 1.0)
    health_state = 1.0 - (age_factor ** 1.8) * np.random.uniform(0.5, 0.95, size=n_samples)
    health_state = np.clip(health_state + np.random.normal(0, 0.08, size=n_samples), 0.02, 1.0)
    
    # Remaining Useful Life (RUL) in operating hours
    # Healthy assets have 1500-4000 hrs; failing assets have < 200 hrs
    rul = health_state * np.random.uniform(2200, 3800, size=n_samples)
    rul = np.clip(np.round(rul, 1), 0.0, 4000.0)
    
    # 4. Sensor Telemetry Generation
    # ISO 10816 Vibration RMS (mm/s): Normal < 2.8, Warning 2.8-7.1, Critical > 7.1
    base_vibration = 1.5 + (1.0 - health_state) * 7.5
    vibration_rms = base_vibration + np.random.normal(0, 0.45, size=n_samples)
    vibration_rms = np.clip(np.round(vibration_rms, 2), 0.40, 18.50)
    
    # Bearing Temperature (°C): Normal 65-80°C, Warning >85°C, Critical >100°C
    base_temp = 68.0 + (1.0 - health_state) * 42.0
    bearing_temp = base_temp + np.random.normal(0, 2.5, size=n_samples)
    bearing_temp = np.clip(np.round(bearing_temp, 1), 45.0, 135.0)
    
    # Discharge Pressure (bar)
    discharge_press = base_pressures * (0.90 + 0.15 * health_state) + np.random.normal(0, 1.8, size=n_samples)
    discharge_press = np.clip(np.round(discharge_press, 2), 10.0, 150.0)
    
    # Suction Pressure (bar)
    suction_press = (base_pressures * 0.28) + np.random.normal(0, 0.8, size=n_samples)
    suction_press = np.clip(np.round(suction_press, 2), 2.0, 45.0)
    
    # Differential Pressure
    diff_press = np.round(discharge_press - suction_press, 2)
    
    # Flow Rate (m³/h)
    flow_rate = (350.0 * (0.85 + 0.15 * health_state)) + np.random.normal(0, 12.0, size=n_samples)
    flow_rate = np.clip(np.round(flow_rate, 1), 80.0, 600.0)
    
    # Rotational Speed (RPM)
    speed_rpm = rated_rpms * (1.0 + np.random.normal(0, 0.015, size=n_samples))
    speed_rpm = np.clip(np.round(speed_rpm, 0), 1000, 16000)
    
    # Lubricant Viscosity (cSt): standard ISO VG 46/68 oil
    lube_viscosity = 46.0 - (1.0 - health_state) * 14.0 + np.random.normal(0, 1.2, size=n_samples)
    lube_viscosity = np.clip(np.round(lube_viscosity, 1), 18.0, 55.0)
    
    # 5. Anomaly & Failure Mode Classification
    is_anomaly = ((vibration_rms > 5.5) | (bearing_temp > 92.0) | (health_state < 0.25)).astype(int)
    
    def assign_failure_mode(vibe, temp, anom):
        if anom == 0:
            return "NORMAL"
        if vibe > 8.0 and temp > 95.0:
            return "BEARING_INNER_RACE_WEAR"
        elif vibe > 7.0:
            return "ROTOR_UNBALANCE"
        elif temp > 92.0:
            return "SEAL_DEGRADATION"
        else:
            return "IMPELLER_CAVITATION"
            
    failure_mode = [assign_failure_mode(v, t, a) for v, t, a in zip(vibration_rms, bearing_temp, is_anomaly)]
    
    df = pd.DataFrame({
        'platform_name': platform_names,
        'basin': basins,
        'latitude': np.round(lats, 4),
        'longitude': np.round(lons, 4),
        'equipment_type': equipment_names,
        'equipment_tag': equipment_tags,
        'operating_hours': np.round(operating_hours, 0),
        'vibration_rms': vibration_rms,
        'bearing_temperature_c': bearing_temp,
        'discharge_pressure_bar': discharge_press,
        'suction_pressure_bar': suction_press,
        'differential_pressure_bar': diff_press,
        'flow_rate_m3h': flow_rate,
        'rotational_speed_rpm': speed_rpm,
        'lubricant_viscosity_cst': lube_viscosity,
        'health_index': np.round(health_state * 100, 1),
        'remaining_useful_life_hrs': rul,
        'is_anomaly': is_anomaly,
        'failure_mode': failure_mode
    })
    
    return df

if __name__ == '__main__':
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_csv = os.path.join(base_dir, 'data', 'raw', 'equipment_telemetry.csv')
    df = generate_telemetry_dataset(n_samples=16000, random_state=42)
    df.to_csv(out_csv, index=False)
    print(f"Generated {len(df)} industrial telemetry records at: {out_csv}")
    print(f"Anomaly Rate: {df['is_anomaly'].mean()*100:.2f}% | Avg RUL: {df['remaining_useful_life_hrs'].mean():.1f} hrs")
