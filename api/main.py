"""
PetroStream: Industrial IoT Telemetry Ingestion & Anomaly Scoring REST API
High-throughput FastAPI microservice for real-time asset condition monitoring,
vibration alert dispatching, and remaining useful life (RUL) estimation.
"""

import sys
import os
from typing import List, Optional, Literal
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add src and database to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'src'))
sys.path.append(os.path.join(base_dir, 'database'))

from maintenance_engine import PetroStreamEngine
from db import init_db, log_telemetry_event, get_recent_telemetry_logs, get_fleet_summary

app = FastAPI(
    title="PetroStream: Industrial IoT Predictive Maintenance API",
    description="Real-time condition monitoring, ISO 10816 vibration classification, and RUL prognostics engine.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
engine = PetroStreamEngine()

class TelemetryPayload(BaseModel):
    platform_name: Literal['Dulang-A Platform', 'Angsi Complex', 'Bekok-C Platform', 'Baram Central', 'Bokor Platform'] = Field(example="Dulang-A Platform")
    equipment_type: Literal['Export Gas Compressor', 'Solar Titan Gas Turbine', 'Multiphase Booster Pump', 'Water Injection Pump'] = Field(example="Export Gas Compressor")
    equipment_tag: str = Field(example="K-101-01", description="Asset tag identifier")
    operating_hours: float = Field(ge=0.0, example=14500.0, description="Cumulative hours since overhaul")
    vibration_rms: float = Field(ge=0.0, le=50.0, example=3.20, description="ISO 10816 Overall Vibration RMS (mm/s)")
    bearing_temperature_c: float = Field(ge=0.0, le=200.0, example=78.5, description="Bearing metal temperature in °C")
    discharge_pressure_bar: float = Field(ge=0.0, example=68.5, description="Discharge pressure in bar")
    suction_pressure_bar: float = Field(ge=0.0, example=18.2, description="Suction pressure in bar")
    differential_pressure_bar: float = Field(ge=0.0, default=50.3, example=50.3)
    flow_rate_m3h: float = Field(ge=0.0, example=340.0, description="Process volumetric flow in m³/h")
    rotational_speed_rpm: float = Field(ge=0.0, example=11480.0, description="Shaft rotational speed in RPM")
    lubricant_viscosity_cst: float = Field(ge=5.0, le=100.0, example=43.0, description="Lube oil viscosity at 40°C")

@app.get("/", tags=["Overview"])
def root():
    return {
        "service": "PetroStream Predictive Maintenance API",
        "status": "ONLINE",
        "documentation": "/docs",
        "version": "1.0.0"
    }

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "OPERATIONAL",
        "model_loaded": True,
        "database_connected": True
    }

@app.post("/api/v1/telemetry/evaluate", tags=["Condition Monitoring"])
def evaluate_asset_telemetry(packet: TelemetryPayload):
    """
    Ingests live sensor telemetry, runs ISO 10816 evaluation, Isolation Forest anomaly scoring,
    and returns Remaining Useful Life (RUL) with maintenance recommendations.
    """
    try:
        data_dict = packet.model_dump()
        evaluation = engine.evaluate_asset(data_dict)
        
        # Log to historian
        log_id = log_telemetry_event(data_dict, evaluation)
        
        return {
            "telemetry_id": log_id,
            "equipment_tag": packet.equipment_tag,
            "platform_name": packet.platform_name,
            **evaluation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/fleet/status", tags=["Fleet Overview"])
def get_fleet_telemetry_status():
    """
    Returns aggregated health index and alarm statistics across the offshore asset fleet.
    """
    return get_fleet_summary()

@app.get("/api/v1/maintenance/work-orders", tags=["Work Order History"])
def get_work_order_history(limit: int = Query(default=25, ge=1, le=200)):
    """
    Returns recent operational telemetry events and dispatched maintenance priorities.
    """
    return get_recent_telemetry_logs(limit=limit)
