"""
PetroStream: SQLite Operational Historian & Work-Order Database
Logs offshore rotating asset telemetry, anomaly alarms, and maintenance dispatch records.
"""

import os
import sqlite3
from datetime import datetime
from typing import Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'telemetry_audit.db')

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS asset_telemetry_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        equipment_tag TEXT NOT NULL,
        equipment_type TEXT NOT NULL,
        platform_name TEXT NOT NULL,
        vibration_rms REAL NOT NULL,
        bearing_temperature_c REAL NOT NULL,
        discharge_pressure_bar REAL NOT NULL,
        health_index REAL NOT NULL,
        status TEXT NOT NULL,
        predicted_rul_hours REAL NOT NULL,
        work_order_priority TEXT NOT NULL,
        action_recommended TEXT
    )
    """)
    conn.commit()
    conn.close()

def log_telemetry_event(telemetry: Dict[str, Any], evaluation: Dict[str, Any]) -> int:
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    action_text = "; ".join(evaluation.get('action_recommendations', []))
    
    cursor.execute("""
    INSERT INTO asset_telemetry_logs (
        timestamp, equipment_tag, equipment_type, platform_name,
        vibration_rms, bearing_temperature_c, discharge_pressure_bar,
        health_index, status, predicted_rul_hours, work_order_priority,
        action_recommended
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        now_str,
        str(telemetry.get('equipment_tag', 'EQ-UNKNOWN')),
        str(telemetry.get('equipment_type', 'Compressor')),
        str(telemetry.get('platform_name', 'Dulang-A Platform')),
        float(telemetry.get('vibration_rms', 0.0)),
        float(telemetry.get('bearing_temperature_c', 0.0)),
        float(telemetry.get('discharge_pressure_bar', 0.0)),
        float(evaluation.get('health_index', 100.0)),
        str(evaluation.get('status', 'HEALTHY')),
        float(evaluation.get('predicted_rul_hours', 0.0)),
        str(evaluation.get('priority', 'P4 - Routine')),
        action_text
    ))
    
    conn.commit()
    row_id = cursor.lastrowid
    conn.close()
    return row_id

def get_recent_telemetry_logs(limit: int = 50) -> List[Dict[str, Any]]:
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM asset_telemetry_logs ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return rows

def get_fleet_summary() -> Dict[str, Any]:
    init_db()
    conn = get_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM asset_telemetry_logs")
    total_logs = cursor.fetchone()[0]
    
    if total_logs == 0:
        conn.close()
        return {
            'total_logs': 0,
            'healthy_count': 0,
            'warning_count': 0,
            'critical_count': 0,
            'avg_health_index': 100.0
        }
        
    cursor.execute("SELECT status, COUNT(*) FROM asset_telemetry_logs GROUP BY status")
    status_counts = dict(cursor.fetchall())
    
    cursor.execute("SELECT AVG(health_index) FROM asset_telemetry_logs")
    avg_health = cursor.fetchone()[0]
    
    conn.close()
    return {
        'total_logs': total_logs,
        'healthy_count': status_counts.get('HEALTHY', 0),
        'warning_count': status_counts.get('WARNING', 0),
        'critical_count': status_counts.get('CRITICAL', 0),
        'avg_health_index': round(avg_health or 100.0, 1)
    }

if __name__ == '__main__':
    init_db()
    print("PetroStream audit database initialized successfully at:", DB_PATH)
