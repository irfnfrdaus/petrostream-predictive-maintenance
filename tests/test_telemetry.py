"""
PetroStream: Automated Unit & Integration Tests
Validates telemetry feature transformations, ISO 10816 zone classification,
health index calculation, and database logging.
"""

import sys
import os
import unittest
import pandas as pd

# Add src and database to path
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(base_dir, 'src'))
sys.path.append(os.path.join(base_dir, 'database'))

from feature_engineering import engineer_telemetry_features
from maintenance_engine import PetroStreamEngine
from db import init_db, log_telemetry_event, get_recent_telemetry_logs

class TestPetroStreamEngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = PetroStreamEngine()
        init_db()

    def test_feature_engineering_transforms(self):
        sample = pd.DataFrame([{
            'platform_name': 'Dulang-A Platform',
            'equipment_type': 'Export Gas Compressor',
            'operating_hours': 10000,
            'vibration_rms': 3.5,
            'bearing_temperature_c': 75.0,
            'discharge_pressure_bar': 70.0,
            'suction_pressure_bar': 20.0,
            'differential_pressure_bar': 50.0,
            'flow_rate_m3h': 350.0,
            'rotational_speed_rpm': 11500,
            'lubricant_viscosity_cst': 44.0
        }])
        
        df_feat = engineer_telemetry_features(sample)
        self.assertIn('vibration_severity_ratio', df_feat.columns)
        self.assertIn('pressure_ratio', df_feat.columns)
        self.assertIn('thermal_gradient', df_feat.columns)
        self.assertIn('hydraulic_power_kw', df_feat.columns)
        self.assertIn('mechanical_stress_index', df_feat.columns)

    def test_healthy_asset_evaluation(self):
        healthy_telemetry = {
            'platform_name': 'Angsi Complex',
            'equipment_type': 'Solar Titan Gas Turbine',
            'equipment_tag': 'GT-201-01',
            'operating_hours': 3000,
            'vibration_rms': 1.4,
            'bearing_temperature_c': 65.0,
            'discharge_pressure_bar': 48.0,
            'suction_pressure_bar': 15.0,
            'differential_pressure_bar': 33.0,
            'flow_rate_m3h': 400.0,
            'rotational_speed_rpm': 9500,
            'lubricant_viscosity_cst': 46.0
        }
        res = self.engine.evaluate_asset(healthy_telemetry)
        self.assertEqual(res['status'], 'HEALTHY')
        self.assertGreater(res['health_index'], 70.0)
        self.assertIn("Zone A", res['iso_zone'])

    def test_critical_vibration_trip(self):
        critical_telemetry = {
            'platform_name': 'Baram Central',
            'equipment_type': 'Export Gas Compressor',
            'equipment_tag': 'K-101-02',
            'operating_hours': 22000,
            'vibration_rms': 10.5, # Exceeds 7.1 mm/s trip limit
            'bearing_temperature_c': 110.0,
            'discharge_pressure_bar': 65.0,
            'suction_pressure_bar': 18.0,
            'differential_pressure_bar': 47.0,
            'flow_rate_m3h': 260.0,
            'rotational_speed_rpm': 11450,
            'lubricant_viscosity_cst': 22.0
        }
        res = self.engine.evaluate_asset(critical_telemetry)
        self.assertEqual(res['status'], 'CRITICAL')
        self.assertIn("Zone D", res['iso_zone'])
        self.assertIn("P1", res['priority'])

    def test_historian_logging(self):
        sample_pkt = {
            'equipment_tag': 'TEST-PUMP-99',
            'equipment_type': 'Multiphase Booster Pump',
            'platform_name': 'Dulang-A Platform',
            'vibration_rms': 2.5,
            'bearing_temperature_c': 70.0,
            'discharge_pressure_bar': 25.0
        }
        eval_dict = {
            'health_index': 88.0,
            'status': 'HEALTHY',
            'predicted_rul_hours': 1800.0,
            'priority': 'P4 - Routine',
            'action_recommendations': ['Test logging action']
        }
        log_id = log_telemetry_event(sample_pkt, eval_dict)
        self.assertGreater(log_id, 0)
        
        recent = get_recent_telemetry_logs(limit=5)
        tags = [r['equipment_tag'] for r in recent]
        self.assertIn('TEST-PUMP-99', tags)

if __name__ == '__main__':
    unittest.main()
