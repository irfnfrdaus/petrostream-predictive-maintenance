"""
PetroStream: Model Training & Asset Prognostics Pipeline
Trains dual-engine predictive maintenance models:
1. Unsupervised Anomaly Detector (Isolation Forest) for zero-day vibration/thermal spikes.
2. Supervised Prognostics Engine (Gradient Boosting Regressor) for Remaining Useful Life (RUL).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest, GradientBoostingRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score
)

from data_generator import generate_telemetry_dataset
from feature_engineering import engineer_telemetry_features, build_telemetry_preprocessor

def train_petrostream_models():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_csv = os.path.join(base_dir, 'data', 'raw', 'equipment_telemetry.csv')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 1. Load or Generate Dataset
    if not os.path.exists(raw_csv):
        print("Generating 16,000 industrial asset telemetry records...")
        df = generate_telemetry_dataset(n_samples=16000, random_state=42)
        df.to_csv(raw_csv, index=False)
    else:
        df = pd.read_csv(raw_csv)
        
    print(f"Loaded telemetry dataset: {len(df)} records. Fleet Anomaly Rate: {df['is_anomaly'].mean()*100:.2f}%")
    
    # 2. Feature Engineering
    df_engineered = engineer_telemetry_features(df)
    processed_parquet = os.path.join(base_dir, 'data', 'processed', 'telemetry_features.parquet')
    df_engineered.to_parquet(processed_parquet, index=False)
    
    # Feature columns for modeling
    drop_cols = ['is_anomaly', 'remaining_useful_life_hrs', 'health_index', 'failure_mode', 'equipment_tag', 'basin', 'latitude', 'longitude']
    X = df_engineered.drop(columns=drop_cols)
    y_anomaly = df_engineered['is_anomaly']
    y_rul = df_engineered['remaining_useful_life_hrs']
    
    # 80/20 Train-Test Split
    X_train, X_test, y_anom_train, y_anom_test, y_rul_train, y_rul_test = train_test_split(
        X, y_anomaly, y_rul, test_size=0.20, random_state=42, stratify=y_anomaly
    )
    
    # ----------------------------------------------------
    # Model 1: Remaining Useful Life (RUL) Regressor
    # ----------------------------------------------------
    print("\n--- Training Remaining Useful Life (RUL) Prognostics Engine ---")
    rul_pipeline = Pipeline([
        ('preprocessor', build_telemetry_preprocessor()),
        ('regressor', GradientBoostingRegressor(
            n_estimators=160,
            learning_rate=0.08,
            max_depth=4,
            subsample=0.85,
            random_state=42
        ))
    ])
    
    rul_pipeline.fit(X_train, y_rul_train)
    y_rul_pred = rul_pipeline.predict(X_test)
    
    r2 = float(r2_score(y_rul_test, y_rul_pred))
    rmse = float(np.sqrt(mean_squared_error(y_rul_test, y_rul_pred)))
    mae = float(mean_absolute_error(y_rul_test, y_rul_pred))
    
    print(f"RUL Prognostics R² Score: {r2:.4f} (Accuracy benchmark: >0.85)")
    print(f"RUL Root Mean Squared Error (RMSE): {rmse:.1f} Operating Hours")
    print(f"RUL Mean Absolute Error (MAE):     {mae:.1f} Operating Hours")
    
    # ----------------------------------------------------
    # Model 2: Unsupervised Zero-Day Anomaly Detector
    # ----------------------------------------------------
    print("\n--- Training Isolation Forest Anomaly Detector ---")
    preproc = build_telemetry_preprocessor()
    X_train_trans = preproc.fit_transform(X_train)
    X_test_trans = preproc.transform(X_test)
    
    iso_forest = IsolationForest(
        n_estimators=150,
        contamination=0.12,
        random_state=42
    )
    iso_forest.fit(X_train_trans)
    
    # Predict (-1 is anomaly, 1 is normal) -> convert to 1 for anomaly, 0 for normal
    raw_preds = iso_forest.predict(X_test_trans)
    y_anom_pred = np.where(raw_preds == -1, 1, 0)
    scores = -iso_forest.score_samples(X_test_trans) # Higher score = more anomalous
    
    f1 = float(f1_score(y_anom_test, y_anom_pred))
    prec = float(precision_score(y_anom_test, y_anom_pred))
    rec = float(recall_score(y_anom_test, y_anom_pred))
    roc_auc = float(roc_auc_score(y_anom_test, scores))
    
    print(f"Anomaly Detection ROC-AUC: {roc_auc:.4f}")
    print(f"Anomaly Detection F1-Score: {f1:.4f}")
    
    # Save Pipeline & Models
    joblib.dump(rul_pipeline, os.path.join(models_dir, 'rul_model.joblib'))
    joblib.dump({'preprocessor': preproc, 'model': iso_forest}, os.path.join(models_dir, 'anomaly_detector.joblib'))
    
    metrics = {
        'rul_prognostics': {
            'r2_score': r2,
            'rmse_hours': rmse,
            'mae_hours': mae
        },
        'anomaly_detection': {
            'roc_auc': roc_auc,
            'f1_score': f1,
            'precision': prec,
            'recall': rec
        }
    }
    
    with open(os.path.join(models_dir, 'model_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("\n[Production Models Saved Successfully]")
    return metrics

if __name__ == '__main__':
    train_petrostream_models()
