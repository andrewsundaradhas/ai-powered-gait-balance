"""
Simplified training script that works without complex dependencies
"""

import json
import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append('.')

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except ImportError:
    import pickle as joblib

def create_synthetic_dataset():
    """Create synthetic gait dataset with multiple classes"""
    print("Creating synthetic gait dataset...")
    
    dataset = []
    gait_types = ["Normal", "Parkinsonian", "Cerebellar", "Hemiplegic", "Antalgic"]
    disorders = ["None", "Parkinsons Disease", "Cerebellar Ataxia", "Stroke/Hemiplegia", "None"]
    balance_levels = ["Stable", "Mildly Impaired", "Moderately Impaired", "Severely Impaired"]
    
    for i in range(200):  # 200 samples total
        gait_type = gait_types[i % len(gait_types)]
        disorder = disorders[i % len(disorders)]
        
        # Generate synthetic features based on gait type
        if gait_type == "Normal":
            features = {
                "left_hip_angle_mean": 25 + np.random.normal(0, 3),
                "right_hip_angle_mean": 25 + np.random.normal(0, 3),
                "left_knee_angle_mean": 60 + np.random.normal(0, 5),
                "right_knee_angle_mean": 60 + np.random.normal(0, 5),
                "left_ankle_angle_mean": 15 + np.random.normal(0, 2),
                "right_ankle_angle_mean": 15 + np.random.normal(0, 2),
                "com_sway_total": 0.05 + np.random.normal(0, 0.01),
                "hip_angle_asymmetry": 2 + np.random.normal(0, 1),
                "knee_angle_asymmetry": 3 + np.random.normal(0, 1),
                "ankle_angle_asymmetry": 1 + np.random.normal(0, 0.5),
            }
            balance_status = "Stable"
            
        elif gait_type == "Parkinsonian":
            features = {
                "left_hip_angle_mean": 15 + np.random.normal(0, 2),
                "right_hip_angle_mean": 15 + np.random.normal(0, 2),
                "left_knee_angle_mean": 50 + np.random.normal(0, 3),
                "right_knee_angle_mean": 50 + np.random.normal(0, 3),
                "left_ankle_angle_mean": 8 + np.random.normal(0, 1),
                "right_ankle_angle_mean": 8 + np.random.normal(0, 1),
                "com_sway_total": 0.08 + np.random.normal(0, 0.02),
                "hip_angle_asymmetry": 5 + np.random.normal(0, 2),
                "knee_angle_asymmetry": 8 + np.random.normal(0, 2),
                "ankle_angle_asymmetry": 3 + np.random.normal(0, 1),
            }
            balance_status = "Moderately Impaired"
            
        elif gait_type == "Cerebellar":
            features = {
                "left_hip_angle_mean": 25 + np.random.normal(0, 8),
                "right_hip_angle_mean": 25 + np.random.normal(0, 8),
                "left_knee_angle_mean": 60 + np.random.normal(0, 10),
                "right_knee_angle_mean": 60 + np.random.normal(0, 10),
                "left_ankle_angle_mean": 15 + np.random.normal(0, 5),
                "right_ankle_angle_mean": 15 + np.random.normal(0, 5),
                "com_sway_total": 0.15 + np.random.normal(0, 0.03),
                "hip_angle_asymmetry": 12 + np.random.normal(0, 3),
                "knee_angle_asymmetry": 15 + np.random.normal(0, 4),
                "ankle_angle_asymmetry": 8 + np.random.normal(0, 2),
            }
            balance_status = "Severely Impaired"
            
        elif gait_type == "Hemiplegic":
            features = {
                "left_hip_angle_mean": 25 + np.random.normal(0, 3),
                "right_hip_angle_mean": 15 + np.random.normal(0, 2),
                "left_knee_angle_mean": 60 + np.random.normal(0, 5),
                "right_knee_angle_mean": 45 + np.random.normal(0, 3),
                "left_ankle_angle_mean": 15 + np.random.normal(0, 2),
                "right_ankle_angle_mean": 8 + np.random.normal(0, 1),
                "com_sway_total": 0.12 + np.random.normal(0, 0.02),
                "hip_angle_asymmetry": 15 + np.random.normal(0, 3),
                "knee_angle_asymmetry": 20 + np.random.normal(0, 4),
                "ankle_angle_asymmetry": 10 + np.random.normal(0, 2),
            }
            balance_status = "Moderately Impaired"
            
        else:  # Antalgic
            features = {
                "left_hip_angle_mean": 20 + np.random.normal(0, 2),
                "right_hip_angle_mean": 18 + np.random.normal(0, 2),
                "left_knee_angle_mean": 55 + np.random.normal(0, 3),
                "right_knee_angle_mean": 50 + np.random.normal(0, 3),
                "left_ankle_angle_mean": 12 + np.random.normal(0, 1),
                "right_ankle_angle_mean": 10 + np.random.normal(0, 1),
                "com_sway_total": 0.09 + np.random.normal(0, 0.02),
                "hip_angle_asymmetry": 8 + np.random.normal(0, 2),
                "knee_angle_asymmetry": 10 + np.random.normal(0, 2),
                "ankle_angle_asymmetry": 5 + np.random.normal(0, 1),
            }
            balance_status = "Mildly Impaired"
        
        # Add some noise to all features
        for key in features:
            features[key] += np.random.normal(0, 0.1)
        
        sample = {
            "subject_id": f"subject_{i}",
            "gait_type": gait_type,
            "disorder": disorder,
            "balance_status": balance_status,
            **features
        }
        dataset.append(sample)
    
    return dataset

def train_models(dataset):
    """Train three classifiers"""
    print("Training models...")
    
    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    
    # Prepare features (exclude labels and subject_id)
    feature_cols = [col for col in df.columns if col not in ['subject_id', 'gait_type', 'disorder', 'balance_status']]
    X = df[feature_cols]
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # 1. Gait Classifier
    print("Training gait classifier...")
    y_gait = df['gait_type']
    X_train, X_test, y_train, y_test = train_test_split(X, y_gait, test_size=0.2, random_state=42, stratify=y_gait)
    
    gait_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    gait_model.fit(X_train, y_train)
    
    # Save gait model
    joblib.dump(gait_model, models_dir / "gait_classifier.pkl")
    print(f"Gait classifier accuracy: {gait_model.score(X_test, y_test):.3f}")
    
    # 2. Balance Classifier
    print("Training balance classifier...")
    y_balance = df['balance_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y_balance, test_size=0.2, random_state=42, stratify=y_balance)
    
    balance_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    balance_model.fit(X_train, y_train)
    
    # Save balance model
    joblib.dump(balance_model, models_dir / "balance_classifier.pkl")
    print(f"Balance classifier accuracy: {balance_model.score(X_test, y_test):.3f}")
    
    # 3. Disorder Classifier
    print("Training disorder classifier...")
    y_disorder = df['disorder']
    X_train, X_test, y_train, y_test = train_test_split(X, y_disorder, test_size=0.2, random_state=42, stratify=y_disorder)
    
    disorder_model = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
    ])
    disorder_model.fit(X_train, y_train)
    
    # Save disorder model
    joblib.dump(disorder_model, models_dir / "disorder_classifier.pkl")
    print(f"Disorder classifier accuracy: {disorder_model.score(X_test, y_test):.3f}")
    
    print("\n✓ All models trained and saved!")
    print("Models saved to:")
    print("  - models/gait_classifier.pkl")
    print("  - models/balance_classifier.pkl")
    print("  - models/disorder_classifier.pkl")
    
    return gait_model, balance_model, disorder_model

def test_predictions(gait_model, balance_model, disorder_model):
    """Test models with sample predictions"""
    print("\nTesting model predictions...")
    
    # Create test features
    test_features = np.array([[25, 25, 60, 60, 15, 15, 0.05, 2, 3, 1]])  # Normal gait
    test_features_parkinsonian = np.array([[15, 15, 50, 50, 8, 8, 0.08, 5, 8, 3]])  # Parkinsonian
    
    # Test predictions
    gait_pred = gait_model.predict(test_features)[0]
    balance_pred = balance_model.predict(test_features)[0]
    disorder_pred = disorder_model.predict(test_features)[0]
    
    print(f"Normal gait test:")
    print(f"  Gait: {gait_pred}")
    print(f"  Balance: {balance_pred}")
    print(f"  Disorder: {disorder_pred}")
    
    gait_pred2 = gait_model.predict(test_features_parkinsonian)[0]
    balance_pred2 = balance_model.predict(test_features_parkinsonian)[0]
    disorder_pred2 = disorder_model.predict(test_features_parkinsonian)[0]
    
    print(f"\nParkinsonian gait test:")
    print(f"  Gait: {gait_pred2}")
    print(f"  Balance: {balance_pred2}")
    print(f"  Disorder: {disorder_pred2}")

if __name__ == "__main__":
    print("=== Gait Analysis Model Training ===")
    
    # Create dataset
    dataset = create_synthetic_dataset()
    print(f"Created dataset with {len(dataset)} samples")
    
    # Train models
    gait_model, balance_model, disorder_model = train_models(dataset)
    
    # Test predictions
    test_predictions(gait_model, balance_model, disorder_model)
    
    print("\n✓ Training complete! Models are ready for use.")
