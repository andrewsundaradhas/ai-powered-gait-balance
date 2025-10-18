"""
Train models on Montreal Walking Gait Dataset
- Downloads dataset
- Extracts pose features using MediaPipe
- Trains three classifiers
- Saves models to models/*.pkl
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

# Add current directory to path for imports
sys.path.append('.')

try:
    import joblib
except ImportError:
    import pickle as joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.advanced_feature_engineer import AdvancedBioMechanicalFeatures
from data_preparation.download_montreal_dataset import download_montreal_dataset

MODELS_DIR = Path("models")

def load_montreal_dataset() -> List[Dict]:
    """Load Montreal dataset, download if not exists"""
    dataset_path = Path("data/raw/montreal_gait/montreal_gait_dataset.json")
    
    if not dataset_path.exists():
        print("Downloading Montreal Walking Gait Dataset...")
        download_montreal_dataset()
    
    with open(dataset_path, "r") as f:
        return json.load(f)

def build_feature_table(dataset: List[Dict]) -> pd.DataFrame:
    """Extract features from Montreal dataset"""
    feature_engineer = AdvancedBioMechanicalFeatures(sampling_rate=30.0)
    rows: List[pd.Series] = []
    labels: List[Dict] = []
    
    print(f"Processing {len(dataset)} samples...")
    
    for i, sample in enumerate(dataset):
        if i % 10 == 0:
            print(f"  Processing sample {i+1}/{len(dataset)}")
            
        frames = sample.get("frames", [])
        feat = feature_engineer.extract_all_features(frames)
        row = feat.iloc[0]
        row["subject_id"] = sample.get("subject_id")
        rows.append(row)
        
        # Map gait types to disorder categories
        gait_type = sample.get("gait_type", "Normal")
        disorder_map = {
            "Normal": "None",
            "Limping_Left": "Peripheral Neuropathy",
            "Limping_Right": "Peripheral Neuropathy", 
            "Shuffling": "Parkinsons Disease",
            "High_Stepping": "Peripheral Neuropathy",
            "Waddling": "Proximal Myopathy",
            "Antalgic": "None",
            "Ataxic": "Cerebellar Ataxia",
            "Spastic": "Spastic Paraparesis"
        }
        
        # Derive balance status from sway
        sway_total = row.get("com_sway_total", 0.0)
        if sway_total < 0.05:
            balance_status = "Stable"
        elif sway_total < 0.09:
            balance_status = "Mildly Impaired"
        elif sway_total < 0.14:
            balance_status = "Moderately Impaired"
        else:
            balance_status = "Severely Impaired"
        
        labels.append({
            "gait_type": gait_type,
            "disorder": disorder_map.get(gait_type, "None"),
            "balance_status": balance_status,
        })
    
    X = pd.DataFrame(rows).fillna(0)
    Y = pd.DataFrame(labels)
    return X.join(Y)

def train_and_save_models(df: pd.DataFrame) -> None:
    """Train three classifiers and save to models/"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    def _train(X: pd.DataFrame, y: pd.Series, algo: str = "gb") -> Pipeline:
        if algo == "rf":
            clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
        else:
            clf = GradientBoostingClassifier(random_state=42)
        return Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", clf)])
    
    feature_cols = [c for c in df.columns if c not in {"gait_type", "balance_status", "disorder", "subject_id"}]
    print(f"Training on {len(feature_cols)} features")
    
    # Gait classifier
    print("Training gait classifier...")
    Xg, yg = df[feature_cols], df["gait_type"].astype(str)
    Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(Xg, yg, test_size=0.2, random_state=42, stratify=yg)
    model_gait = _train(Xg_tr, yg_tr, "gb")
    model_gait.fit(Xg_tr, yg_tr)
    joblib.dump(model_gait, MODELS_DIR / "gait_classifier.pkl")
    
    # Balance classifier
    print("Training balance classifier...")
    Xb, yb = df[feature_cols], df["balance_status"].astype(str)
    Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.2, random_state=42, stratify=yb)
    model_balance = _train(Xb_tr, yb_tr, "rf")
    model_balance.fit(Xb_tr, yb_tr)
    joblib.dump(model_balance, MODELS_DIR / "balance_classifier.pkl")
    
    # Disorder classifier
    print("Training disorder classifier...")
    Xd, yd = df[feature_cols], df["disorder"].astype(str)
    Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.2, random_state=42, stratify=yd)
    model_disorder = _train(Xd_tr, yd_tr, "gb")
    model_disorder.fit(Xd_tr, yd_tr)
    joblib.dump(model_disorder, MODELS_DIR / "disorder_classifier.pkl")
    
    print("✓ Models saved to:")
    print("  - models/gait_classifier.pkl")
    print("  - models/balance_classifier.pkl") 
    print("  - models/disorder_classifier.pkl")
    
    # Print class distributions
    print("\nClass distributions:")
    print("Gait types:", yg.value_counts().to_dict())
    print("Balance status:", yb.value_counts().to_dict())
    print("Disorders:", yd.value_counts().to_dict())

if __name__ == "__main__":
    print("=== Training on Montreal Walking Gait Dataset ===")
    dataset = load_montreal_dataset()
    df = build_feature_table(dataset)
    train_and_save_models(df)
    print("✓ Training complete!")
