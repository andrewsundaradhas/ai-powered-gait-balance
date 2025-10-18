"""
End-to-end training script:
- Loads/generates dataset
- Aggregates per-sample features using AdvancedBioMechanicalFeatures
- Trains three classifiers: gait, balance (derived), disorder
- Saves models to models/*.pkl
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.advanced_feature_engineer import AdvancedBioMechanicalFeatures


DATA_PATH = Path("data/raw/synthetic_gait_dataset.json")
MODELS_DIR = Path("models")


def ensure_dataset() -> List[Dict]:
	if DATA_PATH.exists():
		with DATA_PATH.open("r", encoding="utf-8") as f:
			return json.load(f)
	# Generate if missing
	from data_preparation.generate_synthetic_data import SyntheticGaitDataGenerator
	gen = SyntheticGaitDataGenerator(num_samples=160)
	return gen.create_training_dataset(str(DATA_PATH))


def build_feature_table(dataset: List[Dict]) -> pd.DataFrame:
	feature_engineer = AdvancedBioMechanicalFeatures(sampling_rate=30.0)
	rows: List[pd.Series] = []
	labels: List[Dict] = []
	for sample in dataset:
		frames = sample.get("frames", [])
		feat = feature_engineer.extract_all_features(frames)
		row = feat.iloc[0]
		row["subject_id"] = sample.get("subject_id")
		rows.append(row)
		labels.append(
			{
				"gait_type": sample.get("gait_type", "Unknown"),
				"disorder": sample.get("disorder", "None"),
				# Derive balance status from sway for baseline
				"balance_status": _derive_balance_status(row.get("com_sway_total", 0.0)),
			}
		)
	X = pd.DataFrame(rows).fillna(0)
	Y = pd.DataFrame(labels)
	return X.join(Y)


def _derive_balance_status(sway_total: float) -> str:
	if sway_total < 0.05:
		return "Stable"
	if sway_total < 0.09:
		return "Mildly Impaired"
	if sway_total < 0.14:
		return "Moderately Impaired"
	return "Severely Impaired"


def train_and_save_models(df: pd.DataFrame) -> None:
	MODELS_DIR.mkdir(parents=True, exist_ok=True)

	def _train(X: pd.DataFrame, y: pd.Series, algo: str = "gb") -> Pipeline:
		if algo == "rf":
			clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
		else:
			clf = GradientBoostingClassifier(random_state=42)
		return Pipeline([("scaler", StandardScaler(with_mean=False)), ("clf", clf)])

	feature_cols = [c for c in df.columns if c not in {"gait_type", "balance_status", "disorder", "subject_id"}]

	# Gait
	Xg, yg = df[feature_cols], df["gait_type"].astype(str)
	Xg_tr, Xg_te, yg_tr, yg_te = train_test_split(Xg, yg, test_size=0.2, random_state=42, stratify=yg)
	model_gait = _train(Xg_tr, yg_tr, "gb")
	model_gait.fit(Xg_tr, yg_tr)
	joblib.dump(model_gait, MODELS_DIR / "gait_classifier.pkl")

	# Balance (rf works well on derived discrete)
	Xb, yb = df[feature_cols], df["balance_status"].astype(str)
	Xb_tr, Xb_te, yb_tr, yb_te = train_test_split(Xb, yb, test_size=0.2, random_state=42, stratify=yb)
	model_balance = _train(Xb_tr, yb_tr, "rf")
	model_balance.fit(Xb_tr, yb_tr)
	joblib.dump(model_balance, MODELS_DIR / "balance_classifier.pkl")

	# Disorder
	Xd, yd = df[feature_cols], df["disorder"].astype(str)
	Xd_tr, Xd_te, yd_tr, yd_te = train_test_split(Xd, yd, test_size=0.2, random_state=42, stratify=yd)
	model_disorder = _train(Xd_tr, yd_tr, "gb")
	model_disorder.fit(Xd_tr, yd_tr)
	joblib.dump(model_disorder, MODELS_DIR / "disorder_classifier.pkl")

	print("✓ Models saved to:")
	print("  - models/gait_classifier.pkl")
	print("  - models/balance_classifier.pkl")
	print("  - models/disorder_classifier.pkl")


if __name__ == "__main__":
	dataset = ensure_dataset()
	df = build_feature_table(dataset)
	train_and_save_models(df)


