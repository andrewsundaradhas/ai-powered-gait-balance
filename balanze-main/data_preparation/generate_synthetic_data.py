"""
Synthetic gait data generator (Step 2)
Creates `data/raw/synthetic_gait_dataset.json` and ensures directories exist.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np


class SyntheticGaitDataGenerator:
	def __init__(self, num_samples: int = 1000) -> None:
		self.num_samples = num_samples
		self.frame_rate = 30

	def _linspace(self, num_frames: int) -> np.ndarray:
		return np.linspace(0, 4 * np.pi, num_frames)

	def generate_normal_gait(self, subject_id: str, age: int, num_frames: int = 300) -> Dict:
		t = self._linspace(num_frames)
		stride_length = 1.3 - 0.005 * age
		cadence = 120 - 0.3 * age
		frames: List[Dict] = []
		for i, tt in enumerate(t):
			frames.append(
				{
					"frame_id": i,
					"timestamp": i / self.frame_rate,
					"left_hip_angle": 20 + 25 * np.sin(tt),
					"right_hip_angle": 20 + 25 * np.sin(tt + np.pi),
					"left_knee_angle": 60 + 30 * np.sin(tt + 0.5),
					"right_knee_angle": 60 + 30 * np.sin(tt + 0.5 + np.pi),
					"left_ankle_angle": 10 + 15 * np.sin(tt),
					"right_ankle_angle": 10 + 15 * np.sin(tt + np.pi),
					"left_shoulder_rotation": 20 * np.sin(tt),
					"right_shoulder_rotation": 20 * np.sin(tt + np.pi),
					"pelvis_tilt": 3 * np.sin(2 * tt),
					"pelvis_obliquity": 2 * np.sin(2 * tt + np.pi / 4),
					"com_x": 0.5 * stride_length * np.sin(tt / 2),
					"com_y": 1.0 + 0.03 * np.cos(2 * tt),
					"com_z": 0.0,
					"cadence": float(cadence),
					"stride_length": float(stride_length),
				}
			)
		return {
			"subject_id": subject_id,
			"age": age,
			"gait_type": "Normal",
			"disorder": "None",
			"frames": frames,
		}

	def create_training_dataset(self, output_path: str) -> List[Dict]:
		dataset: List[Dict] = []
		# Normal
		for i in range(60):
			age = int(np.random.randint(20, 80))
			dataset.append(self.generate_normal_gait(f"normal_{i}", age))
		# Parkinsonian-like (reduced amplitudes, shuffling)
		for i in range(40):
			age = int(np.random.randint(55, 85))
			t = self._linspace(300)
			frames = []
			for j, tt in enumerate(t):
				frames.append({
					"frame_id": j,
					"timestamp": j / self.frame_rate,
					"left_hip_angle": 15 + 12 * np.sin(tt),
					"right_hip_angle": 15 + 12 * np.sin(tt + np.pi),
					"left_knee_angle": 50 + 18 * np.sin(tt + 0.5),
					"right_knee_angle": 50 + 18 * np.sin(tt + 0.5 + np.pi),
					"left_ankle_angle": 6 + 6 * np.sin(tt),
					"right_ankle_angle": 6 + 6 * np.sin(tt + np.pi),
					"com_x": 0.15 * np.sin(tt / 2),
					"com_y": 0.96 + 0.02 * np.cos(2 * tt),
					"com_z": 0.0,
				})
			dataset.append({
				"subject_id": f"park_{i}",
				"age": age,
				"gait_type": "Parkinsonian",
				"disorder": "Parkinsons Disease",
				"frames": frames,
			})
		# Cerebellar-like (wider sway, noise)
		for i in range(30):
			age = int(np.random.randint(50, 85))
			t = self._linspace(300)
			frames = []
			for j, tt in enumerate(t):
				noise = np.random.normal(0, 2.0)
				frames.append({
					"frame_id": j,
					"timestamp": j / self.frame_rate,
					"left_hip_angle": 20 + 25 * np.sin(tt) + noise,
					"right_hip_angle": 20 + 25 * np.sin(tt + np.pi) - noise,
					"left_knee_angle": 60 + 30 * np.sin(tt + 0.4) + noise,
					"right_knee_angle": 60 + 30 * np.sin(tt + 0.4 + np.pi) - noise,
					"left_ankle_angle": 10 + 15 * np.sin(tt) + 0.5 * noise,
					"right_ankle_angle": 10 + 15 * np.sin(tt + np.pi) - 0.5 * noise,
					"com_x": 0.35 * np.sin(tt / 2) + 0.1 * np.sin(3 * tt),
					"com_y": 1.0 + 0.08 * np.cos(2 * tt),
					"com_z": 0.02 * np.sin(tt),
				})
			dataset.append({
				"subject_id": f"cere_{i}",
				"age": age,
				"gait_type": "Cerebellar",
				"disorder": "Cerebellar Ataxia",
				"frames": frames,
			})
		# Hemiplegic-like (asymmetry)
		for i in range(30):
			age = int(np.random.randint(45, 85))
			t = self._linspace(300)
			frames = []
			for j, tt in enumerate(t):
				frames.append({
					"frame_id": j,
					"timestamp": j / self.frame_rate,
					"left_hip_angle": 20 + 25 * np.sin(tt),
					"right_hip_angle": 12 + 10 * np.sin(tt),
					"left_knee_angle": 60 + 30 * np.sin(tt + 0.5),
					"right_knee_angle": 42 + 14 * np.sin(tt + 0.5),
					"left_ankle_angle": 10 + 15 * np.sin(tt),
					"right_ankle_angle": 6 + 7 * np.sin(tt),
					"com_x": 0.28 * np.sin(tt / 2),
					"com_y": 1.0 + 0.03 * np.cos(2 * tt),
					"com_z": 0.0,
				})
			dataset.append({
				"subject_id": f"hemi_{i}",
				"age": age,
				"gait_type": "Hemiplegic",
				"disorder": "Stroke/Hemiplegia",
				"frames": frames,
			})

		out = Path(output_path)
		out.parent.mkdir(parents=True, exist_ok=True)
		with out.open("w", encoding="utf-8") as f:
			json.dump(dataset, f)
		return dataset


if __name__ == "__main__":
	gen = SyntheticGaitDataGenerator(num_samples=100)
	gen.create_training_dataset("data/raw/synthetic_gait_dataset.json")


