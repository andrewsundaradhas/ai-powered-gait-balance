"""
Lightweight feature engineering stub used by real-time inference.

Implements a single method `extract_all_features(sequence)` that converts a
list of keypoint dicts into a 1-row numpy array-like structure compatible with
scikit-learn pipelines' `predict`/`predict_proba` expectations.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


class AdvancedBioMechanicalFeatures:
    def __init__(self, sampling_rate: float = 30.0) -> None:
        self.sampling_rate = sampling_rate

    def _safe_series_stats(self, values: List[float], prefix: str) -> Dict[str, float]:
        if len(values) == 0:
            return {
                f"{prefix}_mean": 0.0,
                f"{prefix}_std": 0.0,
                f"{prefix}_min": 0.0,
                f"{prefix}_max": 0.0,
            }
        arr = np.asarray(values, dtype=float)
        return {
            f"{prefix}_mean": float(np.mean(arr)),
            f"{prefix}_std": float(np.std(arr)),
            f"{prefix}_min": float(np.min(arr)),
            f"{prefix}_max": float(np.max(arr)),
        }

    def extract_all_features(self, keypoint_sequence: List[Dict]) -> pd.DataFrame:
        """
        Convert a sequence of keypoint dicts into a single-row feature DataFrame.
        Missing fields are handled gracefully with zeros.
        """
        fields = [
            "left_hip_angle",
            "right_hip_angle",
            "left_knee_angle",
            "right_knee_angle",
            "left_ankle_angle",
            "right_ankle_angle",
            "com_x",
            "com_y",
            "com_z",
        ]

        aggregated: Dict[str, float] = {}
        for f in fields:
            series_vals = [float(frame.get(f, 0.0)) for frame in keypoint_sequence]
            aggregated.update(self._safe_series_stats(series_vals, f))

        # Simple derived metrics
        hip_asym = abs(aggregated.get("left_hip_angle_mean", 0.0) - aggregated.get("right_hip_angle_mean", 0.0))
        knee_asym = abs(aggregated.get("left_knee_angle_mean", 0.0) - aggregated.get("right_knee_angle_mean", 0.0))
        ankle_asym = abs(aggregated.get("left_ankle_angle_mean", 0.0) - aggregated.get("right_ankle_angle_mean", 0.0))
        aggregated["hip_angle_asymmetry"] = float(hip_asym)
        aggregated["knee_angle_asymmetry"] = float(knee_asym)
        aggregated["ankle_angle_asymmetry"] = float(ankle_asym)

        # Center of mass sway approximations
        aggregated["com_sway_ml"] = float(aggregated.get("com_x_std", 0.0))
        aggregated["com_sway_ap"] = float(aggregated.get("com_y_std", 0.0))
        aggregated["com_sway_total"] = float(np.hypot(aggregated["com_sway_ml"], aggregated["com_sway_ap"]))

        # Return a single-row DataFrame; column order is stable
        df = pd.DataFrame([aggregated])
        return df


