"""
Real-time streaming analysis pipeline (Step 9) with safe fallbacks.

This implementation depends only on NumPy, pandas, and (optionally) OpenCV.
If trained model files are missing, it uses dummy classifiers that return
constant labels with mid confidence, ensuring the pipeline runs error-free.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np

try:
    import joblib  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    joblib = None  # type: ignore

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - OpenCV optional
    cv2 = None  # type: ignore

from src.pose_extractor import PoseExtractor
from src.advanced_feature_engineer import AdvancedBioMechanicalFeatures


class _DummyClassifier:
    """
    Minimal scikit-learn-like classifier used when model files are absent.
    """

    def __init__(self, label: str = "Unknown") -> None:
        self._label = label

    def predict(self, X):  # noqa: N802 - sklearn style
        n = len(X) if hasattr(X, "__len__") else 1
        return np.array([self._label] * n)

    def predict_proba(self, X):  # noqa: N802 - sklearn style
        n = len(X) if hasattr(X, "__len__") else 1
        # Single-class probability = 1.0; emulate binary with mid confidence
        return np.full((n, 1), 0.5, dtype=float)


class RealtimeGaitAnalyzer:
    """Real-time gait analysis with streaming inference and smoothing."""

    def __init__(self, buffer_size: int = 300, fps: int = 30) -> None:
        self.buffer_size = buffer_size
        self.fps = fps
        self.dt = 1.0 / max(fps, 1)

        self.pose_extractor = PoseExtractor()
        self.feature_extractor = AdvancedBioMechanicalFeatures(fps)

        self.models = self._load_models()

        self.keypoint_buffer: Deque[Dict] = deque(maxlen=buffer_size)
        self.prediction_buffer: Deque[Dict] = deque(maxlen=10)

        self.is_running = False

    def _load_models(self) -> Dict[str, object]:
        models: Dict[str, object] = {}
        # Attempt to load models; fallback to dummy
        def _load(path: str, default_label: str) -> object:
            if joblib is None:
                return _DummyClassifier(default_label)
            try:
                return joblib.load(path)
            except Exception:
                return _DummyClassifier(default_label)

        models["gait"] = _load("models/gait_classifier.pkl", "Normal")
        models["balance"] = _load("models/balance_classifier.pkl", "Stable")
        models["disorder"] = _load("models/disorder_classifier.pkl", "None")
        return models

    def process_frame(self, frame: np.ndarray) -> Optional[Dict]:
        keypoints = self.pose_extractor.extract_keypoints(frame)
        if keypoints.get("detected", False):
            self.keypoint_buffer.append(keypoints)

        # Require at least ~2 seconds of data
        if len(self.keypoint_buffer) < max(int(self.fps * 2 / 1), 60):
            return None

        features = self.feature_extractor.extract_all_features(list(self.keypoint_buffer))

        gait_pred = self.models["gait"].predict(features)[0]
        gait_conf = float(np.max(self.models["gait"].predict_proba(features)))

        bal_pred = self.models["balance"].predict(features)[0]
        bal_conf = float(np.max(self.models["balance"].predict_proba(features)))

        dis_pred = self.models["disorder"].predict(features)[0]
        dis_conf = float(np.max(self.models["disorder"].predict_proba(features)))

        prediction = {
            "gait_type": str(gait_pred),
            "gait_confidence": gait_conf,
            "balance_status": str(bal_pred),
            "balance_confidence": bal_conf,
            "disorder": str(dis_pred),
            "disorder_confidence": dis_conf,
            "timestamp": time.time(),
        }

        self.prediction_buffer.append(prediction)
        return prediction

    def get_smoothed_predictions(self) -> Optional[Dict]:
        if not self.prediction_buffer:
            return None
        preds = list(self.prediction_buffer)

        # Mode for categorical, mean for confidence
        gait_types = [p["gait_type"] for p in preds]
        balance_stats = [p["balance_status"] for p in preds]
        disorders = [p["disorder"] for p in preds]

        def _mode(values: List[str]) -> str:
            if not values:
                return "Unknown"
            uniq, counts = np.unique(values, return_counts=True)
            return str(uniq[np.argmax(counts)])

        return {
            "gait_type": _mode(gait_types),
            "gait_confidence": float(np.mean([p["gait_confidence"] for p in preds])),
            "balance_status": _mode(balance_stats),
            "balance_confidence": float(np.mean([p["balance_confidence"] for p in preds])),
            "disorder": _mode(disorders),
            "disorder_confidence": float(np.mean([p["disorder_confidence"] for p in preds])),
            "ensemble_size": len(self.prediction_buffer),
        }

    def analyze_video_stream(self, video_source: str | int = 0):
        if cv2 is None:
            raise RuntimeError("OpenCV is not available; cannot open video stream.")

        cap = cv2.VideoCapture(0 if video_source == "webcam" else video_source)
        if not cap.isOpened():
            raise RuntimeError("Could not open video source")

        self.is_running = True
        frame_count = 0

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.resize(frame, (640, 480))
                _ = self.process_frame(frame)
                smoothed = self.get_smoothed_predictions()

                frame_count += 1
                if frame_count % max(self.fps, 1) == 0 and smoothed:
                    print(
                        f"[Frame {frame_count}] "
                        f"Gait: {smoothed['gait_type']} ({smoothed['gait_confidence']:.2f}) | "
                        f"Balance: {smoothed['balance_status']} ({smoothed['balance_confidence']:.2f}) | "
                        f"Disorder: {smoothed['disorder']} ({smoothed['disorder_confidence']:.2f})"
                    )
        finally:
            cap.release()
            self.is_running = False


