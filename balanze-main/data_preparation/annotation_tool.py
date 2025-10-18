"""
Annotation tool scaffold (Step 2.2)
Provides a non-interactive fallback mode to avoid blocking in headless runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


class GaitAnnotationTool:
	GAIT_CATEGORIES = [
		"Normal",
		"Antalgic (Pain-based)",
		"Trendelenburg (Hip weakness)",
		"Parkinsonian (Reduced arm swing)",
		"Cerebellar (Wide-based)",
		"Hemiplegic (Post-stroke)",
		"Spastic (Scissoring)",
		"High-stepping (Foot drop)",
		"Waddling (Proximal weakness)",
		"Steppage (Dorsiflexion weakness)",
	]

	BALANCE_CATEGORIES = [
		"Stable",
		"Mildly Impaired",
		"Moderately Impaired",
		"Severely Impaired",
	]

	DISORDER_SUGGESTIONS = [
		"None",
		"Parkinsons Disease",
		"Cerebellar Ataxia",
		"Peripheral Neuropathy",
		"Spastic Paraparesis",
		"Stroke/Hemiplegia",
		"Proximal Myopathy",
		"Vestibular Dysfunction",
	]

	def __init__(self, video_dir: str, output_dir: str) -> None:
		self.video_dir = Path(video_dir)
		self.output_dir = Path(output_dir)
		self.output_dir.mkdir(parents=True, exist_ok=True)

	def annotate_video(self, video_filename: str, non_interactive: bool = True) -> Dict:
		video_path = self.video_dir / video_filename
		annotations = {
			"video_file": video_filename,
			"gait_type": self.GAIT_CATEGORIES[0],
			"balance_status": self.BALANCE_CATEGORIES[0],
			"suspected_disorder": self.DISORDER_SUGGESTIONS[0],
			"severity": 0.5,
			"notes": "",
			"affected_side": "bilateral",
			"confidence": 0.8,
		}

		if not non_interactive and cv2 is not None:
			cap = cv2.VideoCapture(str(video_path))
			if cap.isOpened():
				cap.release()

		out = self.output_dir / f"{Path(video_filename).stem}_annotations.json"
		with out.open("w", encoding="utf-8") as f:
			json.dump(annotations, f, indent=2)
		return annotations


