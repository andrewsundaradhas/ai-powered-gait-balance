"""
Evaluation utilities scaffold (Step 6)
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


class ModelEvaluationMetrics:
	@staticmethod
	def simple_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
		if not y_true:
			return 0.0
		arr_true = np.asarray(y_true)
		arr_pred = np.asarray(y_pred)
		return float(np.mean(arr_true == arr_pred))

	@staticmethod
	def plot_dummy_curve(save_path: str) -> None:
		x = np.linspace(0, 1, 100)
		y = x
		plt.figure()
		plt.plot(x, y, label="dummy")
		plt.legend()
		plt.tight_layout()
		plt.savefig(save_path)
		plt.close()


