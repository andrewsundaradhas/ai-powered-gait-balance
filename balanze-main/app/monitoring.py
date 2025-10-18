"""
Monitoring and logging setup (Step 12)
"""

from __future__ import annotations

import logging
import time

from prometheus_client import Counter, Gauge, Histogram


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

analysis_counter = Counter("gait_analyses_total", "Total number of gait analyses", ["status"])
analysis_duration = Histogram(
	"gait_analysis_duration_seconds", "Time spent processing analysis", buckets=(1, 5, 10, 30, 60)
)
model_predictions = Histogram(
	"gait_prediction_confidence", "Model prediction confidence scores", buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
)
active_analyses = Gauge("gait_active_analyses", "Number of active analyses")


def monitor_analysis(func):
	def wrapper(*args, **kwargs):
		start_time = time.time()
		active_analyses.inc()
		try:
			result = func(*args, **kwargs)
			analysis_counter.labels(status="success").inc()
			return result
		except Exception as exc:  # noqa: BLE001
			analysis_counter.labels(status="error").inc()
			logger.error("Analysis error: %s", str(exc))
			raise
		finally:
			duration = time.time() - start_time
			analysis_duration.observe(duration)
			active_analyses.dec()
			logger.info("Analysis completed in %.2f seconds", duration)

	return wrapper


