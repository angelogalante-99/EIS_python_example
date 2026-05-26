import numpy as np


def safe_log(value: float) -> float:
    return np.log(max(value, 1e-10))


def normalize_score(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low + 1e-10), 0.0, 1.0))
