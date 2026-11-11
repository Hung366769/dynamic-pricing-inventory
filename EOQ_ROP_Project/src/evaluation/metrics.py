"""Utilities for forecast and policy metrics."""
from __future__ import annotations
import numpy as np


def regression_metrics(actual, predicted) -> dict:
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    err = actual - predicted
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "wape_percent": float(np.sum(np.abs(err)) / max(np.sum(np.abs(actual)), 1e-12) * 100),
        "smape_percent": float(np.mean(2 * np.abs(err) / np.maximum(np.abs(actual) + np.abs(predicted), 1e-12)) * 100),
    }
