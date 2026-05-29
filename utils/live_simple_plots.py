from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils.functions import ACTION_ARROWS_CLIFF_WALKING, ACTION_ARROWS_FROZEN_LAKE, ACTION_ARROWS_TAXI

ACTION_ARROWS = ACTION_ARROWS_FROZEN_LAKE


def rolling_mean(values, window: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return np.arange(1, len(arr) + 1), arr
    kernel = np.ones(window) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    x = np.arange(window, len(arr) + 1)
    return x, smoothed


def rolling_sum(values, window: int) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    if len(arr) < window:
        return np.arange(1, len(arr) + 1), arr
    rolling = np.convolve(arr, np.ones(window), mode="valid")
    x = np.arange(window, len(arr) + 1)
    return x, rolling


from utils.live_simple_dashboard import LiveSimpleDashboard  # noqa: E402

__all__ = [
    "ACTION_ARROWS",
    "ACTION_ARROWS_CLIFF_WALKING",
    "ACTION_ARROWS_FROZEN_LAKE",
    "ACTION_ARROWS_TAXI",
    "LiveSimpleDashboard",
    "rolling_mean",
    "rolling_sum",
]
