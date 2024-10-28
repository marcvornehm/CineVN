"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import numpy as np


def acs(num_cols: int, acs_width: int, center: int | None = None) -> np.ndarray:
    mask = np.zeros(num_cols, dtype=np.uint8)
    if center is None:
        center = num_cols // 2
    acs_low = center - acs_width // 2
    acs_high = center + (acs_width + 1) // 2
    mask[acs_low:acs_high] = 1
    return mask


def full(num_cols: int) -> np.ndarray:
    return np.ones(num_cols, dtype=np.uint8)
