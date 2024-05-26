"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

from typing import Optional

import numpy as np


def equispaced(num_cols: int,
               acceleration: int,
               offset: Optional[int],
                        rng: Optional[np.random.RandomState] = None,
               ) -> np.ndarray:
    if offset is None:
        offset = rng.randint(0, high=round(acceleration))

    mask = np.zeros(num_cols, dtype=np.float32)
    mask[offset::acceleration] = 1

    return mask


def equispaced_fraction(num_cols: int,
                        acceleration: int,
                        offset: Optional[int],
                        center_width: int,
                        rng: Optional[np.random.RandomState] = None,
                        ) -> np.ndarray:
    # determine acceleration rate by adjusting for a fully sampled center
    adjusted_accel = (acceleration * (center_width - num_cols)) / (
            center_width * acceleration - num_cols
    )
    if rng is None:
        rng = np.random.RandomState()

    if offset is None:
        offset = rng.randint(0, high=round(adjusted_accel))

    mask = np.zeros(num_cols)
    accel_samples = np.arange(offset, num_cols, adjusted_accel)
    accel_samples = np.around(accel_samples).astype(np.uint)
    mask[accel_samples] = 1.0

    return mask.astype(np.float32)
