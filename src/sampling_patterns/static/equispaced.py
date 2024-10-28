"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import numpy as np

from .fully_sampled import acs


def equispaced(
        num_cols: int,
        acceleration: int,
        acs_width: int = 0,
        offset: int = -1,
        adjust_acceleration: bool = False,
        center: int | None = None,
        rng: np.random.RandomState | None = None,
) -> np.ndarray:

    if offset == -1:
        if rng is None:
            rng = np.random.RandomState()
        offset = rng.randint(0, high=acceleration)

    if adjust_acceleration and acs_width > 0:
        # adjusted acceleration rate when considering fully-sampled center
        samples_periphery = num_cols / acceleration - acs_width
        if round(samples_periphery) < 1:
            raise ValueError('Center width is too large for the given acceleration rate.')
        acceleration = round((num_cols - acs_width) / samples_periphery)

    mask = np.zeros(num_cols, dtype=np.uint8)
    mask[offset::acceleration] = 1

    # acs
    if acs_width > 0:
        acs_mask = acs(num_cols, acs_width, center=center)
        mask = np.logical_or(mask, acs_mask).astype(np.uint8)

    return mask
