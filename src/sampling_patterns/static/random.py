"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import numpy as np

from .fully_sampled import acs


def uniform(
        num_cols: int,
        acceleration: float | None = None,
        num_samples: int | None = None,
        acs_width: int = 0,
        adjust_acceleration: bool = False,
        center: int | None = None,
        rng: np.random.RandomState | None = None,
) -> np.ndarray:

    if acceleration is not None and num_samples is not None:
        raise ValueError('Acceleration rate and number of samples are mutually exclusive, but both were given.')
    if num_samples is None:
        if acceleration is None:
            raise ValueError('Either an acceleration rate or a number of samples has to be given.')
        num_samples = round(num_cols / acceleration)

    if rng is None:
        rng = np.random.RandomState()

    if adjust_acceleration and acs_width > 0:
        mask = acs(num_cols, acs_width, center=center)
        num_samples -= np.sum(mask)
        p = (1 - mask) / (num_cols - acs_width)
    else:
        mask = np.zeros(num_cols, dtype=np.uint8)
        p = np.ones(num_cols) / num_cols
    idx = rng.choice(num_cols, size=num_samples, replace=False, p=p)
    mask[idx] = 1

    return mask
