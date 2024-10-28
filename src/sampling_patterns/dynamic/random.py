"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import numpy as np

from .. import static


def uniform(
        num_frames: int,
        num_cols: int,
        acceleration: float | None = None,
        num_samples: int | None = None,
        acs_width: int = 0,
        adjust_acceleration: bool = False,
        center: int | None = None,
        rng: np.random.RandomState | None = None,
) -> np.ndarray:

    if rng is None:
        rng = np.random.RandomState()

    mask = np.zeros((num_frames, num_cols), dtype=np.uint8)
    for f in range(num_frames):
        mask[f] = static.uniform(
            num_cols, acceleration=acceleration, num_samples=num_samples, acs_width=acs_width,
            adjust_acceleration=adjust_acceleration, center=center, rng=rng,
        )
    return mask
