"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import numpy as np

from .. import static


def equispaced(
        num_frames: int,
        num_cols: int,
        acceleration: int,
        acs_width: int = 0,
        offset: int = -1,
        roll_offset: bool = False,
        adjust_acceleration: bool = False,
        center: int | None = None,
        rng: np.random.RandomState | None = None,
) -> np.ndarray:

    # do the adjustment here instead of in the static function so that the offset is generated correctly
    if adjust_acceleration and acs_width > 0:
        # adjusted acceleration rate when considering fully-sampled center
        samples_periphery = num_cols / acceleration - acs_width
        if round(samples_periphery) < 1:
            raise ValueError('Center width is too large for the given acceleration rate.')
        acceleration = round((num_cols - acs_width) / samples_periphery)

    if offset == -1:
        if rng is None:
            rng = np.random.RandomState()
        offset = rng.randint(0, high=acceleration)

    mask = np.zeros((num_frames, num_cols), dtype=np.uint8)
    for f in range(num_frames):
        mask[f] = static.equispaced(
            num_cols, acceleration, acs_width=acs_width, offset=offset, adjust_acceleration=False, center=center,
            rng=rng,
        )
        if roll_offset:
            offset = (offset + 1) % acceleration

    return mask
