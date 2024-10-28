"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import numpy as np

from .. import static


def acs(num_frames: int, num_cols: int, acs_width: int, center: int | None = None) -> np.ndarray:
    mask = static.acs(num_cols, acs_width, center=center)
    return np.tile(mask, (num_frames, 1))


def full(num_frames: int, num_cols: int) -> np.ndarray:
    return np.ones((num_frames, num_cols), dtype=np.uint8)
