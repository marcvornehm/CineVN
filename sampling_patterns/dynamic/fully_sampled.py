"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import numpy as np


def center(num_frames: int, num_cols: int, center_width: int = 0) -> np.ndarray:
    """
    Build center mask.

    Args:
        num_frames: The number of temporal frames.
        num_cols: The total number of phase encoding steps.
        center_width: The number of phase encoding steps in the center
            to fully sample.

    Returns:
        A mask for the low spatial frequencies of k-space.
    """
    mask = np.zeros((num_frames, num_cols), dtype=np.float32)
    pad = (num_cols - center_width + 1) // 2
    mask[:, pad:pad + center_width] = 1
    assert mask.sum() == center_width * num_frames

    return mask


def full(num_frames: int, num_cols: int) -> np.ndarray:
    """
    Build a fully sampled mask.

    Args:
        num_frames: The number of temporal frames.
        num_cols: The total number of phase encoding steps.

    Returns:
        A fully sampled k-space mask.
    """
    return np.ones((num_frames, num_cols), dtype=np.float32)
