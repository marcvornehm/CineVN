"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import warnings
from math import ceil, floor

import numpy as np


def round_away_from_zero(x):
    """
    Python 3.x uses "round half to even", also called "Banker's rounding",
    while MatLab uses "round away from zero".
    Fallback for unknown/incompatible types is Python's inbuilt round
    function.
    """
    if isinstance(x, np.ndarray):
        mask = x >= 0
        y = np.zeros_like(x)
        y[mask] = np.floor(x[mask] + 0.5)
        y[~mask] = np.ceil(x[~mask] - 0.5)
        return y
    else:
        # fall back to python version
        try:
            if x >= 0.0:
                return floor(x + 0.5)
            else:
                return ceil(x - 0.5)
        except TypeError:
            warnings.warn('Could not round away from zero. Using inbuilt banker\'s rounding instead.',
                          category=RuntimeWarning)
            return round(x)
