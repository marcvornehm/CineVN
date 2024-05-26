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
        return _round_away_from_zero_numpy(x)
    else:
        # fall back to python version
        try:
            return _round_away_from_zero_python(x)
        except TypeError:
            warnings.warn('Could not round away from zero. Using inbuilt banker\'s rounding instead.',
                          category=RuntimeWarning)
            return round(x)


def _round_away_from_zero_numpy(x):
    mask = x >= 0
    y = np.zeros_like(x)
    y[mask] = np.floor(x[mask] + 0.5)
    y[~mask] = np.ceil(x[~mask] - 0.5)
    return y


def _round_away_from_zero_python(x):
    if x >= 0.0:
        return floor(x + 0.5)
    else:
        return ceil(x - 0.5)
