#  Copyright (c) 2023. Marc Vornehm <marc.vornehm@fau.de>.

from typing import TypeAlias, Union

import numpy as np
try:
    import cupy as cp
    xp = cp
except ImportError:
    # importing cupy fails if no cuda is available
    cp = None
    xp = np


ndarray: TypeAlias = Union[np.ndarray, xp.ndarray]  # type: ignore


def get_ndarray_module(array: ndarray):
    if isinstance(array, np.ndarray):
        return np
    elif cp is not None and isinstance(array, cp.ndarray):
        return cp
    else:
        raise TypeError(f'Array has invalid type {type(array)}')
