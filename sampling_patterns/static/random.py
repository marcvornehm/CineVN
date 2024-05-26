"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import numpy as np
from typing import Optional


def uniform(num_cols: int,
            acceleration: Optional[int] = None,
            prob: Optional[float] = None,
            rng: Optional[np.random.RandomState] = None,
            ) -> np.ndarray:
    assert acceleration is not None or prob is not None, \
        'Either an acceleration rate or a probability has to be given.'
    assert not (acceleration is not None and prob is not None), \
        'Acceleration rate and probability are mutually exclusive, but both were given.'

    if prob is None:
        prob = num_cols / acceleration

    if rng is None:
        rng = np.random.RandomState()

    return rng.uniform(size=num_cols) < prob
