#  Copyright (c) 2023. Marc Vornehm <marc.vornehm@fau.de>.

from typing import Optional

import numpy as np

import sampling_patterns as patterns


class Sampling:
    def __init__(self, name: str, acceleration: Optional[float] = None, center_fraction: Optional[float] = None,
                 mask_type: Optional[str] = None):
        self.name = name
        self.acceleration = acceleration
        self.center_fraction = center_fraction
        self.mask_type = mask_type

    def __repr__(self):
        return f'Sampling("{self.name}", acceleration={self.acceleration}, center_fraction={self.center_fraction}, ' \
               f'mask_type={self.mask_type}'


def get_mask(num_frames: int, num_cols: int, mask_type: str, acceleration: float, center_fraction: float = 0,
             padding_left: int = 0, padding_right: int = 0):
    mask_type = mask_type.lower()

    center_idx_padded = (num_cols + padding_left + padding_right) // 2
    center_idx_unpadded = num_cols // 2
    off_center = (center_idx_padded - padding_left != center_idx_unpadded)

    if mask_type == 'random':
        # center mask
        center_width = round(num_cols * center_fraction)
        num_cols_padded = num_cols + padding_left + padding_right
        center = patterns.dynamic.center(num_frames, num_cols_padded, center_width)
        center = center[..., padding_left:num_cols_padded - padding_right]

        # periphery mask
        accel = patterns.dynamic.uniform(num_frames, num_cols, acceleration=round(acceleration))

        # total mask
        mask = np.maximum(center, accel)

        # remove frame dimension in static case
        if num_frames == 1:
            mask = mask[0]

    elif mask_type == 'vista':
        if off_center:
            raise NotImplementedError('VISTA does not support off-center masking caused by uneven padding')
        num_samples_per_frame = round(num_cols / acceleration)
        param = patterns.dynamic.VISTAParam(PE=num_cols, FR=num_frames, n=num_samples_per_frame, fc=16/num_cols)
        samp = patterns.dynamic.VISTA(param)
        mask = samp.T

    elif mask_type == 'gro':
        if off_center:
            raise NotImplementedError('GRO does not support off-center masking caused by uneven padding')
        num_samples_per_frame = round(num_cols / acceleration)
        param = patterns.dynamic.GROParam(PE=num_cols, FR=num_frames, n=num_samples_per_frame, E=1)
        samp = patterns.dynamic.GRO(param, offset=0)
        mask = samp.T[0]

    elif mask_type == 'cava':
        if off_center:
            raise NotImplementedError('CAVA does not support off-center masking caused by uneven padding')
        num_samples_per_frame = round(num_cols / acceleration)
        param = patterns.dynamic.CAVAParam(PE=num_cols, FR=num_frames, n=num_samples_per_frame, E=1)
        samp = patterns.dynamic.CAVA(param)
        mask = samp.T[0]

    elif mask_type == 'equispaced_fraction':
        # center mask
        center_width = round(num_cols * center_fraction)
        num_cols_padded = num_cols + padding_left + padding_right
        center = patterns.static.center(num_cols_padded, center_width)
        center = center[..., padding_left:num_cols_padded - padding_right]

        # periphery mask
        accel = patterns.static.equispaced_fraction(num_cols, round(acceleration), None, center_width)

        # total mask
        mask = np.maximum(center, accel)
        mask = np.stack([mask] * num_frames)

    else:
        raise RuntimeError(f'Unknown sampling pattern {mask_type}')

    return mask