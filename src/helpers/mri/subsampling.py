#  Copyright (c) 2023. Marc Vornehm <marc.vornehm@fau.de>.

import numpy as np

import sampling_patterns as patterns


class Sampling:
    def __init__(self, name: str, acceleration: float | None = None, mask_type: str | None = None):
        self.name = name
        self.acceleration = acceleration
        self.mask_type = mask_type

    def __repr__(self):
        return f'Sampling("{self.name}", acceleration={self.acceleration}, mask_type={self.mask_type}'


def get_mask(
        num_frames: int,
        num_cols: int,
        mask_type: str,
        acceleration: float,
        padding_left: int = 0,
        padding_right: int = 0,
        rng: np.random.RandomState | None = None,
):
    mask_type = mask_type.lower()

    center_idx_padded = (num_cols + padding_left + padding_right) // 2
    center_idx_unpadded = num_cols // 2
    off_center = (center_idx_padded - padding_left != center_idx_unpadded)

    if mask_type == 'random':
        mask = patterns.dynamic.uniform(num_frames, num_cols, acceleration=acceleration, rng=rng)

    elif mask_type == 'random_acs':
        center = center_idx_padded - padding_left
        mask = patterns.dynamic.uniform(
            num_frames, num_cols, acceleration=acceleration, acs_width=16, adjust_acceleration=True, center=center,
            rng=rng,
        )

    elif mask_type == 'vista':
        if off_center:
            raise NotImplementedError('VISTA does not support off-center masking caused by uneven padding')
        num_samples_per_frame = round(num_cols / acceleration)
        param = patterns.dynamic.VISTAParam(PE=num_cols, FR=num_frames, n=num_samples_per_frame, fc=16/num_cols)
        samp = patterns.dynamic.VISTA(param, rng=rng)
        mask = samp.T

    elif mask_type == 'gro':
        if off_center:
            raise NotImplementedError('GRO does not support off-center masking caused by uneven padding')
        num_samples_per_frame = round(num_cols / acceleration)
        param = patterns.dynamic.GROParam(PE=num_cols, FR=num_frames, n=num_samples_per_frame, E=1)
        samp = patterns.dynamic.GRO(param, offset=0)
        mask = samp.T[0]  # type: ignore

    elif mask_type == 'cava':
        if off_center:
            raise NotImplementedError('CAVA does not support off-center masking caused by uneven padding')
        num_samples_per_frame = round(num_cols / acceleration)
        param = patterns.dynamic.CAVAParam(PE=num_cols, FR=num_frames, n=num_samples_per_frame, E=1)
        samp = patterns.dynamic.CAVA(param)
        mask = samp.T[0]

    elif mask_type == 'equispaced':
        mask = patterns.dynamic.equispaced(
            num_frames, num_cols, round(acceleration), acs_width=0, roll_offset=True, rng=rng,
        )

    elif mask_type == 'equispaced_acs':
        center = center_idx_padded - padding_left
        mask = patterns.dynamic.equispaced(
            num_frames, num_cols, round(acceleration), acs_width=16, roll_offset=True, adjust_acceleration=True,
            center=center, rng=rng,
        )

    else:
        raise RuntimeError(f'Unknown sampling pattern {mask_type}')

    return mask.astype(np.uint8)
