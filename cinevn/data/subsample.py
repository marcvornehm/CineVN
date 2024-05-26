"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import contextlib
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

import sampling_patterns as patterns


@contextlib.contextmanager
def temp_seed(rng: np.random.RandomState, seed: Optional[float | Sequence[float]]):
    """A context manager for temporarily adjusting the random seed."""
    if seed is None:
        try:
            yield
        finally:
            pass
    else:
        state = rng.get_state()
        rng.seed(seed)
        try:
            yield
        finally:
            rng.set_state(state)


class MaskFunc:
    """
    An object for GRAPPA-style sampling masks.

    This crates a sampling mask that densely samples the center while
    subsampling outer k-space regions based on the undersampling factor.

    When called, ``MaskFunc`` uses internal functions create mask by 1)
    creating a mask for the k-space center, 2) create a mask outside of the
    k-space center, and 3) combining them into a total mask. The internals are
    handled by ``sample_mask``, which calls ``calculate_center_mask`` for (1)
    and ``calculate_acceleration_mask`` for (2). The combination is executed
    in the ``MaskFunc`` ``__call__`` function.

    If you would like to implement a new mask, simply subclass
    ``StaticMaskFunc`` or ``DynamicMaskFunc`` and overwrite the ``sample_mask``
    logic. See examples in ``RandomMaskFunc`` and ``EquispacedMaskFunc``.
    """

    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        allow_any_combination: bool = False,
        seed: Optional[int] = None,
    ):
        """
        Args:
            center_fractions: Fraction of low-frequency columns to be retained.
                If multiple values are provided, then one of these numbers is
                chosen uniformly each time. Note that in dynamic masks, this
                applies when the mask is compressed in the temporal dimension.
            accelerations: Amount of under-sampling. This should have the same
                length as center_fractions. If multiple values are provided,
                then one of these is chosen uniformly each time.
            allow_any_combination: Whether to allow cross combinations of
                elements from ``center_fractions`` and ``accelerations``.
            seed: Seed for starting the internal random number generator of the
                ``MaskFunc``.
        """
        if len(center_fractions) != len(accelerations) and not allow_any_combination:
            raise ValueError(
                'Number of center fractions should match number of accelerations '
                'if allow_any_combination is False.'
            )

        self.center_fractions = center_fractions
        self.accelerations = accelerations
        self.allow_any_combination = allow_any_combination
        self.rng = np.random.RandomState(seed)

    def __call__(
        self,
        shape: Sequence[int],
        offset: Optional[int] = None,
        padding_left: int = 0,
        padding_right: int = 0,
        seed: Optional[float | Sequence[float]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, int]:
        """
        Sample and return a k-space mask.

        Args:
            shape: Shape of k-space.
            offset: Offset from 0 to begin mask (for equispaced masks).
                If no offset is given, then one is selected randomly.
            padding_left: Amount of zero-padding on the left. This
                should not be included in ``shape`` but is considered
                when determining the center of k-space. Note that
                certain mask functions might only support certain
                combinations with ``padding_right``.
            padding_right: Amount of zero-padding on the right. This
                should not be included in ``shape`` but is considered
                when determining the center of k-space. Note that
                certain mask functions might only support certain
                combinations with ``padding_left``.
            seed: Seed for random number generator for reproducibility.

        Returns:
            A 2-tuple containing 1) the k-space mask and 2) the number
            of center frequency lines.
        """
        if len(shape) < self.min_ndim:
            raise ValueError(f'Shape should have {self.min_ndim} or more dimensions')

        with temp_seed(self.rng, seed):
            center_mask, accel_mask, num_low_frequencies = self.sample_mask(
                shape, offset, padding_left, padding_right, **kwargs
            )

        # combine masks together
        return torch.max(center_mask, accel_mask), num_low_frequencies

    @property
    def min_ndim(self):
        """The minimal number of dimensions required for a mask"""
        raise NotImplementedError

    def sample_mask(
        self,
        shape: Sequence[int],
        offset: Optional[int],
        padding_left: int,
        padding_right: int,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Sample a new k-space mask.

        This function samples and returns two components of a k-space
        mask:
        1) the center mask (e.g., for sensitivity map calculation) and
        2) the acceleration mask (for the edge of k-space).
        Both of these masks, as well as the integer of low frequency
        samples, are returned.

        Args:
            shape: Shape of the k-space to subsample.
            offset: Offset from 0 to begin mask (for equispaced masks).
            padding_left: Amount of zero-padding on the left.
            padding_right: Amount of zero-padding on the right.

        Returns:
            A 3-tuple containing 1) the mask for the center of k-space,
            2) the mask for the high frequencies of k-space, and 3) the
            integer count of low frequency samples.
        """
        num_cols = shape[-2]
        num_frames = self.get_num_frames(shape)
        center_fraction, acceleration = self.choose_acceleration()
        num_low_frequencies = round(num_cols * center_fraction)
        center_mask = self.reshape_mask(
            self.calculate_center_mask(
                shape, num_low_frequencies, padding_left=padding_left, padding_right=padding_right, **kwargs
            ),
            shape
        )
        acceleration_mask = self.reshape_mask(
            self.calculate_acceleration_mask(
                num_cols, acceleration, padding_left=padding_left, padding_right=padding_right, offset=offset,
                num_low_frequencies=num_low_frequencies, num_frames=num_frames, **kwargs
            ),
            shape,
        )

        return center_mask, acceleration_mask, num_low_frequencies

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        """Reshape mask to desired output shape."""
        raise NotImplementedError

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        **kwargs,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int, padding_left: int = 0, padding_right: int = 0, **kwargs
    ) -> np.ndarray:
        """
        Build center mask based on number of low frequencies.

        Args:
            shape: Shape of k-space to mask.
            num_low_freqs: Number of low-frequency lines to sample.
            padding_left: Amount of zero-padding on the left.
            padding_right: Amount of zero-padding on the right.

        Returns:
            A mask for the low spatial frequencies of k-space.
        """
        num_cols = shape[-2] + padding_left + padding_right
        mask = patterns.static.center(num_cols, num_low_freqs)
        return mask[..., padding_left:num_cols - padding_right]

    def choose_acceleration(self):
        """Choose acceleration based on class parameters."""
        if self.allow_any_combination:
            return self.rng.choice(self.center_fractions), self.rng.choice(
                self.accelerations
            )
        else:
            choice = self.rng.randint(len(self.center_fractions))
            return self.center_fractions[choice], self.accelerations[choice]

    def get_num_frames(self, shape: Sequence[int]) -> int:
        """Determine the number of temporal frames from a given shape."""
        raise NotImplementedError


class StaticMaskFunc(MaskFunc):
    @property
    def min_ndim(self):
        """Static masks require three dimensions (kx, ky, real/complex)"""
        return 3

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        num_cols = shape[-2]
        mask_shape = [1 for _ in shape]
        mask_shape[-2] = num_cols

        return torch.from_numpy(mask.reshape(*mask_shape).astype(np.float32))

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError

    def get_num_frames(self, shape: Sequence[int]) -> int:
        """In a static mask, the number of temporal frames is always 1"""
        return 1


class DynamicMaskFunc(MaskFunc):
    @property
    def min_ndim(self):
        """Dynamic masks require four dimensions (t, kx, ky, real/complex)"""
        return 4

    def reshape_mask(self, mask: np.ndarray, shape: Sequence[int]) -> torch.Tensor:
        # if the mask is one-dimensional, simply propagate it along the
        # temporal dimension (e.g. for the center mask)
        if mask.ndim == 1:
            num_frames = shape[-4]
            num_cols = shape[-2]
            mask_shape = [1 for _ in shape]
            mask_shape[-2] = num_cols
            mask = mask.reshape(*mask_shape).astype(np.float32)
            mask = mask.repeat(num_frames, axis=-4)
        else:
            assert mask.shape[0] == shape[-4], \
                'Temporal size of mask and target shape do not match'
            num_frames = shape[-4]
            num_cols = shape[-2]
            mask_shape = [1 for _ in shape]
            mask_shape[-4] = num_frames
            mask_shape[-2] = num_cols
            mask = mask.reshape(*mask_shape).astype(np.float32)

        return torch.from_numpy(mask)

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        num_frames: int = 1,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError

    def get_num_frames(self, shape: Sequence[int]) -> int:
        """In a dynamic mask, the number of temporal frames is found in
        dimension -4 of the shape"""
        return shape[-4]


class RandomMaskFunc(StaticMaskFunc):
    """
    Creates a random sub-sampling mask of a given shape.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding to low-frequencies.
        2. The other columns are selected uniformly at random with a
        probability equal to: prob = (N / acceleration - N_low_freqs) /
        (N - N_low_freqs). This ensures that the expected number of columns
        selected is equal to (N / acceleration).

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the ``RandomMaskFunc`` object is called.

    For example, if accelerations = [4, 8] and center_fractions = [0.08, 0.04],
    then there is a 50% probability that 4-fold acceleration with 8% center
    fraction is selected and a 50% probability that 8-fold acceleration with 4%
    center fraction is selected.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        padding_left: int = 0,
        padding_right: int = 0,
        num_low_frequencies: int = 0,
        **kwargs,
    ) -> np.ndarray:
        num_cols_periphery = num_cols - num_low_frequencies
        prob_periphery = (num_cols / acceleration - num_low_frequencies) / num_cols_periphery
        center_idx = (num_cols + padding_left + padding_right) // 2 - padding_left
        mask_periphery = patterns.static.uniform(num_cols_periphery, prob=prob_periphery, rng=self.rng)
        mask = np.concatenate([
            mask_periphery[:center_idx - num_low_frequencies // 2],
            np.zeros(num_low_frequencies),
            mask_periphery[center_idx - num_low_frequencies // 2:],
        ], axis=0)

        return mask


class DynamicRandomMaskFunc(DynamicMaskFunc):
    """
    Creates a random dynamic sub-sampling mask of a given shape.

    This mask functions similarly to RandomMaskFunc, but adds a temporal
    dimension to the mask. Individual frames of the mask are independent of
    each other and the number of low frequencies applies to each frame.
    """
    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        padding_left: int = 0,
        padding_right: int = 0,
        num_low_frequencies: int = 0,
        num_frames: int = 1,
        **kwargs,
    ) -> np.ndarray:
        num_cols_periphery = num_cols - num_low_frequencies
        prob_periphery = (num_cols / acceleration - num_low_frequencies) / num_cols_periphery
        center_idx = (num_cols + padding_left + padding_right) // 2 - padding_left
        mask_periphery = patterns.dynamic.uniform(num_frames, num_cols_periphery, prob=prob_periphery, rng=self.rng)
        mask = np.concatenate([
            mask_periphery[:, :center_idx - num_low_frequencies // 2],
            np.zeros((num_frames, num_low_frequencies)),
            mask_periphery[:, center_idx - num_low_frequencies // 2:],
        ], axis=1)

        return mask


class EquispacedMaskFunc(StaticMaskFunc):
    """
    Sample data with equally-spaced k-space lines.

    The lines are spaced exactly evenly, as is done in standard GRAPPA-style
    acquisitions. This means that with a densely-sampled center,
    ``acceleration`` will be greater than the true acceleration rate.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        return patterns.static.equispaced(num_cols, acceleration, offset, self.rng)


class EquispacedFractionMaskFunc(StaticMaskFunc):
    """
    Equispaced mask with exact acceleration matching.

    The mask selects a subset of columns from the input k-space data. If the
    k-space data has N columns, the mask picks out:
        1. N_low_freqs = (N * center_fraction) columns in the center
           corresponding tovlow-frequencies.
        2. The other columns are selected with equal spacing at a proportion
           that reaches the desired acceleration rate taking into consideration
           the number of low frequencies. This ensures that the expected number
           of columns selected is equal to (N / acceleration)

    It is possible to use multiple center_fractions and accelerations, in which
    case one possible (center_fraction, acceleration) is chosen uniformly at
    random each time the EquispacedMaskFunc object is called.

    Note that this function may not give equispaced samples (documented in
    https://github.com/facebookresearch/fastMRI/issues/54), which will require
    modifications to standard GRAPPA approaches. Nonetheless, this aspect of
    the function has been preserved to match the public multicoil data.
    """

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        offset: Optional[int] = None,
        num_low_frequencies: int = 0,
        **kwargs,
    ) -> np.ndarray:
        """
        Produce mask for non-central acceleration lines.

        Args:
            num_cols: Number of columns of k-space (2D subsampling).
            acceleration: Desired acceleration rate.
            offset: Offset from 0 to begin masking. If no offset is specified,
                then one is selected randomly.
            num_low_frequencies: Number of low frequencies. Used to adjust mask
                to exactly match the target acceleration.

        Returns:
            A mask for the high spatial frequencies of k-space.
        """
        return patterns.static.equispaced_fraction(num_cols, acceleration, offset, num_low_frequencies, self.rng)


class DynamicEquispacedFractionMaskFunc(DynamicMaskFunc):
    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        padding_left: int = 0,
        padding_right: int = 0,
        num_low_frequencies: int = 0,
        num_frames: int = 1,
        **kwargs,
    ) -> np.ndarray:
        frames = []
        offset = 0
        for _ in range(num_frames):
            mask_f = patterns.static.equispaced_fraction(num_cols, acceleration, offset=offset,
                                                         center_width=num_low_frequencies, rng=self.rng)
            offset = acceleration - (num_cols - mask_f.nonzero()[0][-1])
            frames.append(mask_f)
        mask = np.stack(frames, axis=0)

        center_idx = (num_cols + padding_left + padding_right) // 2 - padding_left
        mask = np.concatenate([
            mask[:, :center_idx - num_low_frequencies // 2],
            np.zeros((num_frames, num_low_frequencies)),
            mask[:, center_idx - num_low_frequencies // 2:],
        ], axis=1)

        return mask


class DynamicMaskFuncWithoutACS(DynamicMaskFunc):
    def __init__(
        self,
        center_fractions: Sequence[float],
        accelerations: Sequence[int],
        seed: Optional[int] = None,
    ):
        """
        This constructor sets allow_any_combination to True because center_fractions is not relevant in these cases.
        """
        super().__init__(center_fractions, accelerations, allow_any_combination=True, seed=seed)

    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError

    def calculate_center_mask(
        self, shape: Sequence[int], num_low_freqs: int, **kwargs
    ) -> np.ndarray:
        """
        The center mask is all zeros in Masks without auto-calibration
        signal.
        """
        num_cols = shape[-2]
        return patterns.static.center(num_cols, 0)


class VISTAMaskFunc(DynamicMaskFuncWithoutACS):
    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        padding_left: int = 0,
        padding_right: int = 0,
        num_frames: int = 1,
        **kwargs,
    ) -> np.ndarray:
        center_idx_padded = (num_cols + padding_left + padding_right) // 2
        center_idx_unpadded = num_cols // 2
        if center_idx_padded - padding_left != center_idx_unpadded:
            raise NotImplementedError('VISTA does not support off-center masking caused by uneven padding')

        num_samples_per_frame = round(num_cols / acceleration)
        param = patterns.dynamic.VISTAParam(PE=num_cols, FR=num_frames, n=num_samples_per_frame, fc=16/num_cols)
        samp = patterns.dynamic.VISTA(param, rng=self.rng)

        return samp.T


class GROMaskFunc(DynamicMaskFuncWithoutACS):
    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        padding_left: int = 0,
        padding_right: int = 0,
        num_frames: int = 1,
        augment: bool = False,
        **kwargs,
    ) -> np.ndarray:
        center_idx_padded = (num_cols + padding_left + padding_right) // 2
        center_idx_unpadded = num_cols // 2
        if center_idx_padded - padding_left != center_idx_unpadded:
            raise NotImplementedError('GRO does not support off-center masking caused by uneven padding')

        num_samples_per_frame = round(num_cols / acceleration)
        param = patterns.dynamic.GROParam(PE=num_cols, FR=num_frames, n=num_samples_per_frame, E=1)
        offset = self.rng.random() if augment else 0
        samp = patterns.dynamic.GRO(param, offset=offset)
        return samp.T[0]


class CAVAMaskFunc(DynamicMaskFuncWithoutACS):
    def calculate_acceleration_mask(
        self,
        num_cols: int,
        acceleration: int,
        padding_left: int = 0,
        padding_right: int = 0,
        num_frames: int = 1,
        **kwargs,
    ) -> np.ndarray:
        center_idx_padded = (num_cols + padding_left + padding_right) // 2
        center_idx_unpadded = num_cols // 2
        if center_idx_padded - padding_left != center_idx_unpadded:
            raise NotImplementedError('CAVA does not support off-center masking caused by uneven padding')

        num_samples_per_frame = round(num_cols / acceleration)
        param = patterns.dynamic.CAVAParam(PE=num_cols, FR=num_frames, n=num_samples_per_frame, E=1)
        samp = patterns.dynamic.CAVA(param)
        return samp.T[0]


def get_mask_type(mask_type_str: str, mode: str):
    mask_type_str = mask_type_str.lower()
    if mode == 'dynamic':
        if mask_type_str == 'random':
            return DynamicRandomMaskFunc
        elif mask_type_str == 'vista':
            return VISTAMaskFunc
        elif mask_type_str == 'gro':
            return GROMaskFunc
        elif mask_type_str == 'cava':
            return CAVAMaskFunc
        elif mask_type_str == 'equispaced_fraction':
            return DynamicEquispacedFractionMaskFunc
        else:
            raise ValueError(f'{mask_type_str} not supported in dynamic mode')
    elif mode == 'static':
        if mask_type_str == 'random':
            return RandomMaskFunc
        elif mask_type_str == 'equispaced':
            return EquispacedMaskFunc
        elif mask_type_str == 'equispaced_fraction':
            return EquispacedFractionMaskFunc
        else:
            raise ValueError(f'{mask_type_str} not supported in static mode')
    else:
        raise ValueError(f'{mode} not supported')
