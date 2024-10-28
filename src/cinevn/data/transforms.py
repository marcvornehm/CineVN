"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.

Part of this source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import warnings
from typing import NamedTuple, Sequence

with warnings.catch_warnings():  # ismrmrd adds a warning filter that causes all subsequent warnings to be printed even if they should be filtered
    import ismrmrd
import numpy as np
import torch

from ..complex_math import complex_abs
from ..mri import rotate_phase
from .subsample import MaskFunc


def to_tensor(data: np.ndarray, stack_complex_dim: bool = True) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.

    For complex arrays, the real and imaginary parts are stacked along the last
    dimension if `stack_complex_dim` is True.

    Args:
        data: Input numpy array.
        stack_complex_dim: If True, the real and imaginary parts are stacked

    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data) and stack_complex_dim:
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)


def apply_mask(
        data: torch.Tensor,
        mask_func: MaskFunc,
        seed: float | Sequence[float] | None = None,
        padding: tuple[int, int] | None = None,
        flip_mask: bool = False,
        augment_mask: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least four dimensions,
            where dimension -4 is the temporal dimension, dimensions -3 and -2
            are the spatial dimensions, and dimension -1 has size two and
            represents real and imaginary components.
        mask_func: A function that takes a shape (tuple of ints) and a random
            number seed and returns a mask.
        seed: Seed for the random number generator.
        padding: Borders of actually acquired k-space in phase direction (phase
            resolution). k-space lines in front of padding[0] and starting at
            padding[1] are zero-padding. If the phase resolution is 100%, i.e.,
            no padding was applied, padding is (0, data.shape[-2]+1).
            Note that the mask is computed for the smaller k-space without
            padding and then embedded in a tensor corresponding to the k-space
            size with padding. The mask, however, uses padding with ones for
            data consistency.
        flip_mask: If True, the mask is flipped along phase encoding dimension.
        augment_mask: If True, masks are augmented, depending on the
            implementation of mask_func.

    Returns:
        tuple containing:
            masked data: Subsampled k-space data.
            mask: The generated mask.
    """
    if padding is None:
        padding = (0, data.shape[-2])

    # ignore negative padding (phase resolution > 100%)
    padding = (max(0, padding[0]), min(data.shape[-2], padding[1]))

    # generate mask without padding
    num_frames = data.shape[1]
    num_cols = padding[1] - padding[0]
    mask_unpadded = torch.from_numpy(mask_func(  # [frames, cols]
        num_frames, num_cols, padding_left=padding[0], padding_right=data.shape[-2]-padding[1], offset=-1,
        augment=augment_mask, seed=seed,
    ))

    # zero-pad
    shape = (1, num_frames, 1, data.shape[-2], 1)
    mask = torch.ones(size=shape, dtype=mask_unpadded.dtype)
    mask[0, :, 0, padding[0]:padding[1], 0] = mask_unpadded

    # flip the mask along the phase direction. it's important that this is done after zero-padding so the k-space center
    # is correctly flipped
    if flip_mask:
        mask = torch.flip(mask, dims=(-2,))

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask


def shift_phase(k_img: torch.Tensor, dx: float, dy: float) -> torch.Tensor:
    """
    Shift the phase of the fourier transform of an image

    Args:
        k_img: The fourier transform image.
        dx: The phase to shift in x.
        dy: The phase to shift in y.

    Returns:
        Version of k_img with shifted phase.
    """
    k_img = torch.view_as_complex(k_img)
    dims = k_img.shape[-2:]
    y = torch.arange(-dims[0] / 2, dims[0] / 2)
    x = torch.arange(-dims[1] / 2, dims[1] / 2)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    ky = -1j * 2 * torch.pi * yy / dims[0]  # type: torch.Tensor
    kx = -1j * 2 * torch.pi * xx / dims[1]  # type: torch.Tensor
    k_img_shifted = k_img * torch.exp(-(kx * dx + ky * dy))
    return torch.view_as_real(k_img_shifted)


def flip_k(k_img: torch.Tensor, dim: str) -> torch.Tensor:
    """
    Flip the fourier transform of an image around the image center. This
    includes applying a phase shift of a half pixel, flipping, and reapplying a
    phase shift of a half pixel in the other direction.
    Note that the k-space center is located on the other side of the image
    center after this operation!

    Args:
        k_img: The fourier transform image.
        dim: The dimension to flip (either 'x' or 'y').

    Returns:
        Flipped version of k_img.
    """
    if dim == 'x':
        k_img = shift_phase(k_img, -0.5, 0)
        k_img = torch.flip(k_img, dims=(-2,))
        k_img = shift_phase(k_img, 0.5, 0)
    elif dim == 'y':
        k_img = shift_phase(k_img, 0, -0.5)
        k_img = torch.flip(k_img, dims=(-3,))
        k_img = shift_phase(k_img, 0, 0.5)
    else:
        raise RuntimeError
    return k_img


def interpolate_frames(x: torch.Tensor, num_frames: int, frames_axis: int = 0) -> torch.Tensor:
    x = x.moveaxis(frames_axis, -1)
    shape = x.shape  # [..., frame]
    x = x.reshape(1, -1, shape[-1])  # [1, prod(...), frame]
    x = torch.nn.functional.interpolate(x, size=num_frames, mode='linear', align_corners=True)  # [1, prod(...), num_frames]
    x = x.reshape(shape[:-1] + (num_frames,))  # [..., num_frames]
    x = x.moveaxis(-1, frames_axis)
    return x.contiguous()


class VarNetSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        sens_maps: Pre-computed coil sensitivity maps.
        target: The target image (if applicable).
        header: ISMRMRD header as xml string.
        fname: File name.
        slice_num: The slice index.
        max_value: Maximum absolute image value.
        annotations: Annotations [center, axis1, axis2, bbox1, bbox2].
        affine: Rotation angle, horizontal flip and vertical flip to normalize
            image orientation.
        position: Position of the slice in the scanner.
        orientation: Orientation of the slice in the scanner.
        sample_idx: Index of the sample within the dataset.
    """

    masked_kspace: torch.Tensor
    mask: torch.Tensor
    sens_maps: torch.Tensor
    target: torch.Tensor
    header: str
    fname: str
    slice_num: int
    max_value: float
    annotations: np.ndarray
    affine: np.ndarray
    position: np.ndarray
    orientation: np.ndarray
    sample_idx: int


class VarNetDataTransform:
    """
    Data Transformer for training VarNet models.
    """

    def __init__(
            self,
            mask_func: MaskFunc | None = None,
            use_seed: bool = True,
            roll_temporal: bool = False,
            random_flips: bool = False,
            augment_phase: bool = False,
            augment_mask: bool = False,
            augment_tres: bool = False,
            field_strength: float | None = None,
    ):
        """
        Args:
            mask_func: Optional; A function that can create a mask of
                appropriate shape. Defaults to None.
            use_seed: If True, this class computes a pseudo random number
                generator seed from the filename. This ensures that the same
                mask is used for all the slices of a given volume every time.
            roll_temporal: If True, Cine data is randomly rolled along the
                temporal dimension.
            random_flips: If True, images are randomly flipped. Note that the
                affine transformations read from the HDF5 file are not adjusted.
            augment_phase: If True, image phases are randomly rotated.
            augment_mask: If True, masks are augmented, depending on the
                implementation of mask_func.
            augment_tres: If True, the temporal resolution is augmented by
                interpolating frames.
            field_strength: If given, noise is added to the k-space data to
                simulate the given field strength in T. If the given field
                strength is higher than that of the data, no noise is added.
        """
        self.mask_func = mask_func
        self.use_seed = use_seed
        self.roll_temporal = roll_temporal
        self.random_flips = random_flips
        self.augment_phase = augment_phase
        self.augment_mask = augment_mask
        self.augment_tres = augment_tres
        self.field_strength = field_strength

    def __call__(
            self,
            kspace_in: np.ndarray,
            mask_in: np.ndarray | None,
            sens_maps_in: np.ndarray,
            target_in: np.ndarray | None,
            header: ismrmrd.xsd.ismrmrdHeader,
            attrs: dict,
            slice_num: int,
            sample_idx: int,
    ) -> VarNetSample:
        """
        Args:
            kspace_in: Input k-space of shape (coils, frames, rows, cols) for
                multi-coil data.
            mask_in: Mask from the test dataset.
            sens_maps_in: Coil sensitivity maps.
            target_in: Target image.
            header: ISMRMRD header.
            attrs: Additional meta information.
            slice_num: Serial number of the slice.
            sample_idx: Index of the sample within the dataset.

        Returns:
            A VarNetSample with the masked k-space, sampling mask, target
            image, the filename, the slice number, the maximum image value
            (from target), the target crop size, and the number of low
            frequency lines sampled.
        """

        # convert to PyTorch tensors
        kspace    = to_tensor(kspace_in)
        mask      = to_tensor(mask_in)      if mask_in      is not None else None
        sens_maps = to_tensor(sens_maps_in)
        target    = to_tensor(target_in)    if target_in    is not None else torch.empty(0)

        # extract padding information
        # padding values are indices of first non-padded value on the
        # left and first padded value on the right. That means that
        # padding is applied in the ranges [0, padding_left) and
        # [padding_right, enc_size.y).
        enc = header.encoding[0]
        enc_mat_x: int = enc.encodedSpace.matrixSize.x  # type: ignore
        enc_mat_y: int = enc.encodedSpace.matrixSize.y  # type: ignore
        enc_fov_x: float = enc.encodedSpace.fieldOfView_mm.x  # type: ignore
        enc_fov_y: float = enc.encodedSpace.fieldOfView_mm.y  # type: ignore
        if round((enc_fov_x / enc_fov_y) / (enc_mat_x / enc_mat_y)) == 2:
            # for some reason, this step is necessary for some OCMR Free.Max datasets
            enc_mat_y //= 2
        enc_lim_center = enc.encodingLimits.kspace_encoding_step_1.center  # type: ignore
        enc_lim_max = enc.encodingLimits.kspace_encoding_step_1.maximum  # type: ignore
        padding_left = enc_mat_y // 2 - enc_lim_center
        padding_right = padding_left + enc_lim_max + 1

        # extract annotations and affine transformation parameters
        annotations = np.concatenate([
            attrs.get('center', np.full(2, np.nan)).reshape((1, 2)),
            attrs.get('axis1', np.full(2, np.nan)).reshape((1, 2)),
            attrs.get('axis2', np.full(2, np.nan)).reshape((1, 2)),
            attrs.get('bbox', np.full(4, np.nan)).reshape((2, 2)),
        ], dtype=np.float32)  # we need float here to be able to store NaNs
        affine = np.array([
            attrs.get('rotation', 0),
            attrs.get('flip_horizontal', 0),
            attrs.get('flip_vertical', 0),
        ], dtype=np.int32)

        # extract position and orientation
        position = attrs.get('position', np.full(3, np.nan))
        orientation = np.concatenate([
            attrs.get('read_dir', np.full(3, np.nan)).reshape((1, 3)),
            attrs.get('phase_dir', np.full(3, np.nan)).reshape((1, 3)),
            attrs.get('slice_dir', np.full(3, np.nan)).reshape((1, 3)),
        ], dtype=np.float32)

        # random number generator used for augmentation
        seed = None if not self.use_seed else list(map(ord, attrs['fname']))
        rng = np.random.RandomState(np.array(seed)) if seed is not None else np.random

        if self.roll_temporal:
            n_frames = kspace.shape[1]
            n_roll = rng.randint(n_frames)
            kspace = torch.roll(kspace, n_roll, dims=1)  # [coil, frame, ...]
            target = torch.roll(target, n_roll, dims=0)  # [frame, ...]

        flip_mask = False
        if self.random_flips:
            r = rng.random(2)
            height, width = target.shape[-3:-1]

            # horizontal flip - skip in cases with uneven padding
            if r[0] > 0.5 and padding_left == kspace.shape[-2] - padding_right:
                # images
                kspace = flip_k(kspace, 'x')
                sens_maps = torch.flip(sens_maps, dims=(-2,))
                target = torch.flip(target, dims=(-2,))

                # annotations
                # no -1 because of flipping around kspace center
                annotations[:, 0] = np.clip(width - annotations[:, 0], 0, width - 1)

                # During flipping, the k-space center is shifted by one pixel.
                # Horizontal flipping corresponds to flipping in the phase
                # dimension, hence we need to flip the mask as well.
                # We intentionally don't swap padding_left and padding_right
                # to ensure correct flipping of the mask wrt to the k-space
                # center. The mask is later created with the convention that
                # the k-space center is at len//2, then padded and flipped.
                flip_mask = True

                # in horizontal flipping, padding_left and padding_right need to be flipped as well
                padding_right_ = kspace.shape[-2] - padding_left
                padding_left = kspace.shape[-2] - padding_right
                padding_right = padding_right_

            # vertical flip
            if r[1] > 0.5:
                # images
                kspace = flip_k(kspace, 'y')
                sens_maps = torch.flip(sens_maps, dims=(-3,))
                target = torch.flip(target, dims=(-3,))

                # annotations
                # no -1 because of flipping around kspace center
                annotations[:, 1] = np.clip(height - annotations[:, 1], 0, height - 1)

        if self.augment_phase:
            phi = (torch.rand(1) * 2 * torch.pi).reshape(-1)
            kspace = rotate_phase(kspace, phi)
            target = rotate_phase(target, phi)

        if self.augment_tres:
            current_tres = header.sequenceParameters.TR[0]  # type: ignore
            augmented_tres = rng.rand() * (50 - 30) + 30  # random between 30 and 50 ms
            augmented_n_frames = int(current_tres / augmented_tres * kspace.shape[1])

            # interpolate k-space and target
            kspace = interpolate_frames(kspace, augmented_n_frames, frames_axis=1)
            target = interpolate_frames(target, augmented_n_frames, frames_axis=0)

            # update header
            header.sequenceParameters.TR[0] = augmented_tres  # type: ignore
            header.encoding[0].encodingLimits.phase.maximum = augmented_n_frames - 1  # type: ignore

        if self.field_strength:
            noise_var_acq = attrs['noise_var']
            field_strength_acq: float = header.acquisitionSystemInformation.systemFieldStrength_T  # type: ignore
            assert self.field_strength <= field_strength_acq, \
                f'Target field strength ({self.field_strength}T) must be smaller or equal to acquisition field ' \
                f'strength ({field_strength_acq}T).'

            noise_var_add = noise_var_acq * (field_strength_acq ** 2 / self.field_strength ** 2 - 1)
            kspace = kspace + torch.tensor(np.random.normal(loc=0, scale=np.sqrt(noise_var_add), size=kspace.shape))

        if self.mask_func is not None:
            masked_kspace, mask = apply_mask(
                kspace, self.mask_func, seed=seed, padding=(padding_left, padding_right), flip_mask=flip_mask,
                augment_mask=self.augment_mask,
            )
        else:
            if mask is None:
                raise ValueError('No mask given and no mask function specified.')

            masked_kspace = kspace

            mask = mask[None, :, None, :, None]  # [coil=1, frame, readout=1, phase, 1]

            # phase padding with ones
            mask[..., :max(0, padding_left), :] = 1
            mask[..., min(mask.shape[-2], padding_right):, :] = 1

            # flip mask
            if flip_mask:
                mask = torch.flip(mask, dims=(-2,))

            # make sure that k-space and mask match
            mask_check = (torch.mean(complex_abs(kspace).unsqueeze(-1), dim=0, keepdim=True) > 0).to(mask.dtype)
            mask_check[..., :max(0, padding_left), :] = 1
            mask_check[..., min(mask_check.shape[-2], padding_right):, :] = 1
            assert torch.eq(mask_check, mask).all(), \
                f'The masked k-space does not match the mask for file {attrs["fname"]}!'

        sample = VarNetSample(
            masked_kspace=masked_kspace,
            mask=mask,
            sens_maps=sens_maps,
            target=torch.view_as_complex(target),
            header=header.toXML('utf-8'),  # type: ignore
            fname=attrs['fname'],
            slice_num=slice_num,
            max_value=attrs.get('abs_max', np.nan),
            annotations=annotations,
            affine=affine,
            position=position,
            orientation=orientation,
            sample_idx=sample_idx,
        )

        return sample
