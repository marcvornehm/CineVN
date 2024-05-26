"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import warnings
from typing import Dict, NamedTuple, Optional, Sequence, Tuple, Union

with warnings.catch_warnings():  # ismrmrd adds a warning filter that causes all subsequent warnings to be printed even if they should be filtered
    import ismrmrd
import numpy as np
import torch

from ..complex_math import complex_abs
from ..mri import rotate_phase
from .subsample import DynamicMaskFunc, MaskFunc


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
    offset: Optional[int] = None,
    seed: Optional[float | Sequence[float]] = None,
    padding: Optional[Sequence[int]] = None,
    flip_mask: bool = False,
    augment_mask: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Subsample given k-space by multiplying with a mask.

    Args:
        data: The input k-space data. This should have at least 3 dimensions,
            where dimensions -3 and -2 are the spatial dimensions, and the
            final dimension has size 2 (for complex values). If mask_func is of
            type DynamicMaskFunc, data is expected to have an additional
            temporal dimension at -4.
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
            num_low_frequencies: The number of low-resolution frequency samples
                in the mask.
    """
    if padding is None:
        padding = (0, data.shape[-2])

    # ignore negative padding (phase resolution > 100%)
    padding = (max(0, padding[0]), min(data.shape[-2], padding[1]))

    # the mask is computed on the k-space shape without padding and with readout and complex dimensions reduced
    if isinstance(mask_func, DynamicMaskFunc):
        shape = (1, data.shape[1], 1, padding[1] - padding[0], 1)
    else:
        shape = (1, 1, padding[1] - padding[0], 1)
    shape_padded = shape[:-2] + (data.shape[-2], shape[-1])

    mask_unpadded, num_low_frequencies = mask_func(
        shape, offset=offset, padding_left=padding[0], padding_right=data.shape[-2]-padding[1], seed=seed,
        augment=augment_mask
    )

    # apply zero-padding
    mask = torch.ones(size=shape_padded, dtype=mask_unpadded.dtype, layout=mask_unpadded.layout,
                      device=mask_unpadded.device)
    mask[..., padding[0]:padding[1], :] = mask_unpadded

    # flip the mask along the phase direction. it's important that this is done
    # after zero-padding so the k-space center is correctly flipped
    if flip_mask:
        mask = torch.flip(mask, dims=(-2,))

    masked_data = data * mask + 0.0  # the + 0.0 removes the sign of the zeros

    return masked_data, mask, torch.tensor(num_low_frequencies)


def mask_center(x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor) -> torch.Tensor:
    """
    Initializes a mask with the center filled in the second to last dimension.

    Args:
        x: The image / k-space to mask.
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    mask = torch.zeros_like(x)
    mask[..., mask_from:mask_to, :] = x[..., mask_from:mask_to, :]

    return mask


def batched_mask_center(
    x: torch.Tensor, mask_from: torch.Tensor, mask_to: torch.Tensor
) -> torch.Tensor:
    """
    Initializes a mask with the center filled in the second to last dimension.

    Can operate with different masks for each batch element.

    Args:
        x: The images / k-spaces to mask.
        mask_from: Part of center to start filling.
        mask_to: Part of center to end filling.

    Returns:
        A mask with the center filled.
    """
    if not mask_from.shape == mask_to.shape:
        raise ValueError('mask_from and mask_to must match shapes.')
    if not mask_from.ndim == 1:
        raise ValueError('mask_from and mask_to must have 1 dimension.')
    if not mask_from.shape[0] == 1:
        if (not x.shape[0] == mask_from.shape[0]) or (
            not x.shape[0] == mask_to.shape[0]
        ):
            raise ValueError('mask_from and mask_to must have batch_size length.')

    if mask_from.shape[0] == 1:
        mask = mask_center(x, mask_from, mask_to)
    else:
        mask = torch.zeros_like(x)
        for i, (start, end) in enumerate(zip(mask_from, mask_to)):
            mask[i, ..., start:end, :] = x[i, ..., start:end, :]

    return mask


def center_crop(data: torch.Tensor, shape: Tuple[Union[int, torch.Tensor], ...]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should
            have at least as many dimensions as shape has entries and the
            cropping is applied along the last len(shape) dimensions.
        shape: The output shape. The shape should be smaller
            than the corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    if isinstance(shape[0], torch.Tensor):
        shape_stacked = torch.stack(shape).T
        assert all(all(torch.eq(shape_stacked[0], s)) for s in shape_stacked), \
            'Shape needs to be equal for each element of the batch'
        shape = (shape[0][0], shape[1][0])

    slices = ()
    for dim in range(len(shape)):
        if not 0 < shape[-dim-1] <= data.shape[-dim-1]:
            raise ValueError('Invalid shapes.')

        crop_from = int((data.shape[-dim-1] - shape[-dim-1]) / 2)
        crop_to = crop_from + shape[-dim-1]
        slices = (slice(crop_from, crop_to),) + slices

    return data[(..., *slices)]


def center_crop_to_smallest(x: torch.Tensor, y: torch.Tensor, ndim: int = 2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply a center crop on the larger image to the size of the smaller.

    The minimum is taken over the ndim last dimensions (i.e., over dim=-1 and
    dim=-2 if ndim is 2). If x is smaller than y at dim=-1 and y is smaller
    than x at dim=-2, then the returned dimension will be a mixture of the two.

    Args:
        x: The first image.
        y: The second image.
        ndim: The number of dimensions along which to crop, starting with the
            last dimension.

    Returns:
        tuple of tensors x and y, each cropped to the minimum size.
    """
    shape = ()
    for dim in range(ndim):
        smallest = min(x.shape[-dim-1], y.shape[-dim-1])
        shape = (smallest,) + shape
    x = center_crop(x, shape)
    y = center_crop(y, shape)

    return x, y


def batched_crop_to_recon_size(x: torch.Tensor, ismrmrd_headers: Sequence[str | ismrmrd.xsd.ismrmrdHeader]) -> torch.Tensor:
    """Apply a center crop to each of the batched input tensors to the
    reconstruction size as specified in the ISMRMRD headers.

    Args:
        x (torch.Tensor): Reconstructed images.
        ismrmrd_headers (Sequence[str  |  ismrmrd.xsd.ismrmrdHeader]): ISMRMRD headers.

    Returns:
        torch.Tensor: Cropped reconstructions.
    """
    cropped = []
    for image, header in zip(x, ismrmrd_headers):
        cropped.append(crop_to_recon_size(image, header))
    return torch.stack(cropped)


def crop_to_recon_size(x: torch.Tensor, ismrmrd_header: str | ismrmrd.xsd.ismrmrdHeader) -> torch.Tensor:
    """Apply a center crop to the input tensor to the reconstruction size as
    specified in the ISMRMRD header.

    Args:
        x (torch.Tensor): Reconstructed image.
        ismrmrd_header (str | ismrmrd.xsd.ismrmrdHeader): ISMRMRD header.

    Returns:
        torch.Tensor: Cropped reconstruction.
    """
    if isinstance(ismrmrd_header, str):
        ismrmrd_header = ismrmrd.xsd.CreateFromDocument(ismrmrd_header)
    enc_size: ismrmrd.xsd.matrixSizeType = ismrmrd_header.encoding[0].encodedSpace.matrixSize  # type: ignore
    recon_size: ismrmrd.xsd.matrixSizeType = ismrmrd_header.encoding[0].reconSpace.matrixSize  # type: ignore
    if enc_size.x == 2 * recon_size.x:
        # readout oversampling is modelled with double/half readout
        crop_size = (recon_size.x, recon_size.y)
    else:
        # readout oversampling is modelled with double/half phase
        crop_size = (recon_size.x // 2, recon_size.y // 2)
    return center_crop(x, crop_size)


def shift_phase(k_img: torch.Tensor, dx: float, dy: float):
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


def flip_k(k_img: torch.Tensor, dim: str):
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


def interpolate_frames(x: torch.Tensor, num_frames: int) -> torch.Tensor:
    shape = x.shape  # [coil, frame, x, y, 2]
    x = x.reshape(*shape[:2], -1)  # [coil, frame, x*y*2]
    x = torch.movedim(x, -1, 0)  # [x*y*2, coil, frame]
    x = torch.nn.functional.interpolate(x, size=num_frames, mode='linear', align_corners=True)  # [x*y*2, coil, num_frames]
    x = torch.movedim(x, 0, -1)  # [coil, num_frames, x*y*2]
    x = x.reshape(shape[0], num_frames, *shape[2:])  # [coil, num_frames, x, y, 2]
    return x.contiguous()


def apply_affine_to_image(image: torch.Tensor | np.ndarray, rotation: int, flip_horizontal: bool,
                          flip_vertical: bool) -> torch.Tensor | np.ndarray:
    if isinstance(image, np.ndarray):
        if rotation > 0:
            image = np.rot90(image, k=int(rotation) // 90, axes=(-2, -1))
        if flip_horizontal == 1:
            image = np.flip(image, axis=(-1,))
        if flip_vertical == 1:
            image = np.flip(image, axis=(-2,))
    elif isinstance(image, torch.Tensor):
        if rotation > 0:
            image = torch.rot90(image, k=int(rotation) // 90, dims=(-2, -1))
        if flip_horizontal == 1:
            image = torch.flip(image, dims=(-1,))
        if flip_vertical == 1:
            image = torch.flip(image, dims=(-2,))
    else:
        raise TypeError

    return image


def apply_affine_to_annotations(annotations: torch.Tensor | np.ndarray, shape: torch.Size | Tuple[int, ...],
                                rotation: int, flip_horizontal: bool, flip_vertical: bool) -> torch.Tensor | np.ndarray:
    if rotation > 0:
        for _ in range(int(rotation) // 90):
            if isinstance(annotations, np.ndarray):
                rotmat = np.array([[0, -1], [1, 0]], dtype=annotations.dtype)
                trans = np.array([0, shape[1]], dtype=annotations.dtype)
            elif isinstance(annotations, torch.Tensor):
                rotmat = torch.tensor([[0, -1], [1, 0]], device=annotations.device, dtype=annotations.dtype)
                trans = torch.tensor([0, shape[1]], device=annotations.device, dtype=annotations.dtype)
            else:
                raise TypeError
            annotations = annotations @ rotmat + trans
            shape = shape[::-1]
    if flip_horizontal:
        annotations[:, 0] = shape[1] - annotations[:, 0] - 1  # type: ignore
    if flip_vertical:
        annotations[:, 1] = shape[0] - annotations[:, 1] - 1  # type: ignore

    return annotations


class VarNetSample(NamedTuple):
    """
    A sample of masked k-space for variational network reconstruction.

    Args:
        masked_kspace: k-space after applying sampling mask.
        mask: The applied sampling mask.
        num_low_frequencies: The number of samples for the densely-sampled
            center.
        target: The target image (if applicable).
        sens_maps: Pre-computed coil sensitivity maps.
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
    num_low_frequencies: torch.Tensor
    target: torch.Tensor
    sens_maps: torch.Tensor
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

    def __init__(self, mask_func: Optional[MaskFunc] = None, use_seed: bool = True, roll_temporal: bool = False,
                 random_flips: bool = False, augment_phase: bool = False, augment_mask: bool = False,
                 augment_tres: bool = False, field_strength: Optional[float] = None):
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
        mask_in: Optional[np.ndarray],
        target_in: Optional[np.ndarray],
        sens_maps_in: Optional[np.ndarray],
        header: ismrmrd.xsd.ismrmrdHeader,
        attrs: Dict,
        slice_num: int,
        sample_idx: int,
    ) -> VarNetSample:
        """
        Args:
            kspace_in: Input k-space of shape (num_coils, (frames, ) rows, cols)
                for multi-coil data.
            mask_in: Mask from the test dataset.
            target_in: Target image.
            sens_maps_in: Coil sensitivity maps.
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
        mask      = to_tensor(mask_in)                            if mask_in      is not None else None
        target    = to_tensor(target_in, stack_complex_dim=False) if target_in    is not None else torch.empty(0)
        sens_maps = to_tensor(sens_maps_in)                       if sens_maps_in is not None else torch.empty(0)

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
        ], dtype=np.float32)
        affine = np.array([
            attrs.get('rotation', 0),
            attrs.get('flip_horizontal', 0),
            attrs.get('flip_vertical', 0),
        ], dtype=np.float32)

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
            assert kspace.ndim == 5, 'kspace needs temporal dimension for temporal rolling'
            n_frames = kspace.shape[1]
            n_roll = rng.randint(n_frames)
            kspace = torch.roll(kspace, n_roll, dims=1)  # [coil, frame, ...]
            target = torch.roll(target, n_roll, dims=0)  # [frame, ...]

        flip_mask = False
        if self.random_flips:
            r = rng.random(2)
            height, width = target.shape[-2:]

            # horizontal flip - skip in cases with uneven padding
            if r[0] > 0.5 and padding_left == kspace.shape[-2] - padding_right:
                # images
                kspace = flip_k(kspace, 'x')
                target = torch.flip(target, dims=(-1,))
                sens_maps = torch.flip(sens_maps, (-2,)) if sens_maps.numel() > 0 else sens_maps

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
                target = torch.flip(target, dims=(-2,))
                sens_maps = torch.flip(sens_maps, dims=(-3,)) if sens_maps.numel() > 0 else sens_maps

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

            # interpolate k-space
            kspace = interpolate_frames(kspace, augmented_n_frames)

            # interpolate target
            target = torch.view_as_real(target)[None]
            target = interpolate_frames(target, augmented_n_frames)[0]
            target = torch.view_as_complex(target)

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
            masked_kspace, mask, num_low_frequencies = apply_mask(
                kspace, self.mask_func, seed=seed, padding=(padding_left, padding_right), flip_mask=flip_mask,
                augment_mask=self.augment_mask
            )
        else:
            if mask is None:
                raise ValueError('No mask given and no mask function specified.')

            masked_kspace = kspace

            if mask.ndim == 2:  # dynamic data
                mask = mask[None, :, None, :, None]  # [coil=1, frame, readout=1, phase, 1]
            else:  # static data
                mask = mask[None, None, :, None]  # [coil=1, readout=1, phase, 1]

            # extract number of low frequencies
            mask_acs = mask.squeeze()  # [(frame, )phase]
            if mask_acs.ndim == 2:  # dynamic data
                mask_acs = torch.all(mask_acs, dim=0)
            # append and prepend a zero so argmin doesn't return 0 for a fully sampled mask
            mask_acs = torch.cat([torch.zeros(1), mask_acs, torch.zeros(1)])
            acs_front = torch.argmin(torch.flip(mask_acs[:mask_acs.shape[0] // 2], dims=(0,)))
            acs_back = torch.argmin(mask_acs[mask_acs.shape[0] // 2:])
            num_low_frequencies = acs_front + acs_back

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
            mask=mask.to(torch.bool),
            num_low_frequencies=num_low_frequencies,
            target=target,
            sens_maps=sens_maps,
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
