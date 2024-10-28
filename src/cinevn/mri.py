"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.

Part of this source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import warnings
from typing import Sequence, TypeVar

with warnings.catch_warnings():  # ismrmrd adds a warning filter that causes all subsequent warnings to be printed even if they should be filtered
    import ismrmrd
import numpy as np
import torch
import torch.fft

from .complex_math import complex_abs_sq, complex_conj, complex_mul

ArrayType = TypeVar('ArrayType', torch.Tensor, np.ndarray)


def fftnc(data: torch.Tensor, norm: str = 'ortho', ndim: int = 2) -> torch.Tensor:
    """
    Apply centered n-dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data where dimension -1 has size two and
            represents the complex dimension.
        norm: Normalization mode. See ``torch.fft.fft``.
        ndim: Number of back dimensions along which to apply the fft (i.e.
            `ndim`-dimensional Fourier transform), ignoring the last dimension
            of size 2.

    Returns:
        The FFT of the input.
    """
    assert data.shape[-1] == 2, 'Tensor does not have separate complex dim.'
    assert ndim in (1, 2, 3), f'`ndim` must be in (1, 2, 3), but was {ndim}'

    dim = list(range(-ndim, 0))
    data = torch.view_as_complex(data)
    data = torch.fft.ifftshift(data, dim=dim)
    data = torch.fft.fftn(data, dim=dim, norm=norm)
    data = torch.fft.fftshift(data, dim=dim)
    data = torch.view_as_real(data)

    return data


def ifftnc(data: torch.Tensor, norm: str = 'ortho', ndim: int = 2) -> torch.Tensor:
    """
    Apply centered n-dimensional Inverse Fast Fourier Transform.

    Args:
        data: Complex valued input data where dimension -1 has size two and
            represents the complex dimension.
        norm: Normalization mode. See ``torch.fft.ifft``.
        ndim: Number of back dimensions along which to apply the ifft (i.e.
            `ndim`-dimensional inverse Fourier transform), ignoring the last
            dimension of size 2.

    Returns:
        The IFFT of the input.
    """
    assert data.shape[-1] == 2, 'Tensor does not have separate complex dim.'
    assert ndim in (1, 2, 3), f'`ndim` must be in (1, 2, 3), but was {ndim}'

    dim = list(range(-ndim, 0))
    data = torch.view_as_complex(data)
    data = torch.fft.ifftshift(data, dim=dim)
    data = torch.fft.ifftn(data, dim=dim, norm=norm)
    data = torch.fft.fftshift(data, dim=dim)
    data = torch.view_as_real(data)

    return data


def sens_expand(x: torch.Tensor, sens_maps: torch.Tensor, fft_ndim: int = 2) -> torch.Tensor:
    """Expand coil images and transform to k-space.

    Args:
        x (torch.Tensor): Input image of shape (batch, maps,  1, ..., 2)
        sens_maps (torch.Tensor): Coil sensitivity maps of shape (batch, maps,
            coils, ..., 2)

    Returns:
        torch.Tensor: k-space data of shape (batch, coils, ..., 2)
    """
    # expand coils
    x = complex_mul(x, sens_maps)

    # sum over maps dimension
    x = x.sum(dim=1)

    # fft
    x = fftnc(x, ndim=fft_ndim)

    return x


def sens_reduce(x: torch.Tensor, sens_maps: torch.Tensor, fft_ndim: int = 2) -> torch.Tensor:
    """Transform k-space to coil images and combine coils.

    Args:
        x (torch.Tensor): Input k-space of shape (batch, coils, ..., 2)
        sens_maps (torch.Tensor): Coil sensitivity maps of shape (batch, maps,
            coils, ..., 2)

    Returns:
        torch.Tensor: Image of shape (batch, maps, 1, ..., 2)
    """
    # inverse fft
    x = ifftnc(x, ndim=fft_ndim)

    # insert maps dimension
    x = x[:, None]

    # combine coils
    x = complex_mul(x, complex_conj(sens_maps)).sum(dim=2, keepdim=True)

    return x


def rotate_phase(data: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    """
    Compute the phase-rotated version of a complex tensor.

    Args:
        data: A complex valued tensor, either with a complex dtype or where the
            size of the final dimension is 2.
        phi: The angle in radians around which the phase should be rotated.

    Returns:
        Phase-rotated version of data.
    """
    if not torch.is_complex(data) and not data.shape[-1] == 2:
        raise ValueError('Tensor does not have separate complex dim.')

    # use euler's formula to avoid conversion to complex dtype
    phase = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
    phase = phase.reshape((phase.shape[0], *(1,) * (data.ndim - 2), 2))
    data_rotated = complex_mul(data, phase)
    return data_rotated


def rss(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS).

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt((data ** 2).sum(dim))


def rss_complex(data: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute the Root Sum of Squares (RSS) for complex inputs.

    RSS is computed assuming that dim is the coil dimension.

    Args:
        data: The input tensor
        dim: The dimensions along which to apply the RSS transform

    Returns:
        The RSS value.
    """
    return torch.sqrt(complex_abs_sq(data).sum(dim))


def adaptive_combine(data: torch.Tensor, coil_sens: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """
    Compute a SENSE reconstruction by combining coil images with coil
    sensitivities.

    It is computed assuming that dim is the coil dimension.

    Args:
        data: The individual coil images
        coil_sens: The coil sensitivity maps
        dim: The dimension along which to combine the images

    Returns:
        The combined image.
    """

    return complex_mul(data, complex_conj(coil_sens)).sum(dim)


def center_crop(data: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """
    Apply a center crop to the input real image or batch of real images.

    Args:
        data: The input tensor to be center cropped. It should have at least as
            many dimensions as shape has entries and the cropping is applied
            along the last len(shape) dimensions.
        shape: The output shape. The shape should be smaller than the
            corresponding dimensions of data.

    Returns:
        The center cropped image.
    """
    slices = ()
    for dim in range(len(shape)):
        if not 0 < shape[-dim-1] <= data.shape[-dim-1]:
            raise ValueError('Invalid shapes.')

        crop_from = int((data.shape[-dim-1] - shape[-dim-1]) / 2)
        crop_to = crop_from + shape[-dim-1]
        slices = (slice(crop_from, crop_to),) + slices

    return data[(..., *slices)]


def center_crop_to_smallest(x: torch.Tensor, y: torch.Tensor, ndim: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
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


def batched_crop_to_recon_size(x: torch.Tensor, ismrmrd_headers: Sequence[str | ismrmrd.xsd.ismrmrdHeader]) \
        -> torch.Tensor:
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


def apply_affine_to_image(
        image: ArrayType,
        rotation: int,
        flip_horizontal: bool | int,
        flip_vertical: bool | int,
) -> ArrayType:
    if isinstance(image, np.ndarray):
        if rotation > 0:
            image = np.rot90(image, k=int(rotation) // 90, axes=(-2, -1))
        if flip_horizontal:
            image = np.flip(image, axis=(-1,))  # type: ignore
        if flip_vertical:
            image = np.flip(image, axis=(-2,))  # type: ignore
    elif isinstance(image, torch.Tensor):
        if rotation > 0:
            image = torch.rot90(image, k=int(rotation) // 90, dims=(-2, -1))
        if flip_horizontal:
            image = torch.flip(image, dims=(-1,))
        if flip_vertical:
            image = torch.flip(image, dims=(-2,))
    else:
        raise TypeError

    return image


def apply_affine_to_annotations(
        annotations: ArrayType,
        shape: torch.Size | tuple[int, ...],
        rotation: int,
        flip_horizontal: bool | int,
        flip_vertical: bool | int,
) -> ArrayType:
    if rotation > 0:
        for _ in range(int(rotation) // 90):
            if isinstance(annotations, np.ndarray):
                dtype = np.promote_types(annotations.dtype, np.float32)
                rotmat = np.array([[0, -1], [1, 0]], dtype=dtype)
                trans = np.array([0, shape[1]], dtype=dtype)
                annotations = (annotations.astype(dtype) @ rotmat + trans).astype(annotations.dtype)
            elif isinstance(annotations, torch.Tensor):
                dtype = torch.promote_types(annotations.dtype, torch.float32)
                rotmat = torch.tensor([[0, -1], [1, 0]], device=annotations.device, dtype=dtype)
                trans = torch.tensor([0, shape[1]], device=annotations.device, dtype=dtype)
                annotations = (annotations.to(dtype) @ rotmat + trans).to(annotations.dtype)
            else:
                raise TypeError
            shape = shape[::-1]
    if flip_horizontal:
        annotations[:, 0] = int(shape[1]) - annotations[:, 0] - 1  # type: ignore
    if flip_vertical:
        annotations[:, 1] = int(shape[0]) - annotations[:, 1] - 1  # type: ignore

    return annotations
