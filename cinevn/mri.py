"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
import torch.fft

from .complex_math import complex_abs_sq, complex_conj, complex_mul


def fftnc(data: torch.Tensor, norm: str = 'ortho', ndim: int = 2) -> torch.Tensor:
    """
    Apply centered n-dimensional Fast Fourier Transform.

    Args:
        data: Complex valued input data where dimension -1 has size two
            and represents the complex dimension.
        norm: Normalization mode. See ``torch.fft.fft``.
        ndim: Number of back dimensions along which to apply the fft
            (i.e. `ndim`-dimensional Fourier transform), ignoring the
            last dimension of size 2.

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
        data: Complex valued input data where dimension -1 has size two
            and represents the complex dimension.
        norm: Normalization mode. See ``torch.fft.ifft``.
        ndim: Number of back dimensions along which to apply the ifft
            (i.e. `ndim`-dimensional inverse Fourier transform),
            ignoring the last dimension of size 2.

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
        sens_maps (torch.Tensor): Coil sensitivity maps of shape (batch, maps, coils, ..., 2)

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
        sens_maps (torch.Tensor): Coil sensitivity maps of shape (batch, maps, coils, ..., 2)

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
        data: A complex valued tensor, either with a complex dtype or
            where the size of the final dimension is 2.
        phi: The angle in radians around which the phase should be
            rotated.

    Returns:
        Phase-rotated version of data.
    """
    if not torch.is_complex(data) and not data.shape[-1] == 2:
        raise ValueError('Tensor does not have separate complex dim.')

    # use euler's formula to avoid conversion to complex dtype which is not supported by onnx
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


def adaptive_combine(data: torch.Tensor, coil_sens: torch.Tensor,
                     dim: int = 0) -> torch.Tensor:
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
