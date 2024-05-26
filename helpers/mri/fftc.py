#  Copyright (c) 2023. Marc Vornehm <marc.vornehm@fau.de>.

from typing import Sequence

from ..ndarray import get_ndarray_module, ndarray


def fftnc(image: ndarray, axes: Sequence[int]=(-2, -1)) -> ndarray:
    xp = get_ndarray_module(image)

    kspace = xp.fft.fftshift(xp.fft.fftn(xp.fft.ifftshift(image, axes=axes), norm='ortho', axes=axes), axes=axes)

    # np.fft.fftn returns dtype complex128 regardless of input dtype
    if image.dtype in [xp.complex64, xp.float32]:
        kspace = kspace.astype(xp.complex64)

    return kspace


def ifftnc(kspace: ndarray, axes: Sequence[int] = (-2, -1)) -> ndarray:
    xp = get_ndarray_module(kspace)

    image = xp.fft.fftshift(xp.fft.ifftn(xp.fft.ifftshift(kspace, axes=axes), norm='ortho', axes=axes), axes=axes)

    # np.fft.ifftn returns dtype complex128 regardless of input dtype
    if kspace.dtype in [xp.complex64, xp.float32]:
        image = image.astype(xp.complex64)

    return image
