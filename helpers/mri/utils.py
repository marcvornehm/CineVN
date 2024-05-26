#  Copyright (c) 2023. Marc Vornehm <marc.vornehm@fau.de>.

from . import fftc
from ..ndarray import ndarray, get_ndarray_module


def apply_phase_padding(k_data: ndarray, phase_padding_left: int, phase_padding_right: int) -> ndarray:
    assert k_data.ndim == 9, f'k_data has the wrong number of dimensions, should be 9 but was {k_data.ndim}.'

    if phase_padding_left == 0 and phase_padding_right == 0:
        return k_data

    xp = get_ndarray_module(k_data)

    # padding is defined from first to last dimension
    # np.pad cannot handle negative values
    padding_plus = ((0, 0), (max(0, phase_padding_left), max(0, phase_padding_right))) + ((0, 0), ) * 7
    k_data_padded = xp.pad(k_data, padding_plus)
    padding_minus = slice(max(0, -phase_padding_left), k_data_padded.shape[1] - max(0, -phase_padding_right))
    k_data_padded = k_data_padded[:, padding_minus]

    return k_data_padded


def center_crop(img: ndarray, axis: int, target_size: int) -> ndarray:
    low = img.shape[axis] // 2 - target_size // 2
    high = img.shape[axis] // 2 + (target_size + 1) // 2
    slices = [slice(None)] * img.ndim
    slices[axis] = slice(low, high)
    return img[tuple(slices)]


def crop_readout_oversampling(k_data: ndarray) -> ndarray:
    xp = get_ndarray_module(k_data)

    # process slice by slice to avoid OOM errors
    num_slices = k_data.shape[6]
    k_data_processed = []
    for s in range(num_slices):
        k_data_slice = k_data[:, :, :, :, :, :, s, None]  # [kx, ky, kz=1, coil, frame, ...]

        # reconstruct individual coil images
        coil_imgs = fftc.ifftnc(k_data_slice, axes=(0,))  # [x, ky, kz=1, coil, frame, ...]

        # crop coil images and mask
        coil_imgs = center_crop(coil_imgs, 0, coil_imgs.shape[0] // 2)

        # transform back to k-space
        k_data_slice = fftc.fftnc(coil_imgs, axes=(0,))  # [kx, ky, kz=1, coil, frame, ...]

        k_data_processed.append(k_data_slice)

    return xp.concatenate(k_data_processed, axis=6)
