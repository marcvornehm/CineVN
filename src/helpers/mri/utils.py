#  Copyright (c) 2023. Marc Vornehm <marc.vornehm@fau.de>.

from ..ndarray import get_ndarray_module, ndarray
from . import fftc


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

    # extract readouts
    mask = xp.abs(k_data).sum(axis=0) > 0
    readouts = k_data[:, mask]

    # get asymmetric echo
    pre_z = xp.where(xp.abs(readouts[:, 0]) > 0)[0][0]

    # apply ifft along readout
    readouts = fftc.ifftnc(readouts, axes=(0,))  # [x, ky, kz=1, coil, frame, ...]

    # crop
    readouts = center_crop(readouts, 0, readouts.shape[0] // 2)

    # apply fft along readout
    readouts = fftc.fftnc(readouts, axes=(0,))

    # re-apply asymmetric echo
    readouts[:pre_z // 2] = 0

    # insert into new array
    k_data = xp.zeros(shape=(readouts.shape[0],) + mask.shape, dtype=k_data.dtype)
    k_data[:, mask] = readouts

    return k_data
