#  Copyright (c) 2023. Marc Vornehm <marc.vornehm@fau.de>.

from pathlib import Path
from typing import Optional, Sequence, Tuple

try:
    import cupy as cp
except ImportError:
    # importing cupy fails if no cuda is available
    cp = None
import numpy as np
import pygrappa
import yaml
from tqdm import tqdm

from ..ndarray import get_ndarray_module, ndarray
from . import pics_impl, subsampling
from .fftc import fftnc, ifftnc


def rss(
        k_data: ndarray,
        fft_axes: Tuple[int, int] = (-2, -1),
        coil_axis: int = 1,
        keep_coil_dim: bool = False,
) -> ndarray:
    xp = get_ndarray_module(k_data)
    coil_imgs = ifftnc(k_data, axes=fft_axes)
    return xp.sqrt(xp.sum(xp.abs(coil_imgs) ** 2, axis=coil_axis, keepdims=keep_coil_dim))


def sensitivity_weighted(
        k_data: ndarray,
        coil_sens: ndarray,
        fft_axes: Tuple[int, int] = (-2, -1),
        coil_axis: int = 1,
        frame_axis: int = 2,
        keep_coil_dim: bool = False,
        noise_cov: Optional[ndarray] = None,
) -> ndarray:
    """
    f = (C^H * Psi^-1 * C)^-1 * C^H * Psi^-1 * ifft(k)
    :param k_data: k
    :param coil_sens: C
    :param fft_axes: dimensions along which to apply the ifft
    :param coil_axis: coil dimension in all arrays
    :param frame_axis: temporal dimension (can be omitted in coil_sens)
    :param keep_coil_dim: whether the coil dimension should be kept as a
        unitary dimension
    :param noise_cov: Psi
    :return: f
    """
    xp = get_ndarray_module(k_data)

    # use identity matrix if no noise covariance matrix is given
    if noise_cov is None:
        noise_cov = xp.identity(k_data.shape[coil_axis])

    # insert frame dimension in coil sensitivities if they are time-averaged, but the k-space is not
    if k_data.ndim != coil_sens.ndim:
        coil_sens = xp.expand_dims(coil_sens, frame_axis)

    # ifft(k)
    coil_imgs = ifftnc(k_data, axes=fft_axes)

    # move coil dimension to the end and add a unitary dimension
    coil_imgs = xp.moveaxis(coil_imgs, coil_axis, -1)[..., None]  # m [..., coil, 1]
    coil_sens = xp.moveaxis(coil_sens, coil_axis, -1)[..., None]  # C [..., coil, 1]

    coil_sens_herm = xp.moveaxis(coil_sens.conj(), -1, -2)  # C^H [..., 1, coil]
    noise_cov_inv = xp.linalg.inv(noise_cov)  # Psi^-1 [coil, coil]

    norm = coil_sens_herm @ noise_cov_inv @ coil_sens  # [..., 1, 1]
    img = coil_sens_herm @ noise_cov_inv @ coil_imgs  # [..., 1, 1]
    if cp is not None and isinstance(img, cp.ndarray):
        img = cp.where(norm != 0, img / norm, img)
    else:
        # numpy raises zero-division warning if np.where is used
        img = np.divide(img, norm, where=(norm != 0), out=np.zeros_like(img))
    img = img[..., 0]  # remove previously added unitary dimension
    if keep_coil_dim:
        # move coil dimension back to where it was before
        img = xp.moveaxis(img, -1, coil_axis)
    else:
        # remove coil dimension
        img = img[..., 0]
    return img


def pocs(
        k_data: ndarray,
        n_iter: int = 10
) -> ndarray:
    xp = get_ndarray_module(k_data)

    # reshape to remove unnecessary dimensions
    new_shape = (*k_data.shape[:2], *k_data.shape[3:5], k_data.shape[6])
    k_data = k_data.reshape(new_shape)

    # partial fourier mask
    mask = abs(k_data.mean(axis=2, keepdims=True)) > 0
    mask = mask.astype(int)
    mask = mask.any(axis=3).squeeze()  # collapse frame dimension

    # symmetric center mask
    sym_mask = mask.any(axis=0, keepdims=True) * mask.any(axis=1, keepdims=True)
    sym_mask *= xp.rot90(sym_mask, 2)
    nonzeros = sym_mask.nonzero()
    sym_kx = nonzeros[0].max() - nonzeros[0].min() + 1
    sym_ky = nonzeros[1].max() - nonzeros[1].min() + 1

    # hamming window
    ham = xp.hamming(sym_kx.item())[:, None] * xp.hamming(sym_ky.item())[None, :]
    Wc = xp.zeros_like(sym_mask, dtype=xp.float32)
    Wc[sym_mask] = ham.reshape(-1)

    # center phase estimate
    Ic_phase_exp = xp.exp(1j * xp.angle(ifftnc(Wc[..., None, None, None] * k_data, axes=(0, 1))))

    # POCS iterations
    Sn = k_data  # zero-filled
    for _ in tqdm(range(n_iter), leave=False):
        # phase
        In = ifftnc(Sn, axes=(0, 1))
        In = xp.abs(In) * Ic_phase_exp

        # data consistency
        Sn = fftnc(In, axes=(0, 1))
        Sn[mask] = k_data[mask]

    # reinsert unnecessary dimensions
    Sn = Sn[:, :, None, :, :, None, :, None, None]

    return Sn


def pics(
        k_data: ndarray,  # [kx, ky, coil, frame, slice]
        regularizer: Sequence[str],
        recon_dir: Optional[Path] = None,
        coil_sens: Optional[ndarray] = None,  # [x, y, coil, map, frame, slice]
        coil_sens_dir: Optional[Path] = None,
        coil_sens_nmaps: int = 1,
        dynamic: bool = True,
        phase_crop: Optional[int] = None,
        use_gpu: bool = False,
        load_if_exists: bool = True,
        suppress_stdouterr: bool = False,
) -> ndarray:
    xp = get_ndarray_module(k_data)

    if load_if_exists and recon_dir is not None:
        try:
            if dynamic:
                recon_paths = [recon_dir / 'reconstruction.h5']
            else:
                recon_paths = [recon_dir / f'reconstruction_frame{f:02d}.h5' for f in range(k_data.shape[3])]
            recon = pics_impl.read(recon_paths, layout='bart')  # [x, y, frame, slice]
            return xp.asarray(recon)
        except FileNotFoundError:
            pass

    if dynamic:
        if recon_dir is None:
            recon_path = None
        else:
            recon_path = recon_dir / 'reconstruction.h5'
        if coil_sens_dir is None:
            coil_sens_path = None
        else:
            coil_sens_path = coil_sens_dir / 'coil_sens_avg.h5'
        recon = pics_impl.bart_pics(  # [x, y, frame, slice]
            k_data,  # [kx, ky, coil, frame, slice]
            regularizer,
            coil_sens=coil_sens,  # [x, y, coil, map, frame, slice]
            coil_sens_path=coil_sens_path,
            coil_sens_nmaps=coil_sens_nmaps,
            recon_path=recon_path,
            save_img=False,
            phase_crop=phase_crop,
            use_gpu=use_gpu,
            suppress_stdouterr=suppress_stdouterr,
        )
    else:
        if recon_dir is None:
            recon_paths = [None] * k_data.shape[3]
        else:
            recon_paths = [recon_dir / f'reconstruction_frame{f:02d}.h5' for f in range(k_data.shape[3])]
        if coil_sens_dir is None:
            coil_sens_paths = None
        else:
            coil_sens_paths = [coil_sens_dir / f'coil_sens_frame{f:02d}.h5' for f in range(k_data.shape[3])]
        frames = []
        for f in range(k_data.shape[3]):
            if coil_sens is not None:
                coil_sens_ = coil_sens[..., f, None, :]  # [x, y, coil, map, frame=1, slice]
            else:
                coil_sens_ = None
            if coil_sens_paths is not None:
                coil_sens_path = coil_sens_paths[f]
            else:
                coil_sens_path = None
            recon_frame = pics_impl.bart_pics(  # [x, y, frame, slice]
                k_data[:, :, :, f, None, :],  # [kx, ky, coil, frame=1, slice]
                regularizer,
                coil_sens=coil_sens_,
                coil_sens_path=coil_sens_path,
                coil_sens_nmaps=coil_sens_nmaps,
                recon_path=recon_paths[f],
                save_img=False,
                phase_crop=phase_crop,
                use_gpu=use_gpu,
                suppress_stdouterr=suppress_stdouterr,
            )
            frames.append(recon_frame)

        # join frames
        recon = xp.concatenate(frames, axis=2)

    return recon


def grappa(k_data: ndarray, calib: Optional[ndarray] = None, kernel_size: Tuple[int, int] = (5, 5), coil_axis: int = 1,
        frame_axis: int = 2, fft_axes: Tuple[int, int] = (-2, -1), return_kspace: bool = False) -> ndarray:
    # only use numpy in the following
    is_cupy = False
    if cp is not None and isinstance(k_data, cp.ndarray):
        is_cupy = True
        k_data = cp.asnumpy(k_data)
        if isinstance(calib, cp.ndarray):
            calib = cp.asnumpy(calib)

    # move coil and frame dimensions to the front
    k_data = np.moveaxis(k_data, (frame_axis, coil_axis), (0, 1))
    if calib is not None:
        calib = np.moveaxis(calib, (frame_axis, coil_axis), (0, 1))

    # fill missing k-space data frame-by-frame
    k_data_filled = []
    num_frames = k_data.shape[0]
    for f in tqdm(range(num_frames), leave=False, disable=(num_frames <= 1)):
        # pygrappa.grappa can't handle unitary dimensions
        shape = k_data[f].shape
        k_data_frame = k_data[f].squeeze()

        if calib is None:
            # find ACS lines if not given
            calib_frame = pygrappa.find_acs(k_data_frame, coil_axis=0)
        else:
            calib_frame = calib[f].squeeze()

        k_data_frame_filled = pygrappa.grappa(k_data_frame, calib_frame, kernel_size=kernel_size, coil_axis=0)
        assert k_data_frame_filled is not None
        k_data_frame_filled = k_data_frame_filled.reshape(shape)
        k_data_filled.append(k_data_frame_filled)

    # stack frames
    k_data_filled = np.stack(k_data_filled, axis=0)

    # move coil and frame dimensions back to where they were before
    k_data_filled = np.moveaxis(k_data_filled, (0, 1), (frame_axis, coil_axis))

    # convert back to cupy if necessary
    if is_cupy:
        k_data_filled = cp.asarray(k_data_filled)  # type: ignore

    # return either filled k-space or reconstruction
    if return_kspace:
        return k_data_filled
    else:
        return rss(k_data_filled, fft_axes=fft_axes, coil_axis=coil_axis)


def get_pics_regularizer(sampling: subsampling.Sampling) -> Sequence[str]:
    with open(Path(__file__).parent / 'bart_pics_params.yaml') as file:
        pics_params_all = yaml.safe_load(file)
    if sampling.name == 'pse':  # prospectively undersampled
        # search parameters that best match the given acceleration rate
        try:
            pics_params_mask = pics_params_all[sampling.mask_type]
        except KeyError as e:
            raise RuntimeError(
                f'Could not find PICS parameters for mask type {sampling.mask_type}.'
            ) from e
        acc_dist = np.inf
        acc_best = None
        for acc in pics_params_mask:
            assert sampling.acceleration is not None
            if abs(int(acc) - sampling.acceleration) < acc_dist:
                acc_dist = abs(int(acc) - sampling.acceleration)
                acc_best = acc
        pics_regularizer = pics_params_mask[acc_best]
    else:  # retrospectively undersampled
        try:
            pics_regularizer = pics_params_all[sampling.mask_type][sampling.acceleration]
        except KeyError as e:
            raise RuntimeError(
                f'Could not find PICS parameters for mask_type {sampling.mask_type} and acceleration '
                f'{sampling.acceleration}'
            ) from e

    return pics_regularizer.split(' ')
