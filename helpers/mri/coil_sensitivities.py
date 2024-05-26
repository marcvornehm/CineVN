#  Copyright (c) 2023. Marc Vornehm <marc.vornehm@fau.de>.

import os
from pathlib import Path
from typing import Sequence, Union

try:
    import cupy as cp
except ImportError:
    # importing cupy fails if no cuda is available
    cp = None
import h5py
import imageio
import numpy as np
import sigpy.mri

import bart
from ..ndarray import get_ndarray_module, ndarray


def read(
        paths: Union[Path, Sequence[Path]],
        nmaps: int = -1,
        layout: str = 'bart'
) -> np.ndarray:
    assert layout in ['fastmri', 'bart'], f'Unknown layout {layout}'

    if isinstance(paths, Path):
        paths = [paths]

    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError('Could not find file')

        with h5py.File(path, 'r') as hf:
            frame = np.array(hf['coil_sens'])  # [slice, map, coil, (frame=1), x, y]
            if frame.ndim == 5:  # static
                frame = frame[:, :, :, None]  # [slice, map, coil, frame=1, x, y]
            frames.append(frame)

    # join frames
    coil_sens = np.concatenate(frames, axis=3)  # [slice, map, coil, frame, x, y]

    # select desired number of maps
    if nmaps > 0:
        coil_sens = coil_sens[:, :nmaps]

    # adjust layout
    if layout == 'bart':
        coil_sens = np.transpose(coil_sens, [4, 5, 2, 1, 3, 0])  # [x, y, coil, map, frame, slice]

    return coil_sens

def write(
        coil_sens: ndarray,  # [x, y, coil, map, frame, slice]
        path: Path,
        keep_frame_dim: bool
):
    xp = get_ndarray_module(coil_sens)

    coil_sens = xp.transpose(coil_sens, [5, 3, 2, 4, 0, 1])  # [slice, map, coil, frame, x, y]
    if not keep_frame_dim:  # static
        coil_sens = xp.squeeze(coil_sens, axis=3)  # [slice, map, coil, x, y]

    with h5py.File(path, 'w') as hf:
        if cp is not None:
            coil_sens = cp.asnumpy(coil_sens)
        hf.create_dataset('coil_sens', data=coil_sens.astype(np.complex64))

def estimate(
        k_data: ndarray,
        coil_sens_dir: Path,
        backend: str,
        nmaps: int = 1,
        calib_lines: int = 16,
        averaged: bool = True,
        clip: bool = False,
        write_files: bool = True,
        write_png: bool = True,
        suppress_stdouterr: bool = False,
) -> ndarray:
    """

    :param k_data: k-space data with layout [kx, ky, coil, frame, slice]
    :param coil_sens_dir: output directory
    :param backend: backend/method to use
    :param nmaps: number of sensitivity maps to estimate
    :param calib_lines: number of k-space lines to use for calibration
    :param averaged: time-averaged
    :param clip: clip sensitivity maps
    :param write_files: save as h5 files
    :param write_png: additionally write PNG files
    :param suppress_stdouterr: suppress stdout and stderr of bart
    :return: coil sensitivity maps
    """
    xp = get_ndarray_module(k_data)

    # create directory
    os.makedirs(coil_sens_dir, exist_ok=True)

    paths = []
    if averaged:
        # average over time
        paths.append(coil_sens_dir / 'coil_sens_avg.h5')
        sum_ = xp.sum(k_data, axis=3, keepdims=True)
        div = xp.sum(xp.abs(k_data) > 0, axis=3, keepdims=True)
        with np.errstate(invalid='ignore'):
            # `where` keyword argument is not implemented cupy.true_divide, hence we use xp.where. In numpy, however,
            # this triggers a warning due to division by zero. We silence this warning because it is expected in array
            # locations at which the `where` condition does not apply.
            k_data = xp.where(xp.abs(div) > 0, sum_ / div, 0)  # [kx, ky, coil, frame=1, slice]
    else:
        # extract frames
        for i in range(k_data.shape[3]):
            paths.append(coil_sens_dir / f'coil_sens_frame{i:02d}.h5')

    # iterate over frames (only one iteration if averaged == True)
    coil_sens_frames = []
    for i, path in enumerate(paths):
        k_data_frame = k_data[..., i, :]  # [kx, ky, coil, slice]
        match backend:
            case 'bart':
                coil_sens = espirit_bart(  # [x, y, coil, map, frame=1, slice]
                    k_data_frame, nmaps, calib_lines, clip, suppress_stdouterr=suppress_stdouterr,
                )
            case 'sigpy':
                if nmaps > 1:
                    raise NotImplementedError(
                        f'Sigpy currently only supports estimation of one set of coil sensitivity maps, but {nmaps} '
                        f'were requested. Please either set `nmaps` to one or select another backend'
                    )
                coil_sens = espirit_sigpy(  # [x, y, coil, map=1, frame=1, slice]
                    k_data_frame, calib_lines, clip,
                )
            case _:
                raise RuntimeError(f'Unknown backend {backend} for coil sensitivity estimation')

        # save as h5 file
        if write_files:
            write(coil_sens, path, keep_frame_dim=averaged)

        coil_sens_frames.append(coil_sens)

    # combine frames
    coil_sens = xp.concatenate(coil_sens_frames, axis=4)  # [x, y, coil, map, frame, slice]

    # save as pngs
    if write_png:
        coil_sens_ = xp.abs(coil_sens)
        coil_sens_ = (coil_sens_ / xp.max(coil_sens_)) * 255
        coil_sens_ = coil_sens_.astype(np.uint8)
        if cp is not None:
            coil_sens_ = cp.asnumpy(coil_sens_)
        for s in range(coil_sens.shape[-1]):
            png_dir = coil_sens_dir / f'png_slice{s:02d}'
            os.makedirs(png_dir, exist_ok=True)
            for i, name in enumerate(paths):  # equivalent to frame dimension:
                for coil in range(coil_sens_.shape[2]):
                    for nmap in range(coil_sens_.shape[3]):
                        png_name = png_dir / f'{name.name}_map{nmap:02d}_coil{coil:02d}.png'
                        img = coil_sens_[:, :, coil, nmap, i, s]
                        imageio.imwrite(png_name, img)

    return coil_sens


def espirit_sigpy(
        k_data: ndarray,
        calib_lines: int,
        clip: bool,
) -> ndarray:
    xp = get_ndarray_module(k_data)

    crop = 0.8 if clip else 0  # 0.8 is default value in bart
    ksp = xp.transpose(k_data, [2, 0, 1, 3])  # [coil, kx, ky, slice]

    # determine device
    if cp is not None and isinstance(k_data, cp.ndarray):
        device = sigpy.Device(0)
    else:
        device = sigpy.Device(-1)

    app = sigpy.mri.app.EspiritCalib(ksp, calib_width=calib_lines, crop=crop, device=device, show_pbar=False)
    coil_sens = app.run()  # [coil, x, y, slice]

    # convert to bart layout
    coil_sens = xp.transpose(coil_sens, [1, 2, 0, 3])  # type: ignore  # [x, y, coil, slice]
    coil_sens = coil_sens[:, :, :, None, None, :]  # [x, y, coil, map, frame, slice]

    return coil_sens


def espirit_bart(
        k_data: ndarray,
        nmaps: int,
        calib_lines: int,
        clip: bool,
        suppress_stdouterr: bool = False,
) -> ndarray:
    xp = get_ndarray_module(k_data)

    # expand dimensionality to BART format
    # [kx, ky, kz=1, coil, map, ?, ?, ?, ?, ?, frame=1, ?, ?, slice]
    k_data_bart = k_data[:, :, None, :, None, None, None, None, None, None, None, None, None, :]

    # move to cpu if cuda/cupy is available
    if cp is not None:
        k_data_bart = cp.asnumpy(k_data_bart)

    # ESPIRiT
    # -m: number of maps
    # -r: restrict the calibration region to `calib_lines` lines
    # -c0: don't crop the sensitivities outside the body region
    command = f'ecalib -m{nmaps} -r{calib_lines}'
    if not clip:
        command += ' -c0'
    # bart collapses the slice dimension
    coil_sens_sls = []
    for sl in range(k_data_bart.shape[13]):
        # unitary dimensions at the end get removed by bart's python integration
        coil_sens = bart.bart(1, command, k_data_bart[..., sl, None], suppress_stdouterr=suppress_stdouterr)  # [x, y, z, coil, (map)]
        assert isinstance(coil_sens, np.ndarray), 'bart returned unexpected type'
        if nmaps == 1:
            coil_sens = coil_sens[..., None]  # [x, y, z, coil, map]
        coil_sens = np.squeeze(coil_sens, axis=2)  # [x, y, coil, map]
        coil_sens_sls.append(coil_sens)
    coil_sens = np.stack(coil_sens_sls, axis=-1)  # [x, y, coil, map, slice]
    coil_sens = coil_sens[:, :, :, :, None, :]  # [x, y, coil, map, frame, slice]

    # move to gpu if xp is cupy
    coil_sens = xp.asarray(coil_sens)

    return coil_sens
