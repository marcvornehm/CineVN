#  Copyright (c) 2023. Marc Vornehm <marc.vornehm@fau.de>.

from pathlib import Path

try:
    import cupy as cp
except ImportError:  # importing cupy fails if no cuda is available
    cp = None
import h5py
import imageio
import numpy as np
import sigpy.mri

import bart

from ..ndarray import get_ndarray_module, ndarray


def read(fname: Path, nmaps: int = -1, layout: str = 'bart') -> np.ndarray:
    # read from file
    with h5py.File(fname, 'r') as hf:
        coil_sens = np.array(hf['coil_sens'])  # [slice, map, coil, frame=1, x, y]

    # select desired number of maps
    if nmaps > 0:
        coil_sens = coil_sens[:, :nmaps]

    # adjust layout
    if layout == 'bart':
        coil_sens = np.transpose(coil_sens, [4, 5, 2, 1, 3, 0])  # [x, y, coil, map, frame, slice]
    elif layout == 'fastmri':
        pass
    else:
        raise RuntimeError(f'Unknown layout {layout}')

    return coil_sens


def estimate(
        k_data: ndarray,  # [kx, ky, coil, frame, slice]
        backend: str,
        fname: Path | None = None,
        nmaps: int = 1,
        calib_lines: int = 16,
        clip: bool = False,
        write_h5: bool = True,
        write_png: bool = True,
) -> ndarray:
    """

    :param k_data: k-space data with layout [kx, ky, coil, frame, slice]
    :param backend: backend/method to use
    :param fname: output file name
    :param nmaps: number of sensitivity maps to estimate
    :param calib_lines: number of k-space lines to use for calibration
    :param clip: clip sensitivity maps
    :param write_h5: save as h5 file
    :param write_png: save as PNG files
    :return: coil sensitivity maps
    """
    xp = get_ndarray_module(k_data)

    # average over time
    mask = xp.abs(k_data) > 0
    k_data = xp.sum(k_data, axis=3) / (xp.sum(mask, axis=3) + xp.finfo(float).eps)

    # iterate over frames (only one iteration if averaged == True)
    match backend:
        case 'bart':
            coil_sens = espirit_bart(k_data, nmaps, calib_lines, clip)  # [x, y, coil, map, frame=1, slice]
        case 'sigpy':
            if nmaps > 1:
                raise NotImplementedError(
                    f'Sigpy currently only supports estimation of one set of coil sensitivity maps, but {nmaps} '
                    f'were requested. Please either set `nmaps` to one or select another backend'
                )
            coil_sens = espirit_sigpy(k_data, calib_lines, clip)  # [x, y, coil, map=1, frame=1, slice]
        case _:
            raise RuntimeError(f'Unknown backend {backend} for coil sensitivity estimation')

    # save as h5 file
    if write_h5:
        assert fname is not None, 'Output file name must be provided if `write_h5` is True'
        fname.parent.mkdir(parents=True, exist_ok=True)
        coil_sens_ = xp.transpose(coil_sens, [5, 3, 2, 4, 0, 1])  # [slice, map, coil, frame=1, x, y]
        with h5py.File(fname, 'w') as hf:
            if cp is not None:
                coil_sens_ = cp.asnumpy(coil_sens_)
            hf.create_dataset('coil_sens', data=coil_sens_.astype(np.complex64))

    # save as pngs
    if write_png:
        assert fname is not None, 'Output file name must be provided if `write_png` is True'
        fname.parent.mkdir(parents=True, exist_ok=True)
        coil_sens_ = xp.abs(coil_sens)
        coil_sens_ = (coil_sens_ / xp.max(coil_sens_)) * 255
        coil_sens_ = coil_sens_.astype(np.uint8)
        if cp is not None:
            coil_sens_ = cp.asnumpy(coil_sens_)
        for s in range(coil_sens_.shape[-1]):
            for coil in range(coil_sens_.shape[2]):
                for nmap in range(coil_sens_.shape[3]):
                    png_name = fname.parent / f'{fname.stem}_slice{s:02d}_coil{coil:02d}_map{nmap:02d}.png'
                    img = coil_sens_[:, :, coil, nmap, 0, s]
                    imageio.imwrite(png_name, img)

    return coil_sens


def espirit_sigpy(k_data: ndarray, calib_lines: int, clip: bool) -> ndarray:
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


def espirit_bart(k_data: ndarray, nmaps: int, calib_lines: int, clip: bool) -> ndarray:
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
        coil_sens = bart.bart(1, command, k_data_bart[..., sl, None], suppress_stdouterr=True)  # [x, y, z, coil, (map)]
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
