#  Copyright (c) 2023. Marc Vornehm <marc.vornehm@fau.de>.

import os
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

try:
    import cupy as cp
except ImportError:
    # importing cupy fails if no cuda is available
    cp = None
import h5py
with warnings.catch_warnings():  # ismrmrd adds a warning filter that causes all subsequent warnings to be printed even if they should be filtered
    import ismrmrd
import numpy as np
import optuna
from skimage.metrics import structural_similarity

import bart
from .. import utils
from ..ndarray import get_ndarray_module, ndarray
from . import coil_sensitivities, subsampling


def read(
        paths: Union[Path, Sequence[Path]],
        layout: str = 'fastmri',
) -> np.ndarray:
    assert layout in ['fastmri', 'bart'], f'Unknown layout {layout}'

    if isinstance(paths, Path):
        paths = [paths]

    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError('Could not find file')

        with h5py.File(path, 'r') as hf:
            frame = np.array(hf['reconstruction_cs'])  # [slice, (frame=1), x, y]
            if frame.ndim == 3:  # static
                frame = frame[:, :, :, None]  # [slice, frame=1, x, y]
            frames.append(frame)

    # join frames
    arr = np.concatenate(frames, axis=1)  # [slice, frame, x, y]

    # adjust layout
    if layout == 'bart':
        arr = np.transpose(arr, [2, 3, 1, 0])  # [x, y, frame, slice]

    return arr


def write(
        reconstruction: ndarray,  # [x, y, (frame), slice]
        path: Path,
):
    xp = get_ndarray_module(reconstruction)

    if reconstruction.ndim == 3:  # static
        reconstruction = xp.transpose(reconstruction, [2, 0, 1])  # [slice, x, y]
    else:  # dynamic
        reconstruction = xp.transpose(reconstruction, [3, 2, 0, 1])  # [slice, frame, x, y]

    with h5py.File(path, 'w') as hf:
        if cp is not None:
            reconstruction = cp.asnumpy(reconstruction)
        hf.create_dataset('reconstruction_cs', data=reconstruction.astype(np.complex64))


def bart_pics(
        k_data: ndarray,  # [kx, ky, coil, frame, slice]
        regularizer: Sequence[str] | str,
        coil_sens: Optional[ndarray] = None,  # [x, y, coil, map, frame, slice]
        coil_sens_path: Optional[Path] = None,
        coil_sens_nmaps: int = 1,
        recon_path: Optional[Path] = None,
        save_img: bool = True,
        phase_crop: Optional[int] = None,
        use_gpu: bool = True,
        suppress_stdouterr: bool = False,
) -> ndarray:
    xp = get_ndarray_module(k_data)

    # depending on input, read coil sensitivities
    if coil_sens is None:
        assert coil_sens_path is not None, 'Either coil_sens or coil_sens_path must be given'
        coil_sens = coil_sensitivities.read(  # [x, y, coil, map, frame, slice]
            coil_sens_path, nmaps=coil_sens_nmaps, layout='bart'
        )
    else:
        coil_sens = coil_sens[:, :, :, :coil_sens_nmaps]  # [x, y, coil, map, frame, slice]

    # move to cpu if cuda/cupy is available
    if cp is not None:
        k_data = cp.asnumpy(k_data)
        coil_sens = cp.asnumpy(coil_sens)

    # expand dimensionality to BART format
    # [kx, ky, kz=1, coil, map, ?, ?, ?, ?, ?, frame, ?, ?, slice]
    k_data = k_data[:, :, None, :, None, None, None, None, None, None, :, None, None, :]
    coil_sens = coil_sens[:, :, None, :, :, None, None, None, None, None, :, None, None, :]  # type: ignore

    # normalize
    norm = np.linalg.norm(k_data)
    k_data /= norm

    # run bart pics
    if isinstance(regularizer, str):
        regularizer = [regularizer]
    command = f'pics -R {" -R ".join(regularizer)} -S'
    if cp is not None and use_gpu:
        command += ' -g'
    recon = bart.bart(1, command, k_data, coil_sens, suppress_stdouterr=suppress_stdouterr)
    assert isinstance(recon, np.ndarray), 'BART did not return a numpy array'
    recon = recon.squeeze()  # [x, y, (map), (frame), (slice)]
    if coil_sens.shape[4] > 1:
        recon = recon[:, :, 0]  # [x, y, (frame), (slice)]
    if k_data.shape[10] == 1:
        recon = recon[:, :, None]  # [x, y, frame=1, (slice)]
    if k_data.shape[13] == 1:
        recon = recon[..., None]  # [x, y, frame, slice=1]

    # un-normalize
    recon *= norm

    # crop phase padding
    if phase_crop is not None and phase_crop < recon.shape[1]:
        phase_start = recon.shape[1] // 2 - phase_crop // 2
        phase_end = phase_start + phase_crop
        recon = recon[:, phase_start:phase_end]

    # save reconstruction
    if recon_path is not None:
        os.makedirs(recon_path.parent, exist_ok=True)

        # save as h5
        recon_ = recon.squeeze(2) if recon.shape[2] == 1 else recon
        write(recon_, recon_path)

        # save as png/gif
        if save_img:
            for s in range(recon_.shape[-1]):
                img_filename = recon_path.parent / f'reconstruction_slice{s:02d}'
                utils.save(recon_[..., s].transpose(2, 0, 1), img_filename, clip=True)

    return xp.asarray(recon)


def reconstruct_file(
        in_file: Path,
        coil_sens_dir: Path,
        regularizer: Sequence[str],
        out_dir: Optional[Path] = None,
        subsampling_options: Optional[Dict] = None,
        keep_phase_oversampling: bool = False,
        use_gpu: bool = True,
        save_img: bool = True,
) -> np.ndarray:
    with h5py.File(in_file, 'r') as hf:
        # extract ISMRMRD header
        header = ismrmrd.xsd.CreateFromDocument(hf['ismrmrd_header'][()])  # type: ignore
        enc = header.encoding[0]

        # phase oversampling
        if not keep_phase_oversampling:
            if enc.reconSpace.matrixSize.x == enc.encodedSpace.matrixSize.x:  # type: ignore
                # 2D interpolation on
                target_phase_dim = enc.reconSpace.matrixSize.y // 2  # type: ignore
            else:
                # 2D interpolation off
                target_phase_dim = enc.reconSpace.matrixSize.y  # type: ignore
        else:
            target_phase_dim = enc.encodedSpace.matrixSize.y  # type: ignore

        # prepare k-space data
        k_data = np.array(hf['kspace'])  # [slice, coil, (frame), kx, ky]
        if k_data.ndim == 4:  # static data
            k_data = k_data[:, :, None, :, :]  # [slice, coil, frame=1, kx, ky]
        k_data = np.moveaxis(k_data, [3, 4, 0], [0, 1, 4])  # [kx, ky, coil, frame, slice]

    if subsampling_options is not None:
        enc_mat_x: int = enc.encodedSpace.matrixSize.x  # type: ignore
        enc_mat_y: int = enc.encodedSpace.matrixSize.y  # type: ignore
        enc_fov_x: float = enc.encodedSpace.fieldOfView_mm.x  # type: ignore
        enc_fov_y: float = enc.encodedSpace.fieldOfView_mm.y  # type: ignore
        if round((enc_fov_x / enc_fov_y) / (enc_mat_x / enc_mat_y)) == 2:
            # for some reason, this step is necessary for some OCMR Free.Max datasets
            enc_mat_y //= 2
        enc_lim_center = enc.encodingLimits.kspace_encoding_step_1.center  # type: ignore
        enc_lim_max = enc.encodingLimits.kspace_encoding_step_1.maximum  # type: ignore
        phase_padding_left = enc_mat_y // 2 - enc_lim_center
        phase_padding_right = enc_mat_y - phase_padding_left - enc_lim_max - 1
        num_frames = k_data.shape[3]
        num_cols = k_data.shape[1] - max(0, phase_padding_right) - max(0, phase_padding_left)
        mask_unpadded = subsampling.get_mask(  # [frame, ky]
            num_frames, num_cols, subsampling_options['mask_type'], subsampling_options['acceleration'],
            subsampling_options['center_fraction'], padding_left=phase_padding_left, padding_right=phase_padding_right
        )
        mask_unpadded = mask_unpadded.T  # [ky, frame]
        mask = np.ones((k_data.shape[1], num_frames))  # [ky, frame]
        mask[max(0, phase_padding_left):mask.shape[0] - max(0, phase_padding_right), :] = mask_unpadded
        k_data = k_data * mask[None, :, None, :, None]  # [kx, ky, coil, frame, slice]

    # determine paths to reconstruction and coil sensitivities
    if re.search(r'_frame\d{2}', in_file.stem):
        subject, frame = in_file.stem.rsplit('_', 1)
        recon_path = out_dir / subject / f'reconstruction_{frame}.h5' if out_dir is not None else None
        coil_sens_path = coil_sens_dir / subject / f'coil_sens_{frame}.h5'
    else:
        recon_path = out_dir / in_file.stem / 'reconstruction.h5' if out_dir is not None else None
        coil_sens_path = coil_sens_dir / in_file.stem / 'coil_sens_avg.h5'

    # call bart pics
    reconstruction_cs = bart_pics(  # [x, y, frame, slice]
        k_data, regularizer, coil_sens_path=coil_sens_path, recon_path=recon_path, save_img=save_img,
        phase_crop=target_phase_dim, use_gpu=use_gpu, suppress_stdouterr=False
    )

    # move some axes
    reconstruction_cs = np.moveaxis(reconstruction_cs, [2, 3], [1, 0])  # [slice, frame, x, y]

    # squeeze frame axis if it's singleton
    if reconstruction_cs.shape[1] == 1:
        reconstruction_cs = reconstruction_cs.squeeze(axis=1)  # [slice, x, y]

    return reconstruction_cs


def reconstruct_dir(
        in_dir: Path,
        coil_sens_dir: Path,
        out_dir: Path,
        regularizer: Sequence[str],
        subsampling_options: Optional[Dict] = None,
        keep_phase_oversampling: bool = False,
        use_gpu: bool = True,
        save_img: bool = True,
) -> None:
    for file in sorted(in_dir.rglob('*.h5')):
        reconstruct_file(
            file, coil_sens_dir, regularizer, out_dir=out_dir, subsampling_options=subsampling_options,
            keep_phase_oversampling=keep_phase_oversampling, use_gpu=use_gpu, save_img=save_img
        )


def _reconstruct_and_compare(
        in_file: Path,
        coil_sens_dir: Path,
        regularizer: Sequence[str],
        subsampling_options: Dict,
        use_gpu: bool = True,
) -> float:
    # CS reconstruction
    recon = reconstruct_file(  # [slice, (frame), x, y]
        in_file, coil_sens_dir, regularizer, out_dir=None, subsampling_options=subsampling_options,
        keep_phase_oversampling=False, use_gpu=use_gpu, save_img=False,
    )
    recon = np.abs(recon)

    # ground truth reconstruction
    with h5py.File(in_file, 'r') as hf:
        recon_gt = np.array(hf['reconstruction_weighted'])  # [slice, (frame), x, y]
    recon_gt = np.abs(recon_gt)

    assert recon.shape == recon_gt.shape

    # SSIM
    num_slices = recon.shape[0]
    ssim = 0
    for s in range(num_slices):
        ssim += structural_similarity(recon_gt[s], recon[s], data_range=np.max(recon_gt))
    ssim /= num_slices

    return ssim


def _suggest_weights(trials: optuna.Trial, regularizer: Sequence[str]) -> List[str]:
    regularizer_out = []
    for reg in regularizer:
        reg = reg.split(':')
        if len(reg) > 2 and 'x' in reg[2]:
            # this allows to specify multiple regularizers with dependent
            # weights. However, the first regularizer must not have a weight.
            # Dependent regularizers will always use the weight of the first
            # regularizer.
            # Example: ['T:3', 'T:1024:5x'] means that the weight of the second
            # regularizer is 5 times the weight of the first regularizer.
            assert len(regularizer_out) > 0, 'Regularizer with factor must be specified after regularizer without factor'
            factor = float(reg[2].split('x')[0])
            weight = float(regularizer_out[0].split(':')[-1]) * factor
        else:
            weight = trials.suggest_float(f'{reg[0]}_{reg[1]}', 1e-8, 1e-3, log=True)
        regularizer_out.append(f'{reg[0]}:{reg[1]}:0:{weight}')
    return regularizer_out


def _objective_individual(
        trial: optuna.Trial,
        in_file: Path,
        coil_sens_dir: Path,
        regularizer: Sequence[str],
        subsampling_options: Sequence[Dict],
        use_gpu: bool = True,
) -> float:
    regularizer = _suggest_weights(trial, regularizer)
    ssim = 0
    counter = 0
    for subsampling_options_ in subsampling_options:
        ssim += _reconstruct_and_compare(in_file, coil_sens_dir, regularizer, subsampling_options_, use_gpu=use_gpu)
        counter += 1
    ssim /= counter
    return ssim


def _objective_average(
        trial: optuna.Trial,
        in_dir: Path,
        coil_sens_dir: Path,
        regularizer: Sequence[str],
        subsampling_options: Sequence[Dict],
        use_gpu: bool = True,
) -> float:
    regularizer = _suggest_weights(trial, regularizer)
    ssim = 0
    counter = 0
    files = sorted(in_dir.rglob('*.h5'))
    for file in files:
        for subsampling_options_ in subsampling_options:
            ssim += _reconstruct_and_compare(file, coil_sens_dir, regularizer, subsampling_options_, use_gpu=use_gpu)
            counter += 1
    ssim /= counter
    return ssim


def optimize_pics_params(
        scope: str,
        in_dir: Path,
        coil_sens_dir: Path,
        study_name: str,
        regularizer: Sequence[str] | str,
        subsampling_options: Sequence[Dict],
        use_gpu: bool = True,
        n_trials: int = 20,
) -> None:
    if scope == 'individual':
        for file in sorted(in_dir.rglob('*.h5')):
            print(f'Optimizing parameters for file {file.stem}')

            study = optuna.create_study(
                storage=os.path.expandvars('sqlite:///pics_optimization/optuna_pics_params.db'),
                direction='maximize',
                study_name=f'{study_name}/{file.stem}',
                load_if_exists=True,
            )
            completed_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
            if len(completed_trials) < n_trials:
                study.optimize(
                    lambda trial: _objective_individual(
                        trial, file, coil_sens_dir, regularizer, subsampling_options, use_gpu=use_gpu
                    ),
                    n_trials=n_trials - len(completed_trials)
                )

            best_trial = study.best_trial
            print('  Best value: ', best_trial.value)
            print('  Best params: ', best_trial.params)

    elif scope == 'average':
        study = optuna.create_study(
            storage=os.path.expandvars('sqlite:///pics_optimization/optuna_pics_params.db'),
            direction='maximize',
            study_name=study_name,
            load_if_exists=True,
        )
        completed_trials = study.get_trials(deepcopy=False, states=(optuna.trial.TrialState.COMPLETE,))
        if len(completed_trials) < n_trials:
            study.optimize(
                lambda trial: _objective_average(
                    trial, in_dir, coil_sens_dir, regularizer, subsampling_options, use_gpu=use_gpu
                ),
                n_trials=n_trials - len(completed_trials)
            )

        best_trial = study.best_trial
        print('  Best value: ', best_trial.value)
        print('  Best params: ', best_trial.params)

    else:
        raise ValueError(f'Invalid parameter for `scope`: {scope}')
