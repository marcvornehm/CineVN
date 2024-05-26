#  Copyright (c) 2023. Marc Vornehm <marc.vornehm@fau.de>.

import logging
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple, Union

try:
    import cupy as cp
except ImportError:
    # importing cupy fails if no cuda is available
    cp = None
with warnings.catch_warnings():  # ismrmrd adds a warning filter that causes all subsequent warnings to be printed even if they should be filtered
    import ismrmrd
import numpy as np
import sigpy.mri

from ..ndarray import get_ndarray_module, ndarray


class Dataset(ABC):
    split: str
    hdr: ismrmrd.xsd.ismrmrdHeader
    ecg: Optional[np.ndarray]
    _slice_idx: Optional[int]
    k_data_full: np.ndarray
    k_data: ndarray
    has_read_os: bool

    def __init__(self, name: str, filename: Path, device: str, rep_to_frame_dim: bool = False):
        self.name = name
        self.filename = filename
        self._device = device
        self.rep_to_frame_dim = rep_to_frame_dim

        self._read_dir: dict[int, np.ndarray] = {}
        self._phase_dir: dict[int, np.ndarray] = {}
        self._slice_dir: dict[int, np.ndarray] = {}
        self._position: dict[int, np.ndarray] = {}

        self.norm_orientation_rot: Optional[int] = None
        self.norm_orientation_hor: Optional[int] = None
        self.norm_orientation_ver: Optional[int] = None

        self.noise_cov: Optional[np.ndarray] = None

    def read_meta(self):
        self.read_header()
        self.read_ecg()

    @abstractmethod
    def read_header(self):
        raise NotImplementedError

    @abstractmethod
    def read_ecg(self):
        raise NotImplementedError

    def read_kdata(self, whiten: bool = True, select_slice: Optional[int] = None):
        self._read_kdata(select_slice=select_slice)
        if self.rep_to_frame_dim:
            self._move_rep_to_frame_dim()
        self._slice_idx = select_slice
        self.device = self._device
        if whiten:
            self._whiten_kdata()

    @abstractmethod
    def _read_kdata(self, **kwargs):
        raise NotImplementedError

    def _move_rep_to_frame_dim(self):
        self.k_data = self._rep2frame(self.k_data)
        self.k_data_full = self._rep2frame(self.k_data_full)

    @staticmethod
    def _rep2frame(k_data: np.ndarray) -> np.ndarray:
        k_data = np.moveaxis(k_data, 7, 4)  # [kx, ky, kz, coil, rep, frame, set, slice, avg]
        k_data = np.expand_dims(k_data, 8)  # [kx, ky, kz, coil, rep, frame, set, slice, 1, avg]
        new_shape = (*k_data.shape[:4], -1, *k_data.shape[6:])
        k_data = k_data.reshape(*new_shape)  # [kx, ky, kz, coil, rep*frame, set, slice, 1, avg]
        return k_data

    def _whiten_kdata(self):
        if self.noise_cov is None:
            return
        xp = get_ndarray_module(self.k_data)
        k_data_temp = xp.moveaxis(self.k_data, 3, 0)  # [coil, kx, ky, kz, ...]
        norm = xp.linalg.norm(k_data_temp)
        try:
            k_data_whitened = sigpy.mri.whiten(k_data_temp, xp.asarray(self.noise_cov))  # whiten
            if xp.isnan(k_data_whitened).any():
                raise np.linalg.LinAlgError  # error is only automatically raised in numpy, cupy returns nans
            k_data_temp = k_data_whitened
        except np.linalg.LinAlgError:  # noise covariance matrix is not positive definite
            logging.warning('Whitening failed, skipping')
        k_data_temp = k_data_temp / xp.linalg.norm(k_data_temp) * norm  # type: ignore  # re-normalize
        k_data_temp = k_data_temp.astype(self.k_data.dtype)
        self.k_data = xp.moveaxis(k_data_temp, 0, 3)  # [kx, ky, kz, coil, ...]

    @property
    def device(self) -> str:
        return self._device

    @device.setter
    def device(self, device):
        if device == 'cuda':
            assert cp is not None, 'Setting device to `cuda` failed because was not able to import cupy'
            self.k_data = cp.asarray(self.k_data)
        elif device == 'cpu':
            if cp is not None:  # only if cuda/cupy is available, otherwise it should already be a np.ndarray
                self.k_data = cp.asnumpy(self.k_data)
        else:
            raise RuntimeError(f'Unknown device {device}')
        self._device = device

    @property
    def n_slices(self) -> int:
        try:
            return self.hdr.encoding[0].encodingLimits.slice.maximum + 1  # type: ignore
        except AttributeError:
            raise RuntimeError('Number of slices could not be determined from ISMRMRD header')

    @property
    def slice_idx(self) -> Optional[int]:
        return self._slice_idx

    def select_slice(self, slice_idx: int):
        try:
            self.k_data = self.k_data_full[:, :, :, :, :, :, slice_idx, None]
            self._slice_idx = slice_idx
            self.device = self._device
        except IndexError as e:
            raise IndexError(f'Slice index {slice_idx} out of range. The selected slice was probably not loaded.' \
                             'Please select it using `read_kdata(select_slice=slice_idx)`') from e

    def get_read_dir(self, slice_idx: Optional[int]) -> Optional[np.ndarray]:
        if slice_idx in self._read_dir:
            return self._read_dir[slice_idx]
        elif slice_idx is None:
            return self._read_dir[0]
        else:
            return None

    @property
    def read_dir(self) -> Optional[np.ndarray]:
        return self.get_read_dir(self.slice_idx)

    def get_phase_dir(self, slice_idx: Optional[int]) -> Optional[np.ndarray]:
        if slice_idx in self._phase_dir:
            return self._phase_dir[slice_idx]
        elif slice_idx is None:
            return self._phase_dir[0]
        else:
            return None

    @property
    def phase_dir(self) -> Optional[np.ndarray]:
        return self.get_phase_dir(self.slice_idx)

    def get_slice_dir(self, slice_idx: Optional[int]) -> Optional[np.ndarray]:
        if slice_idx in self._slice_dir:
            return self._slice_dir[slice_idx]
        elif slice_idx is None:
            return self._slice_dir[0]
        else:
            return None

    @property
    def slice_dir(self) -> Optional[np.ndarray]:
        return self.get_slice_dir(self.slice_idx)

    def get_position(self, slice_idx: Optional[int]) -> Optional[np.ndarray]:
        if slice_idx in self._position:
            return self._position[slice_idx]
        elif slice_idx is None:
            return self._position[0]
        else:
            return None

    @property
    def position(self) -> Optional[np.ndarray]:
        return self.get_position(self.slice_idx)

    def set_norm_orientation(self, rot: int, hor: int, ver: int):
        self.norm_orientation_rot = rot
        self.norm_orientation_hor = hor
        self.norm_orientation_ver = ver

    def norm_orientation(self, img: ndarray, image_axes: Tuple[int, int]) -> np.ndarray:
        xp = get_ndarray_module(img)
        if self.norm_orientation_rot is not None and self.norm_orientation_rot > 0:
            img = xp.rot90(img, self.norm_orientation_rot // 90, axes=image_axes)
        if self.norm_orientation_hor:
            img = xp.flip(img, axis=image_axes[1])
        if self.norm_orientation_ver:
            img = xp.flip(img, axis=image_axes[0])
        return img

    def estimate_noise(self, return_noise_data: bool = False) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        if self.k_data is None:
            raise RuntimeError('Noise can only be estimated after k-space data is read')

        noise_data = np.concatenate([
            self.k_data[:min(16, round(self.k_data.shape[0] / 8))],
            self.k_data[-min(16, round(self.k_data.shape[0] / 8)):]
        ], axis=0)
        noise_data = np.concatenate([
            noise_data[:, :round(self.k_data.shape[1] / 5)],
            noise_data[:, -round(self.k_data.shape[1] / 5):]
        ], axis=1)
        noise_data = np.concatenate([
            noise_data[:, :, :round(self.k_data.shape[2] / 5)],
            noise_data[:, :, -round(self.k_data.shape[2] / 5):]
        ], axis=2)

        noise_data = np.moveaxis(noise_data, 3, 0)
        noise_data = np.reshape(noise_data, (noise_data.shape[0], -1))
        noise_std = np.std(noise_data, axis=1, where=noise_data != 0)  # type: ignore

        if return_noise_data:
            noise_data = np.stack([n[np.abs(n) > 0] for n in noise_data], axis=0)  # skip data that is not sampled
            return noise_std, noise_data
        else:
            return noise_std
