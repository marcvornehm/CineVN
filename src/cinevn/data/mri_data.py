"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.

Part of this source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import logging
import pickle
import random
import warnings
from pathlib import Path
from typing import Callable

import h5py
with warnings.catch_warnings():  # ismrmrd adds a warning filter that causes all subsequent warnings to be printed even if they should be filtered
    import ismrmrd
import numpy as np
import torch
import torch.utils.data

from .transforms import VarNetDataTransform


def create_minimal_header(kspace: np.ndarray | torch.Tensor) -> ismrmrd.xsd.ismrmrdHeader:
    """Creates a minimal ISMRMRD header from k-space data. The header
    contains only the encoding information and a dummy TR value.

    Args:
        kspace (np.ndarray | torch.Tensor): k-space data.

    Returns:
        ismrmrd.xsd.ismrmrdHeader: Minimal ISMRMRD header.
    """
    warnings.warn(
        'Creating minimal ISMRMRD header. This may lead to incorrect results. Please provide a real header instead.',
        category=UserWarning,
    )
    encodedSpace = ismrmrd.xsd.encodingSpaceType(
        matrixSize=ismrmrd.xsd.matrixSizeType(
            x=kspace.shape[-2] * 2,  # 200% readout oversampling
            y=kspace.shape[-1],
            z=1,
        ),
        fieldOfView_mm=ismrmrd.xsd.fieldOfViewMm(
            x=360,
            y=360 * kspace.shape[-1] / kspace.shape[-2] * 2,
            z=1,
        ),
    )
    reconSpace = ismrmrd.xsd.encodingSpaceType(
        matrixSize=ismrmrd.xsd.matrixSizeType(
            x=kspace.shape[-2],
            y=kspace.shape[-1],
            z=1,
        ),
        fieldOfView_mm=ismrmrd.xsd.fieldOfViewMm(
            x=360,
            y=360 * kspace.shape[-1] / kspace.shape[-2],
            z=1,
        ),
    )
    encodingLimits = ismrmrd.xsd.encodingLimitsType(
        kspace_encoding_step_1=ismrmrd.xsd.limitType(
            minimum=0,
            maximum=kspace.shape[-1] - 1,
            center=kspace.shape[-1] // 2,
        ),
    )
    encoding = ismrmrd.xsd.encodingType(
        encodedSpace=encodedSpace,
        reconSpace=reconSpace,
        encodingLimits=encodingLimits,
    )
    sequenceParameters = ismrmrd.xsd.sequenceParametersType(TR=[50])
    header = ismrmrd.xsd.ismrmrdHeader(
        encoding=[encoding],
        sequenceParameters=sequenceParameters,
    )
    return header


class SliceDataset(torch.utils.data.Dataset):
    """
    A PyTorch Dataset that provides access to MR image slices.
    """

    def __init__(
            self,
            root: str | Path,
            coil_sens_path: Path,
            recons_key: str = 'reconstruction_weighted',
            transform: Callable | None = None,
            use_dataset_cache: bool = False,
            sample_rate: float | None = None,
            volume_sample_rate: float | None = None,
            dataset_cache_file: str | Path = 'dataset_cache.pkl',
            num_espirit_maps: int = 1,
    ):
        """
        Args:
            root: Path to the dataset.
            coil_sens_path: Path to coil sensitivity maps.
            recons_key: Optional; The key in the HDF5 file that contains the
                reconstruction. Defaults to `reconstruction_weighted`.
            transform: Optional; A callable object that pre-processes the raw
                data into appropriate form. The transform function should take
                `kspace`, `target`, `attributes`, `filename`, and `slice` as
                inputs. `target` may be null for test data. If not given, a
                default transform will be used.
            use_dataset_cache: Whether to cache dataset metadata. This is very
                useful for large datasets like the brain data.
            sample_rate: Optional; A float between 0 and 1. This controls what
                fraction of the slices should be loaded. Defaults to 1 if no
                value is given. When creating a sampled dataset either set
                sample_rate (sample by slices) or volume_sample_rate (sample by
                volumes) but not both.
            volume_sample_rate: Optional; A float between 0 and 1. This controls
                what fraction of the volumes should be loaded. Defaults to 1 if
                no value is given. When creating a sampled dataset either set
                sample_rate (sample by slices) or volume_sample_rate (sample by
                volumes) but not both.
            dataset_cache_file: Optional; A file in which to cache dataset
                information for faster load times.
            num_espirit_maps: The number of sets of ESPIRiT coil sensitivity
                maps to use (Soft-SENSE). Usually 1 or 2. Ignored if
                coil_sens_path is None.
        """
        if sample_rate is not None and volume_sample_rate is not None:
            raise ValueError('either set sample_rate (sample by slices) or volume_sample_rate (sample by volumes) but '
                             'not both')

        if transform is None:
            transform = VarNetDataTransform()

        dataset_cache_file = Path(dataset_cache_file)
        self.coil_sens_path = coil_sens_path
        self.num_espirit_maps = num_espirit_maps
        self.transform = transform
        self.recons_key = recons_key
        self.examples = []

        # set default sampling mode if none given
        if sample_rate is None:
            sample_rate = 1.0
        if volume_sample_rate is None:
            volume_sample_rate = 1.0

        # load dataset cache if we have and user wants to use it
        if dataset_cache_file.exists() and use_dataset_cache:
            with open(dataset_cache_file, 'rb') as f:
                dataset_cache = pickle.load(f)
        else:
            dataset_cache = {}

        # check if our dataset is in the cache
        # if there, use that metadata, if not, then regenerate the metadata
        if dataset_cache.get(str(root)) is None or not use_dataset_cache:
            if Path(root).exists():
                files = list(Path(root).iterdir())
                for fname in sorted(files):
                    shapes = self._retrieve_shapes(fname)
                    num_slices, tensor_sizes = shapes[0], shapes[1:]
                    self.examples += [(fname, slice_ind, tensor_sizes) for slice_ind in range(num_slices)]

                if dataset_cache.get(str(root)) is None and use_dataset_cache:
                    dataset_cache[str(root)] = self.examples
                    logging.info(f'Saving dataset cache to {dataset_cache_file}.')
                    with open(dataset_cache_file, 'wb') as f:
                        pickle.dump(dataset_cache, f)
            else:
                logging.info(f'Skipping loading of dataset because {root} does not exists.')
        else:
            logging.info(f'Using dataset cache from {dataset_cache_file}.')
            self.examples = dataset_cache[str(root)]

        # subsample if desired
        if sample_rate < 1.0:  # sample by slice
            random.shuffle(self.examples)
            num_examples = round(len(self.examples) * sample_rate)
            self.examples = self.examples[:num_examples]
        elif volume_sample_rate < 1.0:  # sample by volume
            vol_names = sorted(list(set([f[0].stem for f in self.examples])))
            random.shuffle(vol_names)
            num_volumes = round(len(vol_names) * volume_sample_rate)
            sampled_vols = vol_names[:num_volumes]
            self.examples = [ex for ex in self.examples if ex[0].stem in sampled_vols]

    def _retrieve_shapes(self, fname) -> tuple[int, ...]:
        with h5py.File(fname, 'r') as hf:
            kspace: h5py.Dataset = hf['kspace']  # type: ignore
            recon: h5py.Dataset = hf.get(self.recons_key)  # type: ignore
            shapes = kspace.shape + (recon.shape[1:] if recon is not None else ())

        return shapes

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i: int):
        fname, dataslice, _ = self.examples[i]

        # load data from h5 file
        with h5py.File(fname, 'r', locking=False) as hf:
            kspace = np.asarray(hf['kspace'])[dataslice]  # [coil, frame, readout, phase]

            mask = None
            if 'mask' in hf:
                mask = np.asarray(hf['mask'])  # [frame, phase]

            target = None
            if self.recons_key in hf:
                target = np.asarray(hf[self.recons_key])[dataslice]  # [frame, y, x]

            if 'ismrmrd_header' in hf:
                header = ismrmrd.xsd.CreateFromDocument(hf['ismrmrd_header'][()])  # type: ignore
            else:
                header = create_minimal_header(kspace)

            attrs = dict(hf.attrs)
            attrs.update({'fname': fname.name})

        # load coil sensitivity data
        coil_sens_file = self.coil_sens_path / fname.stem / 'coil_sens_avg.h5'
        with h5py.File(coil_sens_file, 'r', locking=False) as hf:
            sens_maps = np.asarray(hf['coil_sens'])[dataslice]  # [map, coil, frame=1, y, x]

            # detect full-zeros sensitivity maps (due to clipping and no aliasing)
            sens_maps_nonzero = sens_maps.reshape(sens_maps.shape[0], -1).any(axis=1)
            if not all(sens_maps_nonzero):
                max_maps_idx = sens_maps_nonzero.argmin()
            else:
                max_maps_idx = sens_maps.shape[0]

            # number of extracted sets of coil sensitivity maps is determined either by the corresponding parameter
            # or by the number of non-zero coil sensitivity maps
            sens_maps = sens_maps[:min(self.num_espirit_maps, max_maps_idx)]

            if np.all(sens_maps == 0):
                raise ValueError(f'Coil sensitivities are all zeros for file {fname.stem}')

        return self.transform(kspace, mask, sens_maps, target, header, attrs, dataslice, i)
