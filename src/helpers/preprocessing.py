"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import itertools
import logging
from pathlib import Path
from typing import Any

import boto3
try:
    import cupy as cp
except ImportError:  # importing cupy fails if no cuda is available
    cp = None
import h5py
import numpy as np
import pandas as pd
from botocore import UNSIGNED
from botocore.client import Config

import bart

from . import mri
from .datasets import Dataset, IsmrmrdDataset
from .ndarray import ndarray
from .save import save_movie


class Preprocessing:
    # variable that stores all neccessary columns of a csv file that
    # keeps data set information
    REQ_CSV_COLS = [
        'file name', 'scn', 'smp', 'ech', 'dur', 'viw', 'sli', 'fov', 'sub', 'slices',  # OCMR columns
        'excluded slices',  # slices to exclude from processing (e.g. '0-2+4' for slices 0, 1, 2, 4)
        'mask type',  # mask
        'FOV x', 'FOV y', 'FOV z',  # field of view
        'frames', 'coils', 'averages', 'kx', 'ky acq', 'ky enc', 'ky rec',  # kspace dimensionality
        'TRes',  # temporal resolution
        'HR',  # heart rate
        'protocol name',  # protocol name
        'split',  # train/val/test set
        'viw 2',  # view (additional column for 2ch/3ch/4ch distinction)
        'rotation', 'flip horizontal', 'flip vertical',  # operations to be applied to fix image orientation
        'center x', 'center y', 'axis 1 x', 'axis 1 y', 'axis 2 x', 'axis 2 y',  # profiles
        'bbox x low', 'bbox y low', 'bbox x high', 'bbox y high',  # ROI bounding box
    ]

    def __init__(
            self,
            csv_file: Path,
            data_source: str,
            raw_dir: Path,
            target_dir: Path,
            coil_sens_dir: Path,
            cs_dir: Path | None = None,
            csv_query: str | None = None,
            csv_idx_range: list[int | None] = [0, None],
            accelerations: list[int] | None = None,
            mask_types: list[str] | None = None,
            coil_sens_backend: str = 'sigpy',
            n_maps: int = 1,
            try_cuda: bool = True,
            seed: int = 42,
    ):
        """
        Args:
            csv_file (Path): Path to csv file which contains metadata of the
                samples to process
            data_source (str): type of dataset (only `ocmr` allowed for now)
            raw_dir (Path): path to raw unprocessed data files
            target_dir (Path): path to store processed data
            coil_sens_dir (Path): path to store/load coil sensitivities from
            cs_dir (Path | None, optional): path to compressed sensing
                reconstructions. If None no CS recon is performed. Defaults to
                None.
            csv_query (str | None, optional): Query string to filter samples
                from the csv table for processing (see
                https://datagy.io/pandas-query/). Defaults to None.
            csv_idx_range (list[int  |  None], optional): Range of indices to
                filter samples for processing. Defaults to [0, None].
            accelerations (list[int] | None, optional): Acceleration rates for
                retrospective undersampling. Defaults to None.
            mask_types (list[str] | None, optional): Mask types for
                retrospective undersampling. Defaults to None.
            coil_sens_backend (str, optional): Which backend to use to compute
                coil sensitivities (sigpy | bart). Defaults to 'sigpy'.
            n_maps (int, optional): Number of coil sensitivity maps to estimate.
                Defaults to 1.
            try_cuda (bool, optional): Try to use gpu if available. Defaults to
                True.
            seed (int, optional): Seed for random number generator. Defaults to
                42.
        """
        assert csv_file.is_file(), f'{csv_file} is not a file.'
        self.csv_file = csv_file
        self.data_source = data_source
        self.raw_dir = raw_dir
        self.target_dir = target_dir
        self.coil_sens_dir = coil_sens_dir
        self.cs_dir = cs_dir
        self.accelerations = [] if accelerations is None else accelerations
        self.mask_types = [] if mask_types is None else mask_types
        self.coil_sens_backend = coil_sens_backend
        self.n_maps = n_maps
        self.device = 'cpu'
        if try_cuda and cp is not None and cp.cuda.is_available():
            self.device = 'cuda'
        self.rng = np.random.RandomState(seed)

        # read full csv file with all samples of this data source
        self.df_full = pd.read_csv(csv_file, dtype={'excluded slices': str})
        # check if df has all required cols
        missing_cols = [c for c in self.REQ_CSV_COLS if c not in self.df_full.columns]
        assert len(missing_cols) == 0, f'Dataframe misses the following cols: {missing_cols}'

        # filter samples according to arguments
        self.df = self.df_full.loc[csv_idx_range[0]:csv_idx_range[1]]
        if csv_query:
            self.df = self.df.query(csv_query)

    @staticmethod
    def update_csv(df: pd.DataFrame, dset: Dataset, csv_file: Path) -> dict[str, Any]:
        """
        Gets all information from the dset object and fills it into the csv file

        Args:
            df (pd.DataFrame): dataframe containing infos on multiple samples (datasets)
            dset (Dataset): dataset which information is filled in the csv file
            csv_file (Path): csv file that stores infos from the df dataframe

        Returns:
            dict: informatino of the dset as dictionary
        """
        row = df['file name'] == dset.filename.name
        assert sum(row) == 1

        hdr = dset.hdr
        enc = hdr.encoding[0]
        t_res = hdr.sequenceParameters.TR[0]  # type: ignore
        n_frames = enc.encodingLimits.phase.maximum + 1  # type: ignore

        df.loc[row, 'FOV x'] = enc.encodedSpace.fieldOfView_mm.x  # type: ignore
        df.loc[row, 'FOV y'] = enc.encodedSpace.fieldOfView_mm.y  # type: ignore
        df.loc[row, 'FOV z'] = enc.encodedSpace.fieldOfView_mm.z  # type: ignore
        df.loc[row, 'frames'] = n_frames
        df.loc[row, 'coils'] = hdr.acquisitionSystemInformation.receiverChannels  # type: ignore
        df.loc[row, 'averages'] = enc.encodingLimits.average.maximum + 1  # type: ignore
        df.loc[row, 'kx'] = enc.encodedSpace.matrixSize.x  # type: ignore
        df.loc[row, 'ky acq'] = enc.encodingLimits.kspace_encoding_step_1.maximum + 1  # type: ignore
        df.loc[row, 'ky enc'] = enc.encodedSpace.matrixSize.y  # type: ignore
        df.loc[row, 'ky rec'] = enc.reconSpace.matrixSize.y  # type: ignore
        df.loc[row, 'TRes'] = round(t_res, 2)
        if df.loc[row, 'smp'].item() in ['fs']:
            # heart rate can only be computed in segmented acquisitions with exactly one heart cycle
            df.loc[row, 'HR'] = round(60 / (t_res * n_frames / 1000))
        df.loc[row, 'protocol name'] = hdr.measurementInformation.protocolName  # type: ignore
        df.to_csv(csv_file, index=False)

        return df.loc[row].iloc[0].to_dict()

    def load_dataset(self, raw_file: Path, data_attrs: dict[str, Any]) -> tuple[Dataset, dict[str, Any]]:
        """
        Takes file path of raw data and additional attributes of the data
        sample to return a Dataset and the updated sample attributes

        Args:
            raw_file (Path): path to raw data file
            data_attrs (dict[str, Any]): sample attributes. Required keys:
                `split`, `rotation`, `flip horizontal`, `flip vertical`, `dur`,
                `frames` and `sli`

        Raises:
            ValueError: If one of the requires keys not in data_attrs dict
            ValueError: If given self.data_source is not defined

        Returns:
            tuple[Dataset, dict[str, Any]]: Dataset and updated data attributes
            dictionary
        """
        req_keys = ('split', 'rotation', 'flip horizontal', 'flip vertical', 'dur', 'frames', 'sli')
        for key in req_keys:
            assert key in data_attrs, f'{key} not in data_attrs'

        # init dataset (load it if neccessary)
        logging.info(f'Loading {raw_file}')
        if self.data_source.lower() == 'ocmr':
            # download dataset if it doesn't exist yet
            if not raw_file.exists():
                logging.info(f'Did not find {raw_file}. Downloading {raw_file.name} to {raw_file}')
                raw_file.parent.mkdir(exist_ok=True, parents=True)
                s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
                s3_client.download_file('ocmr', f'data/{raw_file.name}', str(raw_file))
            dset = IsmrmrdDataset(name=raw_file.name, filename=raw_file, device=self.device)

        else:
            raise ValueError(f'Unknown data source: {self.data_source}')

        # assign dataset split where to save it
        dset.split = str(data_attrs['split'])

        # read orientation normalization
        if all([pd.notna(data_attrs[k]) for k in ['rotation', 'flip horizontal', 'flip vertical']]):
            dset.set_norm_orientation(
                rot=int(data_attrs['rotation']),
                hor=int(data_attrs['flip horizontal']),
                ver=int(data_attrs['flip vertical'])
            )

        # read meta data
        logging.info('Reading metadata')
        dset.read_meta()

        # update the csv file with new information
        logging.info(f'Updating {self.csv_file}')
        data_attrs = self.update_csv(self.df_full, dset, self.csv_file)

        # preload only small datasets
        if data_attrs['dur'] == 'shr' and float(data_attrs['frames']) < 50:
            logging.info('Reading k-space data')
            dset.read_kdata(whiten=True, select_slice=None)

        return dset, data_attrs

    def get_sampling_patterns(self, acq_sampling: str, acq_mask_type: str, exclude: bool) -> list[mri.Sampling]:
        """
        Creates sampling pattern objects.
        1. The sampling of the original acquisition
        2. Additional sampling patterns that should be created for this data set

        Args:
            acq_sampling (str): how the acquisition was sampled
                (fs: fully sampled | pse: prospectively undersampled)
            acq_mask_type (str): which sampling mask was applied
            exclude (bool): if this sampled should be excluded from the data
                set beeing created and therefore there are no additional
                sampling masks generated

        Raises:
            ValueError: if the acq_sampling is unknown

        Returns:
            list[mri.Sampling]: list of mri.Sampling objects
        """
        # create list of sampling patterns, starting with the original acquisiton sampling pattern
        if acq_sampling.lower() == 'fs':
            sampling_patterns = [mri.Sampling(name='fs', acceleration=1)]
            if not exclude:  # if this dataset is not excluded, add retrospective undersampling patterns
                for mask_type, acc in itertools.product(self.mask_types, self.accelerations):
                    sampling_patterns.append(mri.Sampling(
                        name=f'{mask_type}_{acc:02d}', acceleration=acc, mask_type=mask_type,
                    ))
        elif acq_sampling.lower() == 'pse':
            sampling_patterns = [mri.Sampling(name='pse', mask_type=acq_mask_type)]
        else:
            raise ValueError(f'Unknown acq_sampling: {acq_sampling}')

        logging.info(f'Sampling Patterns: {[s.name for s in sampling_patterns]}')

        return sampling_patterns

    @staticmethod
    def phase_pad_dataset(dset: Dataset) -> tuple[Dataset, int, int]:
        """
        Pads the phase direction of the kspace of a given Dataset

        Args:
            dset (Dataset): dataset with kspace to pad

        Returns:
            tuple[Dataset, int, int]: (
                dataset with padded kspace,
                padding value left,
                padding value right
            )
        """
        # phase padding (phase resolution < 100% or PF) / cropping (phase resolution > 100%)
        enc = dset.hdr.encoding[0]
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

        logging.info(f'Phase padding: {phase_padding_left}, {phase_padding_right}')
        if phase_padding_left < 0 or phase_padding_right < 0:
            logging.warning(
                'Phase padding is negative (i.e., phase resolution > 100%)! The additional k-space lines will be '
                'removed and ignored in the following. This will likely lead to a lower effective acceleration '
                'rate in prospectively undersampled acquisitions and makes comparisons with the product '
                'reconstruction unfair in all sampling schemes!'
            )

        dset.k_data = mri.utils.apply_phase_padding(dset.k_data, phase_padding_left, phase_padding_right)

        return dset, phase_padding_left, phase_padding_right

    @staticmethod
    def get_recon_phase_size(dset: Dataset) -> int:
        """Determines the reconstruction size in phase dimension for a dataset

        Args:
            dset (Dataset): dataset to determine reconstruction size for

        Returns:
            int: Reconstruction size in phase dimension
        """

        # target size without phase oversampling (cropped after reconstruction)
        enc = dset.hdr.encoding[0]
        enc_mat_x: int = enc.encodedSpace.matrixSize.x  # type: ignore
        rec_mat_x: int = enc.reconSpace.matrixSize.x  # type: ignore
        rec_mat_y: int = enc.reconSpace.matrixSize.y  # type: ignore
        if rec_mat_x == enc_mat_x:
            # 2D interpolation on
            recon_phase_size = rec_mat_y // 2
        else:
            # 2D interpolation off
            recon_phase_size = rec_mat_y

        return recon_phase_size

    def mask_kspace(
            self,
            k_data: ndarray,
            sampling: mri.Sampling,
            pad_left: int,
            pad_right: int,
    ) -> tuple[ndarray, ndarray]:
        """
        Applies masking to a kspace array

        Args:
            k_data (ndarray): kdata to mask
            sampling (mri.Sampling): sampling pattern to apply
            pad_left (int): phase padding left
            pad_right (int): phase padding right

        Returns:
            tuple[ndarray, ndarray]: (
                kspace array after masking,
                applied mask
            )
        """
        xp = cp.get_array_module(k_data) if cp is not None else np

        if sampling.name == 'fs':  # fully sampled
            mask = xp.ones((k_data.shape[1], k_data.shape[3]), dtype=xp.uint8)  # [ky, frame]
            kdata_masked = xp.copy(k_data)  # [kx, ky, coil, frame, slice]

        elif sampling.name == 'pse':  # prospectively undersampled
            # obtain mask (zero-padded)
            mask = (abs(xp.mean(xp.abs(k_data), axis=2)) > 0).astype(xp.uint8)  # [kx, ky, frame, slice]

            # collapse kx and slice dimension of mask
            assert xp.all(mask.all(axis=3) | (1 - mask).all(axis=3)), 'Mask is not equal along slice dim'
            mask = mask[mask.shape[0] // 2, :, :, 0]  # [ky, frame]

            # remove zero-padding from mask (ignoring negative padding if phase resolution > 100%)
            mask_unpadded = mask[max(0, pad_left) : mask.shape[0] - max(0, pad_right), :]

            # calc acceleration rate on unpadded mask
            sampling.acceleration = int(mask_unpadded.size / mask_unpadded.sum())
            logging.info(f'Acceleration rate: {sampling.acceleration:.2f}')

            # pad mask with ones
            mask = xp.ones_like(mask)
            mask[max(0, pad_left) : mask.shape[0] - max(0, pad_right), :] = mask_unpadded

            # mask is already applied on k-space data
            kdata_masked = xp.copy(k_data)  # [kx, ky, coil, frame, slice]

        else:  # retrospective undersampling
            # obtain unpadded mask, ignoring negative padding values (phase resolution > 100%)
            num_cols = k_data.shape[1] - max(0, pad_right) - max(0, pad_left)
            assert sampling.mask_type is not None and sampling.acceleration is not None
            mask_unpadded = mri.get_mask(  # [frame, ky]
                k_data.shape[3], num_cols, sampling.mask_type, sampling.acceleration, padding_left=pad_left,
                padding_right=pad_right, rng=self.rng,
            )
            if cp is not None and isinstance(k_data, cp.ndarray):
                mask_unpadded = cp.asarray(mask_unpadded)
            mask_unpadded = mask_unpadded.T  # [ky, frame]

            # effective acc rate
            acc_eff = mask_unpadded.size / mask_unpadded.sum()
            logging.info(f'Effective Acceleration Rate: {acc_eff:5.2f}')

            # pad mask
            mask = xp.ones((k_data.shape[1], k_data.shape[3]), dtype=mask_unpadded.dtype)  # [ky, frame]
            mask[max(0, pad_left) : mask.shape[0] - max(0, pad_right), :] = mask_unpadded

            # apply mask to kspace
            kdata_masked = k_data * mask[None, :, None, :, None]  # [kx, ky, coil, frame, slice]

        return kdata_masked, mask

    def get_coil_sensitivities(self, kdata_masked: ndarray, output_name: str) -> ndarray:
        """
        Computes or loads coil sensitivity maps of an acquisition

        Args:
            kdata_masked (ndarray): acquired / masked kspace
            output_name (str): dir where sens maps will be stored / function
                also looks here for precomputed maps to load

        Returns:
            ndarray: coil sensitivity maps
        """
        coil_sens_fname = self.coil_sens_dir / self.data_source / output_name / 'coil_sens_avg.h5'

        try:
            # try to load previously computed csens maps
            coil_sens = mri.coil_sensitivities.read(  # [x, y, coil, map, frame, slice]
                coil_sens_fname, nmaps=self.n_maps, layout='bart',
            )
            if cp is not None and isinstance(kdata_masked, cp.ndarray):
                coil_sens = cp.asarray(coil_sens)
            logging.info('Loaded previously computed coil sensitivity maps')

        except FileNotFoundError:
            # if none read, compute sens maps
            logging.info(f'Estimating coil sensitivity maps using {self.coil_sens_backend}')
            bart.clear_stdouterr()
            coil_sens = mri.coil_sensitivities.estimate(
                kdata_masked, self.coil_sens_backend, fname=coil_sens_fname, nmaps=self.n_maps, write_png=False,
            )
            if self.coil_sens_backend == 'bart':
                for line in bart.bart.stdout.splitlines():
                    logging.info('Bart: ' + line)
                for line in bart.bart.stderr.splitlines():
                    logging.warning('Bart: ' + line)

        return coil_sens

    def save_data(
            self,
            output_name: str,
            dset: Dataset,
            sampling: mri.Sampling,
            kdata_masked: ndarray,
            mask: ndarray,
            recons: dict[str, ndarray],
            data_attrs: dict[str, Any],
            slice_idx: int,
    ):
        """
        Stores the processed dataset with kspace and corresponding
        reconstructions as a h5 file

        Args:
            output_name (str): name corresponding to the current dataset
            dset (Dataset): Dataset
            sampling (mri.Sampling): sampling that was applied
            kdata_masked (ndarray): sampled kspace
            mask (ndarray): applied mask to get sampled kspace
            recons (dict[str, ndarray]): reconstructions from kspace
            data_attrs (dict[str, Any]): data attributes
            slice_idx (int): current slice idx
        """

        # output file location
        partition = f'test_{sampling.name}' if dset.split == 'test' else f'{dset.split}'
        out_file = self.target_dir / f'{self.data_source}_{partition}' / f'{output_name}.h5'
        out_file.parent.mkdir(exist_ok=True, parents=True)

        logging.info(f'Writing file {out_file}')

        # write file
        with h5py.File(out_file, 'w') as hf:
            # (masked) k-space data
            kspace = kdata_masked.astype(np.complex64)
            if cp is not None:
                kspace = cp.asnumpy(kspace)
            hf.create_dataset('kspace', data=kspace)

            # reference reconstructions
            for key, recon in recons.items():
                if key not in ['rss', 'weighted']:
                    continue
                dtype = np.complex64 if np.iscomplexobj(recon) else np.float32
                recon = recon.astype(dtype)
                if cp is not None:
                    recon = cp.asnumpy(recon)
                hf.create_dataset(f'reconstruction_{key}', data=recon)

            # mask and other information relevant for testing
            if dset.split == 'test':
                mask_ = mask.astype(np.uint8)
                if cp is not None:
                    mask_ = cp.asnumpy(mask_)
                hf.create_dataset('mask', data=mask_)
                hf.attrs['acceleration'] = sampling.acceleration
                hf.attrs['mask_type'] = sampling.mask_type if sampling.mask_type else data_attrs['mask type']

            # some meta information
            if 'weighted' in recons:
                hf.attrs['norm'] = np.linalg.norm(np.abs(recons['weighted'])).item()
                hf.attrs['abs_max'] = np.max(np.abs(recons['weighted'])).item()
            hf.attrs['patient_id'] = dset.filename.stem
            hf.attrs['view'] = data_attrs['viw']
            hf.attrs['slice'] = slice_idx
            if dset.noise_cov is not None:
                hf.attrs['noise'] = dset.noise_cov

            # ISMRMRD header and ECG signal
            hf.create_dataset('ismrmrd_header', data=dset.hdr.toXML('utf-8'))  # type: ignore
            if dset.ecg is not None:
                hf.create_dataset('ecg', data=dset.ecg)

            # annotations
            if 'center x' in data_attrs and not np.isnan(data_attrs['center x']):
                hf.attrs['center'] = (int(data_attrs['center x']), int(data_attrs['center y']))
                hf.attrs['axis1'] = (int(data_attrs['axis 1 x']), int(data_attrs['axis 1 y']))
                hf.attrs['axis2'] = (int(data_attrs['axis 2 x']), int(data_attrs['axis 2 y']))
                hf.attrs['bbox'] = (int(data_attrs['bbox x low']), int(data_attrs['bbox y low']),
                                    int(data_attrs['bbox x high']), int(data_attrs['bbox y high']))

            # base vectors and position
            if dset.read_dir is not None:
                hf.attrs['read_dir'] = dset.read_dir
            if dset.phase_dir is not None:
                hf.attrs['phase_dir'] = dset.phase_dir
            if dset.slice_dir is not None:
                hf.attrs['slice_dir'] = dset.slice_dir
            if dset.position is not None:
                hf.attrs['position'] = dset.position

            # orientation
            if dset.norm_orientation_rot is not None:
                hf.attrs['rotation'] = dset.norm_orientation_rot
            if dset.norm_orientation_hor is not None:
                hf.attrs['flip_horizontal'] = dset.norm_orientation_hor
            if dset.norm_orientation_ver is not None:
                hf.attrs['flip_vertical'] = dset.norm_orientation_ver

    def process_slice_with_sampling(
            self,
            dset: Dataset,
            data_attrs: dict[str, Any],
            sampling: mri.Sampling,
            output_name: str,
            exclude: bool,
            pad_left: int,
            pad_right: int,
            recon_phase_size: int,
    ):
        """
        Process the current slice of a dataset with a specific sampling pattern

        Args:
            dset (Dataset): current dataset
            data_attrs (dict[str, Any]): data attributes
            sampling (mri.Sampling): current sampling pattern
            output_name (str): output name where everything will be stored
            exclude (bool): if this should not be included in the training/testing set
            pad_left (int): kspace phase padding on dataset left
            pad_right (int): kspace phase padding on dataset right
            recon_phase_size (int): size in phase dimension to crop reconstructions to
        """
        # generate masked kspace with this sampling pattern
        kdata_masked, mask = self.mask_kspace(dset.k_data, sampling, pad_left, pad_right)

        # read or compute coil sensitivity maps
        coil_sens = self.get_coil_sensitivities(kdata_masked, output_name)

        # generate reference recons
        recons = {}
        if sampling.name != 'pse':
            # rss and sens weighted recons for all except prospectively undersampled
            logging.info(f'Computing reference reconstructions')
            recons['rss'] = mri.rss(dset.k_data, fft_axes=(0, 1), coil_axis=2)  # [x, y, frame, slice]
            recons['weighted'] = mri.sensitivity_weighted(  # [x, y, frame, slice]
                dset.k_data, coil_sens[:, :, :, 0], fft_axes=(0, 1), coil_axis=2, frame_axis=3,
            )
        if sampling.name != 'fs' and self.cs_dir is not None:
            # cs recon for all undersampled samplings
            logging.info(f'Computing CS reconstruction')
            bart.clear_stdouterr()
            recons['cs'] = mri.pics(
                kdata_masked, mri.get_pics_regularizer(sampling), coil_sens=coil_sens, coil_sens_nmaps=1,
                fname=self.cs_dir / f'{self.data_source}_{sampling.name}' / output_name / 'reconstruction.h5',
                phase_crop=recon_phase_size, use_gpu=('cuda' in self.device), load_if_exists=True,
            )
            if hasattr(bart.bart, 'stdout'):
                for line in bart.bart.stdout.splitlines():
                    logging.info('Bart: ' + line)
            if hasattr(bart.bart, 'stderr'):
                for line in bart.bart.stderr.splitlines():
                    logging.warning('Bart: ' + line)

        # reorder dims to fastMRI format
        mask = mask.T  # [frame, ky]
        logging.info(f'New mask shape: {mask.shape} (frame, ky)')
        kdata_masked = kdata_masked.transpose(4, 2, 3, 0, 1)  # [slice, coil, frame, kx, ky]
        logging.info(f'New k-space shape: {kdata_masked.shape} (slice, coil, frame, kx, ky)')
        for k in recons:
            recons[k] = recons[k].transpose(3, 2, 0, 1)  # [slice, frame, x, y]
            logging.info(f'New recon shape ({k}): {recons[k].shape} (slice, frame, x, y)')

        # crop phase oversampling from recons
        logging.info(f'Cropping phase oversampling')
        for k in recons:
            recons[k] = mri.utils.center_crop(recons[k], 3, recon_phase_size)
            logging.info(f'New recon shape ({k}): {recons[k].shape} (slice, frame, x, y)')

        # save one of the reconstructions as GIF (either sens weighted combination or CS recon)
        if sampling.name in ('fs', 'pse') and ('weighted' in recons or 'cs' in recons):
            recon = recons['weighted'] if sampling.name == 'fs' else recons['cs']
            recon = dset.norm_orientation(recon, image_axes=(-2, -1))
            gif_dir = self.target_dir / f'{self.data_source}_recons'
            if exclude:
                gif_dir /= 'excluded'
            gif_dir.mkdir(exist_ok=True, parents=True)
            if cp is not None:
                recon = cp.asnumpy(recon)
            tres = dset.hdr.sequenceParameters.TR[0]  # type: ignore
            save_movie(recon[0], gif_dir / output_name, clip=True, equalize_histogram=False, tres=tres, apng=False)

        # skip saving if any of the following is true:
        if (exclude                                                        # the dataset is excluded
            or dset.split in [None, 'nan']                                 # the split is not defined
            or (dset.split == 'test' and sampling.name == 'fs')            # test set with fully sampled data
            or (dset.split in ('train', 'val') and sampling.name != 'fs')  # train or val set with undersampled data
        ):
            return

        # save this to the processed dataset
        assert isinstance(dset.slice_idx, int)
        self.save_data(output_name, dset, sampling, kdata_masked, mask, recons, data_attrs, dset.slice_idx)

    def process_slice(self, dset: Dataset, data_attrs: dict[str, Any], slc_idx: int, output_name: str, exclude: bool):
        """
        Processes the current slice of a dataset

        Args:
            dset (Dataset): dataset
            data_attrs (dict[str, Any]): data attributes
            slc_idx (int): current slice idx
            output_name (str): name under which everything will be stored
            exclude (bool): if this should not be excluded from the data set
        """
        # determine which sampling patterns were/will be applied
        sampling_patterns = self.get_sampling_patterns(str(data_attrs['smp']), str(data_attrs['mask type']), exclude)

        # if dataset was already loaded, select this slice, otherwise read kspace data for this slice
        if hasattr(dset, 'k_data_full') and dset.n_slices == dset.k_data_full.shape[6]:
            dset.select_slice(slc_idx)
        else:
            logging.info(f'Reading k-space data for slice {slc_idx + 1}')
            dset.read_kdata(whiten=True, select_slice=slc_idx)

        logging.info(f'K-space shape: {dset.k_data.shape} (kx, ky, kz, coil, frame, set, slice, rep, avg)')

        # the number of frames may differ between slices
        # check for last frame with data
        sum_per_frame = np.moveaxis(np.abs(dset.k_data), 4, 0).reshape(dset.k_data.shape[4], -1).sum(axis=1)
        last_nonzero = np.nonzero(sum_per_frame)[0][-1]
        dset.k_data = dset.k_data[:, :, :, :, :last_nonzero + 1, :, :, :, :]

        # apply phase padding to kdata and get target size without phase oversampling (cropped after recon)
        dset, pad_left, pad_right = self.phase_pad_dataset(dset)

        # in asymetric echo case, apply POCS to get full kspace, only for fully sampled acquisitions
        if data_attrs['ech'] == 'asy':
            if data_attrs['smp'] == 'fs':
                logging.info(f'Running POCS')
                dset.k_data = mri.pocs(dset.k_data)
                # reset phase padded kspace to zero
                left = max(0, pad_left)
                right = min(dset.k_data.shape[1], dset.k_data.shape[1] - pad_right)
                dset.k_data[:, :left] = 0
                dset.k_data[:, right:] = 0
            else:
                logging.warning('POCS only implemented for fully sampled data, skipping')

        # crop readout oversampling -> we assume consistently 200%
        if dset.has_read_os:
            logging.info('Removing readout oversampling')
            dset.k_data = mri.utils.crop_readout_oversampling(dset.k_data)

        # squeeze kz, set, rep, and avg dimensions
        assert dset.k_data.shape[2] == 1, 'kz must be 1'
        assert dset.k_data.shape[5] == 1, 'set must be 1'
        assert dset.k_data.shape[7] == 1, 'rep must be 1'
        assert dset.k_data.shape[8] == 1, 'avg must be 1'
        dset.k_data = dset.k_data.squeeze(axis=(2, 5, 7, 8))  # [kx, ky, coil, frame, slice]
        logging.info(f'K-space shape: {dset.k_data.shape} (kx, ky, coil, frame, slice)')

        # get final reconstruction size in phase dimension
        recon_phase_size = self.get_recon_phase_size(dset)

        # process each slice with given sampling pattern
        for sampling in sampling_patterns:
            logging.info(f'-- Processing sampling \'{sampling.name}\' on slice {slc_idx + 1} of dataset {dset.name} --')
            self.process_slice_with_sampling(
                dset, data_attrs, sampling, output_name, exclude, pad_left, pad_right, recon_phase_size,
            )

    def process_sample(self, data_attrs: dict[str, Any]):
        """
        Process one dataset

        Args:
            data_attrs (dict[str, Any]): data attributes
        """
        # determine raw file path
        raw_file = self.raw_dir / self.data_source / str(data_attrs['file name'])

        # init / load dataset, update csv and get updated version of data attributes
        dset, data_attrs = self.load_dataset(raw_file, data_attrs)

        # determine excluded slices of the data set
        excluded_slices = []
        if not pd.isna(data_attrs['excluded slices']):
            for slc in str(data_attrs['excluded slices']).split('+'):
                if '-' in slc:
                    start, end = slc.split('-')
                    excluded_slices.extend(list(range(int(start), int(end) + 1)))
                else:
                    excluded_slices.append(int(slc))
        logging.info(f'Excluded slices: {excluded_slices}')

        # process each slice individually
        for slc_idx in range(dset.n_slices):
            logging.info(f'---- Processing slice {slc_idx+1} / {dset.n_slices} ----')
            # assign individual save name for this dataset and slice
            output_name = f'{dset.filename.stem}_slice{slc_idx:02d}'
            # determine if this slc should be excluded from processed data set
            exclude = slc_idx in excluded_slices

            self.process_slice(dset, data_attrs, slc_idx, output_name, exclude)

    def run(self):
        """
        Processes all data sets that are in the csv file that corresponds to
        the initialized data source (located in data_attributes dir)
        """
        # iterate all samples of the defined dataframe and process them
        for count, (_, row) in enumerate(self.df.iterrows()):
            data_attrs = row.to_dict()
            logging.info('-'*50)
            logging.info('')
            logging.info('')
            logging.info(f'Processing {data_attrs["file name"]} ({count+1} / {len(self.df)})')
            logging.info('')
            logging.info('')
            self.process_sample(data_attrs)
