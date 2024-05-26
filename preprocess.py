"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import boto3
import h5py
import numpy as np
import pandas as pd
import psutil
from botocore import UNSIGNED
from botocore.client import Config
from PIL import Image

import bart
from helpers import mri, ndarray
from helpers.datasets import Dataset, IsmrmrdDataset

try:
    import cupy as cp
except ImportError:
    cp = None


logging.basicConfig(
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler(log_file)
    ],
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s|%(message)s',
    datefmt='%Y-%m-%d %H-%M-%S'
)
log = logging.getLogger(__name__)

np.random.seed(42)
if cp is not None:
    cp.random.seed(42)


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

    def __init__(self,
                 csv_file: Path,
                 data_source: str,
                 raw_dir: Path,
                 target_dir: Path,
                 coil_sens_dir: Path,
                 cs_dir: Path | None = None,
                 csv_query: str | None = None,
                 csv_idx_range: list[int | None] = [0, None],
                 accelerations: list[int] | None = None,
                 center_fractions: list[float] | None = None,
                 mask_types: list[str] | None = None,
                 mode: str = 'dynamic',
                 whiten: bool = True,
                 coil_sens_backend: str = 'sigpy',
                 n_maps: int = 1,
                 clip_coil_sensitivities: bool = False,
                 try_cuda: bool = True
                 ):
        """
        Args:
            csv_file (Path): Path to csv file which contains metadata of the
                samples to process
            data_source (str): type of dataset (ocmr | diagnostikum | freemax | interactive )
            raw_dir (Path): path to raw unprocessed data files
            target_dir (Path): path to store processed data
            coil_sens_dir (Path): path to store/load coil sensitivities from
            cs_dir (Path | None, optional): path to compressed sensing
                reconstructions. If None no CS recon is perfromed.
                Defaults to None.
            csv_query (str | None, optional): Query string to filter
                samples from the csv table for processing
                (see https://datagy.io/pandas-query/). Defaults to None.
            csv_idx_range (List[int  |  None], optional): Range of indices to
                filter samples for processing. Defaults to [0, None].
            accelerations (List[int] | None, optional): Acceleration rates
                for retrospective undersampling. Defaults to None.
            center_fractions (List[float] | None, optional): Center
                fractions for retrospective undersampling. If None, but
                accelerations rates were passed, defaults to 0 for each
                acceleration rate. Defaults to None.
            mask_types (List[str] | None, optional): Mask types for
                retrospective undersampling. Defaults to None.
            mode (str, optional): Mode that determines if cine frames are
                processed together as a sequence or individually
                (dynamic | static). Defaults to 'dynamic'.
            whiten (bool, optional): Pre-whiten k-space data. Defaults to True.
            coil_sens_backend (str, optional): Which backend to use to compute
                coil sensitivities. Defaults to 'sigpy'.
            n_maps (int, optional): Number of coil sensitivity maps to estimate.
                Defaults to 1.
            clip_coil_sensitivities (bool, optional): Clipping parameter for
                coil sensitiviy map estimation. Defaults to False.
            try_cuda (bool, optional): Try to use gpu if available.
                Defaults to True.
        """
        assert csv_file.is_file(), f'{csv_file} not a file.'
        self.csv_file = csv_file
        self.data_source = data_source
        self.raw_dir = raw_dir
        self.target_dir = target_dir
        self.coil_sens_dir = coil_sens_dir
        self.cs_dir = cs_dir
        self.accelerations = [] if accelerations is None else accelerations
        if center_fractions is None:
            self.center_fractions: list[float] = [0] * len(self.accelerations)
        else:
            assert len(center_fractions) == len(self.accelerations), \
                f'center_fractions (len={len(center_fractions)}) and '\
                f'accelerations (len={len(self.accelerations)}) have to be of '\
                'same size'
            self.center_fractions = center_fractions
        self.mask_types = [] if mask_types is None else mask_types
        self.mode = mode
        self.whiten = whiten
        self.coil_sens_backend = coil_sens_backend
        self.n_maps = n_maps
        self.clip_coil_sensitivities = clip_coil_sensitivities
        self.device = 'cpu'
        if try_cuda and cp is not None and cp.cuda.is_available():
            self.device = 'cuda'

        # read full csv file with all samples of this data source
        self.df_full = pd.read_csv(csv_file, dtype={'excluded slices': str})
        # check if df has all required cols
        missing_cols = [c for c in self.REQ_CSV_COLS
                        if c not in self.df_full.columns]
        assert len(missing_cols) == 0, \
                f'Dataframe misses the following cols: {missing_cols}'

        # filter samples according to arguments
        self.df = self.df_full.loc[csv_idx_range[0]:csv_idx_range[1]]
        if csv_query:
            self.df = self.df.query(csv_query)

    @staticmethod
    def map_system_model(system_model: str) -> str:
        if 'avanto' in system_model.lower():
            return '15avan'
        elif 'sola' in system_model.lower():
            return '15sola'
        elif 'prisma' in system_model.lower():
            return '30pris'
        elif 'vida' in system_model.lower():
            return '30vida'
        elif 'free.max' in system_model.lower():
            return '05freemax'
        else:
            raise RuntimeError(f'Unknown system model {system_model}')

    @staticmethod
    def update_csv(df: pd.DataFrame, dset: Dataset, csv_file: Path
                   ) -> dict[str, str | float | int]:
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

        if not isinstance(dset, IsmrmrdDataset):  # do not overwrite OCMR tags
            df.loc[row, 'scn'] = Preprocessing.map_system_model(hdr.acquisitionSystemInformation.systemModel)  # type: ignore
            df.loc[row, 'dur'] = 'lng' if t_res * n_frames >= 5000 else 'shr'
            df.loc[row, 'sli'] = 'ind' if enc.encodingLimits.slice.maximum == 0 else 'stk'  # type: ignore
            df.loc[row, 'slices'] = enc.encodingLimits.slice.maximum + 1  # type: ignore
        if df.loc[row, 'smp'].item() == 'grappa':
            df.loc[row, 'mask type'] = f'grappa {enc.parallelImaging.accelerationFactor.kspace_encoding_step_1}'  # type: ignore
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
        if df.loc[row, 'smp'].item() in ['fs', 'grappa']:
            # heart rate can only be computed in segmented acquisitions with exactly one heart cycle
            df.loc[row, 'HR'] = round(60 / (t_res * n_frames / 1000))
        df.loc[row, 'protocol name'] = hdr.measurementInformation.protocolName  # type: ignore
        df.to_csv(csv_file, index=False)

        return df.loc[row].iloc[0].to_dict()

    def load_dataset(self,
                     raw_file: Path,
                     data_attrs: Dict[str, str | float | int]
                     ) -> Tuple[Dataset, Dict[str, str | float | int]]:
        """
        Takes file path of raw data and additional attributes of the data
        sample to return a Dataset and the updated sample attributes

        Args:
            raw_file (Path): path to raw data file
            data_attrs (Dict[str, str  |  float  |  int]): sample attributes,
                for this function the keys `split`, `rotation`,
                `flip horizontal`, `flip vertical`, `dur`, `frames` and `sli`
                have to be contained

        Raises:
            ValueError: If one of the requires keys not in data_attrs dict
            ValueError: If given self.data_source is not defined

        Returns:
            Tuple[Dataset, Dict[str, str | float | int]]: Dataset and updated
            data attributes dictionary
        """
        req_keys = ('split', 'rotation', 'flip horizontal', 'flip vertical', 'dur', 'frames', 'sli')
        for key in req_keys:
            assert key in data_attrs, f'{key} not in data_attrs'

        log.info(f'Loading {raw_file}')
        # init dataset (if data source is ocmr, optionally load it if neccessary)
        dataset: Dataset
        match self.data_source:
            case 'ocmr':
                # load dataset if it doesn't exist yet
                if not raw_file.exists():
                    log.info(f'Did not find {raw_file}. Downloading {raw_file.name} to {raw_file}')
                    raw_file.parent.mkdir(exist_ok=True, parents=True)
                    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
                    s3_client.download_file('ocmr', f'data/{raw_file.name}', str(raw_file))

                dataset = IsmrmrdDataset(name=raw_file.name, filename=raw_file, device=self.device)

            case _:
                raise ValueError(f'Unknown data source: {self.data_source}')

        # assign dataset split where to save it
        dataset.split = str(data_attrs['split'])

        # read orientation normalization
        if all([pd.notna(data_attrs[k]) for k in ['rotation', 'flip horizontal', 'flip vertical']]):
            dataset.set_norm_orientation(
                rot=int(data_attrs['rotation']),
                hor=int(data_attrs['flip horizontal']),
                ver=int(data_attrs['flip vertical'])
            )

        # read meta data
        log.info('Reading metadata')
        dataset.read_meta()

        # update the csv file with new information
        log.info(f'Updating {self.csv_file}')
        data_attrs = self.update_csv(self.df_full, dataset, self.csv_file)

        # check whether to preload kdata or load each slice seperately
        if not (
            (data_attrs['dur'] == 'lng' or float(data_attrs['frames']) >= 50)
            and data_attrs['sli'] == 'stk'
        ):
            log.info('Reading k-space data')
            dataset.read_kdata(whiten=self.whiten, select_slice=None)

        return dataset, data_attrs

    def get_sampling_patterns(self,
                              acq_sampling: str,
                              acq_masktype: str,
                              exclude: bool
                              ) -> List[mri.Sampling]:
        """
        Creates sampling pattern objects.
        1. The sampling of the original acquisition
        2. Additional sampling patterns that should be created for this data set

        Args:
            acq_sampling (str): how the acquisition was sampled
                (fs: fully sampled | pse: prospectively undersampled)
            acq_masktype (str): which sampling mask was applied
            exclude (bool): if this sampled should be excluded from the data
                set beeing created and therefore there are no additional
                sampling masks generated

        Raises:
            NotImplementedError: if `grappa` is passed as acq_sampling
            ValueError: if the acq_sampling is unknown

        Returns:
            List[mri.Sampling]: list of mri.Sampling objects
        """
        # create list of sampling patterns, starting with the original acquisiton sampling pattern
        match acq_sampling:
            case 'fs' | 'grappa':
                sampling_patterns = [mri.Sampling(name='fs', acceleration=1)]

                # if this is added to the processed dataset, add additional sampling patterns
                if not exclude:
                    for masktype in self.mask_types:
                        for acc, centfrac in zip(self.accelerations, self.center_fractions):
                            sampling_patterns.append(
                                mri.Sampling(
                                    name=f'{masktype}_{acc:02d}',
                                    acceleration=acc,
                                    center_fraction=centfrac,
                                    mask_type=masktype
                                )
                            )
            case 'pse':
                sampling_patterns = [mri.Sampling(name='pse', mask_type=acq_masktype)]

            case _:
                raise ValueError(f'Unknown dset_sampling: {acq_masktype}')

        log.info(f'Sampling Patterns: {[s.name for s in sampling_patterns]}')

        return sampling_patterns

    @staticmethod
    def phase_pad_dataset(dset: Dataset) -> Tuple[Dataset, int, int]:
        """
        Pads the phase direction of the kspace of a given Dataset

        Args:
            dset (Dataset): dataset with kspace to pad

        Returns:
            Tuple[Dataset, int, int]: (
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

        log.info(f'Phase padding: {phase_padding_left}, {phase_padding_right}')
        if phase_padding_left < 0 or phase_padding_right < 0:
            log.warning(
                'Phase padding is negative (i.e., phase resolution > 100%)! The additional k-space lines will be '
                'removed and ignored in the following. This will likely lead to a lower effective acceleration '
                'rate in prospectively undersampled acquisitions and makes comparisons with the product '
                'reconstruction unfair in all sampling schemes!'
            )

        dset.k_data = mri.utils.apply_phase_padding(
            dset.k_data,
            phase_padding_left,
            phase_padding_right
        )

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

    @staticmethod
    def mask_kspace(k_data: ndarray,
                    sampling: mri.Sampling,
                    pad_left: int,
                    pad_right: int
                    ) -> Tuple[ndarray, ndarray]:
        """
        Applies masking to a kspace array

        Args:
            k_data (ndarray): kdata to mask
            sampling (mri.Sampling): sampling pattern to apply
            pad_left (int): phase padding left
            pad_right (int): phase padding right

        Returns:
            Tuple[ndarray, ndarray]: (
                kspace array after masking,
                applied mask
            )
        """
        xp = cp.get_array_module(k_data) if cp is not None else np

        match sampling.name:

            case 'fs':  # fully sampled
                mask = xp.ones((k_data.shape[1], k_data.shape[3]))  # [ky, frame]
                kdata_masked = xp.copy(k_data)  # [kx, ky, coil, frame, slice]

            case 'pse':  # prospectively undersampled
                # obtain mask (zero-padded)
                mask = (abs(xp.mean(xp.abs(k_data), axis=2)) > 0).astype(int)  # [kx, ky, frame, slice]

                # collapse kx and slice dimension of mask
                assert xp.all(mask.all(axis=0) | (1 - mask).all(axis=0)), 'Mask is not equal along kx dim'
                assert xp.all(mask.all(axis=3) | (1 - mask).all(axis=3)), 'Mask is not equal along slice dim'
                mask = mask[0, :, :, 0]  # [ky, frame]

                # remove zero-padding from mask (ignoring negative padding if phase resolution > 100%)
                mask_unpadded = mask[max(0, pad_left) : mask.shape[0] - max(0, pad_right), :]

                # calc acceleration rate on unpadded mask
                sampling.acceleration = int(mask_unpadded.size / mask_unpadded.sum())
                log.info(f'Acceleration rate: {sampling.acceleration:.2f}')

                # estimate center fraction
                mask_acs = xp.all(mask_unpadded, axis=1)
                mask_acs = xp.concatenate([xp.zeros(1), mask_acs, xp.zeros(1)])
                acs_front = xp.argmin(xp.flip(mask_acs[:mask_acs.shape[0] // 2], axis=0))
                acs_back = xp.argmin(mask_acs[mask_acs.shape[0] // 2:])
                num_low_frequencies = acs_front + acs_back
                sampling.center_fraction = float(num_low_frequencies / mask_unpadded.shape[0])
                log.info(f'Center fraction: {sampling.center_fraction:.2f}')

                # pad mask with ones
                mask = xp.ones(mask.shape)
                mask[max(0, pad_left) : mask.shape[0] - max(0, pad_right), :] = mask_unpadded

                # mask is already applied on k-space data
                kdata_masked = xp.copy(k_data)  # [kx, ky, coil, frame, slice]

            case _:  # retrospective undersampling
                # obtain unpadded mask, ignoring negative padding values (phase resolution > 100%)
                num_cols = k_data.shape[1] - max(0, pad_right) - max(0, pad_left)
                assert sampling.mask_type is not None and sampling.acceleration is not None \
                    and sampling.center_fraction is not None
                mask_unpadded = mri.get_mask(  # [frame, ky]
                    k_data.shape[3],
                    num_cols,
                    sampling.mask_type,
                    sampling.acceleration,
                    sampling.center_fraction,
                    padding_left=pad_left,
                    padding_right=pad_right
                )
                if cp is not None and isinstance(k_data, cp.ndarray):
                    mask_unpadded = cp.asarray(mask_unpadded)
                mask_unpadded = mask_unpadded.T  # [ky, frame]

                # effective acc rate
                acc_eff = mask_unpadded.size / mask_unpadded.sum()
                log.info(f'Effective Acceleration Rate: {acc_eff:5.2f}')

                # pad mask
                mask = xp.ones((k_data.shape[1], k_data.shape[3]))  # [ky, frame]
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
        unique_coil_sens_dir = self.coil_sens_dir / self.mode / self.data_source / output_name

        # determine coil sensitivity file paths
        if self.mode == 'dynamic':
            coil_sens_paths = [unique_coil_sens_dir / 'coil_sens_avg.h5']
        else:
            num_frames = kdata_masked.shape[3]
            coil_sens_paths = [unique_coil_sens_dir / f'coil_sens_frame{f:02d}.h5'
                               for f in range(num_frames)]
        coil_sens = None
        # try to load previously computed csens maps
        try:
            coil_sens = mri.coil_sensitivities.read(  # [x, y, coil, map, frame, slice]
                coil_sens_paths,
                nmaps=self.n_maps,
                layout='bart'
            )
            if cp is not None and isinstance(kdata_masked, cp.ndarray):
                coil_sens = cp.asarray(coil_sens)
            log.info('Loaded previously computed coil sensitivity maps')

        except FileNotFoundError:
            pass

        # if none read, compute csens maps
        if coil_sens is None:
            log.info(f'Estimating coil sensitivity maps using {self.coil_sens_backend}')
            bart.clear_stdouterr()
            coil_sens = mri.coil_sensitivities.estimate(
                k_data=kdata_masked,
                coil_sens_dir=unique_coil_sens_dir,
                backend=self.coil_sens_backend,
                nmaps=self.n_maps,
                averaged=self.mode == 'dynamic',
                clip=self.clip_coil_sensitivities,
                write_png=False,
                suppress_stdouterr=True
            )
            if self.coil_sens_backend == 'bart':
                for line in bart.bart.stdout.splitlines():
                    logging.info('Bart: ' + line)
                for line in bart.bart.stderr.splitlines():
                    logging.warning('Bart: ' + line)

        return coil_sens

    @staticmethod
    def visualize_recon(recon: ndarray,
                        target_dir: Path,
                        output_name: str,
                        exclude: bool,
                        gif_duration: float
                        ):
        """
        Generates tiff and gif representations of reconstructions

        Args:
            recon (ndarray): reconstruction array
            target_dir (Path): dir to store tiff and gif files
            output_name (str): name under which to store
            exclude (bool): if it is excluded from the data set, it should be
                stored in seperate dir
            gif_duration (float): TR value (corresponds to duration of one gif)
        """
        # only use numpy in the following
        if cp is not None:
            recon = cp.asnumpy(recon)

        # magnitude
        if np.iscomplexobj(recon):
            recon = np.abs(recon)

        # get rid of slice dimension (its only one slice anyways)
        recon = recon[0]

        # clip and normalize
        recon = np.clip(recon, np.percentile(recon, 3), np.percentile(recon, 97))
        recon = (recon - np.min(recon)) / (np.max(recon) - np.min(recon))

        # create dirs
        tiff_dir = target_dir / 'tiffs'
        gif_dir = target_dir / 'gifs'
        if exclude:
            tiff_dir /= 'excluded'
            gif_dir /= 'excluded'
        os.makedirs(tiff_dir, exist_ok=True)
        os.makedirs(gif_dir, exist_ok=True)

        # save using PIL
        recon_pillow = [Image.fromarray(recon[j])
                        for j in range(recon.shape[0])]
        recon_pillow[0].save(
            tiff_dir / f'{output_name}.tiff',
            save_all=True,
            append_images=recon_pillow[1:]
        )
        recon_pillow = [Image.fromarray(recon[j] * 255)
                        for j in range(recon.shape[0])]
        recon_pillow[0].save(
            gif_dir / f'{output_name}.gif',
            save_all=True,
            append_images=recon_pillow[1:],
            duration=gif_duration,
            loop=0
        )

    def save_data(self,
                  output_name: str,
                  dataset: Dataset,
                  sampling: mri.Sampling,
                  kdata_masked: ndarray,
                  mask: ndarray,
                  recons: Dict[str, ndarray],
                  data_attrs: Dict[str, str | float | int],
                  slice_idx: int
                  ):
        """
        Stores the processed dataset with kspace and corresponding
        reconstructions as a h5 file

        Args:
            output_name (str): name corresponding to the current dataset
            dataset (Dataset): Dataset
            sampling (mri.Sampling): sampling that was applied
            kdata_masked (ndarray): sampled kspace
            mask (ndarray): applied mask to get sampled kspace
            recons (Dict[str, ndarray]): reconstructions from kspace
            data_attrs (Dict[str, str  |  float  |  int]): data attributes
            slice_idx (int): current slice idx
        """
        num_frames = mask.shape[0]

        if self.mode == 'dynamic':
            frame_slices = [slice(None)]
        else:  # static
            frame_slices = range(num_frames) # type: ignore

        for f in frame_slices:
            # output file location
            partition = f'test_{sampling.name}' if dataset.split == 'test' else f'{dataset.split}'
            out_dir = self.target_dir / self.mode / f'{self.data_source}_{partition}'
            os.makedirs(out_dir, exist_ok=True)
            suffix = f'_frame{f:02d}' if self.mode == 'static' else ''
            out_file = out_dir / f'{output_name}{suffix}.h5'

            log.info(f'Writing file {out_file}')

            # write file
            with h5py.File(out_file, 'w') as hf:
                # (masked) k-space data
                kspace = kdata_masked[:, :, f].astype(np.complex64)
                if cp is not None:
                    kspace = cp.asnumpy(kspace)
                hf.create_dataset('kspace', data=kspace)

                # reference reconstructions
                for key, recon in recons.items():
                    if key not in ['rss', 'weighted']:
                        continue
                    dtype = np.complex64 if np.iscomplexobj(recon) else np.float32
                    recon = recon[:, f].astype(dtype)
                    if cp is not None:
                        recon = cp.asnumpy(recon)
                    hf.create_dataset(f'reconstruction_{key}', data=recon)

                # mask and other information relevant for testing
                if dataset.split == 'test':
                    mask_ = mask[f].astype(np.float32)
                    if cp is not None:
                        mask_ = cp.asnumpy(mask_)
                    hf.create_dataset('mask', data=mask_)
                    hf.attrs['acceleration'] = sampling.acceleration
                    hf.attrs['center_fraction'] = sampling.center_fraction
                    hf.attrs['mask_type'] = sampling.mask_type if sampling.mask_type else data_attrs['mask type']

                # some meta information
                if 'weighted' in recons:
                    hf.attrs['norm'] = np.linalg.norm(np.abs(recons['weighted'][:, f])).item()
                    hf.attrs['abs_max'] = np.max(np.abs(recons['weighted'][:, f])).item()
                hf.attrs['patient_id'] = dataset.filename.stem
                hf.attrs['view'] = data_attrs['viw']
                hf.attrs['frame'] = f if self.mode == 'static' else 0
                hf.attrs['slice'] = slice_idx
                if dataset.noise_cov is not None:
                    hf.attrs['noise'] = dataset.noise_cov

                # ISMRMRD header and ECG signal
                hf.create_dataset('ismrmrd_header', data=dataset.hdr.toXML('utf-8'))  # type: ignore
                if dataset.ecg is not None:
                    hf.create_dataset('ecg', data=dataset.ecg)

                # annotations
                if 'center x' in data_attrs and not np.isnan(data_attrs['center x']):
                    hf.attrs['center'] = (int(data_attrs['center x']), int(data_attrs['center y']))
                    hf.attrs['axis1'] = (int(data_attrs['axis 1 x']), int(data_attrs['axis 1 y']))
                    hf.attrs['axis2'] = (int(data_attrs['axis 2 x']), int(data_attrs['axis 2 y']))
                    hf.attrs['bbox'] = (int(data_attrs['bbox x low']), int(data_attrs['bbox y low']),
                                        int(data_attrs['bbox x high']), int(data_attrs['bbox y high']))

                # base vectors and position
                if dataset.read_dir is not None:
                    hf.attrs['read_dir'] = dataset.read_dir
                if dataset.phase_dir is not None:
                    hf.attrs['phase_dir'] = dataset.phase_dir
                if dataset.slice_dir is not None:
                    hf.attrs['slice_dir'] = dataset.slice_dir
                if dataset.position is not None:
                    hf.attrs['position'] = dataset.position

                # orientation
                if dataset.norm_orientation_rot is not None:
                    hf.attrs['rotation'] = dataset.norm_orientation_rot
                if dataset.norm_orientation_hor is not None:
                    hf.attrs['flip_horizontal'] = dataset.norm_orientation_hor
                if dataset.norm_orientation_ver is not None:
                    hf.attrs['flip_vertical'] = dataset.norm_orientation_ver

    def process_slice_withsampling(self,
                                   dataset: Dataset,
                                   data_attrs: Dict[str, str | float | int],
                                   sampling: mri.Sampling,
                                   output_name: str,
                                   exclude: bool,
                                   pad_left: int,
                                   pad_right: int,
                                   recon_phase_size: int
                                   ):
        """
        Process the current slice of a dataset with a specific sampling pattern

        Args:
            dataset (Dataset): current dataset
            data_attrs (Dict[str, str  |  float  |  int]): data attributes
            sampling (mri.Sampling): current sampling pattern
            output_name (str): output name where everything will be stored
            exclude (bool): if this should not be included in the training/testing set
            pad_left (int): kspace phase padding on dataset left
            pad_right (int): kspace phase padding on dataset right
            recon_phase_size (int): size in phase dimension to crop reconstructions to
        """
        # compute masked kspace with this sampling pattern
        kdata_masked, mask = self.mask_kspace(
            dataset.k_data,
            sampling,
            pad_left,
            pad_right
        )

        # read or compute coil sensitivity maps
        coil_sens = self.get_coil_sensitivities(kdata_masked, output_name)

        # compute reference recons
        recons = {}
        # rss and sens weighted recons for all except prospectively undersampled
        if sampling.name != 'pse':
            logging.info(f'Computing reference reconstructions')
            recons['rss'] = mri.rss(  # [x, y, frame, slice]
                dataset.k_data,
                fft_axes=(0, 1),
                coil_axis=2
            )
            recons['weighted'] = mri.sensitivity_weighted(  # [x, y, frame, slice]
                dataset.k_data,
                coil_sens[:, :, :, 0],
                fft_axes=(0, 1),
                coil_axis=2,
                frame_axis=3
            )

        # cs recon for all undersampled samplings
        if sampling.name != 'fs' and self.cs_dir is not None:
            logging.info(f'Computing CS reconstruction')
            pics_regularizer = mri.get_pics_regularizer(sampling)
            recon_dir = self.cs_dir / self.mode / f'{self.data_source}_{sampling.name}' / output_name
            bart.clear_stdouterr()
            recons['cs'] = mri.pics(
                kdata_masked,
                pics_regularizer,
                recon_dir,
                coil_sens=coil_sens,
                coil_sens_nmaps=1,
                dynamic=(self.mode == 'dynamic'),
                phase_crop=recon_phase_size,
                use_gpu=(self.device == 'cuda'),
                load_if_exists=True,
                suppress_stdouterr=True
            )
            if hasattr(bart.bart, 'stdout'):
                for line in bart.bart.stdout.splitlines():
                    logging.info('Bart: ' + line)
            if hasattr(bart.bart, 'stderr'):
                for line in bart.bart.stderr.splitlines():
                    logging.warning('Bart: ' + line)

        # reorder dims to fastMRI format
        mask = mask.T  # [frame, ky]
        log.info(f'New mask shape: {mask.shape} (frame, ky)')
        kdata_masked = kdata_masked.transpose(4, 2, 3, 0, 1)  # [slice, coil, frame, kx, ky]
        log.info(f'New k-space shape: {kdata_masked.shape} (slice, coil, frame, kx, ky)')
        for k in recons:
            recons[k] = recons[k].transpose(3, 2, 0, 1)  # [slice, frame, x, y]
            log.info(f'New recon shape ({k}): {recons[k].shape} (slice, frame, x, y)')

        # crop phase oversampling from recons
        log.info(f'Cropping phase oversampling')
        for k in recons:
            recons[k] = mri.utils.center_crop(recons[k], 3, recon_phase_size)
            log.info(f'New recon shape ({k}): {recons[k].shape} (slice, frame, x, y)')

        # visualize recons
        if (sampling.name in ('fs', 'pse')
                and self.mode == 'dynamic'
                and ('weighted' in recons or 'cs' in recons)):
            # write either sens weighted combination or CS recon
            recon = recons['weighted'] if sampling.name == 'fs' else recons['cs']

            # fix orientation
            recon = dataset.norm_orientation(recon, image_axes=(-2, -1))

            # visualize recon
            self.visualize_recon(
                recon,
                self.target_dir / 'recons' / self.data_source,
                output_name,
                exclude,
                dataset.hdr.sequenceParameters.TR[0]  # type: ignore
            )

        # save this to the processed dataset
        if not (exclude
                or dataset.split in [None, 'nan']
                or (dataset.split == 'test' and sampling.name == 'fs')
                or (dataset.split in ('train', 'val') and sampling.name != 'fs')):
            # save to dataset
            assert isinstance(dataset.slice_idx, int)
            self.save_data(
                output_name,
                dataset,
                sampling,
                kdata_masked,
                mask,
                recons,
                data_attrs,
                dataset.slice_idx
            )

    def process_slice(self,
                      dataset: Dataset,
                      data_attrs: Dict[str, str | float | int],
                      slc_idx: int,
                      output_name: str,
                      exclude: bool,
                      load_slc_indiv: bool
                      ):
        """
        Processes the current slc of a dataset

        Args:
            dataset (Dataset): dataset
            data_attrs (Dict[str, str  |  float  |  int]): data attributes
            slc_idx (int): current slc idx
            output_name (str): name under which everything will be stored
            exclude (bool): if this should not be excluded from the data set
            load_slc_indiv (bool): if slices should be loaded individually
        """
        # determine which sampling patterns were/will be applied
        sampling_patterns = self.get_sampling_patterns(str(data_attrs['smp']),
                                                       str(data_attrs['mask type']),
                                                       exclude)

        # select slice and
        # if dataset was not loaded or not loaded completely, load this slice
        if load_slc_indiv:
            log.info(f'Reading k-space data for slice {slc_idx + 1}')
            dataset.read_kdata(whiten=self.whiten, select_slice=slc_idx)
        else:
            assert hasattr(dataset, 'k_data')
            dataset.select_slice(slc_idx)

        log.info(f'K-space shape: {dataset.k_data.shape} (kx, ky, kz, coil, frame, set, slice, rep, avg)')

        # the number of frames may differ between slices
        # check for last frame with data
        sum_per_frame = np.moveaxis(np.abs(dataset.k_data), 4, 0).reshape(dataset.k_data.shape[4], -1).sum(axis=1)
        last_nonzero = np.nonzero(sum_per_frame)[0][-1]
        dataset.k_data = dataset.k_data[:, :, :, :, :last_nonzero + 1, :, :, :, :]

        # GRAPPA-interpolated k-space will be considered as fully sampled in the following
        if data_attrs['smp'] == 'grappa':
            logging.info(f'Running GRAPPA interpolation')
            dataset.k_data = mri.grappa(
                dataset.k_data, calib=None,
                coil_axis=3, frame_axis=4, fft_axes=(0, 1), return_kspace=True
            )

        # apply phase padding to kdata and get target size without phase oversampling (cropped after recon)
        dataset, pad_left, pad_right = self.phase_pad_dataset(dataset)

        # in asymetric echo case, apply POCS to get full kspace, only for fully sampled acquisitions
        if data_attrs['ech'] == 'asy':
            if data_attrs['smp'] in ['fs', 'grappa']:
                log.info(f'Running POCS')
                dataset.k_data = mri.pocs(dataset.k_data)
                # reset phase padded kspace to zero
                left = max(0, pad_left)
                right = min(dataset.k_data.shape[1], dataset.k_data.shape[1] - pad_right)
                dataset.k_data[:, :left] = 0
                dataset.k_data[:, right:] = 0
            else:
                log.warning('POCS implemented only for fully sampled data, not editing asymetric echo kspace')

        # crop readout oversampling -> we assume consistently 200%
        if dataset.has_read_os:
            log.info('Removing readout oversampling')
            dataset.k_data = mri.utils.crop_readout_oversampling(dataset.k_data)

        # squeeze kz, set, rep, and avg dimensions
        assert dataset.k_data.shape[2] == 1, 'kz must be 1'
        assert dataset.k_data.shape[5] == 1, 'set must be 1'
        assert dataset.k_data.shape[7] == 1, 'rep must be 1'
        assert dataset.k_data.shape[8] == 1, 'avg must be 1'
        dataset.k_data = dataset.k_data.squeeze(axis=(2, 5, 7, 8))  # [kx, ky, coil, frame, slice]
        log.info(f'K-space shape: {dataset.k_data.shape} (kx, ky, coil, frame, slice)')

        # get final reconstruction size in phase dimension
        recon_phase_size = self.get_recon_phase_size(dataset)

        # process each slice with given sampling pattern
        for sampling in sampling_patterns:
            log.info(f'-- Processing sampling \'{sampling.name}\' on slice {slc_idx + 1} of dataset {dataset.name} --')

            self.process_slice_withsampling(
                dataset,
                data_attrs,
                sampling,
                output_name,
                exclude,
                pad_left,
                pad_right,
                recon_phase_size
            )

    def process_sample(self, data_attrs: Dict[str, str | float | int]):
        """
        Process one dataset

        Args:
            data_attrs (Dict[str, str  |  float  |  int]): data attributes
        """
        # determine raw file path
        raw_file = self.raw_dir / self.data_source / str(data_attrs['file name'])

        # init / load dataset, update csv and get updated version of data attributes
        dataset, data_attrs = self.load_dataset(raw_file, data_attrs)
        load_slc_indiv = not hasattr(dataset, 'k_data')

        # determine excluded slices of the data set
        excluded_slices = []
        if not pd.isna(data_attrs['excluded slices']):
            for slc in str(data_attrs['excluded slices']).split('+'):
                if '-' in slc:
                    start, end = slc.split('-')
                    excluded_slices.extend(
                        list(range(int(float(start)), int(float(end)) + 1))
                    )
                else:
                    excluded_slices.append(int(float(slc)))
        log.info(f'Excluded slices: {excluded_slices}')

        # process each slice individually
        for slc_idx in range(dataset.n_slices):
            log.info(f'---- Processing slice {slc_idx+1} / {dataset.n_slices} ----')
            # assign individual save name for this dataset and slice
            output_name = f'{dataset.filename.stem}_slice{slc_idx:02d}'
            # determine if this slc should be excluded from processed data set
            exclude = slc_idx in excluded_slices

            self.process_slice(
                dataset,
                data_attrs,
                slc_idx,
                output_name,
                exclude,
                load_slc_indiv
            )

        pid = os.getpid()
        memory_info = psutil.Process(pid).memory_info()
        log.info(f'Memory allocated: {memory_info.rss / 1024 / 1024:.2f} MiB')

    def run(self):
        """
        Processes all data sets that are in the csv file that corresponds to
        the initialized data source (located in data_attributes dir)
        """
        # iterate all samples of the defined dataframe and process them
        for count, (_, row) in enumerate(self.df.iterrows()):
            data_attrs = row.to_dict()
            log.info('-'*50)
            log.info('')
            log.info('')
            log.info(f'Processing {data_attrs["file name"]} ({count+1} / {len(self.df)})')
            log.info('')
            log.info('')
            self.process_sample(data_attrs)


def main(data_source: str,
         raw_dir: Path,
         target_dir: Path,
         coil_sens_dir: Path,
         cs_dir: Path,
         csv_query: str | None,
         csv_idx_range: List[int | None],
         accelerations: List[int] | None,
         center_fractions: List[float] | None,
         mask_types: List[str] | None,
         mode: str,
         whiten: bool,
         coil_sens_backend: str,
         n_maps: int,
         clip_coil_sensitivities: bool,
         try_cuda: bool
         ):
    csv_file = Path(__file__).parent / 'data_attributes' / f'{data_source}.csv'
    preprocessing = Preprocessing(
        csv_file,
        data_source,
        raw_dir,
        target_dir,
        coil_sens_dir,
        cs_dir=cs_dir,
        csv_query=csv_query,
        csv_idx_range=csv_idx_range,
        accelerations=accelerations,
        center_fractions=center_fractions,
        mask_types=mask_types,
        mode=mode,
        whiten=whiten,
        coil_sens_backend=coil_sens_backend,
        n_maps=n_maps,
        clip_coil_sensitivities=clip_coil_sensitivities,
        try_cuda=try_cuda
    )
    preprocessing.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # arguments
    parser.add_argument(
        'data_source',
        type=str,
        help='Type of data that will be processed (only `ocmr` implemented in public version)'
    )
    parser.add_argument(
        'raw_dir',
        type=Path,
        help='Dir containing raw input files',
    )
    parser.add_argument(
        'target_dir',
        type=Path,
        help='Dir where processed data samples are stored'
    )
    parser.add_argument(
        'coil_sens_dir',
        type=Path,
        help='Dir where coil sensitivities are located or will be stored, if computed'
    )
    parser.add_argument(
        '--cs_dir',
        nargs='?',
        type=Path,
        help='Dir where CS reconstructions are located or will be stored, if computed'
    )
    parser.add_argument(
        '--csv_query',
        nargs='*',
        type=str,
        default=None,
        help='Pandas query to filter data set'
    )
    parser.add_argument(
        '--csv_idx_range',
        nargs='*',
        type=int,
        default=[0, None],
        help='Indices of data samples to process. (start) | (start, stop)'
    )
    parser.add_argument(
        '--accelerations', '--acceleration',
        nargs='+',
        type=int,
        default=None,
        help='Acceleration rates to use for retrospective undersampling'
    )
    parser.add_argument(
        '--center_fractions',
        nargs='+',
        type=float,
        default=None,
        help='Size of fully sampled k-space center to use for retrospective undersampling'
    )
    parser.add_argument(
        '--mask_types', '--mask_type',
        nargs='+',
        type=str,
        default=None,
        choices=['random', 'vista', 'gro', 'cava', 'equispaced_fraction'],
        help='Types of k-space masks to use for retrospective undersampling'
    )
    parser.add_argument(
        '--mode',
        type=str,
        default='dynamic',
        choices=['static', 'dynamic'],
        help='Output mode. static: Save each frame as indiv file. dynamic: additional frame dim in kspace, recon and mask'
    )
    parser.add_argument(
        '--no_whiten',
        dest='whiten',
        action='store_false',
        help='Do not pre-whiten k-space'
    )
    parser.add_argument(
        '--coil_sens_backend',
        type=str,
        choices=['sigpy', 'bart'],
        default='sigpy',
        help='Which backend to use to calc coil sensitivity maps'
    )
    parser.add_argument(
        '--n_maps',
        type=int,
        default=1,
        help='Number of coil sensitivity maps to estimate'
    )
    parser.add_argument(
        '--clip_coil_sensitivities',
        action='store_true',
        help='Clip coil sensitivity maps'
    )
    parser.add_argument(
        '--no_cuda',
        dest='try_cuda',
        action='store_false',
        help='Disable cuda even if available'
    )
    args = parser.parse_args()

    # process args
    if args.csv_query:
        args.csv_query = ' and '.join(args.csv_query)

    match len(args.csv_idx_range):
        case 0:
            args.csv_idx_range = [0, None]
        case 1:
            args.csv_idx_range = [args.csv_idx_range[0], None] # type: ignore
    assert len(args.csv_idx_range) == 2, \
        f'Invalid csv_idx_range: {args.csv_idx_range}'

    # show values of all arguments
    log.info(f'Arguments: ')
    for k, v in vars(args).items():
        if '__' in k:
            continue
        log.info(f'{k}: {v}')

    main(args.data_source,
         args.raw_dir,
         args.target_dir,
         args.coil_sens_dir,
         args.cs_dir,
         args.csv_query,
         args.csv_idx_range,
         args.accelerations,
         args.center_fractions,
         args.mask_types,
         args.mode,
         args.whiten,
         args.coil_sens_backend,
         args.n_maps,
         args.clip_coil_sensitivities,
         args.try_cuda
         )
