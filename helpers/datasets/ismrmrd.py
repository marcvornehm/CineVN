#  Copyright (c) 2023. Marc Vornehm <marc.vornehm@fau.de>.

import logging
import warnings
from typing import Optional

with warnings.catch_warnings():  # ismrmrd adds a warning filter that causes all subsequent warnings to be printed even if they should be filtered
    import ismrmrd
import numpy as np
import sigpy.mri
from tqdm import tqdm

from .dataset import Dataset


class IsmrmrdDataset(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_read_os = True
        self.dset = ismrmrd.Dataset(self.filename, 'dataset', create_if_needed=False)

    def read_header(self):
        self.hdr = ismrmrd.xsd.CreateFromDocument(self.dset.read_xml_header())

    def read_ecg(self):
        ecg = []
        num_waveforms = self.dset.number_of_waveforms()
        ecg_sample_time = None
        for i in range(num_waveforms):  # type: ignore
            w = self.dset.read_waveform(i)
            head = w.getHead()
            if head.waveform_id in [5, 8] and head.channels == 5:  # ecg signal
                if ecg_sample_time is None:
                    ecg_sample_time = head.sample_time_us
                if ecg_sample_time != head.sample_time_us:
                    logging.warning('aborting ECG signal extraction due to inconsistent sample time')
                    break
                ecg.append(w.data)
        if len(ecg) > 0:
            self.ecg = np.concatenate(ecg, axis=1)
        else:
            self.ecg = np.array(0)

    def _read_kdata(self, select_slice: Optional[int] = None):
        # array size
        enc = self.hdr.encoding[0]
        n_x = enc.encodedSpace.matrixSize.x  # type: ignore
        n_y = enc.encodingLimits.kspace_encoding_step_1.maximum + 1  # no zero padding along ky direction  # type: ignore
        n_z = enc.encodedSpace.matrixSize.z  # type: ignore
        n_coils = self.hdr.acquisitionSystemInformation.receiverChannels  # type: ignore
        if n_coils is None:
            raise RuntimeError('Number of receiver channels is not specified in the header')
        n_slices = enc.encodingLimits.slice.maximum + 1  # type: ignore
        n_reps = enc.encodingLimits.repetition.maximum + 1  # type: ignore
        n_phases = enc.encodingLimits.phase.maximum + 1  # type: ignore
        n_sets = enc.encodingLimits.set.maximum + 1  # type: ignore
        n_average = enc.encodingLimits.average.maximum + 1  # type: ignore

        first_acq = 0

        # look for noise scans
        noise_data = []
        noise_dwelltime_us = -1
        while True:
            acq = self.dset.read_acquisition(first_acq)
            head = acq.getHead()
            if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                noise_data.append(acq.data)
                if noise_dwelltime_us == -1:
                    noise_dwelltime_us = head.sample_time_us
                assert noise_dwelltime_us == head.sample_time_us, 'sample_time_us is inconsistent'
                first_acq += 1
            else:
                break
        noise_data = np.array(noise_data)

        # asymmetric echo
        kx_pre = 0
        acq = self.dset.read_acquisition(first_acq)
        head = acq.getHead()
        if head.center_sample * 2 < n_x:
            kx_pre = n_x - head.number_of_samples

        # read the k-space
        acquisition_dwelltime_us = -1
        if select_slice is None:
            k_data = np.zeros((n_x, n_y, n_z, n_coils, n_phases, n_sets, n_slices, n_reps, n_average), dtype=np.complex64)
        else:
            k_data = np.zeros((n_x, n_y, n_z, n_coils, n_phases, n_sets, 1, n_reps, n_average), dtype=np.complex64)
        for i in tqdm(range(first_acq, self.dset.number_of_acquisitions()), leave=False):  # type: ignore
            acq = self.dset.read_acquisition(i)
            head = acq.getHead()

            if acquisition_dwelltime_us == -1:
                acquisition_dwelltime_us = head.sample_time_us
            assert acquisition_dwelltime_us == head.sample_time_us, 'sample_time_us is inconsistent'

            # Stuff into the buffer
            acq_idx: ismrmrd.EncodingCounters = acq.idx  # type: ignore
            y_idx = acq_idx.kspace_encode_step_1
            z_idx = acq_idx.kspace_encode_step_2
            phase_idx = acq_idx.phase
            set_idx = acq_idx.set
            slice_idx = acq_idx.slice
            rep_idx = acq_idx.repetition
            avg_idx = acq_idx.average

            if acq.isFlagSet(ismrmrd.ACQ_IS_REVERSE):
                raise NotImplementedError('Acquisition is reversed. Flip it before proceeding!')

            if select_slice is None:
                k_data[kx_pre:, y_idx, z_idx, :, phase_idx, set_idx, slice_idx, rep_idx, avg_idx] = np.transpose(acq.data)
            elif slice_idx == select_slice:
                k_data[kx_pre:, y_idx, z_idx, :, phase_idx, set_idx, 0, rep_idx, avg_idx] = np.transpose(acq.data)
            else:
                continue

            read_dir = np.array(head.read_dir)
            read_dir_prev = self.get_read_dir(slice_idx)
            if read_dir_prev is None:
                self._read_dir[slice_idx] = read_dir
            elif not np.allclose(read_dir, read_dir_prev):
                raise ValueError('read_dir is inconsistent')

            phase_dir = np.array(head.phase_dir)
            phase_dir_prev = self.get_phase_dir(slice_idx)
            if phase_dir_prev is None:
                self._phase_dir[slice_idx] = phase_dir
            elif not np.allclose(phase_dir, phase_dir_prev):
                raise ValueError('phase_dir is inconsistent')

            slice_dir = np.array(head.slice_dir)
            slice_dir_prev = self.get_slice_dir(slice_idx)
            if slice_dir_prev is None:
                self._slice_dir[slice_idx] = slice_dir
            elif not np.allclose(slice_dir, slice_dir_prev):
                raise ValueError('slice_dir is inconsistent')

            position = np.array(head.position)
            position_prev = self.get_position(slice_idx)
            if position_prev is None:
                self._position[slice_idx] = position
            elif not np.allclose(position, position_prev):
                raise ValueError('position is inconsistent')

        # discard pilot tone signal, if present
        if any(param.name == 'PilotTone' and param.value == 1 for param in self.hdr.userParameters.userParameterLong):  # type: ignore
            logging.info('Pilot Tone is on, discarding the first 3 and last 1 k-space point for each line')
            k_data[kx_pre:kx_pre+3] = 0
            k_data[-1] = 0

        # discard all but first average (don't use mean of averages in order to preserve SNR of a single scan)
        self.k_data_full = k_data[..., 0, None]
        self.k_data = self.k_data_full

        # estimate noise data
        if noise_data.size == 0:
            _, noise_data = self.estimate_noise(True)
        else:
            noise_data *= np.sqrt(noise_dwelltime_us / acquisition_dwelltime_us)
            noise_data = np.moveaxis(noise_data, 0, 1)
            noise_data = np.reshape(noise_data, (noise_data.shape[0], -1))

        # noise correlation matrix
        self.noise_cov = sigpy.mri.get_cov(noise_data)
