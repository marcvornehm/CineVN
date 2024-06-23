"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import sigpy.mri
import torch

from helpers.datasets import Dataset, IsmrmrdDataset
from helpers import mri, utils
from cinevn.models.varnet import VarNet

try:
    import cupy as cp
except ImportError:
    cp = None


logging.basicConfig(
    handlers=[logging.StreamHandler()],
    level=logging.INFO,
    format='%(asctime)s %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H-%M-%S'
)
log = logging.getLogger(__name__)


def reconstruct_mrd(
    dataset_path: str | Path,
    checkpoint_path: str | Path,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
):
    # load dataset
    logging.info(f'Loading dataset from {dataset_path}')
    dataset_path = Path(dataset_path)
    dataset = IsmrmrdDataset(name=dataset_path.name, filename=dataset_path, device='cpu')  # it will make our life easier in this case if we just use the CPU here
    dataset.read_meta()
    dataset.read_kdata(whiten=True)

    # load model
    logging.info(f'Loading model from {checkpoint_path}')
    checkpoint_path = Path(checkpoint_path)
    model = VarNet.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()


    # iterate over slices
    for slc_idx in range(dataset.n_slices):
        logging.info(f'Processing slice {slc_idx + 1}/{dataset.n_slices}')
        dataset.select_slice(slc_idx)

        # the number of frames may differ between slices
        # check for last frame with data
        sum_per_frame = np.moveaxis(np.abs(dataset.k_data), 4, 0).reshape(dataset.k_data.shape[4], -1).sum(axis=1)
        last_nonzero = np.nonzero(sum_per_frame)[0][-1]
        dataset.k_data = dataset.k_data[:, :, :, :, :last_nonzero + 1, :, :, :, :]

        # remove readout oversampling
        dataset.k_data = mri.utils.crop_readout_oversampling(dataset.k_data)

        # apply phase padding and get target size without phase oversampling (cropped after recon)
        dataset, pad_left, pad_right = phase_pad_dataset(dataset)
        assert dataset.k_data.shape[2] == 1, 'kz must be 1'
        assert dataset.k_data.shape[5] == 1, 'set must be 1'
        assert dataset.k_data.shape[7] == 1, 'rep must be 1'
        assert dataset.k_data.shape[8] == 1, 'avg must be 1'
        dataset.k_data = dataset.k_data.squeeze(axis=(2, 5, 7, 8))  # [kx, ky, coil, frame, slice]
        enc = dataset.hdr.encoding[0]
        if enc.reconSpace.matrixSize.x == enc.encodedSpace.matrixSize.x:  # 2D interpolation on
            recon_size = (enc.reconSpace.matrixSize.x // 2, enc.reconSpace.matrixSize.y // 2)
        else:  # 2D interpolation off
            recon_size = (enc.reconSpace.matrixSize.x, enc.reconSpace.matrixSize.y)

        # get mask
        mask = get_mask(dataset.k_data, pad_left, pad_right)
        acceleration = mask.size / mask.sum()
        logging.info(f'  Acceleration rate: {acceleration:.2f}')

        # estimate coil sensitivities
        logging.info('  Estimating coil sensitivities')
        k_data_avg = dataset.k_data.sum(axis=3) / (mask.sum(axis=1)[None, :, None, None] + np.finfo(float).eps)
        k_data_avg = np.moveaxis(k_data_avg, 2, 0)
        espirit_device = sigpy.Device(0) if (cp is not None and device == 'cuda') else sigpy.Device(-1)
        sens_maps = sigpy.mri.app.EspiritCalib(k_data_avg, calib_width=16, crop=0, show_pbar=False, device=espirit_device).run()
        if cp is not None and cp.get_array_module(sens_maps) == cp:
            sens_maps = cp.asnumpy(sens_maps)

        # convert to required layout and create torch tensors
        k_data = dataset.k_data.transpose(4, 2, 3, 0, 1)  # [slice=1, coil, frame, kx, ky]
        k_data = torch.view_as_real(torch.from_numpy(k_data)).to(device=device, dtype=torch.float32)
        mask = mask.T[None, None, :, None]  # [slice=1, coil=1, frame, kx=1, ky]
        mask = torch.from_numpy(mask)[..., None].to(device=device, dtype=torch.float32)
        sens_maps = sens_maps.transpose(3, 0, 1, 2)[:, None, :, None]  # [slice=1, set=1, coil, frame=1, x, y]
        sens_maps = torch.view_as_real(torch.from_numpy(sens_maps)).to(device=device, dtype=torch.float32)

        # infer model
        logging.info('  Reconstructing')
        with torch.no_grad():
            output = model(k_data, mask, sens_maps=sens_maps).cpu()
        output = torch.view_as_complex(output).numpy()[0]
        vstart = output.shape[-2] // 2 - recon_size[-2] // 2
        vstop = output.shape[-2] // 2 + (recon_size[-2] + 1) // 2
        hstart = output.shape[-1] // 2 - recon_size[-1] // 2
        hstop = output.shape[-1] // 2 + (recon_size[-1] + 1) // 2
        output = output[..., vstart:vstop, hstart:hstop]

        # save as GIF
        output_name = f'{dataset.filename.stem}_slice{slc_idx:02d}.gif'
        logging.info(f'  Saving as {output_name}')
        tr = dataset.hdr.sequenceParameters.TR[0]
        utils.save(output, output_name, clip=True, tres=tr)



def phase_pad_dataset(dset: Dataset) -> tuple[Dataset, int, int]:
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

    dset.k_data = mri.utils.apply_phase_padding(
        dset.k_data,
        phase_padding_left,
        phase_padding_right
    )

    return dset, phase_padding_left, phase_padding_right


def get_mask(k_data: np.ndarray, pad_left: int, pad_right: int) -> np.ndarray:
    # obtain mask (zero-padded)
    mask = (abs(np.mean(np.abs(k_data), axis=2)) > 0).astype(int)  # [kx, ky, frame, slice]

    # collapse kx and slice dimension of mask
    mask = mask[0, :, :, 0]  # [ky, frame]

    # remove zero-padding from mask (ignoring negative padding if phase resolution > 100%)
    mask_unpadded = mask[max(0, pad_left) : mask.shape[0] - max(0, pad_right), :]

    # pad mask with ones
    mask = np.ones(mask.shape)
    mask[max(0, pad_left) : mask.shape[0] - max(0, pad_right), :] = mask_unpadded

    return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('checkpoint', type=Path)
    args = parser.parse_args()

    reconstruct_mrd(args.dataset, args.checkpoint)
