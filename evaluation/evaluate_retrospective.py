"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

import argparse
import warnings
from pathlib import Path
from typing import Any

import h5py
import imageio
with warnings.catch_warnings():  # ismrmrd adds a warning filter that causes all subsequent warnings to be printed even if they should be filtered
    import ismrmrd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.exposure import equalize_adapthist

from cinevn.metrics import hfen, nmse, psnr, ssim, ssim_xt
from cinevn.mri import apply_affine_to_annotations, apply_affine_to_image, ifftnc, rss_complex
from helpers.dicom import array2dicom


def compute_metrics(
        gt: torch.Tensor,
        pred: torch.Tensor,
        maxval: float | None,
        center: tuple[int, int] | np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        'NMSE': nmse(gt, pred).item(),
        'PSNR': psnr(gt, pred, maxval=torch.tensor(maxval)).item(),
        'SSIM': ssim(gt, pred, maxval=torch.tensor(maxval)).item(),
        'HFEN': hfen(gt, pred).item(),
    }
    if center is not None:
        metrics['SSIM_XT'] = ssim_xt(gt, pred, torch.tensor(center), maxval=torch.tensor(maxval)).item()
    return metrics


def process_files(
        experiment: str,
        test_set: str,
        logs_dir: Path,
        gt_dir: Path | None = None,
        cs_dir: Path | None = None,
        out_dir: Path | None = None,
        csv_path: Path | None = None,
        gt_key: str = 'reconstruction_weighted',
        equalize_histogram: bool = True,
):
    # complete paths
    vn_dir = logs_dir / experiment / 'reconstructions' / test_set
    if gt_dir is not None:
        gt_dir = gt_dir / test_set
    if cs_dir is not None:
        cs_dir = cs_dir / test_set.replace('_test', '')
    if out_dir is not None:
        out_dir = out_dir / experiment / test_set
    else:
        out_dir = logs_dir / experiment / 'evaluations' / test_set
    if csv_path is None:
        csv_path = out_dir / 'evaluation.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    df: pd.DataFrame | None = None
    files_vn = sorted(vn_dir.glob('*.h5'))
    if len(files_vn) == 0:
        raise RuntimeError(f'No VN reconstructions found in {vn_dir}')

    for f_vn in files_vn:
        print(f'Processing file {f_vn.stem}')

        # dictionary for reconstructions
        recons: dict[str, torch.Tensor] = {}

        with h5py.File(f_vn, 'r') as hf_vn:
            # load VarNet reconstruction
            recons['vn'] = torch.as_tensor(hf_vn['reconstruction'][:])  # [slice, frame, x, y]  # type: ignore
            recons['vn'] = torch.abs(recons['vn'])

            # load affine transformation parameters for orientation normalization
            affine = np.stack([
                int(hf_vn.attrs.get('rotation', 0)),
                int(hf_vn.attrs.get('flip_horizontal', 0)),
                int(hf_vn.attrs.get('flip_vertical', 0)),
            ])

            # load slice position and orientation information
            if 'position' in hf_vn:
                position = np.array(hf_vn.get('position'))  # [slice, dims]
            else:
                position = np.array([[0., 0., 0.]])
            if 'orientation' in hf_vn:
                orientation = np.array(hf_vn.get('orientation'))  # [slice, direction, dims]
            else:
                orientation = np.array([[[1., 0., 0.], [0., 1., 0.], [0., 0., -1.]]])

            # read MRD header and temporal resolution
            mrd_header = ismrmrd.xsd.CreateFromDocument(hf_vn['ismrmrd_header'][()])  # type: ignore
            t_res = mrd_header.sequenceParameters.TR[0]  # type: ignore

        annotations: np.ndarray | None = None
        has_gt = False
        if gt_dir is not None:
            f_gt = gt_dir / f_vn.name
            if f_gt.exists():
                with h5py.File(f_gt, 'r') as hf_gt:
                    # load ground truth
                    if gt_key in hf_gt:
                        recons['gt'] = torch.as_tensor(hf_gt[gt_key][:])  # [slice, frame, x, y]  # type: ignore
                        recons['gt'] = torch.abs(recons['gt'])
                        has_gt = True

                    # load downsampled kspace
                    kspace = torch.as_tensor(hf_gt['kspace'][:])  # [slice, coil, frame, kx, ky]  # type: ignore

                    # reconstruct zero-filling solution
                    recons['zf'] = rss_complex(ifftnc(torch.view_as_real(kspace), ndim=2), dim=1)

                    # apply phase cropping
                    enc = mrd_header.encoding[0]
                    enc_mat_x: int = enc.encodedSpace.matrixSize.x  # type: ignore
                    rec_mat_x: int = enc.reconSpace.matrixSize.x  # type: ignore
                    rec_mat_y: int = enc.reconSpace.matrixSize.y  # type: ignore
                    if rec_mat_x == enc_mat_x:
                        # 2D interpolation on
                        target_phase_dim = rec_mat_y // 2
                    else:
                        # 2D interpolation off
                        target_phase_dim = rec_mat_y
                    crop_from = int((recons['zf'].shape[-1] - target_phase_dim) / 2)
                    crop_to = crop_from + target_phase_dim
                    recons['zf'] = recons['zf'][..., crop_from:crop_to]

                    # load annotations
                    if 'center' in hf_gt.attrs and 'bbox' in hf_gt.attrs:
                        annotations = np.concatenate([
                            np.array(hf_gt.attrs['center']).reshape((1, 2)),
                            np.array(hf_gt.attrs['bbox']).reshape((2, 2)),
                        ])

        if cs_dir is not None:
            # read from hdf5 file
            f_cs = cs_dir / f_vn.stem / 'reconstruction.h5'
            with h5py.File(f_cs, 'r') as hf_cs:
                # load CS reconstruction
                recons['cs'] = torch.as_tensor(hf_cs['reconstruction_cs'][:])  # [slice, frame, x, y]  # type: ignore
                recons['cs'] = torch.abs(recons['cs'])

        # make sure that slice dimension is 1 in all reconstructions and squeeze this dimension
        for tag, recon in recons.items():
            recons[tag] = recon.squeeze(0)
        position = position.squeeze(0)
        orientation = orientation.squeeze(0)

        # apply transformations to annotations
        center: np.ndarray | None = None
        bbox: np.ndarray | None = None
        if annotations is not None:
            input_shape = tuple(recons['vn'].shape[-2:])
            annotations = apply_affine_to_annotations(annotations, input_shape, *affine)
            center = annotations[0]
            bbox = annotations[1:3].flatten()
            bbox = np.array([
                min(bbox[0], bbox[2]),
                min(bbox[1], bbox[3]),
                max(bbox[0], bbox[2]),
                max(bbox[1], bbox[3]),
            ])

        # apply transformations to reconstructions
        for tag, recon in recons.items():
            recons[tag] = apply_affine_to_image(recon, *affine)

        # windowing values
        maxval = torch.max(recons['gt'] if has_gt else recons['vn']).item()
        minval = torch.min(recons['gt'] if has_gt else recons['vn']).item()

        # extract ROIs
        rois = {}
        for tag, recon in recons.items():
            if bbox is not None:
                rois[tag] = extract_roi(recon, bbox)

        # error images
        errors = {}
        for tag, recon in recons.items():
            if has_gt and tag not in ['gt', 'zf']:
                errors[tag] = abs(recons['gt'] - recon)

        # output directory
        p = out_dir / f'{f_vn.stem}'
        p.mkdir(parents=True, exist_ok=True)

        # compute evaluation metrics
        # SSIM in 3D mode (use channel_axis=0 for mean of SSIMs per frame)
        metrics = {}
        for tag, recon in recons.items():
            if has_gt and tag not in ['gt', 'zf']:
                metrics[tag] = compute_metrics(recons['gt'], recon, maxval, center=center)
        metrics_roi = {}
        for tag, roi in rois.items():
            if has_gt and tag not in ['gt', 'zf']:
                metrics_roi[tag] = compute_metrics(rois['gt'], roi, maxval)

        # save reconstructions and errors as image files
        metrics_params = {'xy': (5, 5), 'anchor': 'la', 'fill': (255, 255, 0), 'font': ImageFont.truetype('arial', 12)}
        if '4ch' in p.name or 'fs_0059' in p.name:
            metrics_params.update({'xy': (5, recons['vn'].shape[-2] - 15), 'anchor': 'ld'})
        for tag in recons.keys():
            # for zf reconstruction we use a different windowing because of the higher dynamic range and because
            # visual comparability is not as important for this reconstruction
            maxval_ = torch.quantile(recons['zf'], 0.98).item() if tag == 'zf' else maxval
            write_images(
                recons[tag].numpy(), minval, maxval_, p, f'recon_{tag}', center, t_res=t_res,
                equalize_histogram=(tag != 'zf'), metrics=metrics.get(tag, None), metrics_params=metrics_params,
            )
        for tag in errors.keys():
            write_images(
                errors[tag].numpy(), 0, maxval / 20, p, f'error_{tag}', center, colored=True, t_res=t_res,
                equalize_histogram=False,
            )
        if center is not None and bbox is not None:
            # save overview images
            write_overview_images(
                recons['gt'].numpy() if has_gt else recons['vn'].numpy(), minval, maxval, center, bbox, p,
                t_res=t_res, equalize_histogram=equalize_histogram,
            )

        # save reconstructions as DICOM files
        for i, tag in enumerate(recons):
            recon = recons[tag]
            rot90 = affine[0] in [90, 270]
            img = recon.numpy()
            img = np.around(img * 32767 / img.max()).astype(np.int16)
            images = [img[j] for j in range(img.shape[0])]
            positions = [position] * img.shape[0]
            orientations = [orientation] * img.shape[0]
            dcm = array2dicom(
                images, mrd_header, positions, orientations, series_number=i + 1, instance_number=1,
                series_description_appendix=f'_{tag}', rot90=rot90, enhanced=True,
            )
            assert isinstance(dcm, pydicom.Dataset)
            dcm.save_as(p / f'recon_{tag}.dcm')

        # write image metrics if there is a reference
        if has_gt:
            # create empty dataframe if not created previously
            if df is None:
                columns = []
                metric_keys = metrics['vn'].keys()
                assert all([metric_keys == metric_dict.keys() for metric_dict in metrics.values()])
                columns.extend([f'{met.upper()} {tag.upper()}' for met in metric_keys for tag in metrics.keys()])
                if 'vn' in metrics_roi:
                    metric_roi_keys = metrics_roi['vn'].keys()
                    assert all([metric_roi_keys == m.keys() for m in metrics_roi.values()])
                    columns.extend(
                        [f'{met.upper()} {tag.upper()} ROI' for met in metric_roi_keys for tag in metrics_roi.keys()]
                    )
                df = pd.DataFrame(columns=columns)

            new_row = {}
            for tag, metrics_recon in metrics.items():
                for metric_name, metric_value in metrics_recon.items():
                    new_row[f'{metric_name.upper()} {tag.upper()}'] = metric_value
            for tag, metrics_roi_recon in metrics_roi.items():
                for metric_name, metric_value in metrics_roi_recon.items():
                    new_row[f'{metric_name.upper()} {tag.upper()} ROI'] = metric_value
            df.loc[f_vn.stem] = pd.Series(new_row)

    if df is not None:
        csv_path.unlink(missing_ok=True)  # delete old csv file
        df.to_csv(csv_path)

        print(f'Results for experiment {experiment} and test set {test_set}:')
        print(df.mean())


def adjust_contrast(image: np.ndarray, minval: float, maxval: float, equalize_histogram: bool = True):
    out = np.clip(image, minval, maxval).astype(image.dtype)
    out = (out - minval) / (maxval - minval)
    if equalize_histogram:
        out = equalize_adapthist(out, clip_limit=0.02)
    return out


def extract_roi(image: torch.Tensor, bbox: np.ndarray) -> torch.Tensor:
    return image[..., bbox[1]:bbox[3] + 1, bbox[0]:bbox[2] + 1]


def create_profiles(cine: np.ndarray, center: np.ndarray, stretch_factor: int = 5) -> tuple[np.ndarray, np.ndarray]:
    assert cine.ndim == 3

    xt_profile = cine[:, center[1], :]
    yt_profile = cine[:, :, center[0]]

    # stretch
    xt_profile = np.repeat(xt_profile, stretch_factor, axis=0)
    yt_profile = np.repeat(yt_profile, stretch_factor, axis=0)

    # transpose y-t profile
    yt_profile = yt_profile.T

    return xt_profile, yt_profile


def draw_crosshairs(image: np.ndarray, center: np.ndarray):
    # draw crosshairs in red
    color = np.array([1, 0, 0])
    image[..., center[1], :, :] = color
    image[..., center[0], :] = color

    return image


def draw_bbox(image: np.ndarray, bbox: np.ndarray):
    # draw bounding box in yellow
    x_low = bbox[0]
    y_low = bbox[1]
    x_high = bbox[2]
    y_high = bbox[3]
    color = np.array([1, 1, 0])
    image[..., y_low, x_low:x_high + 1, :] = color
    image[..., y_high, x_low:x_high + 1, :] = color
    image[..., y_low:y_high + 1, x_low, :] = color
    image[..., y_low:y_high + 1, x_high, :] = color

    return image


def write_images(
        image: np.ndarray,
        minval: float,
        maxval: float,
        path: Path,
        name: str,
        center: np.ndarray | None,
        colored: bool = False,
        t_res: float | None = None,
        equalize_histogram: bool = True,
        metrics: dict[str, float] | None = None,
        metrics_params: dict[str, Any] | None = None,
        metrics_format: str | None = None,
):
    if metrics_format is None:
        metrics_format = 'SSIM: {SSIM:02.2%}\nPSNR: {PSNR:02.2f}dB'

    # adjust contrast
    image_np = adjust_contrast(image, minval, maxval, equalize_histogram=equalize_histogram)

    # save as TIFF and GIF
    cine_pil = [Image.fromarray(image_np[i] * 255).convert('RGB') for i in range(image_np.shape[0])]
    save_tiff(cine_pil, path / name)
    save_gif(cine_pil, path / name, t_res)

    # TIFF and GIF with metrics
    if metrics is not None:
        if metrics_params is None or 'xy' not in metrics_params:
            raise ValueError('`metrics_params` must contain the key `xy`.')
        cine_draw = [ImageDraw.Draw(img) for img in cine_pil]
        for d in cine_draw:
            d.multiline_text(text=metrics_format.format(**metrics), **metrics_params)
        save_tiff(cine_pil, path / f'{name}_metrics')
        save_gif(cine_pil, path / f'{name}_metrics', t_res)

    # save as png with pyplot default colormap
    if colored:
        path_ = path / f'{name}_png'
        path_.mkdir(exist_ok=True)
        for i in range(image_np.shape[0]):
            plt.imsave(str(path_ / f'{name}_{i:02d}.png'), image_np[i], vmin=0, vmax=1)

    # xt-/yt-profiles
    if center is not None:
        profile_xt, profile_yt = create_profiles(image_np, center)
        if colored:
            plt.imsave(str(path / f'{name}_xt.png'), profile_xt, vmin=0, vmax=1)
            plt.imsave(str(path / f'{name}_yt.png'), profile_yt, vmin=0, vmax=1)
        else:
            imageio.imwrite(path / f'{name}_xt.png', (profile_xt * 255).astype(np.uint8))
            imageio.imwrite(path / f'{name}_yt.png', (profile_yt * 255).astype(np.uint8))


def write_overview_images(
        recon_ref: np.ndarray,
        minval: float,
        maxval: float,
        center: np.ndarray,
        bbox: np.ndarray,
        path: Path,
        t_res: float | None = None,
        equalize_histogram: bool = True,
):
    overview = adjust_contrast(  # [frame, y, x, channel=1]
        recon_ref[..., None], minval, maxval, equalize_histogram=equalize_histogram
    )
    overview = np.repeat(overview, 3, axis=-1)  # [frame, y, x, channel=3]

    overview_crosshairs = draw_crosshairs(overview.copy(), center)
    overview_crosshairs = (overview_crosshairs * 255).astype(np.uint8)
    imageio.imwrite(path / 'overview_crosshairs.png', overview_crosshairs[0])
    overview_crosshairs = [Image.fromarray(overview_crosshairs[i], 'RGB') for i in range(overview.shape[0])]
    save_gif(overview_crosshairs, path / 'overview_crosshairs', t_res)

    overview_bbox = draw_bbox(overview.copy(), bbox)
    overview_bbox = (overview_bbox * 255).astype(np.uint8)
    imageio.imwrite(path / 'overview_bbox.png', overview_bbox[0])
    overview_bbox = [Image.fromarray(overview_bbox[i], 'RGB') for i in range(overview.shape[0])]
    save_gif(overview_bbox, path / 'overview_bbox', t_res)

    overview_both = draw_bbox(draw_crosshairs(overview.copy(), center), bbox)
    overview_both = (overview_both * 255).astype(np.uint8)
    imageio.imwrite(path / 'overview_both.png', overview_both[0])
    overview_both = [Image.fromarray(overview_both[i], 'RGB') for i in range(overview.shape[0])]
    save_gif(overview_both, path / 'overview_both', t_res)


def save_tiff(images: list[Image.Image], filename: Path):
    if filename.suffix == '':
        filename = filename.with_suffix('.tiff')
    if len(images) == 1:
        images[0].save(filename)
    else:
        images[0].save(filename, save_all=True, append_images=images[1:])


def save_gif(images: list[Image.Image], filename: Path, t_res: float | None):
    if filename.suffix == '':
        filename = filename.with_suffix('.gif')
    images[0].save(filename, save_all=True, append_images=images[1:], duration=125 if t_res is None else t_res, loop=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('experiment', type=str,
                        help='Name of the experiment.')
    parser.add_argument('test_set', type=str,
                        help='Name of the test set.')
    parser.add_argument('logs_dir', type=Path,
                        help='Directory containing the network logs and outputs.')
    parser.add_argument('--gt_dir', type=Path,
                        help='Directory containing sub-folders for the test sets with ground truth reconstructions.')
    parser.add_argument('--cs_dir', type=Path,
                        help='Directory containing sub-folders for the test sets with CS reconstructions.')
    parser.add_argument('--out_dir', type=Path,
                        help='Directory for output images/movies. If not given, `vn_dir` is used.')
    parser.add_argument('--csv_path', type=Path,
                        help='Path to CSV file where evaluation metrics are written. If not given, the CSV file is '
                             'written in out_dir.')
    parser.add_argument('--gt_key', type=str, default='reconstruction_weighted',
                        help='Key of ground truth reconstruction in HDF5 file.')
    parser.add_argument('--equalize_histogram', action='store_true', default=True,
                        help='Equalize histogram of output images.')
    parser.add_argument('--no-equalize_histogram', dest='equalize_histogram', action='store_false',
                        help='Do not equalize histogram of output images.')

    args = parser.parse_args()

    process_files(
        args.experiment, args.test_set, args.logs_dir, gt_dir=args.gt_dir, cs_dir=args.cs_dir, out_dir=args.out_dir,
        csv_path=args.csv_path, gt_key=args.gt_key, equalize_histogram=args.equalize_histogram,
    )
