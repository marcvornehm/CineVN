"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.

Part of this source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import warnings
from collections import defaultdict
from pathlib import Path

import h5py
with warnings.catch_warnings():  # ismrmrd adds a warning filter that causes all subsequent warnings to be printed even if they should be filtered
    import ismrmrd
import numpy as np
import pytorch_lightning as pl
import torch
from optuna.integration import PyTorchLightningPruningCallback

from cinevn import mri
from helpers import save_movie

from .data_module import FastMriDataModule


class ValImageLogger(pl.Callback):
    def __init__(self, num_log_images: int = 16, logging_interval: int = 10, log_always_after: float = 0.8):
        """
        Args:
            num_log_images: Number of validation images to log. Defaults to 16.
            logging_interval: After how many epochs to log validation images. Defaults to 10.
            log_always_after: After what percentage of trainer.max_epochs to log images in every epoch.
        """
        super().__init__()

        self.num_log_images = num_log_images
        self.val_log_indices = None

        self.logging_interval = logging_interval
        self.log_always_after = log_always_after

    @property
    def state_key(self) -> str:
        return self._generate_state_key(
            num_log_images=self.num_log_images, val_log_indices=self.val_log_indices,
            logging_interval=self.logging_interval, log_always_after=self.log_always_after,
        )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not isinstance(outputs, dict):
            raise RuntimeError('Expected outputs to be a dict')

        # pick a set of images to log if we don't have one already
        if self.val_log_indices is None:
            if isinstance(trainer.val_dataloaders, list):
                num_samples = sum(len(dataloader.dataset) for dataloader in trainer.val_dataloaders)
            elif trainer.val_dataloaders is not None:
                num_samples = len(trainer.val_dataloaders.dataset)
            else:
                raise RuntimeError('Could not determine number of validation samples')
            self.val_log_indices = list(np.random.permutation(num_samples)[:self.num_log_images])

        # log images to tensorboard (less often in the beginning to reduce log file size)
        max_epochs = trainer.max_epochs if trainer.max_epochs is not None else -1
        if trainer.current_epoch % self.logging_interval == 0 \
                or trainer.current_epoch >= max_epochs * self.log_always_after:
            if isinstance(batch.sample_idx, int):
                sample_indices = [batch.sample_idx]
            else:
                sample_indices = batch.sample_idx
            for i, sample_index in enumerate(sample_indices):
                if sample_index in self.val_log_indices:
                    log_key = f'{sample_index}_{batch.fname[i]}_{batch.slice_num[i]}'
                    affine = batch.affine[i]
                    annotations = batch.annotations[i]
                    output = outputs['output'][i]
                    target = outputs['target'][i]

                    if annotations.isnan().any():
                        annotations = None
                    else:
                        annotations = annotations.to(torch.int32)
                        annotations = mri.apply_affine_to_annotations(annotations, target.shape[-2:], *affine)
                    output = mri.apply_affine_to_image(output, *affine)
                    target = mri.apply_affine_to_image(target, *affine)
                    header = ismrmrd.xsd.CreateFromDocument(batch.header[i])
                    t_res = header.sequenceParameters.TR[0]  # type: ignore

                    if torch.is_complex(output) and torch.is_complex(target):
                        # phase maps
                        output_phase = torch.angle(output)
                        target_phase = torch.angle(target)

                        # phase error
                        error_phase_zero = torch.abs(target_phase - output_phase)
                        error_phase_plus = torch.abs(target_phase - (output_phase + 2 * np.pi))
                        error_phase_minus = torch.abs(target_phase - (output_phase - 2 * np.pi))
                        error_phase = torch.min(error_phase_zero, torch.min(error_phase_plus, error_phase_minus))

                        # normalize
                        output_phase = (output_phase + np.pi) / (2 * np.pi)
                        target_phase = (target_phase + np.pi) / (2 * np.pi)
                        error_phase = error_phase / np.pi

                        self.log_images(trainer, log_key + '_phase', target_phase, output_phase, error_phase, t_res=t_res)

                    # magnitude images
                    if torch.is_complex(output):
                        output = torch.abs(output)
                    if torch.is_complex(target):
                        target = torch.abs(target)

                    # magnitude error
                    error = torch.abs(target - output)

                    # clamp and normalize
                    error = torch.clamp(error / target.max() * 20, 0, 1)
                    output = torch.clamp(output, torch.quantile(output, 0.03), torch.quantile(output, 0.97))
                    output = (output - output.min()) / (output.max() - output.min())
                    target = torch.clamp(target, torch.quantile(target, 0.03), torch.quantile(target, 0.97))
                    target = (target - target.min()) / (target.max() - target.min())

                    self.log_images(trainer, log_key, target, output, error, annotations=annotations, t_res=t_res)

    def log_images(
            self,
            trainer: pl.Trainer,
            key: str,
            target: torch.Tensor,
            output: torch.Tensor,
            error: torch.Tensor,
            annotations: torch.Tensor | np.ndarray | None = None,
            t_res: float = 0,
    ):
        assert target.ndim == output.ndim == error.ndim

        fps = 1000 / t_res if t_res > 0 else 8
        if trainer.current_epoch == 0:  # log target only once
            self.log_video(trainer, f'{key}/target', target, fps=fps)
        self.log_video(trainer, f'{key}/reconstruction', output, fps=fps)
        self.log_video(trainer, f'{key}/error', error, fps=fps)

        if annotations is not None:
            center = (int(annotations[0][0].item()), int(annotations[0][1].item()))

            if trainer.current_epoch == 0:  # log target and profiles location only once
                target_profiles = self.create_profiles(target, center)
                self.log_image(trainer, f'{key}/xt_target', target_profiles[0])
                self.log_image(trainer, f'{key}/yt_target', target_profiles[1])

                profiles_location = self.draw_crosshairs(target, center)
                self.log_image(trainer, f'{key}/profiles_location', profiles_location)

            output_profiles = self.create_profiles(output, center)
            self.log_image(trainer, f'{key}/xt_reconstruction', output_profiles[0])
            self.log_image(trainer, f'{key}/yt_reconstruction', output_profiles[1])

    @staticmethod
    def log_image(trainer: pl.Trainer, name: str, image: torch.Tensor, global_step: int | None = None):
        """
        expected shape of image: [y, x] or [channel, y, x]
        """
        if image.ndim == 2:
            image = image.unsqueeze(dim=0)  # [channel, y, x]
        step = global_step if global_step else trainer.global_step
        trainer.logger.experiment.add_image(name, image, global_step=step)  # type: ignore

    @staticmethod
    def log_video(trainer: pl.Trainer, name: str, video: torch.Tensor, global_step: int | None = None, fps: float = 8):
        """
        expected shape of video: [frame, y, x] or [frame, channel, y, x]
        """
        if video.ndim == 3:
            video = video.unsqueeze(dim=1)  # [frame, channel, y, x]
            video = video.expand(-1, 3, -1, -1)  # expand to 3 channels
        video = video.unsqueeze(dim=0)  # [batch, frame, channel, y, x]
        step = global_step if global_step else trainer.global_step
        trainer.logger.experiment.add_video(name, video, global_step=step, fps=fps)  # type: ignore

    @staticmethod
    def create_profiles(cine: torch.Tensor, center: tuple[int, int], stretch_factor: int = 5) \
            -> tuple[torch.Tensor, torch.Tensor]:
        xt_profile = cine[:, center[1], :]
        yt_profile = cine[:, :, center[0]]

        # stretch
        xt_profile = torch.repeat_interleave(xt_profile, stretch_factor, dim=0)
        yt_profile = torch.repeat_interleave(yt_profile, stretch_factor, dim=0)

        return xt_profile, yt_profile

    @staticmethod
    def draw_crosshairs(cine: torch.Tensor, center: tuple[int, int]) -> torch.Tensor:
        crosshairs = cine[0, :, :]  # static image (first frame) [y, x]
        crosshairs = torch.stack(3 * [crosshairs])  # stack along channel dimension to [3, y, x]
        crosshairs[:, center[1], :] = torch.tensor([[1], [0], [0]])
        crosshairs[:, :, center[0]] = torch.tensor([[1], [0], [0]])
        return crosshairs


class SaveReconstructions(pl.Callback):
    def __init__(self, save_gif: bool = False):
        super().__init__()
        self.save_gif = save_gif
        self.test_batch_outputs = []

    @property
    def state_key(self) -> str:
        return self._generate_state_key(save_gif=self.save_gif)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not isinstance(outputs, dict):
            raise RuntimeError('Expected outputs to be a dict')
        headers = [ismrmrd.xsd.CreateFromDocument(header) for header in batch.header]
        self.test_batch_outputs.append({
            'fname': batch.fname,
            'slice': batch.slice_num,
            'output': outputs['output'],
            'affine': batch.affine,
            'position': batch.position,
            'orientation': batch.orientation,
            'headers': headers,
        })

    def on_test_epoch_end(self, trainer, pl_module):
        outputs = defaultdict(dict)
        affine = dict()
        headers = dict()
        positions = defaultdict(dict)
        orientations = defaultdict(dict)

        # use dicts for aggregation to handle duplicate slices in ddp mode
        for log in self.test_batch_outputs:
            for i, (fname, slice_num) in enumerate(zip(log['fname'], log['slice'])):
                output = log['output'][i]
                slice_num_ = int(slice_num.cpu())
                affine[fname] = log['affine'][i].cpu().numpy()
                outputs[fname][slice_num_] = output.cpu().numpy()
                headers[fname] = log['headers'][i]
                positions[fname][slice_num_] = log['position'][i].cpu().numpy()
                orientations[fname][slice_num_] = log['orientation'][i].cpu().numpy()

        # stack/concatenate all the slices for each file
        outputs_stacked = dict()
        positions_stacked = dict()
        orientations_stacked = dict()
        for fname in outputs:
            outputs_stacked[fname] = np.stack([out for _, out in sorted(outputs[fname].items())])
            positions_stacked[fname] = np.stack([pos for _, pos in sorted(positions[fname].items())])
            orientations_stacked[fname] = np.stack([ori for _, ori in sorted(orientations[fname].items())])

        # determine output directory
        datamodule: FastMriDataModule = trainer.datamodule  # type: ignore
        if hasattr(datamodule, 'test_source') and hasattr(datamodule, 'test_split'):
            split = f'{datamodule.test_source}_{datamodule.test_split}'
            save_path = Path(trainer.default_root_dir) / 'reconstructions' / split
        else:
            save_path = Path.cwd() / 'reconstructions'
        pl_module.print(f'Saving reconstructions to {save_path}')

        self.save_reconstructions(save_path, outputs_stacked, affine, headers, positions_stacked, orientations_stacked)

        self.test_batch_outputs.clear()  # free memory

    def save_reconstructions(
            self, out_dir: Path,
            reconstructions: dict[str, np.ndarray],
            affines: dict[str, np.ndarray],
            headers: dict[str, ismrmrd.xsd.ismrmrdHeader],
            positions: dict[str, np.ndarray],
            orientations: dict[str, np.ndarray],
    ):
        """
        Save reconstruction images.

        This function writes to h5 files. Additionally, GIFs can be saved of the
        magnitude and the phase images for rapid evaluation.

        Args:
            out_dir: Path to the output directory where the reconstructions should
                be saved.
            reconstructions: A dictionary mapping input filenames to corresponding
                reconstructions.
            affines: A dictionary mapping input filenames to corresponding affine
                transformation parameters.
            headers: A dictionary mapping input filenames to corresponding ISMRMRD
                headers.
            positions: A dictionary mapping input filenames to corresponding
                slice positions.
            orientations: A dictionary mapping input filenames to corresponding
                slice orientations.
        """
        assert reconstructions.keys() == affines.keys() == headers.keys() == positions.keys() == orientations.keys()

        out_dir.mkdir(exist_ok=True, parents=True)

        for fname in reconstructions:
            recon = reconstructions[fname]
            affine = affines[fname]
            header = headers[fname]
            position = positions[fname]
            orientation = orientations[fname]

            # save as hdf5
            with h5py.File(out_dir / fname, 'w') as hf:
                hf.create_dataset('reconstruction', data=recon)
                hf.create_dataset('ismrmrd_header', data=header.toXML('utf-8'))  # type: ignore
                if not np.isnan(position).any():
                    hf.create_dataset('position', data=position)
                if not np.isnan(orientation).any():
                    hf.create_dataset('orientation', data=orientation)
                hf.attrs['rotation'] = affine[0]
                hf.attrs['flip_horizontal'] = affine[1]
                hf.attrs['flip_vertical'] = affine[2]

            # save as gif files
            if self.save_gif:
                gif_dir = out_dir / 'gif'
                gif_dir.mkdir(exist_ok=True)

                # normalize orientation
                recon = mri.apply_affine_to_image(recon, *affine)

                # get temporal resolution
                try:
                    tres = headers[fname].sequenceParameters.TR[0]  # type: ignore
                except:
                    tres = 50  # arbitrary default value

                num_slices = recon.shape[0]
                for s in range(num_slices):
                    if 'slice' in Path(fname).stem and num_slices == 1:
                        fname_ = Path(fname).stem
                    else:
                        fname_ = f'{Path(fname).stem}_slice{s:02d}'

                    # save magnitude image
                    recon_mag = np.abs(recon[s])
                    save_movie(recon_mag, gif_dir / f'{fname_}_mag', clip=False, equalize_histogram=True, tres=tres)

                    # save phase image
                    recon_phi = np.angle(recon[s])
                    save_movie(
                        recon_phi, gif_dir / f'{fname_}_phi', clip=False, equalize_histogram=False, vmin=-np.pi,
                        vmax=np.pi, tres=tres,
                    )


class MyPyTorchLightningPruningCallback(PyTorchLightningPruningCallback):
    """
    Version of the PyTorchLightningPruningCallback with a quick fix for
    an incompatibility issue between optuna and PyTorch Lightning
    """

    def on_validation_end(self, trainer, pl_module):
        trainer.training_type_plugin = trainer.strategy
        super().on_validation_end(trainer, pl_module)
        del trainer.training_type_plugin


class MetricMonitor(pl.Callback):
    """
    Callback that monitors a metric during training
    """

    def __init__(self, monitor: str, mode: str):
        super().__init__()
        self.monitor = monitor
        assert mode in ['max', 'min'], f'Unknown mode {mode}'
        self.mode = mode
        self.best = None

    @property
    def state_key(self) -> str:
        return self._generate_state_key(monitor=self.monitor, mode=self.mode, best=self.best)

    def on_validation_end(self, trainer, pl_module):
        current_metric = trainer.callback_metrics[self.monitor]
        if (self.best is None
            or (self.mode == 'max' and current_metric > self.best)
            or (self.mode == 'min' and current_metric < self.best)
        ):
            self.best = current_metric


class AccelerationScheduler(pl.Callback):
    def __init__(
            self,
            acc_target: int,
            acc_start: int = 2,
            acc_incr: int = 1,
            step_size: int = 1,
            target_epoch: int | None = None,
    ):
        """
        Args:
            acc_target: Target acceleration rate.
            acc_start: Acceleration rate at first epoch.
            acc_incr: Value by which to increment the acceleration rate
                every ``step_size`` epochs. Ignored if ``target_epoch``
                is given.
            step_size: Number of epochs after which the acceleration
                rate is increased by ``acc_incr``. Ignored if
                ``target_epoch`` is given.
            target_epoch: The epoch at which ``acc_target`` should be
                reached. If given, ``acc_incr`` and ``step_size`` are
                ignored.
        """
        super().__init__()
        if target_epoch is not None:
            self._acc_incr = (acc_target - acc_start) // target_epoch + 1
            num_steps = (acc_target - acc_start - 1) // self._acc_incr + 1
            self._step_size = target_epoch // num_steps
        else:
            self._acc_incr = acc_incr
            self._step_size = step_size
        self._acc_target = acc_target
        self._acc_start = acc_start

    def on_train_epoch_start(self, trainer, pl_module):
        new_acc = min(self._acc_target, trainer.current_epoch // self._step_size * self._acc_incr + self._acc_start)
        try:
            assert len(trainer.train_dataloader.dataset.transform.mask_func.accelerations) == 1  # type: ignore
            trainer.train_dataloader.dataset.transform.mask_func.accelerations = [new_acc]  # type: ignore
        except AttributeError as e:
            raise RuntimeError('Could not set new acceleration rate') from e
