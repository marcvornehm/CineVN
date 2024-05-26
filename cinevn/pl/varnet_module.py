"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
import warnings
from typing import Iterable, Optional

import pytorch_lightning as pl
import torch
import torchmetrics

from .. import losses, metrics
from ..data import transforms
from ..models import VarNet


class VarNetModule(pl.LightningModule):
    """
    VarNet training module.

    This can be used to train variational networks from the paper:

    A. Sriram et al. End-to-end variational networks for accelerated MRI
    reconstruction. In International Conference on Medical Image Computing and
    Computer-Assisted Intervention, 2020.

    which was inspired by the earlier paper:

    K. Hammernik et al. Learning a variational network for reconstruction of
    accelerated MRI data. Magnetic Resonance inMedicine, 79(6):3055-3071, 2018.
    """

    def __init__(
        self,
        mode: str = 'static',
        num_cascades: int = 12,
        chans: int = 18,
        pools: int = 4,
        conv_size: int = 3,
        conv_mode: str = '2d',
        pool_temporal: bool | None = True,
        conv_size_temp: int | None = 3,
        pad_mode_temp: str | None = 'circular',
        residual_unet_blocks: bool = True,
        two_convs_per_unet_block: bool = True,
        view_sharing: int = 0,
        phase_cycling: bool = False,
        fit_reg_landscape: bool = False,
        clamp_dc_weight: bool = False,
        cgdc_interval: Optional[int | str] = None,
        cgdc_mu: Optional[float] = None,
        cgdc_iter: int = 10,
        cgdc_autograd: bool = True,
        median_temp_size: int = 1,
        use_sens_net: bool = True,
        sens_chans: int = 8,
        sens_pools: int = 4,
        mask_center: bool = True,
        gradient_checkpointing: bool = False,
        normalize_for_loss: bool = True,
        lr: float = 0.0003,
        lr_step_size: int = 40,
        lr_gamma: float = 0.1,
        weight_decay: float = 0.0,
        loss_fns: Iterable[str] = ('ssim',),
        ssim_mode: Optional[str] = None,
        compile_model: bool = False,
    ):
        """
        Args:
            mode: 'static' or 'dynamic'.
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            conv_size: Convolution kernel size in spatial dimensions.
            conv_mode: Convolution mode ('2d', '2d+t', or '3d').
            pool_temporal: Down-/Upsample the temporal dimension (ignored if
                `mode` is 'static' or `conv_mode` is '2d')
            conv_size_temp: Convolution kernel size in temporal dimension
                (ignored if `mode` is 'static' or `conv_mode` is '2d')
            pad_mode_temp: Padding mode for temporal dimension (ignored if
                `mode` is 'static' or `conv_mode` is '2d')
            residual_unet_blocks: Use residual connections in each cascade's
                U-Net.
            two_convs_per_unet_block: Use two convolutions per block in
                each cascade's U-Net.
            view_sharing: Number of columns in k-space center on which view
                sharing should be applied as initial estimate. Set to `0` to
                disable view sharing.
            phase_cycling: Apply random phase shifts in each iteration.
            fit_reg_landscape: Fit a weighting function/landscape in k/k-t
                space that is multiplied to the regularization term.
            clamp_dc_weight: Clamp the data consistency weight to [0, 1].
            cgdc_interval: Apply CGDC every n cascades. Set to `None` to
                disable. Set to `before` to apply CGDC before first cascade
                only. Set to `after` to apply CGDC after last cascade only.
                Set to `both` to apply CGDC before first and after last
                cascade.
            cgdc_mu: The mu parameter for the conjugate gradient data
                consistency. If None, this is a learnable parameter.
            cgdc_iter: Number of conjugate gradient iterations for data
                consistency.
            cgdc_autograd: Whether to use the default PyTorch autograd function
                for conjugate gradient data consistency. This gives an exact
                gradient but is less memory efficient.
            median_temp_size: Kernel size of a median filter over the temporal
                dimension, applied after reconstruction.
            use_sens_net: Estimate coil sensitivity maps using a network. If
                False, pre-computed sensitivity maps are used.
            sens_chans: Number of channels for sensitivity map U-Net.
            sens_pools: Number of downsampling and upsampling layers for
                sensitivity map U-Net.
            mask_center: Whether to mask center of k-space for sensitivity map
                calculation.
            gradient_checkpointing: Whether gradient checkpointing should be
                applied.
            normalize_for_loss: Normalize target and output for loss
                computation. Note that this does not affect normalization of
                the input to the underlying U-Net, which happens regardless of
                the value of `normalize_for_loss`. This parameter ensures that
                examples with different signal levels contribute similarly to
                the loss computation in case the loss function depends on the
                signal level.
            lr: Learning rate.
            lr_step_size: Learning rate step size.
            lr_gamma: Learning rate gamma decay.
            weight_decay: Parameter for penalizing weights norm.
            loss_fns: Tuple of strings specifying loss functions to be used.
            ssim_mode: SSIM loss mode. Ignored if `mode` is 'static'. If None,
                same value as for `conv_mode` is used.
            compile_model: Use torch.compile
        """
        super().__init__()

        if mode not in ('static', 'dynamic'):
            raise ValueError(f'Unknown mode {mode}')
        if mode == 'static' and conv_mode != '2d':
            raise ValueError('Static mode only supports 2D convolution mode')

        self.save_hyperparameters()

        self.mode = mode
        self.normalize_for_loss = normalize_for_loss
        self.lr = lr
        self.lr_step_size = lr_step_size
        self.lr_gamma = lr_gamma
        self.weight_decay = weight_decay
        self.ssim_mode = conv_mode if ssim_mode is None else ssim_mode

        # model
        self.varnet = VarNet(
            num_cascades=num_cascades,
            chans=chans,
            pools=pools,
            conv_size=conv_size,
            conv_mode=conv_mode,
            pool_temporal=pool_temporal,
            conv_size_temp=conv_size_temp,
            pad_mode_temp=pad_mode_temp,
            residual_unet_blocks=residual_unet_blocks,
            two_convs_per_unet_block=two_convs_per_unet_block,
            view_sharing=view_sharing,
            phase_cycling=phase_cycling,
            clamp_dc_weight=clamp_dc_weight,
            fit_reg_landscape=fit_reg_landscape,
            cgdc_interval=cgdc_interval,
            cgdc_mu=cgdc_mu,
            cgdc_iter=cgdc_iter,
            cgdc_autograd=cgdc_autograd,
            median_temp_size=median_temp_size,
            use_sens_net=use_sens_net,
            sens_chans=sens_chans,
            sens_pools=sens_pools,
            mask_center=mask_center,
            gradient_checkpointing=gradient_checkpointing,
        )
        if compile_model:
            try:
                self.varnet = torch.compile(self.varnet)
            except RuntimeError:
                # torch.compile is not yet supported on Windows
                pass

        # loss functions
        losses_list = []
        for loss_fn_str in loss_fns:
            match loss_fn_str.lower():
                case 'ssim':
                    loss_fn = losses.SSIMLoss(mode=self.ssim_mode)
                case 'perp' | 'perpendicular':
                    loss_fn = losses.PerpendicularLoss()
                case 'mse':
                    loss_fn = losses.ComplexMSELoss()
                case _:
                    raise RuntimeError('Unknown loss type')

            if any([isinstance(loss, type(loss_fn)) for loss in losses_list]):
                warnings.warn(
                    f'Loss function of type {type(loss_fn)} already exists. Are you sure you want to add it twice?',
                    category=UserWarning
                )
            losses_list.append(loss_fn)
        self.loss = losses.MultiTaskLoss(losses_list)

        # evaluation metrics
        self.val_loss = torchmetrics.MeanMetric()
        self.nmse = metrics.NMSEMetric()
        self.ssim = metrics.SSIMMetric()
        self.psnr = metrics.PSNRMetric()
        self.hfen = metrics.HFENMetric()
        self.ssim_xt = metrics.SSIMXTMetric() if self.mode == 'dynamic' else None

    def forward(self, masked_kspace, mask, num_low_frequencies, sens_maps=None):
        assert all(torch.eq(n, num_low_frequencies[0]) for n in num_low_frequencies), \
            'num_low_frequencies must be equal for all elements in the batch'
        output = self.varnet(masked_kspace, mask, num_low_frequencies[0], sens_maps)
        return torch.view_as_complex(output)

    def train_val_forward(self, batch):
        if torch.any(batch.max_value == 0) or torch.any(batch.max_value.isnan()):
            raise ValueError('max_value must not be zero or nan during training or validation')

        # forward through network
        output = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.sens_maps)

        # crop phase oversampling
        target, output = transforms.center_crop_to_smallest(batch.target, output, ndim=2)

        # compute loss
        if self.normalize_for_loss:
            norm_val = target.abs().flatten(1).max(dim=1).values
            target_for_loss = target / norm_val[(...,) + (None,) * (target.ndim - 1)]
            output_for_loss = output / norm_val[(...,) + (None,) * (output.ndim - 1)]
            data_range = batch.max_value / norm_val
        else:
            target_for_loss = target
            output_for_loss = output
            data_range = batch.max_value
        loss = self.loss(pred=output_for_loss.unsqueeze(1), targ=target_for_loss.unsqueeze(1), data_range=data_range)

        return target, output, loss

    def training_step(self, batch, batch_idx):
        _, _, loss = self.train_val_forward(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False, sync_dist=True)
        for i, cgdc in self.varnet.cgdc.items():
            if cgdc.mu.requires_grad:
                self.log(f'params/cgdc_mu_{i}', cgdc.mu.item(), on_step=True, on_epoch=False, sync_dist=True)  # type: ignore
        for i, casc in enumerate(self.varnet.cascades):
            self.log(f'params/dc_weight_{i}', casc.dc_weight.item(), on_step=True, on_epoch=False, sync_dist=True)  # type: ignore
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        target, output, loss = self.train_val_forward(batch)
        return {'output': output, 'target': target, 'val_loss': loss}

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if not isinstance(outputs, dict):
            raise RuntimeError('outputs must be a dict')
        # update metrics
        target = outputs['target'].abs()
        output = outputs['output'].abs()
        maxval = [v.reshape(-1) for v in batch.max_value]
        if batch.annotations.isnan().any():
            center = None
        else:
            center = [annotation[0].to(int) for annotation in batch.annotations]
        self.val_loss.update(outputs['val_loss'])
        self.nmse.update(batch.fname, batch.slice_num, target, output)
        self.ssim.update(batch.fname, batch.slice_num, target, output, maxvals=maxval)
        self.psnr.update(batch.fname, batch.slice_num, target, output, maxvals=maxval)
        self.hfen.update(batch.fname, batch.slice_num, target, output)
        if self.ssim_xt is not None and center is not None:
            self.ssim_xt.update(batch.fname, batch.slice_num, target, output, center, maxvals=maxval)

    def on_validation_epoch_end(self):
        # logging
        self.log('validation_loss', self.val_loss, prog_bar=True)
        self.log('val_metrics/nmse', self.nmse)
        self.log('val_metrics/ssim', self.ssim, prog_bar=True)
        self.log('val_metrics/psnr', self.psnr)
        self.log('val_metrics/hfen', self.hfen)
        if self.ssim_xt is not None and self.ssim_xt.update_count > 0:
            self.log('val_metrics/ssim_xt', self.ssim_xt)

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        output = self(batch.masked_kspace, batch.mask, batch.num_low_frequencies, batch.sens_maps)
        output = transforms.batched_crop_to_recon_size(output, batch.header)
        return {'output': output}

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, self.lr_step_size, self.lr_gamma)

        return [optim], [scheduler]
