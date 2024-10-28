"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.

Part of this source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import inspect
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..mri import rotate_phase, sens_expand, sens_reduce
from .cgdc import CGDC
from .unet import NormUnet, Unet2d, Unet2dPlusTime, Unet3d


class VarNet(nn.Module):
    """
    A full variational network model.

    This model applies a combination of soft data consistency with a U-Net
    regularizer. To use non-U-Net regularizers, use VarNetBlock.
    """

    def __init__(
            self,
            num_cascades: int = 15,
            chans: int = 16,
            pools: int = 2,
            conv_size: int = 3,
            conv_mode: str = '2d+t',
            pool_temporal: bool | None = False,
            conv_size_temp: int | None = 3,
            pad_mode_temp: str | None = 'circular',
            residual_unet_blocks: bool = True,
            two_convs_per_unet_block: bool = True,
            view_sharing: int = 0,
            phase_cycling: bool = True,
            fit_reg_landscape: bool = True,
            clamp_dc_weight: bool = False,
            cgdc_interval: int | str | None = 3,
            cgdc_mu: float | None = None,
            cgdc_iter: int = 10,
            cgdc_autograd: bool = True,
            gradient_checkpointing: bool = True,
    ):
        """
        Args:
            num_cascades: Number of cascades (i.e., layers) for variational
                network.
            chans: Number of channels for cascade U-Net.
            pools: Number of downsampling and upsampling layers for cascade
                U-Net.
            conv_size: Convolution kernel size in spatial dimensions.
            conv_mode: Convolution mode ('2d', '2d+t', or '3d').
            pool_temporal: Down-/Upsample the temporal dimension (ignored if
                `conv_mode` is '2d')
            conv_size_temp: Convolution kernel size in temporal dimension
                (ignored if `conv_mode` is '2d')
            pad_mode_temp: Padding mode for temporal dimension (ignored if
                `conv_mode` is '2d')
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
            gradient_checkpointing: Whether gradient checkpointing should be applied.
        """
        super().__init__()
        self.view_sharing = view_sharing

        if conv_mode.lower() in ['2d+t', '3d']:
            assert pool_temporal is not None, 'pool_temporal must be set if conv_mode is not 2d'
            assert conv_size_temp is not None, 'conv_size_temp must be set if conv_mode is not 2d'
            assert pad_mode_temp is not None, 'pad_mode_temp must be set if conv_mode is not 2d'

        # U-net args
        unet_args = {
            'in_chans': 2,
            'out_chans': 2,
            'chans': chans,
            'num_pool_layers': pools,
            'conv_size': conv_size,
            'residual_blocks': residual_unet_blocks,
            'two_convs_per_block': two_convs_per_unet_block,
        }
        unet_spatiotemporal_args = {
            **unet_args,
            'pool_temporal': pool_temporal,
            'conv_size_temp': conv_size_temp,
            'pad_mode_temp': pad_mode_temp,
        }

        # set up cascades
        self.cascades = []
        for _ in range(num_cascades):
            # set up U-Net
            match conv_mode.lower():
                case '2d':
                    unet = Unet2d(**unet_args)
                case '2d+t':
                    unet = Unet2dPlusTime(**unet_spatiotemporal_args)
                case '3d':
                    unet = Unet3d(**unet_spatiotemporal_args)
                case _:
                    raise ValueError(f'Unknown conv_mode {conv_mode}')

            # wrap U-Net in normalization
            model = NormUnet(unet)

            # set up VN block
            block = VarNetBlock(
                model=model,
                phase_cycling=phase_cycling,
                gradient_checkpointing=gradient_checkpointing,
                fit_reg_landscape=fit_reg_landscape,
                clamp_dc_weight=clamp_dc_weight,
            )
            self.cascades.append(block)
        self.cascades = nn.ModuleList(self.cascades)

        # set up CGDC blocks
        self.cgdc = {}
        if isinstance(cgdc_interval, int) and cgdc_interval > 0:
            for i in range(cgdc_interval-1, num_cascades, cgdc_interval):
                self.cgdc[str(i)] = CGDC(autograd=cgdc_autograd, mu=cgdc_mu, n_iter=cgdc_iter, fft_ndim=2)
        if isinstance(cgdc_interval, str) and cgdc_interval.lower() in ['before', 'both']:
            self.cgdc['-1'] = CGDC(autograd=cgdc_autograd, mu=cgdc_mu, n_iter=cgdc_iter, fft_ndim=2)
        if isinstance(cgdc_interval, str) and cgdc_interval.lower() in ['after', 'both']:
            self.cgdc[str(num_cascades - 1)] = CGDC(autograd=cgdc_autograd, mu=cgdc_mu, n_iter=cgdc_iter, fft_ndim=2)
        self.cgdc = nn.ModuleDict(self.cgdc)

    def forward(
            self,
            masked_kspace: torch.Tensor,
            mask: torch.Tensor,
            sens_maps: torch.Tensor,
            return_kspace: bool = False,
    ) -> torch.Tensor:
        # view sharing
        if self.view_sharing != 0:
            masked_kspace = self.apply_view_sharing(masked_kspace, mask)

        kspace_pred = masked_kspace.clone()

        # CG-based data consistency
        if '-1' in self.cgdc:
            cgdc_block: CGDC = self.cgdc['-1']  # type: ignore
            kspace_pred = self.run_cgdc_block(cgdc_block, None, masked_kspace, sens_maps, mask)

        # iterate through cascades
        for i, cascade in enumerate(self.cascades):
            kspace_pred = cascade(kspace_pred, masked_kspace, mask, sens_maps)

            # CG-based data consistency
            if str(i) in self.cgdc:
                cgdc_block: CGDC = self.cgdc[str(i)]  # type: ignore
                kspace_pred = self.run_cgdc_block(cgdc_block, kspace_pred, masked_kspace, sens_maps, mask)

        # optionally return k-space before ifft and coil combination
        if return_kspace:
            return kspace_pred

        # final ifft and coil combination (only necessary if we did not do the CG-DC before)
        # no need for soft-SENSE at this point, so just use first set of sensitivity maps
        reconstruction = sens_reduce(kspace_pred, sens_maps[:, 0, None], fft_ndim=2)

        # squeeze coil dimension and select reconstruction from first set of sensitivity maps
        reconstruction = reconstruction[:, 0, 0]

        return reconstruction

    def apply_view_sharing(self, masked_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        center_col = masked_kspace.shape[-2] // 2
        assert self.view_sharing % 2 == 1, 'Number of columns for view sharing must be odd'
        n_frames = masked_kspace.shape[2]
        if self.view_sharing > 0:
            col_min = center_col - self.view_sharing // 2
            col_max = center_col + self.view_sharing // 2 + 1
        elif self.view_sharing == -1:
            col_min = 0
            col_max = masked_kspace.shape[-2]
        else:
            raise ValueError(f'Invalid value for `view_sharing`: {self.view_sharing}')

        for frame in range(n_frames):
            for col in range(col_min, col_max):
                if mask.squeeze()[frame, col]:
                    continue

                measured_frames = mask.squeeze()[:, col].nonzero()[:, 0]
                if len(measured_frames) == 0:
                    continue
                measured_frames = torch.cat([measured_frames - n_frames, measured_frames, measured_frames + n_frames])

                # find next and previous measured frames
                next_frame = measured_frames[measured_frames > frame].min()
                prev_frame = measured_frames[measured_frames < frame].max()

                # interpolate k-space
                next_data = masked_kspace[:, :, next_frame % n_frames, :, col] * (frame - prev_frame)
                prev_data = masked_kspace[:, :, prev_frame % n_frames, :, col] * (next_frame - frame)
                masked_kspace[:, :, frame, :, col] = (next_data + prev_data) / (next_frame - prev_frame)

        return masked_kspace

    def run_cgdc_block(
            self,
            cgdc_block: CGDC,
            current_kspace: torch.Tensor | None,
            masked_kspace: torch.Tensor,
            sens_maps: torch.Tensor,
            mask: torch.Tensor,
            fft_ndim: int = 2,
    ) -> torch.Tensor:
        if current_kspace is None:
            current_image = None
        else:
            current_image = sens_reduce(current_kspace, sens_maps, fft_ndim=fft_ndim)
        updated_image = cgdc_block(masked_kspace, current_image, sens_maps, mask)
        updated_kspace = sens_expand(updated_image, sens_maps, fft_ndim=fft_ndim)
        return updated_kspace

    @staticmethod
    def load_from_checkpoint(ckpt_path: Path, map_location: str | torch.device, **overwrite_args) -> 'VarNet':
        ckpt = torch.load(ckpt_path, map_location=map_location)
        ckpt['hyper_parameters'].update(overwrite_args)
        model = VarNet(**{k: v for k, v in ckpt['hyper_parameters'].items() if k in inspect.getfullargspec(VarNet).args})
        state_dict = {k.lstrip('varnet.'): v for k, v in ckpt['state_dict'].items() if k.startswith('varnet.')}
        model.load_state_dict(state_dict)
        model = model.to(map_location)
        return model


class VarNetBlock(nn.Module):
    """
    Model block for end-to-end variational network.

    This model applies a combination of soft data consistency with the input
    model as a regularizer. A series of these blocks can be stacked to form
    the full variational network.

    k_n = k_n-1 - M * lambda_n * (k_n-1 - k_0) - R_n{k_n-1}
    """

    def __init__(
            self,
            model: nn.Module,
            phase_cycling: bool = False,
            gradient_checkpointing: bool = False,
            fit_reg_landscape: bool = False,
            clamp_dc_weight: bool = False,
    ):
        """
        Args:
            model: Module for "regularization" component of variational
                network.
            phase_cycling: Whether phase cycling should be used.
            gradient_checkpointing: Whether gradient checkpointing should be
                applied.
            fit_reg_landscape: Fit a weighting function/landscape in k/k-t
                space that is multiplied to the regularization term.
            clamp_dc_weight: Clamp the data consistency weight to [0, 1].
        """
        super().__init__()

        self.model = model
        self.phase_cycling = phase_cycling
        self.gradient_checkpointing = gradient_checkpointing
        self.dc_weight = nn.Parameter(torch.ones(1))  # lambda_n
        self.clamp_dc_weight = clamp_dc_weight

        self.reg_landscape = None
        if fit_reg_landscape:
            self.reg_landscape = nn.Parameter(torch.ones(10, 10, 10))

    @staticmethod
    def rotate_phase(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        phi = (torch.rand(x.shape[0]) * 2 * torch.pi).to(x.device)
        x = rotate_phase(x, phi)
        return x, phi

    @staticmethod
    def unrotate_phase(x: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
        return rotate_phase(x, -phi)

    def dc_term(self, current_kspace: torch.Tensor, ref_kspace: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # clamp data consistency weight
        if self.clamp_dc_weight:
            dc_weight = torch.clamp(self.dc_weight, 0, 1)
        else:
            dc_weight = self.dc_weight
        return mask * (current_kspace - ref_kspace) * dc_weight

    def reg_term(self, current_kspace: torch.Tensor, sens_maps: torch.Tensor) -> torch.Tensor:
        # IFFT and reduce coils
        current_image = sens_reduce(current_kspace, sens_maps, fft_ndim=2)

        # move sensitivity map dimension into batch dimension
        n_batch, n_maps = current_image.shape[0:2]
        current_image = current_image.reshape((n_batch * n_maps, *current_image.shape[2:]))

        # apply phase cycling
        phi = torch.empty(0)
        if self.phase_cycling:
            current_image, phi = self.rotate_phase(current_image)

        # apply model
        if self.gradient_checkpointing and self.training:
            # the following does not work in combination with batch_size > 1
            # due to a bug (https://github.com/pytorch/pytorch/issues/48439)
            # current_image = checkpoint(self.model, current_image)
            current_image: torch.Tensor = checkpoint(lambda inp: self.model(inp.view_as(inp)), current_image, use_reentrant=False)  # type: ignore
        else:
            current_image = self.model(current_image)

        # apply inverse phase cycling
        if self.phase_cycling:
            current_image = self.unrotate_phase(current_image, phi)

        # separate sensitivity map dimension from batch dimension
        current_image = current_image.reshape((n_batch, n_maps, *current_image.shape[1:]))

        # expand coils and FFT
        reg = sens_expand(current_image, sens_maps, fft_ndim=2)

        # multiply with learnable weight function
        if self.reg_landscape is not None:
            reg_landscape = torch.nn.functional.interpolate(
                self.reg_landscape[None, None], reg.shape[-4:-1], mode='trilinear', align_corners=True,
            )[..., None]
            reg = reg * reg_landscape

        return reg

    def forward(
            self,
            current_kspace: torch.Tensor,
            ref_kspace: torch.Tensor,
            mask: torch.Tensor,
            sens_maps: torch.Tensor,
    ) -> torch.Tensor:

        # data consistency term
        dc = self.dc_term(current_kspace, ref_kspace, mask)

        # regularization term
        reg = self.reg_term(current_kspace, sens_maps)

        return current_kspace - dc - reg
