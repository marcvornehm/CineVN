"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import inspect
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_laplace


class SSIMLoss(nn.Module):
    """
    SSIM loss module.
    """

    def __init__(self, win_size: int | Tuple[int, int] = 7, k1: float = 0.01, k2: float = 0.03, mode: str = '2d'):
        """
        Args:
            win_size: Window size for SSIM calculation. If `mode` is '3d', two
                values may be given as a tuple to denote differing window sizes
                in spatial and temporal dimensions.
            k1: k1 parameter for SSIM calculation.
            k2: k2 parameter for SSIM calculation.
            mode: Convolution mode for SSIM calculation ('2d' or '3d'). If '2d'
                and input has a temporal dimension, the SSIM is calculated for
                each frame and averaged. If '3d', the SSIM is calculated for
                the whole volume.
        """
        super().__init__()
        if isinstance(win_size, int):
            self.win_size = [win_size]
        else:
            self.win_size = win_size
        if mode == '3d' and len(self.win_size) == 1:
            self.win_size = 2 * self.win_size
        self.k1, self.k2 = k1, k2

        if mode == '2d':
            w = torch.ones(1, 1, 1, self.win_size[0], self.win_size[0])
        elif mode == '3d':
            w = torch.ones(1, 1, self.win_size[1], self.win_size[0], self.win_size[0])
        else:
            raise ValueError(f'Unsupported mode {mode}')
        NP = w.numel()
        self.register_buffer('w', w / NP)
        self.cov_norm = NP / (NP - 1)

    def forward(self, pred: torch.Tensor, targ: torch.Tensor, data_range: torch.Tensor):
        assert isinstance(self.w, torch.Tensor)

        if torch.is_complex(pred):
            pred = torch.abs(pred)
        if torch.is_complex(targ):
            targ = torch.abs(targ)

        if pred.ndim == 4:
            pred = pred[:, :, None]
        if targ.ndim == 4:
            targ = targ[:, :, None]

        data_range = data_range[:, None, None, None, None].type(pred.dtype)
        C1 = (self.k1 * data_range) ** 2
        C2 = (self.k2 * data_range) ** 2

        ux = F.conv3d(pred, self.w)  # typing: ignore
        uy = F.conv3d(targ, self.w)  #
        uxx = F.conv3d(pred * pred, self.w)
        uyy = F.conv3d(targ * targ, self.w)
        uxy = F.conv3d(pred * targ, self.w)
        vx = self.cov_norm * (uxx - ux * ux)
        vy = self.cov_norm * (uyy - uy * uy)
        vxy = self.cov_norm * (uxy - ux * uy)
        A1, A2, B1, B2 = (
            2 * ux * uy + C1,
            2 * vxy + C2,
            ux ** 2 + uy ** 2 + C1,
            vx + vy + C2,
        )
        D = B1 * B2
        S = (A1 * A2) / D

        S_loss = 1 - S

        return S_loss.mean()


class HFENLoss(nn.Module):
    """
    HFEN loss module.
    """

    def __init__(self, kernel_size: int = 11, sigma: int = 2):
        assert kernel_size % 2 == 1, 'kernel size must be odd'
        super().__init__()

        kernel = np.zeros((kernel_size, kernel_size, kernel_size))
        mid = kernel_size // 2
        kernel[mid, mid, mid] = 1
        kernel = gaussian_laplace(kernel, sigma)[None, None, ...]
        self.register_buffer('kernel', torch.from_numpy(kernel).to(torch.float))

    def forward(self, pred: torch.Tensor, targ: torch.Tensor):
        assert isinstance(self.kernel, torch.Tensor)

        pred_filtered = F.conv3d(pred, self.kernel, padding='same')
        targ_filtered = F.conv3d(targ, self.kernel, padding='same')

        return torch.norm(pred_filtered - targ_filtered) / torch.norm(targ_filtered)


class PerpendicularLoss(nn.Module):
    """
    https://doi.org/10.1016/j.media.2022.102509
    """
    def forward(self, pred: torch.Tensor, targ: torch.Tensor):
        # equation 3
        l_perp = torch.abs(pred.real * targ.imag - pred.imag * targ.real) / (torch.abs(pred) + 1e-8)

        # equation 4 (ensure smoothness)
        pred_phi = torch.angle(pred)
        targ_phi = torch.angle(targ)
        phi_diff = torch.min(
            torch.abs(pred_phi - targ_phi),
            torch.min(
                torch.abs(pred_phi - (targ_phi - 2 * torch.pi)),
                torch.abs(pred_phi - (targ_phi + 2 * torch.pi))
            )
        )
        l_perp = torch.where(phi_diff < torch.pi / 2, l_perp, 2 * torch.abs(targ) - l_perp)

        return l_perp.mean()


class ComplexMSELoss(nn.Module):
    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs(pred - targ) ** 2)


class MultiTaskLoss(nn.Module):
    """
    According to Liebel et al. (https://arxiv.org/abs/1805.06334), which is a
    modification of Kendall et al. (https://arxiv.org/abs/1705.07115v1) such
    that negative loss values are prevented.
    Note that in later versions, Kendall et al. use log(weight) instead of
    log(weight^2), which would consequently mean changing log(1+weight^2) to
    log(1+weight) as well.
    """
    def __init__(self, losses: List[nn.Module]):
        super().__init__()

        self.losses = nn.ModuleList(losses)
        self.weights = nn.Parameter(torch.ones(len(losses))) if len(losses) > 1 else None

    def forward(self, **kwargs):
        val = [self._call_loss(loss, **kwargs) for loss in self.losses]
        if self.weights is not None:
            val = torch.stack(val) / (2 * self.weights ** 2)
            val = val.sum() + torch.log((1 + self.weights ** 2).prod())
        else:
            val = val[0]
        return val

    @staticmethod
    def _call_loss(loss: nn.Module, **kwargs):
        func_args = inspect.getfullargspec(loss.forward).args
        kwargs_ = {key: kwargs[key] for key in func_args if key in kwargs}
        return loss(**kwargs_)
