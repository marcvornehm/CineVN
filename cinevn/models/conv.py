"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.
"""

from typing import Tuple

import torch
from torch import nn


class ConvNdWithFlexiblePadding(nn.Module):
    """
    This class implements a convolution layer where different padding modes can be applied along different dimensions
    """
    def __init__(self, N: int, padding: Tuple[int, ...], padding_modes: Tuple[str, ...], **kwargs):
        super().__init__()

        if len(padding) != N:
            raise ValueError(f'`padding` must have length {N}, but was {len(padding)}')
        if len(padding_modes) != N:
            raise ValueError(f'`padding_modes` must have length {N}, but was {len(padding_modes)}')

        match N:
            case 1:
                self.conv = nn.Conv1d(**kwargs)
            case 2:
                self.conv = nn.Conv2d(**kwargs)
            case 3:
                self.conv = nn.Conv3d(**kwargs)
            case _:
                raise NotImplementedError(f'ConvNdWithFlexiblePadding only implemented for N={{1, 2, 3}}, but was {N}')

        self.N = N
        self.padding_ = padding
        self.padding_modes = padding_modes

    def pad(self, image: torch.Tensor) -> torch.Tensor:
        # iterate from right to left
        for i in range(self.N):
            p = self.padding_[-i - 1]
            pad = (0, 0) * i + (p, p) + (0, 0) * (self.N - i - 1)
            mode = self.padding_modes[-i - 1]
            image_padded = torch.nn.functional.pad(image, pad, mode=mode)

            # replace
            image = image_padded

        return image

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        image = self.pad(image)
        image = self.conv.forward(image)
        return image


class Conv3dSpatiotemporal(ConvNdWithFlexiblePadding):
    def __init__(self, padding: Tuple[int, int, int], pad_mode_temp: str, **kwargs):
        # `pad_mode_temp` in temporal dimension, 'replicate' in readout dimension, 'circular' in phase dimension
        super().__init__(N=3, padding=padding, padding_modes=(pad_mode_temp, 'replicate', 'circular'), **kwargs)


class Conv2dSpatial(ConvNdWithFlexiblePadding):
    def __init__(self, padding: Tuple[int, int], **kwargs):
        # 'replicate' in readout dimension', 'circular' in phase dimension
        super().__init__(N=2, padding=padding, padding_modes=('replicate', 'circular'), **kwargs)


class Conv1dTemporal(ConvNdWithFlexiblePadding):
    def __init__(self, padding: Tuple[int, ], pad_mode_temp: str, **kwargs):
        # `pad_mode_temp` in temporal dimension
        super().__init__(N=1, padding=padding, padding_modes=(pad_mode_temp, ), **kwargs)


class Conv3dSpatial(Conv2dSpatial):
    def forward(self, image: torch.Tensor):
        # move temporal dimension to batch dimension
        b, c, t, x, y = image.shape
        image = image.permute(0, 2, 1, 3, 4).reshape(b * t, c, x, y)

        # conv
        image = super().forward(image)

        # reshape back
        bt, c, x, y = image.shape
        assert b * t == bt
        image = image.view(b, t, c, x, y).permute(0, 2, 1, 3, 4).contiguous()

        return image


class Conv3dTemporal(Conv1dTemporal):
    def forward(self, image: torch.Tensor):
        # move spatial dimensions to batch dimension
        b, c, t, x, y = image.shape
        image = image.permute(0, 3, 4, 1, 2).reshape(b * x * y, c, t)

        # conv
        image = super().forward(image)

        # reshape back
        bxy, c, t = image.shape
        assert b * x * y == bxy
        image = image.view(b, x, y, c, t).permute(0, 3, 4, 1, 2).contiguous()

        return image
