#  Copyright (c) 2023. Marc Vornehm <marc.vornehm@fau.de>.

from pathlib import Path
from typing import Optional

import imageio
import numpy as np
import torch
from PIL import Image


def save(image: np.ndarray | torch.Tensor, filename: Path | str, clip: bool = True, tres: Optional[float] = 50,
         vmin: Optional[float] = None, vmax: Optional[float] = None):
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if image.shape[-1] == 2:
        image = image[..., 0] + 1j * image[..., 1]
    if np.iscomplexobj(image):
        image = np.abs(image)
    filename = Path(filename)

    if clip:
        image = np.clip(image, np.percentile(image, 3), np.percentile(image, 97))
    if vmin is None:
        vmin = np.min(image)
    if vmax is None:
        vmax = np.max(image)
    image = (image - vmin) / (vmax - vmin) * 255  # type: ignore

    if image.ndim == 2:  # static
        image = image.astype(np.uint8)  # type: ignore
        imageio.imwrite(filename.with_suffix('.png'), image)
    elif image.ndim == 3:  # dynamic
        images_pil = [Image.fromarray(img) for img in image]
        images_pil[0].save(str(filename.with_suffix('.gif')), save_all=True, append_images=images_pil[1:], duration=tres, loop=0)
    else:
        raise ValueError('Invalid number of dimensions.')
