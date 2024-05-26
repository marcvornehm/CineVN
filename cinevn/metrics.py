"""
Copyright (c) Facebook, Inc. and its affiliates.
Copyright (c) Marc Vornehm <marc.vornehm@fau.de>.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""
from abc import abstractmethod
from collections import defaultdict
from typing import Any, Optional, Sequence

try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_laplace as gaussian_laplace_cuda
except ImportError:
    # importing cupy fails if no cuda is available
    cp = None
import torch
from scipy.ndimage import gaussian_laplace as gaussian_laplace_cpu
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.metric import Metric
from torchvision.transforms.functional import rotate


def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute Mean Squared Error (MSE)"""
    return torch.mean((gt - pred) ** 2)


def nmse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """Compute Normalized Mean Squared Error (NMSE)"""
    return torch.linalg.norm(gt - pred) ** 2 / torch.linalg.norm(gt) ** 2


def psnr(gt: torch.Tensor, pred: torch.Tensor, maxval: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = torch.max(gt)

    mse_val = mse(gt, pred)
    return 20 * torch.log10(maxval) - 10 * torch.log10(mse_val)


def ssim(gt: torch.Tensor, pred: torch.Tensor, maxval: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute Structural Similarity Index Metric (SSIM)"""
    if maxval is None:
        maxval = torch.max(gt)

    # expand dimensions
    # for static data, this results in [B=1, C=1, H, W]
    # for dynamic data, this results in [B=1, C, H, W] where C is the number of frames
    match gt.ndim:
        case 2:  # static data
            gt = gt[None, None]
            pred = pred[None, None]
        case 3:  # dynamic data
            gt = gt[None]
            pred = pred[None]
        case _:
            raise RuntimeError('Unexpected number of dimensions')

    # torchmetrics.functional.structural_similarity_index_measure returns wrong results for dtype float32 inputs on cuda
    pred = pred.to(torch.float64)
    gt = gt.to(torch.float64)

    # choose parameters to match default parameters of skimage.metrics.structural_similarity
    ssim = StructuralSimilarityIndexMeasure(data_range=maxval.item(), gaussian_kernel=False, kernel_size=7).to(pred.device)
    return ssim(pred, gt)


def ssim_xt(gt: torch.Tensor, pred: torch.Tensor, center: torch.Tensor, n_profiles: int = 8,
            maxval: Optional[torch.Tensor] = None) -> torch.Tensor:
    assert (n_profiles & (n_profiles - 1) == 0) and n_profiles > 0, \
        f'n_profiles must be a power of two and larger than zero, but was {n_profiles}'
    assert gt.ndim == pred.ndim == 3, \
        f'Expected number of dimensions is 3, but was {gt.ndim} for gt and {pred.ndim} for pred.'
    assert center.shape == (2,), f'`center` must be of size 2 but was {center.shape}.'

    angle_increment = 360 / (2 * n_profiles)
    radius = torch.tensor(gt.shape[1:]).min() // 4
    h, w = gt.shape[1:]

    ssim_vals = []

    for i in range(n_profiles // 2):
        gt_rot = rotate(gt, i * angle_increment)
        pred_rot = rotate(pred, i * angle_increment)

        x_low = torch.max(torch.tensor(0), center[0] - radius)
        x_high = torch.min(torch.tensor(w), center[0] + radius)
        y_low = torch.max(torch.tensor(0), center[1] - radius)
        y_high = torch.min(torch.tensor(h), center[1] + radius)
        gt_profile1 = gt_rot[:, center[1], x_low:x_high]
        gt_profile2 = gt_rot[:, y_low:y_high, center[0]]
        pred_profile1 = pred_rot[:, center[1], x_low:x_high]
        pred_profile2 = pred_rot[:, y_low:y_high, center[0]]

        ssim_vals.append(ssim(gt_profile1, pred_profile1, maxval=maxval))
        ssim_vals.append(ssim(gt_profile2, pred_profile2, maxval=maxval))

    return torch.tensor(ssim_vals).mean()


def hfen(gt: torch.Tensor, pred: torch.Tensor, sigma: float = 1.5) -> torch.Tensor:
    """
    See 10.1109/TMI.2010.2090538 for details.
    """
    if gt.is_cuda and cp is not None:
        device = gt.device
        filtered = gaussian_laplace_cuda(cp.asarray(gt) - cp.asarray(pred), sigma=sigma)
        return torch.linalg.norm(torch.as_tensor(filtered, device=device))
    else:
        gt = gt.cpu()
        pred = pred.cpu()
        filtered = gaussian_laplace_cpu(gt.numpy() - pred.numpy(), sigma=sigma)
        return torch.linalg.norm(torch.as_tensor(filtered))


class _VolumeAverageMetric(Metric):
    """
    Abstract class for distributed metrics averaged over image volumes.
    Each given value is associated to a volume. Upon computing, values
    are averaged for each volume before an average is calculated over
    these resultant values.
    """
    values: list[torch.Tensor]
    fname_hashes: list[torch.Tensor]
    slice_nums: list[torch.Tensor]

    def __init__(self):
        super().__init__()
        self.add_state('values', default=[], dist_reduce_fx='cat')
        self.add_state('fname_hashes', default=[], dist_reduce_fx='cat')  # use hash because str is not allowed
        self.add_state('slice_nums', default=[], dist_reduce_fx='cat')

    @abstractmethod
    def update(self, *_: Any, **__: Any) -> None:
        """Override this method. This method should internally call the `update_keys` method and append a value to the
        `values` state."""

    def update_keys(self, fname: str, slice_num: torch.Tensor):
        assert slice_num.ndim == 0, '`slice_num` has to be a pytorch scalar'
        self.fname_hashes.append(torch.as_tensor(hash(fname), device=self.device))
        self.slice_nums.append(torch.as_tensor(slice_num, device=self.device))

    def compute(self):
        values_dict = defaultdict(dict)
        assert len(self.values) == len(self.fname_hashes) == len(self.slice_nums)
        for value, fname_hash, slice_num in zip(self.values, self.fname_hashes, self.slice_nums):
            values_dict[fname_hash][slice_num] = value
        values_sum = 0
        for values in values_dict.values():
            values_sum += torch.mean(torch.cat([v.view(-1) for v in values.values()]))
        if len(values_dict) == 0:
            return torch.as_tensor(values_sum)
        else:
            return torch.as_tensor(values_sum / len(values_dict))


class MSEMetric(_VolumeAverageMetric):
    def update(self, fnames: Sequence[str], slice_nums: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor):
        assert len(fnames) == slice_nums.shape[0] == targets.shape[0] == predictions.shape[0]
        for i in range(len(fnames)):
            self.update_keys(fnames[i], slice_nums[i])
            self.values.append(mse(targets[i], predictions[i]))


class NMSEMetric(_VolumeAverageMetric):
    def update(self, fnames: Sequence[str], slice_nums: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor):
        assert len(fnames) == slice_nums.shape[0] == targets.shape[0] == predictions.shape[0]
        for i in range(len(fnames)):
            self.update_keys(fnames[i], slice_nums[i])
            self.values.append(nmse(targets[i], predictions[i]))


class PSNRMetric(_VolumeAverageMetric):
    def update(self, fnames: Sequence[str], slice_nums: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor,
               maxvals: Sequence[Optional[torch.Tensor]]):
        assert len(fnames) == slice_nums.shape[0] == targets.shape[0] == predictions.shape[0] == len(maxvals)
        for i in range(len(fnames)):
            self.update_keys(fnames[i], slice_nums[i])
            self.values.append(psnr(targets[i], predictions[i], maxval=maxvals[i]))


class SSIMMetric(_VolumeAverageMetric):
    def update(self, fnames: Sequence[str], slice_nums: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor,
               maxvals: Sequence[Optional[torch.Tensor]]):
        assert len(fnames) == slice_nums.shape[0] == targets.shape[0] == predictions.shape[0] == len(maxvals)
        for i in range(len(fnames)):
            self.update_keys(fnames[i], slice_nums[i])
            self.values.append(ssim(targets[i], predictions[i], maxval=maxvals[i]))


class SSIMXTMetric(_VolumeAverageMetric):
    def __init__(self, n_profiles: int = 8):
        super().__init__()
        self.n_profiles = n_profiles

    def update(self, fnames: Sequence[str], slice_nums: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor,
               centers: Sequence[Optional[torch.Tensor]], maxvals: Sequence[Optional[torch.Tensor]]):
        assert len(fnames) == slice_nums.shape[0] == targets.shape[0] == predictions.shape[0] == len(centers) \
               == len(maxvals)
        for i in range(len(fnames)):
            if centers[i] is None:
                continue
            self.update_keys(fnames[i], slice_nums[i])
            self.values.append(ssim_xt(targets[i], predictions[i],
                centers[i],  # type: ignore
                n_profiles=self.n_profiles, maxval=maxvals[i]
            ))


class HFENMetric(_VolumeAverageMetric):
    def __init__(self, sigma: float = 1.5):
        super().__init__()
        self.sigma = sigma

    def update(self, fnames: Sequence[str], slice_nums: torch.Tensor, targets: torch.Tensor, predictions: torch.Tensor):
        assert len(fnames) == slice_nums.shape[0] == targets.shape[0] == predictions.shape[0]
        for i in range(len(fnames)):
            self.update_keys(fnames[i], slice_nums[i])
            self.values.append(hfen(targets[i], predictions[i], sigma=self.sigma))
