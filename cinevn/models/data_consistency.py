"""
Copyright (c) Marc Vornehm <marc.vornehm@fau.de> and Florian Fuernrohr.
"""

from typing import Optional

import torch
import torch.nn as nn

from ..complex_math import complex_conj, complex_mul
from ..mri import sens_expand, sens_reduce


def zdot_reduce_sum(input_x: torch.Tensor, input_y: torch.Tensor):
    # take only real part
    dims = tuple(range(len(input_x.shape[:-1])))
    inner = complex_mul(complex_conj(input_x), input_y).sum(dims)
    return inner[0]


def EhE_Op(x: torch.Tensor, csm: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, fft_ndim: int):
    """
    Computes (E^H*E + mu*I) x
    """
    kspace = sens_expand(x[None], csm[None], fft_ndim=fft_ndim)[0] * mask  # E*x
    image = sens_reduce(kspace[None], csm[None], fft_ndim=fft_ndim)[0]  # E^H*E*x
    return image + mu * x


def rhs_Op(y: torch.Tensor, z: torch.Tensor, csm: torch.Tensor, mu: torch.Tensor, fft_ndim: int):
    """
    Computes E^H*y + mu*z
    """
    image = sens_reduce(y[None], csm[None], fft_ndim=fft_ndim)[0]  # E^H*y
    return image + mu * z


def conjgrad_iterations(x0: torch.Tensor, csm: torch.Tensor, mask: torch.Tensor, rhs: torch.Tensor,
                        mu: torch.Tensor, n_iter: int, fft_ndim: int, eps: float = 1e-15):
    """
    Performs Conjugate Gradients optimization

    Ax = b, where  A == EhE_Op  and  b == rhs
    """
    x = x0
    r, p = rhs, rhs
    rsnot = zdot_reduce_sum(r, r)
    rsold, rsnew = rsnot, rsnot

    max_iters = n_iter if n_iter >= 0 else 100
    for _ in range(max_iters):
        Ap = EhE_Op(p, csm, mask, mu, fft_ndim)
        pAp = zdot_reduce_sum(p, Ap)
        alpha = (rsold / pAp)
        x = x + alpha * p
        r = r - alpha * Ap
        rsnew = zdot_reduce_sum(r, r)
        beta = (rsnew / rsold)
        rsold = rsnew
        p = beta * p + r

        # stop if number of iterations was not given and residual is small enough
        if n_iter < 0 and rsnew < eps:
            break

    return x


def conjgrad(y: Optional[torch.Tensor], z: Optional[torch.Tensor], rhs: Optional[torch.Tensor], csm: torch.Tensor,
             mask: torch.Tensor, mu: torch.Tensor, cg_iter: int, fft_ndim: int, init_x_at_z: bool):
    x = []
    n_batch = mask.shape[0]
    for i in range(n_batch):
        if rhs is None:
            assert y is not None, 'If rhs is None, y must be provided.'
            z_ = z[i] if z is not None else torch.zeros_like(y[i])
            rhs_ = rhs_Op(y[i], z_, csm[i], mu, fft_ndim)
        else:
            assert y is None and z is None, 'If rhs is provided, y and z must be None.'
            rhs_ = rhs[i]
        if init_x_at_z:
            assert z is not None, 'If init_x_at_z is True, z must be provided.'
            x0 = z[i]
        else:
            x0 = torch.zeros_like(rhs_)
        xi = conjgrad_iterations(x0, csm[i], mask[i], rhs_, mu, cg_iter, fft_ndim)
        x.append(xi)
    x = torch.stack(x)

    return x


class CGDCAutogradFun(torch.autograd.Function):
    """
    DC block employs conjugate gradient for data consistency. Use of custom backward function for memory efficient training.
    """
    @staticmethod
    def forward(ctx, y: torch.Tensor, z: torch.Tensor, csm: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor,
                cg_iter: int, fft_ndim: int, init_x_at_z: bool):
        fft_ndim_tensor = torch.tensor(fft_ndim, requires_grad=False)
        ctx.save_for_backward(csm, mask, mu, fft_ndim_tensor)
        return conjgrad(y, z, None, csm, mask, mu, cg_iter, fft_ndim, init_x_at_z)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        csm, mask, mu, fft_ndim = ctx.saved_tensors
        fft_ndim = fft_ndim.item()
        x_grads = conjgrad(None, None, grad_output, csm, mask, mu, -1, fft_ndim, False)
        return None, x_grads, None, None, None, None, None, None


class CGDC(nn.Module):
    """
    Data consistency block employing conjugate gradient descent.
    """

    def __init__(self, autograd: bool = True, mu: Optional[float] = None,
                 n_iter: int = 10, fft_ndim: int = 2,
                 start_at_current_estimate: bool = False):
        """
        Args:
            autograd: Use PyTorch's autograd for backpropagation. If
                False, the gradient is approximated by employing a CG
                optimization. This is less accurate but is more
                memory-efficient. Defaults to True.
            mu: Tikhonov regularization weight. If None, this weight is
                a learnable parameter. Learning this weight is not
                possible if use_autograd is False. In this case,
                defaults to 0.05. Defaults to None.
            n_iter: Number of CG iterations. If -1, CG is run until
                convergence. Defaults to 10.
            fft_ndim: Number of spatial dimensions of the FFT. Defaults
                to 2.
            start_at_current_estimate: If True, the CG optimization
                starts at the current estimate. If False, the CG
                optimization starts at zero. Defaults to False.
        """
        super().__init__()
        self.autograd = autograd
        if self.autograd and mu is None:
            self.mu = nn.Parameter(torch.tensor(0.05))
        else:
            mu = 0.05 if mu is None else mu
            self.register_buffer('mu', torch.tensor(mu))
        self.n_iter = n_iter
        self.fft_ndim = fft_ndim
        self.start_at_current_estimate = start_at_current_estimate

    def forward(
        self,
        kspace_meas: torch.Tensor,
        recons_pred: torch.Tensor,
        csm: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        if self.autograd:
            return conjgrad(
                kspace_meas, recons_pred, None, csm, mask, self.mu, self.n_iter, self.fft_ndim,
                self.start_at_current_estimate
            )
        else:
            return CGDCAutogradFun.apply(  # type: ignore
                kspace_meas, recons_pred, csm, mask, self.mu, self.n_iter, self.fft_ndim, self.start_at_current_estimate
            )
