"""
Copyright (c) Facebook, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

import torch


def complex_mul(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Complex multiplication.

    This multiplies two complex tensors assuming that they are both stored as
    real arrays with the last dimension being the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.
        y: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == y.shape[-1] == 2:
        raise ValueError('Tensors do not have separate complex dim.')

    x = torch.view_as_complex(x)
    y = torch.view_as_complex(y)

    return torch.view_as_real(x * y)


def complex_conj(x: torch.Tensor) -> torch.Tensor:
    """
    Complex conjugate.

    This applies the complex conjugate assuming that the input array has the
    last dimension as the complex dimension.

    Args:
        x: A PyTorch tensor with the last dimension of size 2.

    Returns:
        A PyTorch tensor with the last dimension of size 2.
    """
    if not x.shape[-1] == 2:
        raise ValueError('Tensor does not have separate complex dim.')

    x = torch.view_as_complex(x)
    x = torch.resolve_conj(torch.conj(x))

    return torch.view_as_real(x)


def complex_abs(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the absolute value of a complex valued input tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError('Tensor does not have separate complex dim.')

    return torch.abs(torch.view_as_complex(data))


def complex_abs_sq(data: torch.Tensor) -> torch.Tensor:
    """
    Compute the squared absolute value of a complex tensor.

    Args:
        data: A complex valued tensor, where the size of the final dimension
            should be 2.

    Returns:
        Squared absolute value of data.
    """
    if not data.shape[-1] == 2:
        raise ValueError('Tensor does not have separate complex dim.')

    return (data ** 2).sum(dim=-1)
