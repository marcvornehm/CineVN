"""
Copyright (c) 2019 - The Ohio State University.
Copyright (c) 2023 - Marc Vornehm <marc.vornehm@fau.de>.
"""

from math import ceil, sqrt

import numpy as np

from .rounding import round_away_from_zero as round


class GROParam():
    """
    Parameters for GRO sampling pattern.
    """
    def __init__(
        self,
        PE: int     = 160,  # Size of phase encoding (PE) grid
        FR: int     = 64,   # Number of frames
        n: int      = 12,   # Number of samples (readouts) per frame
        E: int      = 1,    # Number of encoding, E=1 for cine, E=2 for flow (phase-contrast MRI)
        tau: float  = 1,    # Extent of shift between frames, tau = 1 or 2: golden ratio shift, tau>2: tiny golden ratio shift
        s: float    = 2.2,  # s>=1. larger values means higher sampling density in the middle
        alph: float = 3,    # alph>1. larger alpha means sharper transition from high-density to low-density regions
        PF: int     = 0,    # for partial Fourier; discards PF samples from one side
    ):
        self.PE   = PE
        self.FR  = FR
        self.n   = n
        self.E    = E
        self.tau  = tau
        self.s    = s
        self.alph = alph
        self.PF   = PF


def GRO(param: GROParam, offset: float = 0, return_indices: bool = False) \
        -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    GRO (Golden Ratio offset) sampling pattern

    offset added by Marc Vornehm. Should be between 0 and 1.

    Reference:
        Rizwan Ahmad, Ning Jin, Orlando Simonetti, Yingmin Liu, and Adam
        Rich. “Cartesian sampling for dynamic magnetic resonance imaging
        (MRI)”, U.S. Patent Application No. 16/984,351 (pub. February
        4th 2021). https://patents.justia.com/patent/20210033689.

    Original MatLab implementation:
        https://github.com/OSU-CMR/GRO-CAVA
        https://github.com/OSU-CMR/cmr-sampling/tree/main/functions/GRO

    Original author:
        Rizwan Ahmad (ahmad.46@osu.edu)
    """
    n   = param.n     # Number of phase encoding (PE) lines per frame
    FR  = param.FR    # Frames
    N   = param.PE    # Size of of PE grid
    E   = param.E     # Number of encoding, E=1 for cine, E=2 for flow
    tau = param.tau
    PF  = param.PF
    s   = param.s
    a   = param.alph

    gr = (1 + sqrt(5)) / 2  # golden ratio
    gr = 1 / (gr + tau - 1)  # golden angle, sqrt(2) works equally well

    # Size of the smaller pseudo-grid which after stretching gives the true grid size
    Ns = ceil(N * 1 / s)  # Size of shrunk PE grid
    k = (N/2 - Ns/2) / ((Ns/2)**a)  # location specific displacement

    samp = np.zeros((N, FR, E))  # sampling on k-t grid
    PEInd = np.zeros(((n-PF) * FR, E))  # The ordered sequence of PE indices and encoding
    FRInd = np.zeros(((n-PF) * FR, 1))  # The ordered sequence of frame indices

    v0 = np.arange(1/2 + 1e-10, Ns + 1/2 - 1e-10, Ns / (n+PF))  # Start with uniform sampling for each frame
    v0 += offset * Ns / (n+PF)  # Note: offset is not implemented in the original MatLab code
    for e in range(E):
        v0 = v0 + 1 / E * Ns / (n+PF)  # Start with uniform sampling for each frame
        kk = E + 1 - (e + 1)
        for j in range(FR):
            v = ((v0 + j * Ns / (n+PF) * gr) - 1) % Ns + 1  # In each frame, shift by golden shift of Ns/TR*ga
            v = v - Ns * (v >= Ns + 0.5)

            if N % 2 == 0:  # if even, shift by 1/2 pixel
                vC = v - k * np.sign((Ns/2 + 1/2) - v) * np.abs((Ns/2 + 1/2) - v)**a + (N-Ns) / 2 + 1/2
                vC = vC - N * (vC >= N + 0.5)
            elif N % 2 == 1:  # if odd don't shift
                vC = v - k * np.sign((Ns/2 + 1/2) - v) * np.abs((Ns/2 + 1/2) - v)**a + (N-Ns) / 2
            vC = round(np.sort(vC))
            vC = vC[PF:]

            if (j + 1) * n > PEInd.shape[0]:  # required when PF>0
                PEInd_temp = np.zeros(((j + 1) * n, PEInd.shape[1]))
                PEInd_temp[:PEInd.shape[0], :PEInd.shape[1]] = PEInd
                PEInd = PEInd_temp
            if (j + 1) % 2 == 1:
                PEInd[j * n:(j + 1) * n, e] = vC - 1
            else:
                PEInd[j * n:(j + 1) * n, e] = vC[::-1] - 1
            FRInd[j * n:(j + 1) * n] = j

            samp[vC.astype(int) - 1, j, e] += kk

    if return_indices:
        return PEInd, FRInd, samp
    else:
        return samp
