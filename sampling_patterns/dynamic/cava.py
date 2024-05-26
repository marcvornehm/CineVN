"""
Copyright (c) 2019 - The Ohio State University.
Copyright (c) 2023 - Marc Vornehm <marc.vornehm@fau.de>.
"""

from math import ceil, floor, sqrt

import numpy as np

from .rounding import round_away_from_zero as round


class CAVAParam():
    """
    Parameters for CAVA sampling pattern.
    """
    def __init__(
        self,
        PE: int     = 120,  # Size of phase encoding (PE) grid
        FR: int     = 48,   # Nominal number of frames (for display only)
        n: int      = 6,    # Nominal number of samples per frame per encoding (for display one)
        E: int      = 2,    # Number of encoding, E=1 for cine, E=2 for flow (phase-contrast MRI)
        tau: float  = 1,    # Size of jumps, tau = 1 or 2: golden ratio jump, tau>2: tiny golden ratio jump
        s: float    = 2.2,  # s>=1. larger values means higher sampling density in the middle
        alph: float = 3,    # alph>1. larger alpha means sharper transition from high-density to low-density regions
    ):
        self.PE   = PE
        self._FR  = FR
        self._n   = n
        self._M   = FR * n  # Total number of samples
        self.E    = E
        self.tau  = tau
        self.s    = s
        self.alph = alph

    @property
    def FR(self):
        return self._FR
    
    @FR.setter
    def FR(self, value):
        self._FR = value
        self._M = self.FR * self.n

    @property
    def n(self):
        return self._n
    
    @n.setter
    def n(self, value):
        self._n = value
        self._M = self.FR * self.n

    @property
    def M(self):
        # no setter for M as it is a derived property
        return self._M


# def CAVA(n_in: int, FR_in: int, PE_in: int, E_in: int = 1):
def CAVA(param: CAVAParam) -> np.ndarray:
    """
    CAVA (CArtesian sampling with Variable density and Adjustable temporal
    resolution) sampling pattern

    Reference:
        Adam Rich, Michael Gregg, Ning Jin, Yingmin Liu, Lee Potter,
        Orlando Simonetti, and Rizwan Ahmad. “CArtesian Sampling with
        Variable Density and Adjustable Temporal Resolution (CAVA).”
        Magnetic Resonance in Medicine 83, no. 6 (June 2020): 2015-25.
        https://doi.org/10.1002/mrm.28059.


    Original MatLab implementation:
        https://github.com/OSU-CMR/GRO-CAVA
        https://github.com/OSU-CMR/cmr-sampling/tree/main/functions/CAVA

    Original author:
        Rizwan Ahmad (ahmad.46@osu.edu)
    """
    n   = param.n     # Number of phase encoding (PE) lines per frame. This can be changed post-acquisition
    M   = param.M     # Total number of samples
    N   = param.PE    # Size of of PE grid
    E   = param.E     # Number of encoding, E=1 for cine, E=2 for flow
    tau = param.tau
    s   = param.s
    a   = param.alph

    R = N / n  # Initial guess of acceleration

    gr = (1 + sqrt(5)) / 2  # golden ratio F_{PE+1}/F_{PE}
    gr = 1 / (gr + tau - 1)  # golden angle 
    Ns = ceil(N * 1 / s)  # Size of shrunk PE grid

    # Size of the smaller pseudo-grid which after stretching gives the true grid size
    k = (N/2 - Ns/2) / ((Ns/2)**a)  # location specific displacement

    # Let's populate the grid;
    samp = np.zeros((N, ceil(M / n), E))  # sampling on k-t grid
    PEInd = np.zeros((M, E))  # Final PE index used for MRI
    FRInd = np.zeros((M, 1))  # Frame index (can be changed post acquisition)

    ind = np.zeros((M, E))  # "hidden" index on a uniform grid

    for e in range(E + 1):
        kk = e + 1
        for i in range(M):
            if e < E:
                if i == 0:
                    ind[i, e] = (floor(Ns / 2) + 1 + e * sqrt(11) * gr * Ns / E - 1) % Ns + 1
                elif i > 0:
                    ind[i, e] = ((ind[i-1, e] + Ns * gr) - 1) % Ns + 1
                ind[i, e] = ind[i, e] - Ns * (ind[i, e] >= (Ns + 0.5))

                if N % 2 == 0:  # if even, shift by 1/2 pixel
                    indC = ind[i, e] - k * np.sign((Ns/2 + 1/2) - ind[i, e]) * (np.abs((Ns/2 + 1/2) - ind[i, e]))**a + (N-Ns) / 2 + 1/2
                    indC = indC - N * (indC >= (N + 0.5))
                elif N % 2 == 1:  # if odd don't shift
                    indC = ind[i, e] - k * np.sign((Ns/2 + 1/2) - ind[i, e]) * (np.abs((Ns/2 + 1/2) - ind[i, e]))**a + (N-Ns) / 2
                PEInd[i, e] = round(indC)
                samp[PEInd[i, e].astype(int) - 1, ceil((i+1) / n) - 1, e] += kk
            FRInd[i] = ceil((i+1) / n)

    return samp
