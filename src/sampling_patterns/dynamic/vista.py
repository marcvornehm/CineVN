"""
Copyright (c) 2014 - The Ohio State University.
Copyright (c) 2023 - Marc Vornehm <marc.vornehm@fau.de>.
"""

import copy
import warnings
from math import ceil, floor, log10
from pathlib import Path

import h5py
import numpy as np

from .rounding import round_away_from_zero as round


class VISTAParam():
    """
    Parameters for VISTA sampling pattern.
    """
    def __init__(
        self,
        PE: int           = 160,   # Size of phase encoding (PE) grid
        FR: int           = 64,    # Number of frames
        n: int            = 12,    # Number of samples (readouts) per frame
        s: float          = 1.6,   # 1<=s<=10 controls sampling density; 1: uniform density, 10: maximally non-uniform density
        sig: float | None = None,  # Std of the Gaussian envelope for sampling density (default: PE/6)
        w: float | None   = None,  # Scaling of time dimension; frames are "w" units apart (default: max((PE/n)/10+0.25, 1))
        beta: float       = 1.4,   # Exponent of the potenital energy term.
        sd: int           = 10,    # Seed to generate random numbers; a fixed seed should reproduce VISTA
        nIter: int        = 120,   # Number of iterations for VISTA
        ss: float         = 0.25,  # Step-size for gradient descent
        tf: float         = 0.0,   # Step-size in time direction wrt to phase-encoding direction; use zero for constant temporal resolution
        g: int | None     = None,  # Every gth iteration is relocated on a Cartesian grid (default: nIter//6)
        uni: int | None   = None,  # At uni iteration, reset to equivalent uniform sampling (default: round(nIter/2))
        fs: int           = 1,     # Does time average has to be fully sampled, 0 for no, 1 for yes
        fc: float         = 1,     # What fraction of the time-averaged should be fully sampled
        fl: int | None    = None,  # Start checking fully sampledness at fl^th iteration (default: round(5/6*nIter))
    ):
        self._PE    = PE
        self._FR    = FR
        self._n     = n
        self._R     = PE / n          # acceleration rate
        self._M     = FR * n          # Total number of samples
        self.s      = s
        self.sig    = PE / 6 if sig is None else sig
        self.w      = max((self.PE / self.n) / 10 + 0.25, 1) if w is None else w
        self.beta   = beta
        self.sd     = sd
        self._nIter = nIter
        self.ss     = ss
        self.tf     = tf
        self.g      = nIter // 6 if g is None else g
        self.uni    = round(nIter / 2) if uni is None else uni
        self.fs     = fs
        self.fc     = fc
        self.fl     = round(5 / 6 * nIter) if fl is None else fl

        assert self.PE >= 2 and self.PE % 1 == 0, 'The value assigned to PE must be an integer greater than 1'
        VISTAParam._check_FR(self.FR)
        assert 1 <= self.R <= self.PE, 'The value assigned to R must be an integer between 1 and PE'
        assert 1 <= self.s <= 10, 'The value assigned to s must be between 1 and 10'
        assert self.sig > 0, 'The value assigned to sig must be greater than zero'
        assert 20 <= self.nIter <= 1024 and self.nIter % 1 == 0, 'The value assigned to nIter must be an integer between 20 and 1024'
        assert self.ss > 0, 'The value assigned to ss must be greater than zero'
        assert self.tf >= 0, 'The value assigned to tf must be greater than or equql to zero'
        if self.tf > 0:
            warnings.warn('tf > 0 will destroy the constanct temporal resolution')
        assert 0 < self.beta <= 10, 'The value assigned to beta must be between 0 and 10'
        assert 5 <= self.g <= round(self.nIter / 4) and self.g % 1 == 0, 'The value assigned to g must be between 1 and nIter/4'
        assert round(self.nIter / 4) <= self.uni <= round(self.nIter / 2) and self.uni % 1 == 0, 'The value assigned to uni must be between nIter/4 and nIter/2'
        assert self.nIter / 2 < self.fl <= self.nIter and self.fl % 1 == 0, 'The value assigned to fl must be between nIter/2 and nIter'
        assert self.w > 0, 'The value assigned to w must be greater than zero'
        assert (self.fs == 0 or self.fs == 1 or self.fs >= self.R) and self.fs % 1 == 0, 'The value assigned to fs must be a non-negative integer'
        assert 0 < self.fc <= 1, 'The value assigned to fc must be between 0 and 1'

    @property
    def PE(self) -> int:
        # no setter for PE as other parameters depend on it
        return self._PE

    @property
    def FR(self) -> int:
        return self._FR

    @FR.setter
    def FR(self, value: int):
        VISTAParam._check_FR(value)
        self._FR = value
        self._M = self.n * self.FR

    @staticmethod
    def _check_FR(value: int):
        assert value >= 2 and value % 1 == 0, 'The value assigned to FR must be an integer greater than 1'
        if value > 32:
            warnings.warn('For faster processing, reduce the number of frames (to lets say 32) and then cyclically reuse these frames to achieve any arbitrarilty number of frames.')

    @property
    def n(self) -> int:
        # no setter for n as other parameters depend on it
        return self._n

    @property
    def R(self) -> float:
        # no setter for R as it is a derived property
        return self._R

    @property
    def M(self) -> int:
        # no setter for M as it is a derived property
        return self._M

    @property
    def nIter(self) -> int:
        # no setter for nIter as other parameters depend on it
        return self._nIter

    def generate_key(self, exclude_seed: bool = False) -> str:
        s = 'VISTA'
        for key, value in vars(self).items():
            if key.startswith('_'):
                key = key[1:]  # remove leading underscore
            if key in ['R', 'M']:
                continue  # skip R and M because they are derived from PE and n
            if key == 'sd' and exclude_seed:
                continue  # skip sd if exclude_seed is True
            s += f'_{key}_{value}'

        return s


def _open_cache_file(cache_file: Path, mode: str) -> h5py.File:
    # wait for file to be unlocked
    while True:
        try:
            hf = h5py.File(cache_file, mode)
            break
        except OSError as e:
            if 'unable to lock file' in str(e).lower():
                continue
            else:
                raise e
    return hf


def VISTA(
        param: VISTAParam,
        rng: np.random.RandomState | None = None,
        max_FR: int = 32,
        cache: bool = True,
        cache_ignore_seed: bool = True,
) -> np.ndarray:
    """
    VISTA (Variable Density Incoherent Spatiotemporal Acquisition)
    sampling pattern

    Reference:
        Rizwan Ahmad, Hui Xue, Shivraman Giri, Yu Ding, Jason Craft, and
        Orlando P. Simonetti. “Variable Density Incoherent
        Spatiotemporal Acquisition (VISTA) for Highly Accelerated
        Cardiac MRI: VISTA for Highly Accelerated Cardiac MRI.” Magnetic
        Resonance in Medicine 74, no. 5 (November 2015): 1266-78.
        https://doi.org/10.1002/mrm.25507.

    Original MatLab implementation:
        https://github.com/OSU-CMR/VISTA
        https://github.com/OSU-CMR/cmr-sampling/tree/main/functions/VISTA

    Original author:
        Rizwan Ahmad (ahmad.46@osu.edu)
    """
    param_tmp = copy.deepcopy(param)
    if cache:
        param_tmp.FR = max_FR  # always compute sampling pattern with max_FR frames for caching
        key = param_tmp.generate_key(exclude_seed=cache_ignore_seed)
        cache_file = Path(__file__).parent / Path('./vista_cache.h5')
        samp = None
        if cache_file.exists():
            # try to read from cache
            hf = _open_cache_file(cache_file, 'r')
            if key in hf:
                samp = hf[key][:]
            hf.close()
        if samp is None:
            # compute and cache
            _, _, samp = _VISTA_impl(param_tmp, rng)
            hf = _open_cache_file(cache_file, 'a')
            if key not in hf:
                hf.create_dataset(key, data=samp)
            hf.close()
    else:
        param_tmp.FR = min(max_FR, param.FR)  # compute sampling pattern with at most max_FR frames
        _, _, samp = _VISTA_impl(param_tmp, rng)

    # tile samp to param.FR frames
    samp_out = np.repeat(samp, param.FR // max_FR, axis=1)
    samp_out = np.concatenate([samp_out, samp[:, :param.FR % max_FR]], axis=1)
    return samp_out


def _VISTA_impl(param: VISTAParam, rng: np.random.RandomState | None = None) \
        -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    n     = param.n             # Number of samples per frame
    N     = param.PE            # Number of phase encoding steps
    FR    = param.FR            # Number of frames
    tf    = param.tf            # Relative step-size in "time" dimension wrt to phase-encoding direction
    w     = param.w             # Scaling of time dimension
    fs    = param.fs            # To make time average fully sampled or not
    uni   = param.uni           # Iteration where sampling is reset to jittered uniform
    R     = param.PE / param.n  # Net acceleration factor
    nIter = param.nIter         # Number of iterations
    s     = log10(param.s)      # 1=uniform density, s>1 for variable density
    sig   = param.sig           # Std of the Gaussian envelope that defines variable density
    beta  = param.beta          # Exponent of force term
    g     = param.g             # At every g^th iteration, sampled are relocated to the nearest Cartesian grid
    fl    = param.fl            # At/beyond this iteration, start checking for fully-sampledness
    ss    = param.ss            # Gradient descent step-size
    sd    = param.sd            # Seed for random number generation
    M     = param.M             # Total number of samples

    # Let's handle the special case of R = 1
    if R == 1:
        samp = np.ones((N, FR))
        return samp

    # Use radom variable density sampling as initialization for VISTA
    p1 = np.arange(-np.floor(N / 2), np.ceil(N / 2))
    t1 = np.empty(0)
    Ind = 0
    ti = np.zeros(n * FR)
    ph = np.zeros(n * FR)
    prob = (0.1 + s / (1 - s + 1e-10) * np.exp(-(p1)**2 / (1 * sig**2)))

    if rng is None:
        rng = np.random.RandomState(sd)
    rng_state = rng.get_state()
    tmpSd = round(1e6 * rng.rand(FR)).astype(int)
    for i in range(-floor(FR / 2), ceil(FR / 2)):
        a = np.nonzero(t1 == i)[0]
        n_tmp = n - a.size
        prob_tmp = prob
        prob_tmp[a] = 0
        rng2 = np.random.RandomState(tmpSd[i + floor(FR / 2)])
        p_tmp = rng2.choice(np.arange(N), size=n_tmp, replace=True, p=prob_tmp / prob_tmp.sum()) - N // 2
        ti[Ind:Ind + n_tmp] = i
        ph[Ind:Ind + n_tmp] = p_tmp
        Ind = Ind + n_tmp

    # Displacement parameters
    stp = np.ones(nIter)  # Gradient descent displacement
    a = w * np.ones(nIter)  # Temporal axis scaling

    dis_ext = np.zeros(M)  # Extent of displacement
    for i in range(nIter):
        ph, ti = _tile(ph[:M], ti[:M], param)
        for j in range(M):
            # Distances
            dis = np.sqrt(np.abs(ph - ph[j])**2 + np.abs(a[i] * (ti - ti[j]))**2)
            nanloc = dis == 0
            dis[nanloc] = np.inf

            # Scaling
            scl = 1 - s * np.exp(-(ph)**2 / (2 * sig**2))
            scl = scl + (1 - scl[0])
            dscl = 1 / sig**2 * s * ph[j] * np.exp(-(ph[j]**2) / (2 * sig**2))  # Differentiation of scl wrt to "ph"

            # Force and resulting displacement
            fx = beta * ((ph[j] - ph) * (scl[j] * scl / dis**(beta + 2))) - dscl * scl / dis**beta
            fy = beta * (a[i]**2 * (ti[j] - ti) * (scl[j] * scl / dis**(beta + 2))) * tf
            ph[j] = (ph[j] + max(min(stp[i] * np.sum(fx), R/4), -R/4))
            ti[j] = (ti[j] + max(min(stp[i] * np.sum(fy), R/4), -R/4))

            # Ensure that the samples stay in bounds
            if ph[j] < -floor(N/2) - 1/2:
                ph[j] = ph[j] + N
            elif ph[j] > (ceil(N/2) - 1/2):
                ph[j] = ph[j] - N

            if ti[j] < -floor(FR/2) - 1/2:
                ti[j] = ti[j] + FR
            elif ti[j] > (ceil(FR/2) - 1/2):
                ti[j] = ti[j] - FR

            # Displacing samples to nearest Cartesian location
            if (i + 1) % g == 0 or (i + 1) == nIter:
                ph[j] = round(ph[j])
                ti[j] = round(ti[j])

            # Measuring the displacement
            if i == 1:
                dis_ext[j] = np.abs(stp[i] * np.sum(fx))

        # Normalizing the step-size to a reasonable value
        if i == 2:
            stp = ss * (1 + R/4) / np.median(dis_ext) * stp

        # At uni-th iteration, reset to jittered uniform sampling
        ti = ti[:M]
        ph = ph[:M]
        if (i + 1) == uni:
            tmp = np.zeros((n, FR))
            for k in range(FR):
                tmp[:, k] = np.sort(ph[k*n:(k+1)*n])
            tmp = round(np.mean(tmp, axis=1))  # Find average distances between adjacent phase-encoding samples
            ph = np.tile(tmp, FR)  # Variable density sampling with "average" distances
            rng2 = np.random.RandomState()
            rng2.set_state(rng_state)
            rndTmp = rng2.rand(FR)
            for k in range(-floor(FR/2), ceil(FR/2)):
                tmp = ti == k
                ptmp = ph[tmp] + round(1/2 * R**1 * (rndTmp[k + floor(FR/2)] - 0.5))  # Add jitter
                ptmp[ptmp > ceil(N/2) - 1] = ptmp[ptmp > ceil(N/2) - 1] - N  # Ensure the samples don't run out of the k-t grid
                ptmp[ptmp < -floor(N/2)] = ptmp[ptmp < -floor(N/2)] + N
                ph[tmp] = ptmp
            # Temporary stretch the time axis to avoid bad local minima
            a[i + 1:] = a[i + 1:] * (1 + np.exp(-(np.arange(i+1, nIter) - (i+1)) / ceil(nIter / 60)))

        # Displace the overlapping points so that one location has only one sample
        if (i + 1) % g == 0 or (i + 1) == nIter:
            ph, ti = _dispdup(ph[:M], ti[:M], param)

        # Check/ensure time-average is fully sampled
        if ((i + 1) % g == 0 or (i + 1) == nIter) and (i + 1) >= fl:
            ph = ph[:M]
            ti = ti[:M]
            if fs == 1:  # Ensuring fully sampledness at average all
                ph, ti = _fillK(ph, ti, ph.copy(), ti.copy(), param)
            elif fs > 1:  # Ensuring fully sampledness for "fs" frames
                for m in range(floor(FR / fs)):
                    tmp = np.arange(m * n * fs, (m + 1) * n * fs)
                    ph, ti = _fillK(ph[tmp], ti[tmp], ph.copy(), ti.copy(), param)

    ph, ti = _dispdup(ph[:M], ti[:M], param)

    # create binary mask 'samp'
    PEInd = round(N * (ti + floor(FR/2)) + (ph + floor(N/2))).astype(int)  # shift center to N/2
    samp = np.zeros((FR, N))
    samp.flat[PEInd] = 1
    samp = samp.T

    # flip the ky ordering of alternate frames to reduce jumps
    # FRind = PEInd.shape
    row, col = np.unravel_index(PEInd, (N, FR), order='F')
    PEInd = row
    FRInd = col
    for i in range(FR):
        if i % 2 == 1:
            PEInd[i * n:(i + 1) * n] = np.sort(PEInd[i * n:(i + 1) * n])[::-1]
        else:
            PEInd[i * n:(i + 1) * n] = np.sort(PEInd[i * n:(i + 1) * n])
        FRInd[i * n:(i + 1) * n] = i

    samp = samp.astype(bool)

    return PEInd, FRInd, samp


def _tile(ph, ti, param):
    """
    Replicate the sampling pattern in each direction. Probablity, this is
    not an efficient way to impose periodic boundary condition because it
    makes the problem size grow by 9 fold.
    """
    N = param.PE
    FR = param.FR
    po = np.concatenate([ph, ph - N, ph, ph + N, ph - N, ph + N, ph - N, ph, ph + N])
    to = np.concatenate([ti, ti - FR, ti - FR, ti - FR, ti, ti, ti + FR, ti + FR, ti + FR])
    return po, to


def _dispdup(ph, ti, param):
    """
    If multiple samples occupy the same location, this routine displaces the
    duplicate samples to the nearest vacant location so that there is no
    more than one smaple per location on the k-t grid
    """
    ph = ph + ceil((param.PE + 1) / 2)
    ti = ti + ceil((param.FR + 1) / 2)
    pt = (ti - 1) * param.PE + ph

    uniquept, countOfpt = np.unique(pt, return_counts=True)
    indexToRepeatedValue = np.where(countOfpt != 1)
    repeatedValues = uniquept[indexToRepeatedValue]
    dupind = []
    for i in range(repeatedValues.size):
        tmp = np.where(pt == repeatedValues[i])[0]
        dupind.extend(tmp[1:])  # Indices of locations which have more than one sample

    empind = np.setdiff1d(np.arange(1, param.PE * param.FR + 1), pt)  # Indices of vacant locations

    for i in range(len(dupind)):  # Let's go through all 'dupind' one by one
        newind = _nearestvac(pt[dupind[i]], empind, param)
        pt[dupind[i]] = newind
        empind = np.setdiff1d(empind, newind)

    ph, ti = _ind2xy(pt, param.PE)
    ph = ph - ceil((param.PE + 1) / 2)
    ti = ti - ceil((param.FR + 1) / 2)

    return ph, ti


def _nearestvac(dupind, empind, param):
    """
    For a given 'dupind', this function finds the nearest vacant location
    among 'empind'
    """
    x0, y0 = _ind2xy(dupind, param.PE)
    x, y = _ind2xy(empind, param.PE)
    dis1 = (x - x0) ** 2
    dis2 = (y - y0) ** 2; dis2[dis2 > np.finfo(float).eps] = np.inf
    dis = np.sqrt(dis1 + dis2)
    b = dis.argmin()
    n = empind[b]
    return n


def _ind2xy(ind, X):
    """
    Index to (x,y)
    """
    x = ind - np.floor((ind - 1) / X) * X
    y = np.ceil(ind / X)
    return x, y


def _fillK(P, T, Pacc, Tacc, param):
    """
    Ensures time-average of VISTA is fully sampled (except for the outer most
    region)
    """

    p = param.PE
    fc = param.fc  # fraction of k-space to be fully sampled

    # empty locations;
    tmp = np.setdiff1d(np.arange(-floor(fc * p / 2), ceil(fc * p / 2)), P)
    ord = np.argsort(np.abs(tmp))
    tmp2 = np.abs(tmp)[ord]
    tmp2 = tmp2 * np.sign(tmp[ord])  # Sorted (from center-out) empty locations

    while tmp2.size > 0:
        ind = tmp2[0]  # the hole (PE location) to be filled

        eps = np.finfo(float).eps
        can = P[(np.sign(ind + eps) * P) > (np.sign(ind + eps) * ind)]  # Find candidates which are on the boundary side of "ind"
        if can.size == 0:
            break  # If there is nothing on the boundary side
        Pcan = can[np.sign(ind + eps) * (can - ind) == np.min(np.sign(ind + eps) * (can - ind))]  # P index of candidates that can be used to cover the void
        Tcan = T[P == Pcan[0]]                                                                    # T index of candidates that can be used to cover the void

        U = np.inf
        for i in range(Pcan.size):
            Ptmp = Pacc.copy()
            Ttmp = Tacc.copy()
            Ptmp[(Pacc == Pcan[i]) & (Tacc == Tcan[i])] = ind
            Utmp = _computeU(Ptmp, Ttmp, param)  # Compute engergy U for the ith candidate
            if Utmp <= U:
                slc = i
                U = Utmp

        Pacc[(Pacc == Pcan[slc]) & (Tacc == Tcan[slc])] = ind  # Fill the hole with the appropriate candidate
        P[(P == Pcan[slc]) & (T == Tcan[slc])] = ind  # Fill the hole with the approprate candidate
        tmp = np.setdiff1d(np.arange(-floor(fc * p / 2), ceil(fc * p / 2)), P)
        tmp = _excludeOuter(tmp, p)
        ord = np.argsort(np.abs(tmp))
        tmp2 = np.abs(tmp)[ord]
        tmp2 = tmp2 * np.sign(tmp[ord])  # Find new holes

    return Pacc, Tacc


def _computeU(P, T, param):
    """
    Compute potential energy (U) of the distribution
    """
    N    = P.size
    s    = param.s
    sig  = param.sig
    a    = param.w
    beta = param.beta

    U = 0
    for i in range(N):
        k = np.concatenate([np.arange(i), np.arange(i+1, N)])
        U = U + np.sum(1/2 * ((1 - s * np.exp(-(P[i])**2 / (2 * sig**2))) * (1 - s * np.exp(-(P[k])**2 / (2 * sig**2)))) /
                        ((P[i] - P[k])**2 + (a * (T[i] - T[k]))**2)**(beta/2))

    return U


def _excludeOuter(tmp, p):
    """
    Remove the "voids" that are contiguous to the boundary
    """
    tmp = np.sort(tmp)[::-1]
    cnd = np.abs(np.diff(np.insert(tmp, 0, ceil(p/2))))
    if cnd.size > 0:
        if np.max(cnd) == 1:
            tmp = np.empty(0)
        else:
            tmp = tmp[np.argmax(cnd > 1):]

    tmp = np.sort(tmp)
    cnd = np.abs(np.diff(np.insert(tmp, 0, -floor(p/2) - 1)))
    if cnd.size > 0:
        if np.max(cnd) == 1:
            tmp = np.empty(0)
        else:
            tmp = tmp[np.argmax(cnd > 1):]

    return tmp
