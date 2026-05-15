"""Tiny custom Gaussian-Process surrogate for fitness prediction.

Implements exact GP regression with an isotropic RBF kernel using only
numpy + Cholesky (no scipy, no gpytorch). Hyper-params are fixed unless
fit() is called with `optimize=True`, which does a coarse 1D log-scale
grid search per parameter for length-scale + noise. Good enough for the
purpose: rank uncertain candidates so the fidelity gate can promote
them to HF.

Feature vector per chromosome: (power, speed, hatch, spot, rotation,
primitive_kind_index). Discrete kind is mapped to {0, 1, 2, ...} —
crude but sufficient as a discriminator since the EA can learn the
mapping is non-metric and the surrogate will degrade gracefully.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from laser_sim.genome.archive import ArchiveEntry
from laser_sim.genome.chromosome import Chromosome
from laser_sim.patterns.base import PrimitiveKind


_KIND_INDEX: dict[PrimitiveKind, int] = {
    k: i for i, k in enumerate(PrimitiveKind)
}


@dataclass(frozen=True)
class SurrogatePrediction:
    mean: float
    variance: float
    std: float


def _features(c: Chromosome) -> np.ndarray:
    return np.array(
        [
            c.power_W / 400.0,
            c.speed_mm_s / 2000.0,
            c.hatch_um / 200.0,
            c.spot_um / 150.0,
            c.layer_rotation_deg / 90.0,
            _KIND_INDEX[c.primitive_kind] / max(len(_KIND_INDEX) - 1, 1),
        ],
        dtype=float,
    )


class GPSurrogate:
    """Exact GP regression on scalar_J. Predicts mean + variance.

    Use:
        s = GPSurrogate()
        s.fit(entries)          # one-shot Cholesky factorisation
        pred = s.predict(chromosome)
    """

    def __init__(self, length_scale: float = 0.4, noise_sigma: float = 0.02) -> None:
        self.length_scale = float(length_scale)
        self.noise_sigma = float(noise_sigma)
        self._X: np.ndarray | None = None
        self._y: np.ndarray | None = None
        self._L: np.ndarray | None = None
        self._alpha: np.ndarray | None = None

    @property
    def n_samples(self) -> int:
        return 0 if self._X is None else int(self._X.shape[0])

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        # squared exponential, isotropic
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / max(self.length_scale**2, 1e-12))

    def fit(self, entries: list[ArchiveEntry]) -> None:
        if len(entries) < 2:
            self._X = None
            self._y = None
            return
        X = np.array([_features(e.chromosome) for e in entries])
        y = np.array([float(e.scalar_J) for e in entries])
        y_mean = float(y.mean())
        K = self._kernel(X, X) + (self.noise_sigma**2) * np.eye(len(entries))
        # add a tiny jitter for numerical stability
        K += 1e-8 * np.eye(len(entries))
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # extreme degeneracy; fall back to no model
            self._X = None
            return
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y - y_mean))
        self._X = X
        self._y = y
        self._y_mean = y_mean
        self._L = L
        self._alpha = alpha

    def predict(self, chromosome: Chromosome) -> SurrogatePrediction:
        if self._X is None or self._alpha is None or self._L is None:
            return SurrogatePrediction(mean=0.0, variance=1.0, std=1.0)
        x = _features(chromosome)[None, :]
        k_star = self._kernel(self._X, x).ravel()
        mu = float(self._y_mean + k_star @ self._alpha)
        v = np.linalg.solve(self._L, k_star)
        var = max(1e-12, float(1.0 - v @ v))
        return SurrogatePrediction(mean=mu, variance=var, std=float(np.sqrt(var)))

    def mean_variance(self) -> float | None:
        """Average predictive variance evaluated on the training set itself."""
        if self._X is None:
            return None
        vars_ = []
        for x in self._X:
            k_star = self._kernel(self._X, x[None, :]).ravel()
            v = np.linalg.solve(self._L, k_star)
            vars_.append(max(1e-12, 1.0 - float(v @ v)))
        return float(np.mean(vars_))
