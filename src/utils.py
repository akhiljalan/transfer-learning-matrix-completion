 matrix_completion.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Callable
import logging
import numpy as np
from numpy.typing import NDArray
from scipy.linalg import svd as dense_svd
from scipy.linalg import expm
from scipy.optimize import minimize

try:
    from sklearn.utils.extmath import randomized_svd as skl_randomized_svd
except Exception:  # optional
    skl_randomized_svd = None

logger = logging.getLogger(__name__)

SVDBackend = Literal["auto", "numpy", "randomized", "svds"]

def _validate_obs(X: NDArray, obs: Optional[NDArray], *, missing_value: Optional[float]) -> NDArray[np.bool_]:
    if obs is not None:
        obs = obs.astype(bool)
        if obs.shape != X.shape:
            raise ValueError("obs mask must match X.shape")
        return obs
    if missing_value is None:
        # treat NaN as missing
        return ~np.isnan(X)
    # else treat specific value as missing
    return ~(X == missing_value)

def _apply_center_scale(
    X: NDArray, obs: NDArray, *, center: bool, scale: bool
) -> Tuple[NDArray, Optional[float], Optional[float]]:
    Xw = X.copy()
    Xw[~obs] = 0.0
    mu = np.nanmean(Xw[obs]) if center else None
    if center and mu is not None:
        Xw[obs] = Xw[obs] - mu
    s = np.nanmax(np.abs(Xw[obs])) if scale else None
    if scale and s and s > 0:
        Xw[obs] = Xw[obs] / s
    return Xw, mu, s

def _invert_center_scale(Y: NDArray, mu: Optional[float], s: Optional[float]) -> NDArray:
    Z = Y.copy()
    if s and s > 0:
        Z = Z * s
    if mu is not None:
        Z = Z + mu
    return Z

def _truncated_svd(
    M: NDArray, rank: Optional[int], backend: SVDBackend = "auto", random_state: Optional[int] = None
) -> Tuple[NDArray, NDArray, NDArray]:
    m, n = M.shape
    k = min(rank or min(m, n), min(m, n))
    if backend == "auto":
        backend = "randomized" if (skl_randomized_svd and max(m, n) > 500 and k < min(m, n)//2) else "numpy"
    if backend == "randomized":
        if skl_randomized_svd is None:
            logger.warning("randomized SVD requested but sklearn not available; falling back to numpy")
            backend = "numpy"
        else:
            U, s, Vt = skl_randomized_svd(M, n_components=k, random_state=random_state)
            return U[:, :k], s[:k], Vt[:k, :]
    # numpy dense
    U, s, Vt = dense_svd(M, full_matrices=False)
    return U[:, :k], s[:k], Vt[:k, :]