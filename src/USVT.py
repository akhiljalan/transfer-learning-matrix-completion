from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Callable
from BaseConfig import USVTConfig 
from MatrixCompletionSolver import MatrixCompletionSolver 
from numpy.typing import NDArray
import numpy as np 
from utils import _validate_obs, _apply_center_scale, _invert_center_scale, _truncated_svd

'''
Implementation of USVT algorithm, from the paper: 
CHATTERJEE, SOURAV. "MATRIX ESTIMATION BY UNIVERSAL SINGULAR VALUE THRESHOLDING." The Annals of Statistics 43.1 (2015): 177-214.
'''
class USVT(MatrixCompletionSolver):
    '''
    Universal Singular Value Thresholding (USVT). 
    '''
    def __init__(self, config: USVTConfig = USVTConfig()):
        super().__init__(config)
        self.U = None 
        self.s = None
        self.Vt = None
        self._mu = None 
        self._s = None
        self._obs = None

    def fit(self, X: NDArray, obs: Optional[NDArray] = None, *, missing_value: Optional[float] = None):
        '''
        Fits a low-rank USVT estimate of the noisy observed matrix X. 
        '''
        cfg: USVTConfig = self.config  # type: ignore
        obs = _validate_obs(X, obs, missing_value=missing_value)
        Xw, mu, s = _apply_center_scale(X, obs, center=cfg.center, scale=cfg.scale)
        self._obs, self._mu, self._s = obs, mu, s

        m, n = Xw.shape
        p_hat = obs.mean() if cfg.use_p_hat else 1.0
        tau = cfg.tau if cfg.tau is not None else cfg.tau_multiplier * np.sqrt(max(m, n) * p_hat)

        # fill missing with zeros (standard USVT)
        Xw_filled = np.where(obs, Xw, 0.0)
        U, svals, Vt = _truncated_svd(Xw_filled, cfg.rank, cfg.svd_backend, cfg.random_state)
        mask = (svals > tau)
        if not np.any(mask):
            logger.info("USVT threshold removed all components; returning zero matrix.")
            self._Y = np.zeros_like(Xw_filled)
        else:
            s_trunc = svals * mask
            self._Y = (U[:, :len(s_trunc)] @ np.diag(s_trunc) @ Vt[:len(s_trunc), :]) / max(p_hat, 1e-8)

        self._fitted = True
        return self

    def predict(self, *, clip: Optional[Tuple[float, float]] = None) -> NDArray:
        self._check_fitted()
        Y = _invert_center_scale(self._Y, self._mu, self._s)
        if clip is not None:
            Y = np.clip(Y, clip[0], clip[1])
        return Y