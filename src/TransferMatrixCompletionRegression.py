from BaseConfig import TransferRegConfig
from MatrixCompletionSolver import MatrixCompletionSolver
from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Callable
from numpy.typing import NDArray

'''
Main transfer learning algorithm from Jalan et al. 2025. 
'''

class TransferMatrixCompletionRegression(MatrixCompletionSolver):
    '''
    Learn low-rank truncations of AP, AQ and estimate 
    the shift within low-rank subspace via regression. 
    Final estimate of Q combines low-rank subspace of P 
    and shift from P to Q. 
    '''
    def __init__(self, config: TransferRegConfig = TransferRegConfig()):
        super().__init__(config)
        self.U = None
        self.Vt = None
        self.beta = None
        self._mu = None
        self._s = None
        self._shape = None

    def fit(
        self,
        AP: NDArray,
        AQ: NDArray,
        obs_Q: Optional[NDArray] = None, 
        *,
        missing_value: Optional[float] = None, # Percent missing entries 
    ):
        cfg: TransferRegConfig = self.config  # type: ignore
        if AP.shape != AQ.shape:
            raise ValueError("AP and AQ must have the same shape.")
        self._shape = AQ.shape

        # learn subspaces from AP 
        U, s_ap, Vt = _truncated_svd(AP, cfg.rank, cfg.svd_backend, cfg.random_state)
        self.U, self.Vt = U, Vt

        # normalize/center/scale AQ on observed entries
        obs = _validate_obs(AQ, obs_Q, missing_value=missing_value)
        AQw, mu, s = _apply_center_scale(AQ, obs, center=cfg.center, scale=cfg.scale)
        self._mu, self._s = mu, s

        # Build regression ONLY over observed entries
        m, n = AQw.shape
        ii, jj = np.nonzero(obs)

        # Each observed (i,j) corresponds to row (j * m + i) of kron(V, U).
        # Build rows lazily to avoid dense kron formation. 
        # We parameterize S as a vector of length rank^2.
        k = U.shape[1]
        num_obs = len(ii)
        X_rows = np.empty((num_obs, k * k), dtype=AQw.dtype)
        for t, (i, j) in enumerate(zip(ii, jj)):
            # row = kron(V[j, :], U[i, :])  (1 x k^2)
            X_rows[t, :] = np.kron(Vt.T[j, :], U[i, :])

        y = AQw[ii, jj]
        # Solve least squares
        beta, *_ = np.linalg.lstsq(X_rows, y, rcond=None)
        self.beta = beta
        self._fitted = True
        return self

    def predict(self, *, clip: Optional[Tuple[float, float]] = None) -> NDArray:
        self._check_fitted()
        # Singular vectors from AP 
        U, V = self.U, self.Vt.T 

        # Rank = k 
        k = U.shape[1]

        # reconstruct S_hat from beta
        S_hat = self.beta.reshape(k, k, order="F")

        # Key step is to use the S_hat as the "central" matrix 
        # in the modified SVD (where U, V come from AP)
        Y = U @ S_hat @ V.T
        Y = _invert_center_scale(Y, self._mu, self._s)

        # Optionally clip values (e.g. in adjacency matrices)
        if clip is not None:
            Y = np.clip(Y, *clip)
        return Y