from typing import Optional, Tuple, Literal, Callable
from BaseConfig import BaseConfig 

'''
Base solver class and interface. 
Basic requirements for all solvers: fit, predict, check_fitted. 
'''

class MatrixCompletionSolver:
    """Abstract base class for matrix completion."""
    def __init__(self, config: BaseConfig = BaseConfig()):
        self.config = config
        self._fitted = False

    def fit(self, X: NDArray, obs: Optional[NDArray] = None, *, missing_value: Optional[float] = None):
        raise NotImplementedError

    def predict(self, *, clip: Optional[Tuple[float, float]] = None) -> NDArray:
        raise NotImplementedError

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before predict().")