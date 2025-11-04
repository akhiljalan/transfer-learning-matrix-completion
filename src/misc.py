# matrix_completion.py
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



# ----------------------------- base ---------------------------------



# ----------------------------- USVT ----------------------------------

