from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Callable
SVDBackend = Literal["auto", "numpy", "randomized", "svds"]

'''
Base data classes that contain state information for different 
solvers. 
'''

@dataclass
class BaseConfig:
    rank: Optional[int] = None
    svd_backend: SVDBackend = "auto"
    center: bool = False
    scale: bool = True
    random_state: Optional[int] = None

@dataclass
class TransferRegConfig(BaseConfig):
    use_mask_in_target: bool = True

@dataclass
class USVTConfig(BaseConfig):
    tau: Optional[float] = None # Threshold value for USVT 
    tau_multiplier: float = 2.02  # typical constant
    use_p_hat: bool = True # Whether to estimate p (missing probability)