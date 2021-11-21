import numpy as np
from abc import ABC
from typing import Tuple


class Regularization(ABC):
    def __init__(self, **kwargs):
        raise NotImplementedError

    def __call__(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        # returns loss and gradient, respectively
        raise NotImplementedError

class L2Regularization(Regularization):
    def __init__(self, **kwargs):
        pass

    def __call__(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        gradient = 2 * w
        gradient[0] = 0
        return np.square(w)[1:].sum(), gradient
