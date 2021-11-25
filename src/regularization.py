import numpy as np
from abc import ABC
from typing import Tuple


class Regularization(ABC):
    def __call__(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        # returns loss and gradient, respectively
        raise NotImplementedError


class L1Regularization(Regularization):
    def __call__(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        gradient = np.sign(w)
        gradient[0] = 0
        return np.absolute(w)[1:].sum(), gradient


class L2Regularization(Regularization):
    def __call__(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        gradient = 2 * w
        gradient[0] = 0
        return np.square(w)[1:].sum(), gradient
