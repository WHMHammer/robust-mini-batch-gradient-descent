import numpy as np
from abc import ABC
from typing import Tuple


class Regularization(ABC):
    def __call__(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        # returns loss and gradient, respectively
        raise NotImplementedError


class NullRegularization(Regularization):
    def __call__(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        return 0, 0


class L1Regularization(Regularization):
    def __init__(self, regularization_weight: float) -> None:
        self.regularization_weight = regularization_weight

    def __call__(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        gradient = np.sign(w) * self.regularization_weight
        gradient[0] = 0
        return np.absolute(w)[1:].sum() * self.regularization_weight, gradient


class L2Regularization(Regularization):
    def __init__(self, regularization_weight: float) -> None:
        self.regularization_weight = regularization_weight

    def __call__(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        gradient = 2 * w * self.regularization_weight
        gradient[0] = 0
        return np.square(w)[1:].sum() * self.regularization_weight, gradient
