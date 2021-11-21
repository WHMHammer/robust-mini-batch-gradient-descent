import numpy as np
from abc import ABC
from math import ceil
from typing import Tuple


class Loss(ABC):
    def __init__(self, epsilon: float, **kwargs):
        self.epsilon = epsilon

    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        # returns loss and gradient, respectively
        raise NotImplementedError


class SquaredLoss(Loss):
    def __init__(self, epsilon: float, **kwargs):
        self.epsilon = epsilon

    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        kept_size = ceil(X.shape[0] * (1 - self.epsilon))
        residuals = X.dot(w) - y
        losses = np.square(residuals)
        kept_indices = np.argsort(losses)[:kept_size]
        return sum(losses[kept_indices]) / kept_size, 2 / kept_size * X[kept_indices].transpose().dot(residuals[kept_indices])
