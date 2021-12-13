import numpy as np
from abc import ABC
from math import ceil
from typing import Tuple


class Loss(ABC):
    def __init__(self, epsilon: float):
        self.epsilon: float = epsilon

    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        # returns loss and gradient, respectively
        raise NotImplementedError


class EpsilonTrimmedSquaredLoss(Loss):
    def __init__(self, epsilon: float):
        self.epsilon: float = epsilon

    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        kept_size = ceil(X.shape[0] * (1 - self.epsilon))
        residuals = X.dot(w) - y
        losses = np.square(residuals)
        kept_indices = np.argsort(losses)[:kept_size]
        return (
            losses[kept_indices].sum() / 2 / kept_size,
            X[kept_indices].T.dot(residuals[kept_indices]) / kept_size
        )


class EpsilonTrimmedHuberLoss(Loss):
    def __init__(self, epsilon: float, threshold: float):
        self.epsilon: float = epsilon
        self.threshold: float = threshold

    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        kept_size = ceil(X.shape[0] * (1 - self.epsilon))
        residuals = X.dot(w) - y
        absolute_residuals = np.absolute(residuals)
        kept_indices = np.argsort(absolute_residuals)[:kept_size]
        filter = absolute_residuals[kept_indices] <= self.threshold
        return (
            np.where(
                filter,
                np.square(residuals[kept_indices]) / 2,
                absolute_residuals[kept_indices] * self.threshold - self.threshold * self.threshold / 2
            ).sum() / kept_size,
            np.where(
                np.repeat(filter.reshape((-1, 1)), X.shape[1], 1),
                (X[kept_indices].T * residuals[kept_indices]).T,
                (X[kept_indices].T * np.sign(residuals[kept_indices])).T * self.threshold
            ).sum(0) / kept_size
        )
