import numpy as np
from abc import ABC
from math import ceil
from typing import Tuple, Union


class Loss(ABC):
    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray]:
        # returns loss and gradient, respectively
        raise NotImplementedError


class SquaredLoss(Loss):
    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray, residuals: Union[np.ndarray, None] = None) -> Tuple[float, np.ndarray]:
        if residuals is None:
            residuals = X.dot(w) - y
        losses = np.square(residuals)
        return (
            losses.sum() / 2 / X.shape[0],
            X.T.dot(residuals) / X.shape[0]
        )


class HuberLoss(Loss):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray, residuals: Union[np.ndarray, None] = None) -> Tuple[float, np.ndarray]:
        if residuals is None:
            residuals = X.dot(w) - y
        absolute_residuals = np.absolute(residuals)
        filter = absolute_residuals <= self.threshold
        return (
            np.where(
                filter,
                np.square(residuals) / 2,
                absolute_residuals * self.threshold - self.threshold * self.threshold / 2
            ).sum() / X.shape[0],
            np.where(
                np.repeat(filter.reshape((-1, 1)), X.shape[1], 1),
                (X.T * residuals).T,
                (X.T * np.sign(residuals)).T * self.threshold
            ).sum(0) / X.shape[0]
        )


class EpsilonTrimmedLoss(Loss):
    def __init__(self, loss: Loss, epsilon: float):
        self.loss = loss
        self.epsilon = epsilon

    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray, residuals: Union[np.ndarray, None] = None) -> Tuple[float, np.ndarray]:
        if residuals is None:
            residuals = X.dot(w) - y
        absolute_residuals = np.absolute(residuals)
        kept_indices = np.argsort(absolute_residuals)[
            :ceil(X.shape[0] * (1 - self.epsilon))]
        return self.loss(X[kept_indices], w, y[kept_indices], residuals[kept_indices])


def absolute_z_score(x: np.ndarray) -> np.ndarray:
    return np.absolute((x - np.mean(x)) / np.std(x))


class ZScoreTrimmedLoss(Loss):
    def __init__(self, loss: Loss, threshold: float):
        self.loss = loss
        self.threshold = threshold

    def __call__(self, X: np.ndarray, w: np.ndarray, y: np.ndarray, residuals: Union[np.ndarray, None] = None) -> Tuple[float, np.ndarray]:
        if residuals is None:
            residuals = X.dot(w) - y
        residuals_copy = np.copy(residuals)
        kept_filter = np.full(X.shape[0], True)
        half = int(X.shape[0] / 2)
        z_scores_filter = absolute_z_score(residuals) > self.threshold
        while np.count_nonzero(kept_filter) > half and np.any(z_scores_filter):
            kept_filter[z_scores_filter] = False
            residuals[z_scores_filter] = np.nan
            z_scores_filter = absolute_z_score(residuals) > self.threshold
        if np.count_nonzero(kept_filter) <= half:
            return self.loss(X, w, y, residuals_copy)
        return self.loss(X[kept_filter], w, y[kept_filter], residuals[kept_filter])
