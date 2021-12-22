import numpy as np

from .regularization import Regularization
from .loss import Loss


class MiniBatchGradientDescent:
    def __init__(
        self,
        regularization: Regularization,
        loss: Loss,
        learning_rate: float,
        batch_size: int,
        max_iter: int
    ):
        self.regularization: Regularization = regularization
        self.loss: Loss = loss
        self.learning_rate: float = learning_rate,
        self.batch_size: int = batch_size
        self.max_iter: int = max_iter
        self.rng = np.random.default_rng()

    def fit(self, X: np.ndarray, y: np.ndarray):
        sample_size, dimension = X.shape
        X = np.c_[np.ones(sample_size), X]
        self.w = np.ones(dimension + 1)
        indices = self.rng.choice(sample_size, self.batch_size, False)
        prev_loss, gradient = self.loss(
            X[indices],
            self.w,
            y[indices]
        )
        regularization_loss, regularization_gradient = self.regularization(
            self.w)
        prev_loss += regularization_loss
        gradient += regularization_gradient
        for _ in range(self.max_iter):
            self.w -= self.learning_rate * gradient
            indices = self.rng.choice(sample_size, self.batch_size, False)
            loss, gradient = self.loss(
                X[indices],
                self.w,
                y[indices]
            )
            regularization_loss, regularization_gradient = self.regularization(
                self.w)
            loss += regularization_loss
            gradient += regularization_gradient
            if abs(loss - prev_loss) / prev_loss < 1e-5:  # TODO: fix magic number
                return
            prev_loss = loss

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.c_[np.ones(X.shape[0]), X].dot(self.w)
