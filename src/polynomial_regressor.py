import numpy as np
from statistics import stdev

from .preprocessor import Preprocessor
from .regularization import Regularization
from .loss import Loss
from .mini_batch_gradient_descent import MiniBatchGradientDescent


def power_expand(x: np.ndarray, power: int) -> np.ndarray:
    # | x0 |    | x0 x0^2 ... |
    # | x1 | => | x1 x1^2 ... |
    # | .. |    | .. .... ... |
    X = np.empty((x.shape[0], power))
    X[:, 0] = x
    for i in range(1, power):
        X[:, i] = x ** (i + 1)
    return X


class PolynomialRegressor:
    def __init__(
        self,
        fitted_power: int,
        regularization: Regularization,
        loss: Loss,
        learning_rate: float,
        batch_size: int,
        max_iter: int
    ):
        self.fitted_power: int = fitted_power
        self.model: MiniBatchGradientDescent = MiniBatchGradientDescent(
            regularization,
            loss,
            learning_rate,
            batch_size,
            max_iter
        )

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.model.fit(power_expand(x, self.fitted_power), y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict(power_expand(x, self.fitted_power))


def generate_random_weights(power: int, w_low: float, w_high: float) -> np.ndarray:
    rng = np.random.default_rng()
    return rng.uniform(w_low, w_high, power + 1)


def generate_random_samples(
    w: np.ndarray,
    noise_level: float,
    sample_size: int
):
    # return x and y, respectively
    rng = np.random.default_rng()
    x = rng.uniform(-1, 1, sample_size)
    y = np.c_[np.ones(sample_size), power_expand(x, w.shape[0] - 1)].dot(w)
    y += rng.normal(0, stdev(y) * noise_level, sample_size)
    return x, y
