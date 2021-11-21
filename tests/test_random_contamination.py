import numpy as np
from .utils import *


def test_random_contamination(power: int, w_low: float, w_high: float, x_low: float, x_high: float, noise_level: float, epsilon: float, training_size: int, testing_size: int, regressor: PolynomialRegressor):
    rng = np.random.default_rng()
    w = generate_random_weights(power, w_low, w_high)
    x_training, y_training = generate_random_samples(
        w, x_low, x_high, noise_level, training_size)
    contamination_size = int(training_size * epsilon)
    contaminated_indices = rng.choice(training_size, contamination_size, False)
    x_contamination = rng.uniform(
        x_training.min(), x_training.max(), contamination_size)
    y_contamination = rng.uniform(
        y_training.min(), y_training.max(), contamination_size)
    x_training[contaminated_indices] = x_contamination
    y_training[contaminated_indices] = y_contamination
    x_testing, y_testing = generate_random_samples(
        w, x_low, x_high, 0, testing_size)
    test_model(x_training, y_training, contaminated_indices, x_testing,
               y_testing, regressor, "random contamination")
