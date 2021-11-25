import numpy as np

from .utils import *


def test_edge_contamination(
    power: int,
    w_low: float,
    w_high: float,
    x_low: float,
    x_high: float,
    noise_level: float,
    epsilon: float,
    training_size: int,
    testing_size: int,
    regressor: PolynomialRegressor
):
    rng = np.random.default_rng()
    w = generate_random_weights(power, w_low, w_high)
    x_training, y_training = generate_random_samples(
        w,
        x_low,
        x_high,
        noise_level,
        training_size
    )
    contamination_size = int(training_size * epsilon)
    contaminated_indices = rng.choice(training_size, contamination_size, False)
    x_contamination = rng.uniform(
        x_training.max() - (x_training.max() - x_training.min()) / 10,
        x_training.max(),
        contamination_size
    )
    y_contamination = rng.uniform(
        y_training.max() + (y_training.max() - y_training.min()) * 1.9,
        y_training.max() + (y_training.max() - y_training.min()) * 2,
        contamination_size
    )
    x_training[contaminated_indices] = x_contamination
    y_training[contaminated_indices] = y_contamination
    x_testing, y_testing = generate_random_samples(
        w,
        x_low,
        x_high,
        0,
        testing_size
    )
    test_model(
        x_training,
        y_training,
        contaminated_indices,
        x_testing,
        y_testing,
        power,
        regressor,
        "edge contamination"
    )


def test_begin_contamination(
    power: int,
    w_low: float,
    w_high: float,
    x_low: float,
    x_high: float,
    noise_level: float,
    epsilon: float,
    training_size: int,
    testing_size: int,
    regressor: PolynomialRegressor
):
    w = generate_random_weights(power, w_low, w_high)
    x_training, y_training = generate_random_samples(
        w,
        x_low,
        x_high,
        noise_level,
        training_size
    )
    contamination_size = int(training_size * epsilon)
    contaminated_indices = \
        np.argsort(x_training)[np.arange(contamination_size)]
    y_training[contaminated_indices] += \
        (y_training.max() - y_training.min()) * 2
    x_testing, y_testing = generate_random_samples(
        w,
        x_low,
        x_high,
        0,
        testing_size
    )
    test_model(
        x_training,
        y_training,
        contaminated_indices,
        x_testing,
        y_testing,
        power,
        regressor,
        "begin contamination"
    )


def test_end_contamination(
    power: int,
    w_low: float,
    w_high: float,
    x_low: float,
    x_high: float,
    noise_level: float,
    epsilon: float,
    training_size: int,
    testing_size: int,
    regressor: PolynomialRegressor
):
    w = generate_random_weights(power, w_low, w_high)
    x_training, y_training = generate_random_samples(
        w,
        x_low,
        x_high,
        noise_level,
        training_size
    )
    contamination_size = int(training_size * epsilon)
    contaminated_indices = np.argsort(x_training)[
        np.arange(training_size - contamination_size, training_size)
    ]
    y_training[contaminated_indices] += \
        (y_training.max() - y_training.min()) * 2
    x_testing, y_testing = generate_random_samples(
        w,
        x_low,
        x_high,
        0,
        testing_size
    )
    test_model(
        x_training,
        y_training,
        contaminated_indices,
        x_testing,
        y_testing,
        power,
        regressor,
        "end contamination"
    )


def test_mid_contamination(
    power: int,
    w_low: float,
    w_high: float,
    x_low: float,
    x_high: float,
    noise_level: float,
    epsilon: float,
    training_size: int,
    testing_size: int,
    regressor: PolynomialRegressor
):
    w = generate_random_weights(power, w_low, w_high)
    x_training, y_training = generate_random_samples(
        w,
        x_low,
        x_high,
        noise_level,
        training_size
    )
    contamination_size = int(training_size * epsilon)
    contaminated_indices = np.argsort(x_training)[np.arange(
        training_size // 2 - contamination_size // 2,
        training_size // 2 + contamination_size // 2
    )]
    y_training[contaminated_indices] += \
        (y_training.max() - y_training.min()) * 2
    x_testing, y_testing = generate_random_samples(
        w,
        x_low,
        x_high,
        0,
        testing_size
    )
    test_model(
        x_training,
        y_training,
        contaminated_indices,
        x_testing,
        y_testing,
        power,
        regressor,
        "mid contamination"
    )


def test_mid_rand_contamination(
    power: int,
    w_low: float,
    w_high: float,
    x_low: float,
    x_high: float,
    noise_level: float,
    epsilon: float,
    training_size: int,
    testing_size: int,
    regressor: PolynomialRegressor
):
    rng = np.random.default_rng()
    w = generate_random_weights(power, w_low, w_high)
    x_training, y_training = generate_random_samples(
        w,
        x_low,
        x_high,
        noise_level,
        training_size
    )
    contamination_size = int(training_size * epsilon)
    num = rng.integers(2, 10)
    contaminated_indices = np.argsort(x_training)[np.arange(
        training_size // num,
        training_size // num + contamination_size
    )]
    y_training[contaminated_indices] += \
        (y_training.max() - y_training.min()) * 2
    x_testing, y_testing = generate_random_samples(
        w,
        x_low,
        x_high,
        0,
        testing_size
    )
    test_model(
        x_training,
        y_training,
        contaminated_indices,
        x_testing,
        y_testing,
        power,
        regressor,
        "mid rand contamination"
    )
