from .utils import *


def test_no_noise_no_contamination(
    power: int,
    w_low: float,
    w_high: float,
    x_low: float,
    x_high: float,
    training_size: int,
    testing_size: int,
    regressor: PolynomialRegressor
):
    w = generate_random_weights(power, w_low, w_high)
    x_training, y_training = generate_random_samples(
        w,
        x_low,
        x_high, 0,
        training_size
    )
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
        None,
        x_testing,
        y_testing,
        power,
        regressor,
        "no noise no contamination"
    )


def test_no_contamination(
    power: int,
    w_low: float,
    w_high: float,
    x_low: float,
    x_high: float,
    noise_level: float,
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
        None,
        x_testing,
        y_testing,
        power,
        regressor,
        "no contamination"
    )
