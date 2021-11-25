from src.regularization import *
from src.loss import *
from .utils.polynomial_regressor import *

from .test_no_contamination import *
from .test_random_contamination import *
from .test_parallel_line_contamination import *
from .test_edge_contamination import *

true_power = 5
fitted_power = 9
epsilon = 0.49
huber_loss_threshold = 1000

test_no_noise_no_contamination(
    true_power,
    -10,
    10,
    -1,
    1,
    1000,
    1000,
    PolynomialRegressor(
        fitted_power,
        L2Regularization(),
        0,
        HuberLoss(epsilon, huber_loss_threshold),
        0.01,
        100,
        100000
    )
)

test_no_contamination(
    true_power,
    -10,
    10,
    -1,
    1,
    1,
    1000,
    1000,
    PolynomialRegressor(
        fitted_power,
        L2Regularization(),
        0,
        HuberLoss(epsilon, huber_loss_threshold),
        0.01,
        100,
        100000
    )
)

test_random_contamination(
    true_power,
    -10,
    10,
    -1,
    1,
    1,
    epsilon,
    1000,
    1000,
    PolynomialRegressor(
        fitted_power,
        L2Regularization(),
        0,
        HuberLoss(epsilon, huber_loss_threshold),
        0.01,
        100,
        100000
    )
)

test_parallel_line_contamination(
    true_power,
    -10,
    10,
    -1,
    1,
    1,
    epsilon,
    1000,
    1000,
    PolynomialRegressor(
        fitted_power,
        L2Regularization(),
        0,
        HuberLoss(epsilon, huber_loss_threshold),
        0.01,
        100,
        100000
    )
)

epsilon = 0.25

test_edge_contamination(
    true_power,
    -10,
    10,
    -1,
    1,
    1,
    epsilon,
    1000,
    1000,
    PolynomialRegressor(
        fitted_power,
        L2Regularization(),
        0,
        HuberLoss(epsilon, huber_loss_threshold),
        0.01,
        100,
        100000
    )
)

test_begin_contamination(
    true_power,
    -10,
    10,
    -1,
    1,
    1,
    epsilon,
    1000,
    1000,
    PolynomialRegressor(
        fitted_power,
        L2Regularization(),
        0,
        HuberLoss(epsilon, huber_loss_threshold),
        0.01,
        100,
        100000
    )
)

test_end_contamination(
    true_power,
    -10,
    10,
    -1,
    1,
    1,
    epsilon,
    1000,
    1000,
    PolynomialRegressor(
        fitted_power,
        L2Regularization(),
        0,
        HuberLoss(epsilon, huber_loss_threshold),
        0.01,
        100,
        100000
    )
)

test_mid_contamination(
    true_power,
    -10,
    10,
    -1,
    1,
    1,
    epsilon,
    1000,
    1000,
    PolynomialRegressor(
        fitted_power,
        L2Regularization(),
        0,
        HuberLoss(epsilon, huber_loss_threshold),
        0.01,
        100,
        100000
    )
)

test_mid_rand_contamination(
    true_power,
    -10,
    10,
    -1,
    1,
    1,
    epsilon,
    1000,
    1000,
    PolynomialRegressor(
        fitted_power,
        L2Regularization(),
        0,
        HuberLoss(epsilon, huber_loss_threshold),
        0.01,
        100,
        100000
    )
)
