from src.regularization import *
from src.loss import *
from .utils.polynomial_regressor import *

from .test_no_contamination import *
from .test_random_contamination import *
from .test_parallel_line_contamination import *
from .test_edge_contamination import *

true_power = 5
training_size = 1000
testing_size = 1000
fitted_power = 9
regularization_weight = 0
epsilon = 0
huber_loss_threshold = 20
learning_rate = 0.01
batch_size = 100
max_iter = 100000

test_no_noise_no_contamination(
    true_power,
    -10,
    10,
    -1,
    1,
    training_size,
    testing_size,
    PolynomialRegressor(
        fitted_power,
        L1Regularization(),
        regularization_weight,
        HuberLoss(epsilon, huber_loss_threshold),
        learning_rate,
        batch_size,
        max_iter
    )
)

test_no_contamination(
    true_power,
    -10,
    10,
    -1,
    1,
    1,
    training_size,
    testing_size,
    PolynomialRegressor(
        fitted_power,
        L1Regularization(),
        regularization_weight,
        HuberLoss(epsilon, huber_loss_threshold),
        learning_rate,
        batch_size,
        max_iter
    )
)

epsilon = 0.49

test_random_contamination(
    true_power,
    -10,
    10,
    -1,
    1,
    1,
    epsilon,
    training_size,
    testing_size,
    PolynomialRegressor(
        fitted_power,
        L1Regularization(),
        regularization_weight,
        HuberLoss(epsilon, huber_loss_threshold),
        learning_rate,
        batch_size,
        max_iter
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
    training_size,
    testing_size,
    PolynomialRegressor(
        fitted_power,
        L1Regularization(),
        regularization_weight,
        HuberLoss(epsilon, huber_loss_threshold),
        learning_rate,
        batch_size,
        max_iter
    )
)

epsilon = 0.33

test_edge_contamination(
    true_power,
    -10,
    10,
    -1,
    1,
    1,
    epsilon,
    training_size,
    testing_size,
    PolynomialRegressor(
        fitted_power,
        L1Regularization(),
        regularization_weight,
        HuberLoss(epsilon, huber_loss_threshold),
        learning_rate,
        batch_size,
        max_iter
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
    training_size,
    testing_size,
    PolynomialRegressor(
        fitted_power,
        L1Regularization(),
        regularization_weight,
        HuberLoss(epsilon, huber_loss_threshold),
        learning_rate,
        batch_size,
        max_iter
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
    training_size,
    testing_size,
    PolynomialRegressor(
        fitted_power,
        L1Regularization(),
        regularization_weight,
        HuberLoss(epsilon, huber_loss_threshold),
        learning_rate,
        batch_size,
        max_iter
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
    training_size,
    testing_size,
    PolynomialRegressor(
        fitted_power,
        L1Regularization(),
        regularization_weight,
        HuberLoss(epsilon, huber_loss_threshold),
        learning_rate,
        batch_size,
        max_iter
    )
)
