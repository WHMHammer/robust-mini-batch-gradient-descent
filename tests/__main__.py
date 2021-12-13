from .utils import *
from .test_no_contamination import *
from .test_random_contamination import *
from .test_parallel_line_contamination import *
from .test_edge_contamination import *

true_power = 9
w_low = -10
w_high = 10
x_low = -1
x_high = 1
noise_level = 1
training_size = 1000
testing_size = 1000
kernel_size = (0.1, 5)  # to be tuned
strides = (0.02, 1)  # to be tuned
preprocessor_threshold = 0.01  # to be tuned
fitted_power = 5
regularization_weight = 0
epsilon = 0
huber_loss_threshold = 20
learning_rate = 0.01
batch_size = 100
max_iter = 100000

test_no_noise_no_contamination(
    true_power,
    w_low,
    w_high,
    x_low,
    x_high,
    training_size,
    testing_size,
    PolynomialRegressor(
        NullPreprocessor(),
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
    w_low,
    w_high,
    x_low,
    x_high,
    noise_level,
    training_size,
    testing_size,
    PolynomialRegressor(
        NullPreprocessor(),
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
    w_low,
    w_high,
    x_low,
    x_high,
    noise_level,
    epsilon,
    training_size,
    testing_size,
    PolynomialRegressor(
        NullPreprocessor(),
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
    w_low,
    w_high,
    x_low,
    x_high,
    noise_level,
    epsilon,
    training_size,
    testing_size,
    PolynomialRegressor(
        NullPreprocessor(),
        fitted_power,
        L1Regularization(),
        regularization_weight,
        HuberLoss(epsilon, huber_loss_threshold),
        learning_rate,
        batch_size,
        max_iter
    )
)

test_edge_contamination(
    true_power,
    w_low,
    w_high,
    x_low,
    x_high,
    noise_level,
    epsilon,
    training_size,
    testing_size,
    PolynomialRegressor(
        NullPreprocessor(),
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
    w_low,
    w_high,
    x_low,
    x_high,
    noise_level,
    epsilon,
    training_size,
    testing_size,
    PolynomialRegressor(
        NullPreprocessor(),
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
    w_low,
    w_high,
    x_low,
    x_high,
    noise_level,
    epsilon,
    training_size,
    testing_size,
    PolynomialRegressor(
        NullPreprocessor(),
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
    w_low,
    w_high,
    x_low,
    x_high,
    noise_level,
    epsilon,
    training_size,
    testing_size,
    PolynomialRegressor(
        NullPreprocessor(),
        fitted_power,
        L1Regularization(),
        regularization_weight,
        HuberLoss(epsilon, huber_loss_threshold),
        learning_rate,
        batch_size,
        max_iter
    )
)
