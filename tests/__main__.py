from src.regularization import *
from src.loss import *
from .utils.polynomial_regressor import *

from .test_no_contamination import *
from .test_random_contamination import *
from .test_parallel_line_contamination import *
from .test_edge_contamination import *

power = 9
epsilon = 0.49

test_no_noise_no_contamination(power, -10, 10, -1, 1, 1000, 1000, PolynomialRegressor(
    power, L2Regularization(), 0, SquaredLoss(0), 0.01, 100, 100000))

test_no_contamination(power, -10, 10, -1, 1, 1, 1000, 1000, PolynomialRegressor(
    power, L2Regularization(), 0, SquaredLoss(0), 0.01, 100, 100000))

test_random_contamination(power, -10, 10, -1, 1, 1, epsilon, 1000, 1000, PolynomialRegressor(
    power, L2Regularization(), 0, SquaredLoss(epsilon), 0.01, 100, 100000))

test_parallel_line_contamination(power, -10, 10, -1, 1, 1, epsilon, 1000, 1000, PolynomialRegressor(
    power, L2Regularization(), 0, SquaredLoss(epsilon), 0.01, 100, 100000))

epsilon = 0.25

test_edge_contamination(power, -10, 10, -1, 1, 1, epsilon, 1000, 1000, PolynomialRegressor(
    power, L2Regularization(), 0, SquaredLoss(epsilon), 0.01, 100, 100000))

test_begin_contamination(power, -10, 10, -1, 1, 1, epsilon, 1000, 1000, PolynomialRegressor(
    power, L2Regularization(), 0, SquaredLoss(epsilon), 0.01, 100, 100000))

test_end_contamination(power, -10, 10, -1, 1, 1, epsilon, 1000, 1000, PolynomialRegressor(
    power, L2Regularization(), 0, SquaredLoss(epsilon), 0.01, 100, 100000))

test_mid_contamination(power, -10, 10, -1, 1, 1, epsilon, 1000, 1000, PolynomialRegressor(
    power, L2Regularization(), 0, SquaredLoss(epsilon), 0.01, 100, 100000))

test_mid_rand_contamination(power, -10, 10, -1, 1, 1, epsilon, 1000, 1000, PolynomialRegressor(
    power, L2Regularization(), 0, SquaredLoss(epsilon), 0.01, 100, 100000))
