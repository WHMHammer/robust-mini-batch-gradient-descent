from src.regularization import *
from src.loss import *
from .utils.polynomial_regressor import *
from .test_no_contamination import *

test_no_noise_no_contamination(5, -10, 10, -1, 1, 1000, 1000, PolynomialRegressor(
    5, L2Regularization(), 0, SquaredLoss(0), 0.01, 100, 100000))
test_no_contamination(5, -10, 10, -1, 1, 1, 1000, 1000, PolynomialRegressor(5,
                      L2Regularization(), 0, SquaredLoss(0), 0.01, 100, 100000))
