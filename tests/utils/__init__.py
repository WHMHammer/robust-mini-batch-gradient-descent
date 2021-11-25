from os import makedirs
from os.path import join

from src.loss import SquaredLoss
from .polynomial_regressor import *
from .figures import *


def mean_square_error(y: np.ndarray, y_bar: np.ndarray) -> float:
    return np.square(y_bar - y).sum() / y_bar.shape[0]


def test_model(
    x_training: np.ndarray,
    y_training: np.ndarray,
    contaminated_indices: np.ndarray,
    x_testing: np.ndarray,
    y_testing: np.ndarray,
    true_power: int,
    regressor: PolynomialRegressor,
    test_name: str
):
    print(f"Testing {test_name}.")
    dir_name = join("test_results", test_name.replace(" ", "_"))
    try:
        makedirs(dir_name)
    except FileExistsError:
        pass

    regressor.fit(x_training, y_training)
    y_bar_training = regressor.predict(x_training)
    y_bar_testing = regressor.predict(x_testing)
    mse = mean_square_error(y_testing, y_bar_testing)
    export_figure(
        x_training,
        y_training,
        y_bar_training,
        contaminated_indices,
        f"{test_name} (robust)",
        f"training set, true power={true_power}, fitted power={regressor.power}, ε={regressor.model.loss.epsilon}",
        (x_training.min(), x_training.max()),
        (y_training.min(), y_training.max()),
        join(dir_name, "robust_training")
    )
    export_figure(
        x_testing,
        y_testing,
        y_bar_testing,
        None,
        f"{test_name} (robust)",
        f"testing set, true power={true_power}, fitted power={regressor.power}, ε={regressor.model.loss.epsilon}, MSE={mse}",
        (x_training.min(), x_training.max()),
        (y_training.min(), y_training.max()),
        join(dir_name, "robust_testing")
    )

    regressor.model.regularization_weight = 0
    regressor.model.loss = SquaredLoss(0)
    regressor.fit(x_training, y_training)
    y_bar_training = regressor.predict(x_training)
    y_bar_testing = regressor.predict(x_testing)
    mse = mean_square_error(y_testing, y_bar_testing)
    export_figure(
        x_training,
        y_training,
        y_bar_training,
        contaminated_indices,
        f"{test_name} (naïve)",
        f"training set, true power={true_power}, fitted power={regressor.power}, ε={regressor.model.loss.epsilon}",
        (x_training.min(), x_training.max()),
        (y_training.min(), y_training.max()),
        join(dir_name, "naïve_training")
    )
    export_figure(
        x_testing,
        y_testing,
        y_bar_testing,
        None,
        f"{test_name} (naïve)",
        f"testing set, true power={true_power}, fitted power={regressor.power}, ε={regressor.model.loss.epsilon}, MSE={mse}",
        (x_training.min(), x_training.max()),
        (y_training.min(), y_training.max()),
        join(dir_name, "naïve_testing")
    )
