from os import makedirs
from os.path import join

from src.preprocessor import *
from src.regularization import *
from src.loss import *
from src.mini_batch_gradient_descent import *
from src.polynomial_regressor import *
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

    x_training_transformed, y_training_transformed = regressor.preprocessor(
        x_training, y_training)
    regressor.fit(x_training_transformed, y_training_transformed)
    y_bar_training = regressor.predict(x_training)
    y_bar_testing = regressor.predict(x_testing)
    mse_preprocess = mean_square_error(y_testing, y_bar_testing)
    export_figure(
        x_training,
        y_training,
        x_training_transformed,
        y_training_transformed,
        y_bar_training,
        contaminated_indices,
        f"{test_name} (preprocess)",
        f"training set, true power={true_power}, fitted power={regressor.power}, ε={regressor.model.loss.epsilon}",
        (x_training.min(), x_training.max()),
        (y_training.min(), y_training.max()),
        join(dir_name, "preprocess_training")
    )
    export_figure(
        x_testing,
        y_testing,
        None,
        None,
        y_bar_testing,
        None,
        f"{test_name} (preprocess)",
        f"testing set, true power={true_power}, fitted power={regressor.power}, ε={regressor.model.loss.epsilon}, MSE={mse_preprocess}",
        (x_training.min(), x_training.max()),
        (y_training.min(), y_training.max()),
        join(dir_name, "preprocess_testing")
    )

    regressor.preprocessor = NullPreprocessor()
    regressor.model.regularization_weight = 0
    regressor.fit(x_training, y_training)
    y_bar_training = regressor.predict(x_training)
    y_bar_testing = regressor.predict(x_testing)
    mse_robust = mean_square_error(y_testing, y_bar_testing)
    export_figure(
        x_training,
        y_training,
        # x_training_transformed,
        # y_training_transformed,
        None,
        None,
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
        None,
        None,
        y_bar_testing,
        None,
        f"{test_name} (robust)",
        f"testing set, true power={true_power}, fitted power={regressor.power}, ε={regressor.model.loss.epsilon}, MSE={mse_robust}",
        (x_training.min(), x_training.max()),
        (y_training.min(), y_training.max()),
        join(dir_name, "robust_testing")
    )


    regressor.model.regularization_weight = 0
    regressor.model.loss = SquaredLoss(0)
    regressor.fit(x_training, y_training)
    y_bar_training = regressor.predict(x_training)
    y_bar_testing = regressor.predict(x_testing)
    mse_naive = mean_square_error(y_testing, y_bar_testing)
    export_figure(
        x_training,
        y_training,
        None,
        None,
        y_bar_training,
        contaminated_indices,
        f"{test_name} (naive)",
        f"training set, true power={true_power}, fitted power={regressor.power}, ε={regressor.model.loss.epsilon}",
        (x_training.min(), x_training.max()),
        (y_training.min(), y_training.max()),
        join(dir_name, "naive_training")
    )
    export_figure(
        x_testing,
        y_testing,
        None,
        None,
        y_bar_testing,
        None,
        f"{test_name} (naive)",
        f"testing set, true power={true_power}, fitted power={regressor.power}, ε={regressor.model.loss.epsilon}, MSE={mse_naive}",
        (x_training.min(), x_training.max()),
        (y_training.min(), y_training.max()),
        join(dir_name, "naive_testing")
    )
    return (mse_robust, mse_naive)
