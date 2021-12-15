from ..utils import *


def test_random_contamination(
    preprocessor: Union[Preprocessor, None],
    regressor: PolynomialRegressor,
    x_training: np.ndarray,
    y_training: np.ndarray,
    contamination_indices: np.ndarray,
    x_testing: np.ndarray,
    y_testing: np.ndarray,
    test_name: str,
    base_filename: str
):
    # train
    if preprocessor is None:
        transformed_x = None
        transformed_y = None
        regressor.fit(x_training, y_training)
    else:
        transformed_x, transformed_y = preprocessor(x_training, y_training)
        regressor.fit(transformed_x, transformed_y)

    # test on the training set
    predicted_y_training = regressor.predict(x_training)
    export_training_figure(
        x_training,
        y_training,
        contamination_indices,
        transformed_x,
        transformed_y,
        predicted_y_training,
        test_name,
        base_filename + "_training"
    )
