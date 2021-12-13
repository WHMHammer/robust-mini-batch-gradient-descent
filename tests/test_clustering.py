from tests.utils import *
from tests.test_no_contamination import *
from tests.test_random_contamination import *
from tests.test_parallel_line_contamination import *
from tests.test_edge_contamination import *
from tests.test_mse import *




def test_clustering(power: int, w_low: float, w_high: float, x_low: float, x_high: float, noise_level: float,
                      epsilon: float, y_distance: float, x_distance: float, training_size: int, testing_size: int,
                      regressor: PolynomialRegressor):
    rng = np.random.default_rng()
    w = generate_random_weights(power, w_low, w_high)
    # x_training, y_training = generate_random_samples(w, x_low, x_high, noise_level, training_size)
    x_training, y_training = generate_incomplete_samples(w, x_low, x_high, noise_level, 0.5, 0.5, training_size)
    contamination_size = int(training_size * epsilon)
    contaminated_indices = rng.choice(training_size, contamination_size, False)
    x_contamination = rng.uniform(
        x_training.mean() + (x_training.max() - x_training.min()) * x_distance,
        x_training.mean() + (x_training.max() - x_training.min()) * x_distance + (
                x_training.max() - x_training.min()) * 0.1,
        contamination_size)
    y_contamination = rng.uniform(
        y_training.mean() + (y_training.max() - y_training.min()) * y_distance,
        y_training.mean() + (y_training.max() - y_training.min()) * y_distance + (
                y_training.max() - y_training.min()) * 0.1,
        contamination_size)
    x_training[contaminated_indices] = x_contamination
    y_training[contaminated_indices] = y_contamination
    regressor.preprocessor.export_figure(x_training, y_training)

if __name__ == "__main__":
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
    epsilon = 0.49
    huber_loss_threshold = 20
    learning_rate = 0.01
    batch_size = 100
    max_iter = 100000
    regressor =PolynomialRegressor(
                                ClusteringPreprocessor(mode="DBSCAN", eps=2, min_samples=400),
                                fitted_power,
                                L1Regularization(),
                                regularization_weight,
                                HuberLoss(epsilon, huber_loss_threshold),
                                learning_rate,
                                batch_size,
                                max_iter
                            )
    # test_clustering(true_power,w_low,w_high,x_low,x_high,noise_level,
    #                         epsilon, 0.2, 0.2,
    #                         training_size,testing_size,
    #                         regressor
    # )
    test_mse_distance(true_power, w_low, w_high, x_low, x_high, noise_level,
                      0.49, 0.2, 0.2, training_size, testing_size, regressor)
