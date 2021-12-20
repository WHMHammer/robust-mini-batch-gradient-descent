import numpy as np
from math import ceil

from numpy.core.numeric import base_repr
from tests.utils import *

power = 9
w = generate_random_weights(power, -10, 10)
epsilon = 0.49
rng = np.random.default_rng()

training_size = 1000
x_training, y_training = generate_random_samples(w, 1, training_size)
contamination_size = ceil(epsilon * training_size)
contamination_indices = rng.choice(training_size, contamination_size, False)
x_contamination = rng.uniform(-1, 1, contamination_size)
y_contamination = rng.uniform(
    y_training.min(), y_training.max(), contamination_size)
x_training[contamination_indices] = x_contamination
y_training[contamination_indices] = y_contamination

x_testing, y_testing = generate_random_samples(w, 0, 1000)

markdown_str = "| Random Contamination |"

# naive model test begin
regressor = PolynomialRegressor(
    power,
    NullRegularization(),
    EpsilonTrimmedSquaredLoss(0),
    0.01,
    100,
    100000
)
regressor.fit(x_training, y_training)

predicted_y_training = regressor.predict(x_training)
predicted_y_testing = regressor.predict(x_testing)
markdown_str += export_figures(
    x_training,
    y_training,
    contamination_indices,
    None,
    None,
    predicted_y_training,
    x_testing,
    y_testing,
    predicted_y_testing,
    "Random Contamination (Naive)",
    join("random_contamination", "naive")
)
# naive model test end

# huber loss test begin
regressor = PolynomialRegressor(
    power,
    NullRegularization(),
    EpsilonTrimmedHuberLoss(0, 10),
    0.01,
    100,
    100000
)
regressor.fit(x_training, y_training)

predicted_y_training = regressor.predict(x_training)
predicted_y_testing = regressor.predict(x_testing)
markdown_str += export_figures(
    x_training,
    y_training,
    contamination_indices,
    None,
    None,
    predicted_y_training,
    x_testing,
    y_testing,
    predicted_y_testing,
    "Random Contamination (Huber Loss)",
    join("random_contamination", "huber_loss")
)
# huber loss test end

# epsilon-trimmed huber loss test begin
regressor = PolynomialRegressor(
    power,
    NullRegularization(),
    EpsilonTrimmedHuberLoss(epsilon, 10),
    0.01,
    100,
    100000
)
regressor.fit(x_training, y_training)

predicted_y_training = regressor.predict(x_training)
predicted_y_testing = regressor.predict(x_testing)
markdown_str += export_figures(
    x_training,
    y_training,
    contamination_indices,
    None,
    None,
    predicted_y_training,
    x_testing,
    y_testing,
    predicted_y_testing,
    "Random Contamination (ε-trimmed Huber Loss)",
    join("random_contamination", "epsilon_trimmed_huber_loss")
)
# epsilon-trimmed huber loss test end

# mean-kernel preprocessor test begin
preprocessor = MeanKernelPreprocessor(
    (0.2, 2),
    (0.02, 0.2),
    0.01
)
regressor = PolynomialRegressor(
    power,
    NullRegularization(),
    EpsilonTrimmedSquaredLoss(0),
    0.01,
    100,
    100000
)
transformed_x, transformed_y = preprocessor(x_training, y_training)
regressor.fit(transformed_x, transformed_y)

predicted_y_training = regressor.predict(x_training)
predicted_y_testing = regressor.predict(x_testing)
markdown_str += export_figures(
    x_training,
    y_training,
    contamination_indices,
    transformed_x,
    transformed_y,
    predicted_y_training,
    x_testing,
    y_testing,
    predicted_y_testing,
    "Random Contamination (Mean-kernel Preprocessor)",
    join("random_contamination", "mean_kernel_preprocessor")
)
# mean-kernel preprocessor test end

# epsilon-trimmed huber loss with mean-kernel preprocessor test begin
preprocessor = MeanKernelPreprocessor(
    (0.2, 2),
    (0.02, 0.2),
    0.01
)
regressor = PolynomialRegressor(
    power,
    NullRegularization(),
    EpsilonTrimmedHuberLoss(epsilon, 10),
    0.01,
    100,
    100000
)
transformed_x, transformed_y = preprocessor(x_training, y_training)
regressor.fit(transformed_x, transformed_y)

predicted_y_training = regressor.predict(x_training)
predicted_y_testing = regressor.predict(x_testing)
markdown_str += export_figures(
    x_training,
    y_training,
    contamination_indices,
    transformed_x,
    transformed_y,
    predicted_y_training,
    x_testing,
    y_testing,
    predicted_y_testing,
    "Random Contamination (ε-trimmed Huber Loss with Mean-kernel Preprocessor)",
    join("random_contamination",
         "epsilon_trimmed_huber_loss_with_mean_kernel_preprocessor")
)
# epsilon-trimmed huber loss with mean-kernel preprocessor test end

# print(markdown_str)
