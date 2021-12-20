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
preprocessor = None
regressor = PolynomialRegressor(
    power,
    NullRegularization(),
    EpsilonTrimmedSquaredLoss(0),
    0.01,
    100,
    100000
)

if preprocessor is None:
    transformed_x = None
    transformed_y = None
    regressor.fit(x_training, y_training)
else:
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
    "Random Contamination (naive)",
    join("random_contamination", "naive")
)
# naive model test end

preprocessor = None
regressor = PolynomialRegressor(
    power,
    NullRegularization(),
    EpsilonTrimmedHuberLoss(epsilon, 20),
    0.01,
    100,
    100000
)

if preprocessor is None:
    transformed_x = None
    transformed_y = None
    regressor.fit(x_training, y_training)
else:
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
    "Random Contamination (ε-trimmed huber loss)",
    join("random_contamination", "epsilon_trimmed_huber_loss")
)

# print(markdown_str)
