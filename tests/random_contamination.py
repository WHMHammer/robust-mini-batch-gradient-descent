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
y_contamination = rng.uniform(y_training.min(), y_training.max() * 2 - y_training.min(), contamination_size)
x_training[contamination_indices] = x_contamination
y_training[contamination_indices] = y_contamination

x_testing, y_testing = generate_random_samples(w, 0, 1000)

test_all(
    power,
    x_training,
    y_training,
    contamination_indices,
    x_testing,
    y_testing,
    "Random Contamination",
    "random_contamination"
)
