from tests.utils import *

w = generate_random_weights(9, -10, 10)
epsilon = 0.49
rng = np.random.default_rng()

training_size = 1000
x_training, y_training = generate_random_samples(w, 1, training_size)
contamination_size = ceil(epsilon * training_size)
contamination_indices = rng.choice(training_size, contamination_size, False)
y_training[contamination_indices] += y_training.max() - y_training.min()

x_testing, y_testing = generate_random_samples(w, 0, 1000)

test_all(
    5,
    x_training,
    y_training,
    contamination_indices,
    x_testing,
    y_testing,
    "Parallel-line Contamination",
    "parallel_line_contamination"
)
