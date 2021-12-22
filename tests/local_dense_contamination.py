from tests.utils import *

w = generate_random_weights(9, -10, 10)
epsilon = 0.49
rng = np.random.default_rng()

training_size = 1000
x_training, y_training = generate_random_samples(w, 1, training_size)
contamination_size = ceil(epsilon * training_size)
contamination_indices = rng.choice(training_size, contamination_size, False)
x_contamination = rng.uniform(0.8, 1, contamination_size)
y_contamination = rng.uniform(y_training.max() * 1.9 - y_training.min()
                              * 0.9, y_training.max() * 2 - y_training.min(), contamination_size)
x_training[contamination_indices] = x_contamination
y_training[contamination_indices] = y_contamination

x_testing, y_testing = generate_random_samples(w, 0, 1000)

test_all(
    5,
    x_training,
    y_training,
    contamination_indices,
    x_testing,
    y_testing,
    "Local Dense Contamination",
    "local_dense_contamination"
)
