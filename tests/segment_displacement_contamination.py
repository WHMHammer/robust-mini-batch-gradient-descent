from tests.utils import *

power = 9
w = generate_random_weights(power, -10, 10)
epsilon = 0.49
rng = np.random.default_rng()

training_size = 1000
x_training, y_training = generate_random_samples(w, 1, training_size)
contamination_size = ceil(epsilon * training_size)
contamination_indices = np.argsort(x_training)[int(
    training_size * (1 - epsilon) * 0.5):int(training_size * (1 + epsilon) * 0.5)]
y_training[contamination_indices] += y_training.max() - y_training.min()

x_testing, y_testing = generate_random_samples(w, 0, 1000)

test_all(
    power,
    x_training,
    y_training,
    contamination_indices,
    x_testing,
    y_testing,
    "Segment Displacement Contamination",
    "segment_displacement_contamination"
)
