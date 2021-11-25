import numpy as np
from .utils import *

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
epsilon = 0.49

rng = np.random.default_rng()
w = generate_random_weights(true_power, w_low, w_high)
x_training, y_training = generate_random_samples(
    w,
    x_low,
    x_high,
    noise_level,
    training_size
)
contamination_size = int(training_size * epsilon)
contaminated_indices = rng.choice(training_size, contamination_size, False)
x_contamination = rng.uniform(
    x_training.min(),
    x_training.max(),
    contamination_size
)
y_contamination = rng.uniform(
    y_training.min(),
    y_training.max(),
    contamination_size
)
x_training[contaminated_indices] = x_contamination
y_training[contaminated_indices] = y_contamination
preprocessor = MeanKernelPreprocessor(kernel_size, strides, preprocessor_threshold)
new_x, new_y = preprocessor(x_training, y_training)

plt.figure()
plt.title("Mean Kernel Preprocessor")
plt.scatter(
    np.delete(x_training, contaminated_indices),
    np.delete(y_training, contaminated_indices),
    s=4,
    c="blue",
    label="True Samples"
)
plt.scatter(
    x_training[contaminated_indices],
    y_training[contaminated_indices],
    s=4,
    color="gray",
    label="Contamination"
)
plt.scatter(
    new_x,
    new_y,
    s=16,
    c="red",
    label="Transformed Samples",
)
plt.legend()
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("mean_kernel_preprocessor")
plt.close()
