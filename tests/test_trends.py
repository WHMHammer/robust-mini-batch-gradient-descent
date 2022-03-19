from tests.utils import *

w = generate_random_weights(9, -10, 10)
epsilon = 0.49
rng = np.random.default_rng()

naive_regressor = PolynomialRegressor(
    5,
    NullRegularization(),
    SquaredLoss(),
    0.01,
    100,
    100000
)

robust_regressor = PolynomialRegressor(
    5,
    NullRegularization(),
    EpsilonTrimmedLoss(ZScoreTrimmedLoss(HuberLoss(10), 2), 0),
    0.01,
    100,
    100000
)

training_size = 1000
x_training_raw, y_training_raw = generate_random_samples(w, 1, training_size)
x_testing, y_testing = generate_random_samples(w, 0, 1000)

epsilons = np.arange(0, 0.5, 0.01)

naive_mses = list()
robust_mses = list()

for epsilon in epsilons:
    print(epsilon)
    x_training = np.copy(x_training_raw)
    y_training = np.copy(y_training_raw)
    contamination_size = ceil(epsilon * training_size)
    contamination_indices = rng.choice(
        training_size, contamination_size, False)
    x_contamination = rng.uniform(-1, 1, contamination_size)
    y_contamination = rng.uniform(
        y_training.min(), y_training.max() * 2 - y_training.min(), contamination_size)
    x_training[contamination_indices] = x_contamination
    y_training[contamination_indices] = y_contamination

    naive_regressor.fit(x_training, x_training_raw)
    naive_mses.append(mean_squared_error(
        naive_regressor.predict(x_testing), y_testing))

    robust_regressor.model.loss.epsilon = epsilon
    robust_regressor.fit(x_training, x_training_raw)
    robust_mses.append(mean_squared_error(
        robust_regressor.predict(x_testing), y_testing))

plt.figure()
plt.suptitle("MSE vs. Epsilon")
plt.title("Random Contamination")
plt.grid()
plt.plot(epsilons, naive_mses, c="red", label="Naïve")
plt.plot(epsilons, robust_mses, c="blue", label="Robust")
plt.xlabel("Epsilon")
plt.xlim(0, 0.5)
plt.ylabel("MSE")
plt.legend()
plt.savefig("mse_vs_epsilon")
plt.close()
