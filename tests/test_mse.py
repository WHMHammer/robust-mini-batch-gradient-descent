

from tests.utils import *
from tests.test_no_contamination import *
from tests.test_random_contamination import *
from tests.test_parallel_line_contamination import *
from tests.test_edge_contamination import *
import matplotlib.pyplot as plt
import numpy as np

# global parameters
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
epsilon = 0
huber_loss_threshold = 20
learning_rate = 0.01
batch_size = 100
max_iter = 100000

def test_mse_distance(power: int, w_low: float, w_high: float, x_low: float, x_high: float, noise_level: float,
                      epsilon: float, x_distance:float, y_distance:float,
                      training_size: int, testing_size: int, regressor: PolynomialRegressor):
    rng = np.random.default_rng()
    w = generate_random_weights(power, w_low, w_high)
    x_training, y_training = generate_random_samples( w, x_low, x_high, noise_level, training_size)
    contamination_size = int(training_size * epsilon)
    contaminated_indices = rng.choice(training_size, contamination_size, False)
    x_contamination = rng.uniform(
        x_training.mean() + (x_training.max()-x_training.min())*x_distance,
        x_training.mean() + (x_training.max()-x_training.min())*x_distance + (x_training.max()-x_training.min())*0.1,
        contamination_size)
    y_contamination = rng.uniform(
        y_training.mean() + (y_training.max()-y_training.min())*y_distance,
        y_training.mean() + (y_training.max()-y_training.min())*y_distance + (y_training.max()-y_training.min())*0.1,
        contamination_size)
    x_training[contaminated_indices] = x_contamination
    y_training[contaminated_indices] = y_contamination
    x_testing, y_testing = generate_random_samples(w, x_low, x_high, 0, testing_size)
    (mse_robust, mse_naive)=test_model(x_training, y_training, contaminated_indices,
               x_testing, y_testing, power, regressor, "mse_distance e="+str(epsilon) +" x="+str(x_distance)+" y="+str(y_distance))
    return (mse_robust, mse_naive)

def test_mse_density(power: int, w_low: float, w_high: float, x_low: float, x_high: float, noise_level: float,
                     x_begin, x_end, mode,
                      training_size: int, testing_size: int, regressor: PolynomialRegressor):
    rng = np.random.default_rng()
    w = generate_random_weights(power, w_low, w_high)
    if mode == "incomplete":
        x_training, y_training = generate_incomplete_samples(w, x_low, x_high, noise_level, x_begin, x_end, training_size)
    elif mode == "dense":
        x_training, y_training = generate_dense_samples(w, x_low, x_high, noise_level, x_begin, x_end, training_size)
    x_testing, y_testing = generate_random_samples(w, x_low, x_high, 0, testing_size)
    (mse_robust, mse_naive)=test_model(x_training, y_training, [],
               x_testing, y_testing, power, regressor, "mse_"+mode+" "+str(x_begin)+"-"+str(x_end))
    return (mse_robust, mse_naive)

def explore_figure(parameter, mse_naive, mse_robust, x_label, y_lable, title):
    plt.figure()
    plt.plot(parameter, mse_naive, label="robust")
    plt.plot(parameter, mse_robust, label="naive")
    # plt.suptitle(suptitle)
    # plt.title(title)
    plt.legend()
    plt.xlabel(x_label)
    plt.ylabel(y_lable)
    plt.title(title)
    plt.savefig("test_results/mse_figure/"+str(title)+".png")

def draw_mse_epsilon(true_power,w_low,w_high,x_low,x_high,noise_level,training_size,testing_size):
    mse_robust_list = []
    mse_naive_list = []
    repeat_times = 30
    x_distance = 0.3
    y_distance = 0.3
    parameter_list = np.around(np.arange(0.05, 0.49, 0.05), decimals=2)
    for e in parameter_list:
        sum_robust = 0
        sum_naive = 0
        for i in range(0, repeat_times):
            regressor = PolynomialRegressor(
                NullPreprocessor(),
                fitted_power,
                L1Regularization(),
                regularization_weight,
                HuberLoss(epsilon, huber_loss_threshold),
                learning_rate,
                batch_size,
                max_iter
            )
            (mse_robust, mse_naive)=test_mse_distance(true_power,w_low,w_high,x_low,x_high,noise_level,
                                              e, x_distance, y_distance,training_size,testing_size,regressor)
            sum_naive += mse_naive
            sum_robust += mse_robust
        mse_naive_list.append(sum_naive/repeat_times)
        mse_robust_list.append(sum_robust/repeat_times)
    explore_figure(parameter_list, mse_naive_list, mse_robust_list, "epsilon", "mse",
                   "mse_vs_epsilon_x=" + str(x_distance) + "_y=" + str(y_distance))

def draw_mse_y_distance(true_power,w_low,w_high,x_low,x_high,noise_level,training_size,testing_size):
    mse_robust_list = []
    mse_naive_list = []
    repeat_times = 20
    epsilon = 0.49
    x_distance = 0
    parameter_list = np.around(np.arange(-0.5, 0.5, 0.05), decimals=2)
    for y in parameter_list:
        sum_robust = 0
        sum_naive = 0
        for i in range(0, repeat_times):
            regressor = PolynomialRegressor(
                NullPreprocessor(),
                fitted_power,
                L1Regularization(),
                regularization_weight,
                HuberLoss(epsilon, huber_loss_threshold),
                learning_rate,
                batch_size,
                max_iter
            )
            (mse_robust, mse_naive)=test_mse_distance(true_power,w_low,w_high,x_low,x_high,
                                                      noise_level,epsilon, x_distance, y, training_size,testing_size,regressor)
            sum_naive += mse_naive
            sum_robust += mse_robust
        mse_naive_list.append(sum_naive/repeat_times)
        mse_robust_list.append(sum_robust/repeat_times)
    explore_figure(parameter_list, mse_naive_list, mse_robust_list, "y_distance", "mse",
                   "mse_vs_y_distance x=" + str(x_distance) + "_e=" + str(epsilon))


def draw_mse_x_distance(true_power,w_low,w_high,x_low,x_high,noise_level,training_size,testing_size):
    mse_robust_list = []
    mse_naive_list = []
    repeat_times = 20
    epsilon = 0.49
    y_distance = 0.2
    parameter_list = np.around(np.arange(-0.5, 0.5, 0.05), decimals=2)
    for x in parameter_list:
        sum_robust = 0
        sum_naive = 0
        for i in range(0, repeat_times):
            regressor = PolynomialRegressor(
                NullPreprocessor(),
                fitted_power,
                L1Regularization(),
                regularization_weight,
                HuberLoss(epsilon, huber_loss_threshold),
                learning_rate,
                batch_size,
                max_iter
            )
            (mse_robust, mse_naive)=test_mse_distance(true_power,w_low,w_high,x_low,x_high,noise_level,
                                              epsilon, x, y_distance, training_size,testing_size,regressor)
            sum_naive += mse_naive
            sum_robust += mse_robust
        mse_naive_list.append(sum_naive/repeat_times)
        mse_robust_list.append(sum_robust/repeat_times)
    explore_figure(parameter_list, mse_naive_list, mse_robust_list, "x_distance", "mse",
                   "mse_vs_x_distance y=" + str(y_distance) + "_e=" + str(epsilon))

if __name__ == "__main__":
    true_power = 9
    w_low = -10
    w_high = 10
    x_low = -1
    x_high = 1
    noise_level = 1
    training_size = 1000
    testing_size = 1000
    fitted_power = 5
    regularization_weight = 0
    epsilon = 0.49
    huber_loss_threshold = 20
    learning_rate = 0.01
    batch_size = 100
    max_iter = 100000
    regressor =PolynomialRegressor(
                                NullPreprocessor(),
                                fitted_power,
                                L1Regularization(),
                                regularization_weight,
                                HuberLoss(epsilon, huber_loss_threshold),
                                learning_rate,
                                batch_size,
                                max_iter
                            )

    # test_mse_distance(true_power, w_low, w_high, x_low, x_high, noise_level,
    #                   0.4, 0.3, 0.3, training_size, testing_size, regressor)
    # test_mse_density(true_power, w_low, w_high, x_low, x_high, noise_level,
    #                  0.4, 0.5, "incomplete",training_size, testing_size, regressor)
    # draw_mse_epsilon(true_power, w_low, w_high, x_low, x_high, noise_level, training_size, testing_size)

    # draw_mse_y_distance(true_power, w_low, w_high, x_low, x_high, noise_level, training_size, testing_size)
    # draw_mse_x_distance(true_power, w_low, w_high, x_low, x_high, noise_level, training_size, testing_size)