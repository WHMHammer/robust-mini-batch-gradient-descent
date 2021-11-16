# Mini-Batch Gradient Descent with Trimmed Losses

## Introduction

This is for the final project of COMP_SCI 496: Foundations of Reliability and Robustness in Machine Learning at Northwestern University.

Mini Batch Gradient Descent (MBGD) is a simple yet effective machine learning model as a linear (and polynomial) regressor. However, the naïve MBGD model with squared losses is very sensitive to outliers, making it vulnerable to adversary samples.

Our group is proposing to add a trimming procedure based on the losses when calculating the gradients to make the MBGD model more robust. We will measure the robustness of the modified model under the ε-contamination model by calculating the mean squared error (MSE) on the training sets.

We will test with random adversaries, adversaries attempting to affect the slopes, and adversaries attempting to affect the bias.

## Algorithm

    Procedure fit(X, y, ε, batch_size, η, max_iter) -> w:
        Initialize w
        While not converged and not exceeding max_iter iterations:
            Randomly select batch_size samples without replacement
            Calculate the squared losses of all the samples in the batch
            Calculate the gradient of the losses with respect to w, ignoring the effects of the ε ⋅ batch_size samples with the largest losses
            Update w := w - η ⋅ gradient
        Return w

## Run

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 main.py
```

## Test Results

| Condition | Training Set (with trimming) | Testing Set (with trimming) | Training Set (without trimming) | Testing Set (without trimming) |
| - | - | - | - | - |
| No noise, no contamination | ![](test_result_img/No%20Noise%20No%20Contamination%20Training%20with%20trimming.png) | ![](test_result_img/No%20Noise%20No%20Contamination%20Testing%20with%20trimming.png) | ![](test_result_img/No%20Noise%20No%20Contamination%20Training%20without%20trimming.png) | ![](test_result_img/No%20Noise%20No%20Contamination%20Testing%20without%20trimming.png) |
| No contamination | ![](test_result_img/No%20Contamination%20Training%20with%20trimming.png) | ![](test_result_img/No%20Contamination%20Testing%20with%20trimming.png) | ![](test_result_img/No%20Contamination%20Training%20without%20trimming.png) | ![](test_result_img/No%20Contamination%20Testing%20without%20trimming.png) |
| Random contamination | ![](test_result_img/Random%20Contamination%20Training%20with%20trimming.png) | ![](test_result_img/Random%20Contamination%20Testing%20with%20trimming.png) | ![](test_result_img/Random%20Contamination%20Training%20without%20trimming.png) | ![](test_result_img/Random%20Contamination%20Testing%20without%20trimming.png) |

## TODO

- Test with more epsilons on training data without contamination.

- Test with more epsilons on training data with contamination having x and y following uniform distributions within their ranges in the authentic data.

- Test with more epsilons on training data with adversary contaminations attempting to affect specific weights.

- Verify the model with higher dimensions (powers).
