# Mini-Batch Gradient Descent with Trimmed Losses

## Introduction

This is for the final project of COMP_SCI 496: Foundations of Reliability and Robustness in Machine Learning at Northwestern University.

Mini Batch Gradient Descent (MBGD) is a simple yet effective machine learning model as a linear regressor. However, the naïve MBGD model with squared losses is very sensitive to outliers, making it vulnerable to adversary samples.

Our group is proposing to add loss and gradient trimming to the fitting procedure to make the MBGD model more robust. We will measure the robustness of the modified model under the epsilon-contamination model by calculating the mean squared error (MSE) on the training sets.

We will test with random adversaries, adversaries attempting to affect the slopes, and adversaries attempting to affect the bias.

## Algorithm

    Method fit(X, y, ε, batch_size, η, max_iter) -> w:
        Initialize w
        While not converged and not exceeding max_iter iterations:
            For each batch:
                Calculate the squared losses of all samples
                Trim the ε ⋅ batch_size samples with the largest losses
                Calculate the gradient of w, ignoring the effects of the trimmed samples
                Update w
        Return w

## Run

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
python3 main.py
```

## Test Results

| Condition | Training Set | Testing Set |
| - | - | - |
| No noise, no contamination | ![](00%20No%20Noise%20No%20Contamination%20Training.png) | ![](00%20No%20Noise%20No%20Contamination%20Testing.png) |
| No contamination | ![](01%20No%20Contamination%20Training.png) | ![](01%20No%20Contamination%20Testing.png) |
| Random contamination | ![](02%20Random%20Contamination%20Training.png) | ![](02%20Random%20Contamination%20Testing.png) |
