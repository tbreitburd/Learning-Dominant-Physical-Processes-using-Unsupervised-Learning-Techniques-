import numpy as np
from scipy.stats import multivariate_normal


class CustomGMM:
    def __init__(
        self,
        n_components: int,
        n_features: int,
        max_iter: int,
        tolerance: 1e-6 | float,
        random_state: None | int,
    ):
        self.n_components = n_components
        self.n_features = n_features
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state

        # Set the seeds
        np.random.seed(self.random_state)

        # Initialise the distributions mixture
        # initialise the weights, means and covariances
        self.weights = np.ones(self.n_components) / self.n_components
        # define multivariate normal distributions
        normal = multivariate_normal()  # noqa: F841

    def fit(self, features):
        max_iter = 100  # noqa: F841
        # loop over iterations
        # for i in range(self.max_iter):

        # Expectation step
        # Calculate the probability of each component for each sample
        # probs = evaluate each multivariate normal distribution for each sample
        # Maximisation step
        # Update the weights
        # https://towardsdatascience.com/how-to-code-gaussian-mixture-models-from-scratch-in-python-9e7975df5252
        # Update the means
        # Update the covariances
        # Check the tolerance
        # if new log likelihood - old log likelihood < tolerance:
        # break

    def predict(self, features):
        # for each sample, calculate the probability of each component
        probs = None  # evaluate each multivariate normal distribution for each sample
        # return the component with the highest probability

        return np.argmax(probs)
