import numpy as np
from scipy.stats import multivariate_normal

# https://towardsdatascience.com/how-to-code-gaussian-mixture-models-from-scratch-in-python-9e7975df5252


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

    def _initialise_parameters(self, features):
        # Initialise the weights as uniform
        self.weights_ = np.ones(self.n_components) / self.n_components

        # Initialise the means as random samples from the features
        mean_samples = np.random.choice(range(features.shape[0]), self.n_components)
        self.means = features[mean_samples]

        # Initialise the covariances, as (conservative choice) identity matrices
        self.covariances = np.array([np.eye(self.n_features)] * self.n_components)

    def expectation_step(self, features):
        # Initialise the likelihoods
        likelihoods = np.zeros((features.shape[0], self.n_components))

        # Calculate the likelihood of each sample for each component
        for i in range(self.n_components):
            likelihoods[:, i] = self.weights_[i] * multivariate_normal.pdf(
                features, self.means[i], self.covariances[i]
            )

        # Calc likelihood of sample belonging to certain component
        # by calc likelihood for component k/sum of likelihoods for all components
        total_likelihoods = likelihoods.sum(axis=1)
        probabilities = likelihoods / total_likelihoods

        return probabilities

    def maximisation_step(self, features, probabilities):
        # Update the weights
        self.weights_ = probabilities.mean(axis=0)

        # Update the means, dot product takes care of element-wise multiplication and sum
        self.means = np.dot(probabilities.T, features) / probabilities.sum(axis=0)

        # Update the covariances
        for i in range(self.n_components):
            self.covariances_[i] = (
                np.sum(
                    probabilities[:, i]
                    * (features - self.means_[i]).T
                    * (features - self.means_[i]),
                    axis=0,
                )
                / probabilities[:, i].sum()
            )

    def fit(self, features):
        # Calc first log likelihood
        probabilities = self.expectation_step(features)
        log_likelihood = np.sum(np.log(np.sum(probabilities, axis=0)))
        # loop over iterations
        for i in range(self.max_iter):
            # Expectation step
            # Calculate the probability of each component for each sample
            probabilities = self.expectation_step(features)

            # Maximisation step
            # Update the weights
            self.maximisation_step(features, probabilities)

            # Check the tolerance
            new_log_likelihood = np.sum(np.log(np.sum(probabilities, axis=0)))
            if new_log_likelihood - log_likelihood < self.tolerance:
                break
            log_likelihood = new_log_likelihood

    def predict(self, features):
        # for each sample, calculate the probability of each component
        probs = None  # evaluate each multivariate normal distribution for each sample
        # return the component with the highest probability

        return np.argmax(probs)
