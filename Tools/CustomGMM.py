import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans

# https://towardsdatascience.com/how-to-code-gaussian-mixture-models-from-scratch-in-python-9e7975df5252
# https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html


class CustomGMM:
    def __init__(
        self,
        n_components: int,
        n_features: int,
        max_iter: int = 1000,
        tolerance: float = 1e-6,
        random_state: None | int = None,
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
        self.weights = np.ones(self.n_components) / self.n_components

        # Initialise the means using k-means, as done in sklearn
        kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state)
        kmeans.fit(features)
        self.means = kmeans.cluster_centers_

        print(self.means)
        print(self.means.shape)

        # Initialise the covariances, based on the k-means clusters
        labels = kmeans.predict(features)

        print("labels predicted")

        self.covariances = np.zeros(
            (self.n_components, self.n_features, self.n_features)
        )

        for i in range(self.n_components):
            cluster = features[labels == i]
            self.covariances[i] = np.cov(cluster, rowvar=False)

        print(self.covariances)
        print(self.covariances.shape)

    def expectation_step(self, features):
        # Initialise the likelihoods
        likelihoods = np.zeros((features.shape[0], self.n_components))

        # Calculate the likelihood of each sample for each component
        for i in range(self.n_components):
            likelihoods[:, i] = self.weights[i] * multivariate_normal.pdf(
                features, self.means[i], self.covariances[i]
            )

        # Calc likelihood of sample belonging to certain component
        # by calc likelihood for component k/sum of likelihoods for all components
        total_likelihoods = np.array([likelihoods.sum(axis=1)] * self.n_components).T
        probabilities = likelihoods / total_likelihoods

        return probabilities

    def maximisation_step(self, features, probabilities):
        # Update the weights
        self.weights = probabilities.mean(axis=0)

        # Update the means, dot product takes care of element-wise multiplication and sum
        self.means = np.dot(probabilities.T, features) / probabilities.sum(axis=0)

        # Update the covariances
        for i in range(self.n_components):
            self.covariances[i] = (
                np.dot(
                    probabilities[:, i] * (features - self.means[i]).T,
                    (features - self.means[i]),
                )
                / probabilities[:, i].sum()
            )
        print(self.weights)
        print(self.means)
        print(self.means.shape)
        print(self.covariances)
        print(self.covariances.shape)

    def fit(self, features):
        # Initialise the parameters
        self._initialise_parameters(features)

        # Calc first log likelihood
        probabilities = self.expectation_step(features)
        # log_likelihood = np.sum(np.log(np.sum(probabilities, axis=0)))
        log_likelihood = 0

        # loop over iterations
        for i in range(self.max_iter):
            # Expectation step
            # Calculate the probability of each component for each sample
            probabilities = self.expectation_step(features)
            print(probabilities.shape)

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
        probabilites = self.expectation_step(features)

        return np.argmax(probabilites, axis=1)
