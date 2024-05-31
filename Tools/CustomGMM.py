"""!@file CustomGMM.py

@brief This file contains the CustomGMM class, which is a custom implementation
of the Gaussian Mixture Model clustering algorithm.

@details The CustomGMM class is a custom implementation of the Gaussian Mixture
Model clustering algorithm. The class is designed to be similar to the
GaussianMixture class from the scikit-learn library, following the same initialisation,
fitting, and prediction methods. The class is designed to be used as an alternative
for the GaussianMixture class, with the same parameters and methods. It was written
following the tutorial by Michael Galarnyk on Towards Data Science,
(https://towardsdatascience.com/how-to-code-gaussian-
mixture-models-from-scratch-in-python-9e7975df5252)
and the documentation for the GaussianMixture class from scikit-learn
(https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).

@author Created by T.Breitburd on 29/05/2023
"""
import numpy as np
from scipy.stats import multivariate_normal
from sklearn.cluster import KMeans


class CustomGMM:
    """!@brief Custom implementation of the Gaussian Mixture Model clustering algorithm.

    @details The CustomGMM class is a custom implementation of the Gaussian Mixture
    Model clustering algorithm. The class is designed to be similar to the
    GaussianMixture class from the sc
    ikit-learn library, following the same initialisation,
    fitting, and prediction methods. The class is designed
    to be used as an alternative for the GaussianMixture class,
    with the same parameters and methods. It was written
    following the tutorial by Michael Galarnyk on Towards Data Science,
    (https://towardsdatascience.com/how-to-code-gaussian-
    mixture-models-from-scratch-in-python-9e7975df5252)
    and the documentation for the GaussianMixture class from scikit-learn
    (https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html).

    @param n_components The number of components in the mixture model
    @param n_features The number of features in the data
    @param max_iter The maximum number of iterations for the EM algorithm
    @param tolerance The tolerance for the convergence of the EM algorithm
    @param random_state The random seed for the algorithm

    @return A CustomGMM object

    @attribute n_components The number of components in the mixture model,
    i.e. the number of clusters
    @attribute n_features The number of features in the data
    @attribute max_iter The maximum number of iterations for the EM algorithm
    @attribute tolerance The tolerance for the convergence of the EM algorithm
    @attribute random_state The random seed for the algorithm
    @attribute weights The weights of each gaussian component in the mixture model
    @attribute means The means of each gaussian component in the mixture model
    @attribute covariances The covariances of each gaussian component in the mixture model

    @method fit Fit the Gaussian Mixture Model to the data using the E-M algorithm
    @method predict Predict the cluster a certain sample in the same feature space belongs to.
    """

    # Initialise the CustomGMM object
    def __init__(
        self,
        n_components: int,
        n_features: int,
        max_iter: int = 100,
        tolerance: float = 1e-10,
        random_state: None | int = None,
    ):
        self.n_components = n_components
        self.n_features = n_features
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.random_state = random_state

        # Set the seeds
        np.random.seed(self.random_state)

        # Check the parameters
        if self.n_components < 1:
            raise ValueError(
                "Arg n_components must be at least 1, but has been given:",
                self.n_components,
            )
        if not isinstance(self.n_components, int):
            raise TypeError(
                "n_components must be an integer, but is type:", type(self.n_components)
            )
        if self.n_features < 1:
            raise ValueError(
                "n_features must be at least 1-dimensional but has been given:",
                self.n_features,
            )
        if not isinstance(self.n_features, int):
            raise TypeError(
                "n_features must be an integer, but is type:", type(self.n_features)
            )

        if self.max_iter < 1:
            raise ValueError(
                "max_iter must be at least 1, but has been given:", self.max_iter
            )
        if not isinstance(self.max_iter, int):
            raise TypeError(
                "max_iter must be an integer, but is type:", type(self.max_iter)
            )

        if self.tolerance <= 0:
            raise ValueError(
                "tolerance must be at greater than 0, but has been given:",
                self.tolerance,
            )
        if not isinstance(self.random_state, (int, type(None))):
            raise TypeError(
                "random_state must be an integer or None, but is type:",
                type(self.random_state),
            )

    # Initialise the parameters
    def _initialise_parameters(self, features):
        # Check the shape of the features
        if features.shape[1] != self.n_features:
            raise ValueError(
                "The number of features in the data does not match the number of features"
                + "given in the CustomGMM object. Expected",
                self.n_features,
                "but got",
                features.shape[1],
            )

        if features.shape[0] < self.n_components:
            raise ValueError(
                "The number of samples in the data is less than the number of components"
                + "given in the CustomGMM object. Expected at least",
                self.n_components,
                "but got",
                features.shape[0],
                "\n Try reducing the number of components.",
            )

        # Check for nan, inf and complex values
        if np.isnan(features).any():
            raise ValueError(
                "Invalid value in features: nan. \n Make sure there are no ",
                "nan values in the features.",
            )

        if np.isinf(features).any():
            raise ValueError(
                "Invalid value in features: inf. \n Check for inf values in the features."
            )

        if np.iscomplex(features).any():
            raise ValueError(
                "The GMM algorithm cannot handle complex values. \n",
                " Check for complex values in the features.",
            )

        # Initialise the weights as uniform
        self.weights = np.ones(self.n_components) / self.n_components

        # Initialise the means using k-means, as done in sklearn
        kmeans = KMeans(n_clusters=self.n_components, random_state=self.random_state)
        kmeans.fit(features)
        self.means = kmeans.cluster_centers_

        # Initialise the covariances, based on the k-means clusters
        labels = kmeans.predict(features)
        self.covariances = np.zeros(
            (self.n_components, self.n_features, self.n_features)
        )
        for i in range(self.n_components):
            cluster = features[labels == i]
            self.covariances[i] = np.cov(cluster, rowvar=False)

    def _expectation_step(self, features):
        # Initialise the likelihoods
        likelihoods = np.zeros((features.shape[0], self.n_components))

        # Calculate the likelihood of each sample for each component
        for i in range(self.n_components):
            likelihoods[:, i] = self.weights[i] * multivariate_normal.pdf(
                features, self.means[i], self.covariances[i]
            )

        # Calc likelihood of sample belonging to certain component:
        total_likelihoods = np.array([likelihoods.sum(axis=1)] * self.n_components).T
        probabilities = likelihoods / total_likelihoods

        return probabilities

    def _maximisation_step(self, features, probabilities):
        # Update the weights
        self.weights = probabilities.mean(axis=0)

        for i in range(self.n_components):
            # Update the means, dot product takes care of element-wise multiplication and sum
            self.means[i] = np.dot(probabilities[:, i], features) / probabilities[
                :, i
            ].sum(axis=0)

            # Update the covariances
            self.covariances[i] = (
                np.dot(
                    probabilities[:, i] * (features - self.means[i]).T,
                    (features - self.means[i]),
                )
                / probabilities[:, i].sum()
            )

    def fit(self, features):
        """!@brief Fit the Gaussian Mixture Model to the data using the E-M algorithm.

        @details This method fits the Gaussian Mixture Model to the data using the
        Expectation-Maximisation algorithm. The method initialises the parameters,
        calculates the log likelihood, and then loops over the iterations, calculating
        the probabilities of each component for each sample, and updating the weights,
        means, and covariances of the components. The method stops when the log likelihood
        converges or the maximum number of iterations is reached.

        @param features The data to fit the model to
        """
        # Initialise the parameters
        self._initialise_parameters(features)

        # Calc first log likelihood
        probabilities = self._expectation_step(features)
        # log_likelihood = np.sum(np.log(np.sum(probabilities, axis=0)))
        log_likelihood = 0

        # loop over iterations
        for i in range(self.max_iter):
            # Expectation step:
            probabilities = self._expectation_step(features)

            # Maximisation step:
            self._maximisation_step(features, probabilities)

            # Check the tolerance
            new_log_likelihood = np.sum(np.log(np.sum(probabilities, axis=0)))
            if new_log_likelihood - log_likelihood < self.tolerance:
                print(f"Converged after {i} iterations")
                break
            log_likelihood = new_log_likelihood

    def predict(self, features):
        """!@brief Predict the cluster a certain sample in the same feature space belongs to.

        @details This method predicts the cluster a certain sample in the same feature space
        belongs to. The method calculates the probability of each component for each sample,
        and returns the component with the highest probability.

        @param features The data to predict the cluster for

        @return The cluster the sample belongs to
        """

        # For each sample, calculate the probability of each component
        probabilities = self._expectation_step(features)

        return np.argmax(probabilities, axis=1)
