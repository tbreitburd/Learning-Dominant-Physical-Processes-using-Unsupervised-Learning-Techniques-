"""!@file stability_assessment.py

@brief Module to store the routines of the dominant balance identification method.

@details This module contains functions that together implement the dominant balance
identification method. The method and code is based on the work of Callaham et al. (2021) and
is simply put into functions for clarity when using the method multiple times when
conducting the stability assessment of the method.

@author T.Breitburd and Callaham et al."""

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import SparsePCA
from joblib import Parallel, delayed


# Set the random seed
np.random.seed(75016)


def get_clusters(n_clusters, features, train_frac):
    """
    !@brief Cluster the points in equation space using a Gaussian Mixture Model.

    @param n_clusters: The number of clusters, int
    @param features: The equation space data with each term being a feature, np.array
    @param train_frac: The fraction of the data to train on, float

    @return cluster_idx: The cluster index for each point, np.array
    """
    # Initialize the model
    model = GaussianMixture(n_components=n_clusters, random_state=75016)

    # Train on only a subset train_frac of the data
    sample_pct = train_frac
    mask = np.random.permutation(features.shape[0])[
        : int(sample_pct * features.shape[0])
    ]
    model.fit(features[mask, :])

    # Predict the cluster index for each point
    cluster_idx = model.predict(features)

    return cluster_idx


def get_spca_residuals(alphas, n_clusters, cluster_idx, eq_space_data):
    """
    !@brief Calculate the error for each alpha value using Sparse PCA.

    @param alphas: The alpha values to test, np.array
    @param n_clusters: The number of clusters, int
    @param cluster_idx: The cluster index for each point, np.array
    @param eq_space_data: The data with each term being a feature, np.array

    @return err: The error for each alpha value, np.array
    """

    # Initialize the error array
    err = np.zeros([len(alphas)])

    # Define the function to calculate the error for each alpha value
    def spca_err(alpha, cluster_idx, features, nc):
        # Initialize the error
        err_ = 0

        # Loop over each cluster
        for i in range(nc):
            # Identify points in the field corresponding to each cluster
            feature_idx = np.where(cluster_idx == i)[0]
            cluster_features = features[feature_idx, :]

            # Conduct Sparse PCA
            spca = SparsePCA(n_components=1, alpha=alpha)
            spca.fit(cluster_features)

            # Identify active and inactive terms
            inactive_terms = np.where(spca.components_[0] == 0)[0]

            # Calculate the error, as the sum of the norms of the inactive terms
            err_ += np.sqrt(np.sum((cluster_features[:, inactive_terms].ravel()) ** 2))

        return err_

    # Calculate the error for each alpha value
    err = Parallel(n_jobs=4)(
        delayed(spca_err)(alpha, cluster_idx, eq_space_data, n_clusters)
        for alpha in alphas
    )

    return err


def get_spca_active_terms(
    alpha_opt, n_clusters, cluster_idx, eq_space_data, n_features
):
    """
    !@brief Get the active terms for each cluster using Sparse PCA.

    @param alpha_opt: The optimal alpha value, float
    @param n_clusters: The number of clusters, int
    @param cluster_idx: The cluster index for each point, np.array
    @param eq_space_data: The data with each term being a feature, np.array
    @param n_features: The number of features, int

    @return spca_model: The active terms for each cluster, np.array
    """

    # Initialize the models array
    spca_model = np.zeros([n_clusters, n_features])

    for i in range(n_clusters):
        # Identify points in the field corresponding to each cluster
        feature_idx = np.where(cluster_idx == i)[0]
        cluster_features = eq_space_data[feature_idx, :]

        # Apply Sparse PCA
        spca = SparsePCA(n_components=1, alpha=alpha_opt)
        spca.fit(cluster_features)

        # Identify active terms
        active_terms = np.where(spca.components_[0] != 0)[0]
        if len(active_terms) > 0:
            spca_model[i, active_terms] = 1  # Set the active terms to 1

    return spca_model


def get_unique_balance_models(spca_model, labels):
    """
    !@brief Get the unique balance models and the mapping to the original model.

    @param spca_model: The active terms for each cluster, np.array
    @param labels: The labels for each term, np.array

    @return balance_models: The unique balance models, np.array
    @return model_index: The mapping to the original model, np.array
    @return nmodels: The number of unique models, int
    @return gridmap: The mapping of the unique models to the original model, np.array
    @return grid_labels: The labels for each term in the unique models, np.array
    """

    # Get the unique balance models
    balance_models, model_index = np.unique(spca_model, axis=0, return_inverse=True)
    nmodels = balance_models.shape[0]

    # Have each model be represented by an integer
    gridmap = balance_models.copy()
    gridmask = gridmap == 0
    gridmap = (gridmap.T * np.arange(nmodels)).T + 1
    gridmap[gridmask] = 0

    # Delete unused terms
    grid_mask = np.nonzero(np.all(gridmap == 0, axis=0))[0]
    gridmap = np.delete(gridmap, grid_mask, axis=1)
    grid_labels = np.delete(labels, grid_mask)

    return balance_models, model_index, nmodels, gridmap, grid_labels
