import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import SparsePCA

np.random.seed(75016)


def get_clusters(n_clusters, features, train_frac):
    """
    Cluster the points in the equation space using a Gaussian Mixture Model.

    Params
    ----------
    n_clusters : The number of clusters to use, int
    features : The features to cluster, np.array
    train_frac : The fraction of the data to train on, float

    Returns
    -------
    cluster_idx : The cluster index for each point, np.array
    """
    model = GaussianMixture(n_components=n_clusters, random_state=75016)

    # Train on only a subset (10%) of the data
    sample_pct = train_frac
    mask = np.random.permutation(features.shape[0])[
        : int(sample_pct * features.shape[0])
    ]
    model.fit(features[mask, :])

    cluster_idx = model.predict(features)

    return cluster_idx


def get_spca_residuals(alphas, n_clusters, cluster_idx, eq_space_data, n_features):
    """
    Calculate the SPCA residuals for each cluster, for each alpha value.

    Params
    ----------
    alphas : The alpha values to test, list
    n_clusters : The number of clusters, int
    cluster_idx : The cluster index for each point, np.array
    eq_space_data : The data with each term being a feature, np.array
    n_features : The number of features, int

    Returns
    -------
    err : The error for each alpha, np.array
    """

    err = np.zeros([len(alphas)])

    for k in range(len(alphas)):
        for i in range(n_clusters):
            # Identify points in the field corresponding to each cluster
            feature_idx = np.nonzero(cluster_idx == i)[0]
            cluster_features = eq_space_data[feature_idx, :]

            # Conduct Sparse PCA
            spca = SparsePCA(n_components=1, alpha=alphas[k], random_state=75016)
            spca.fit(cluster_features)

            # Identify active and terms
            active_terms = np.nonzero(spca.components_[0])[0]
            inactive_terms = [
                feat for feat in range(n_features) if feat not in active_terms
            ]

            # Calculate the error, as the sum of the norms of the inactive terms
            err[k] += np.linalg.norm(cluster_features[:, inactive_terms])

    return err


def get_spca_active_terms(
    alpha_opt, n_clusters, cluster_idx, eq_space_data, n_features
):
    """
    Get the active terms for each cluster using Sparse PCA, now using the optimal alpha.

    Params
    ----------
    alpha_opt : The optimal alpha value, float
    n_clusters : The number of clusters, int
    cluster_idx : The cluster index for each point, np.array
    eq_space_data : The data with each term being a feature, np.array
    n_features : The number of features, int

    Returns
    -------
    spca_model : The active terms for each cluster, np.array
    """

    spca_model = np.zeros(
        [n_clusters, n_features]
    )  # Store the active terms for each cluster

    for i in range(n_clusters):
        feature_idx = np.nonzero(cluster_idx == i)[0]
        cluster_features = eq_space_data[feature_idx, :]

        spca = SparsePCA(
            n_components=1, alpha=alpha_opt, random_state=75016
        )  # normalize_components=True
        spca.fit(cluster_features)

        active_terms = np.nonzero(spca.components_[0])[0]
        if len(active_terms) > 0:
            spca_model[i, active_terms] = 1  # Set the active terms to 1

    return spca_model


def get_unique_balance_models(spca_model):
    """
    Get the unique balance models and the gridmap for the balance models.

    Params
    ----------
    spca_model : The active terms for each cluster, np.array

    Returns
    -------
    balance_models : The unique balance models, np.array
    model_index : The index of the balance model for each cluster, np.array
    nmodels : The number of unique balance models, int
    gridmap : The gridmap for the balance models, np.array
    grid_labels : The labels for the balance models, np.array
    """

    labels = [
        r"$\bar{u} \bar{u}_x$",
        r"$\bar{v}\bar{u}_y$",
        r"$\rho^{-1} \bar{p}_x$",
        r"$\nu \nabla^2 \bar{u}$",
        r"$\overline{(u^\prime v^\prime)}_y$",
        r"$\overline{({u^\prime} ^2)}_x$",
    ]

    balance_models, model_index = np.unique(spca_model, axis=0, return_inverse=True)
    nmodels = balance_models.shape[0]

    gridmap = balance_models.copy()
    gridmask = gridmap == 0
    gridmap = (gridmap.T * np.arange(nmodels)).T + 1
    gridmap[gridmask] = 0

    # Delete unused terms
    grid_mask = np.nonzero(np.all(gridmap == 0, axis=0))[0]
    gridmap = np.delete(gridmap, grid_mask, axis=1)
    grid_labels = np.delete(labels, grid_mask)

    return balance_models, model_index, nmodels, gridmap, grid_labels
