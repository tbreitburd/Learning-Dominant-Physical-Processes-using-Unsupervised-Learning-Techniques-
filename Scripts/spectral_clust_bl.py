# ---------------------------------------------
# Import Modules
# ---------------------------------------------

import numpy as np
import h5py
import sys
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import SparsePCA
from joblib import Parallel, delayed


# adding Tools to the system path
sys.path.insert(0, "../Tools/")
import plot_funcs as pf  # noqa: E402
import preprocessing as pp  # noqa: E402


# Take the fraction of the data to be used
# as the first input argument

sample_pct = float(sys.argv[1])


# ---------------------------------------------
# Preprocessing
# ---------------------------------------------

# Load the data from http://turbulence.pha.jhu.edu/Transition_bl.aspx
file = h5py.File("../Data/Transition_BL_Time_Averaged_Profiles.h5", "r")

# Get arrays for variables and the Reynold's averages
x = np.array(file["x_coor"])
y = np.array(file["y_coor"])
u_bar = np.array(file["um"])
v_bar = np.array(file["vm"])
p_bar = np.array(file["pm"])
R_uu = np.array(file["uum"]) - u_bar**2
R_uv = np.array(file["uvm"]) - u_bar * v_bar
R_vv = np.array(file["uvm"]) - v_bar**2


# Visualize the wall-normal Reynolds stress
X, Y = np.meshgrid(x, y)

# Include line of 99% of free flow mean velocity
# Values from http://turbulence.pha.jhu.edu/docs/README-transition_bl.pdf
U_inf = 1
nu = 1 / 800
Re = (U_inf / nu) * x

# ------- Get the derivatives --------
print("----------------------------------")
print("Getting derivatives")
print("----------------------------------")

# Get space steps
dx = x[1] - x[0]
dy = y[1:] - y[:-1]

nx = len(x)  # Number of points in x
ny = len(y)  # Number of points in y

# Get y space steps:
# The y step is not constant, so we need to calculate it for each point
dy = np.diff(y[::-1])
dy = np.append(dy, dy[-1])

# Get the derivatives
u_x, u_y, lap_u, v_y, p_x, R_uux, R_uvy = pp.get_derivatives_numpy(
    nx, ny, dx, y, u_bar, y, p_bar, R_uu, R_uv
)

# Flatten arrays for matrix multiplication, using fortran ordering
u_bar = u_bar.flatten("F")
v_bar = v_bar.flatten("F")
p_bar = p_bar.flatten("F")
R_uu = R_uu.flatten("F")
R_uv = R_uv.flatten("F")

# Flatten the derivative terms arrays for the rest of the notebook
lap_u = lap_u.flatten("F")
R_uux = R_uux.flatten("F")
R_uvy = R_uvy.flatten("F")
u_x = u_x.flatten("F")
u_y = u_y.flatten("F")
v_y = v_y.flatten("F")
p_x = p_x.flatten("F")

# Labels of terms in the RANS equation
labels = [
    r"$\bar{u} \bar{u}_x$",
    r"$\bar{v}\bar{u}_y$",
    r"$\rho^{-1} \bar{p}_x$",
    r"$\nu \nabla^2 \bar{u}$",
    r"$\overline{(u^\prime v^\prime)}_y$",
    r"$\overline{({u^\prime} ^2)}_x$",
]

# ---------------------------------------------
# Obtain the equation space representation
# of the RANS equation
# ---------------------------------------------

# Get terms stored as features, keeping track of the space position
X_flat = X.flatten("F")
Y_flat = Y.flatten("F")
features = (
    1e3
    * np.vstack(
        [
            u_bar * u_x,
            v_bar * u_y,
            p_x,
            nu * lap_u,
            R_uvy,
            R_uux,
            1e-3 * X_flat,
            1e-3 * Y_flat,
        ]
    ).T
)
nfeatures = features.shape[1] - 2

# ---------------------------------------------
# Cluster using Spectral Clustering
# ---------------------------------------------
print("----------------------------------")
print("Clustering with Spectral Clustering")
print("----------------------------------")

# Initialise the Spectral Clustering model
nc = int(sys.argv[2])  # Number of clusters
seed = 75016  # Keep a seed for debugging/plotting
np.random.seed(seed)
model = SpectralClustering(
    n_clusters=nc, affinity="nearest_neighbors", n_neighbors=8, random_state=seed
)

# Train on only a subset of the data, limited by the high memory cost of spectral clustering
mask = np.random.permutation(features.shape[0])[: int(sample_pct * features.shape[0])]
features_sc = features[mask, :]
model.fit_predict(features_sc[:, :6])

# Get covariances in each cluster
covs = np.zeros((nc, nfeatures, nfeatures))
for i in range(nc):
    mask_ = model.labels_ == i
    covs[i, :, :] = np.cov(features_sc[mask_, :6].T)

# Plot the covariances
pf.plot_cov_mat(
    covs, nfeatures, nc, labels, "Other", f"BL/SC_CovMat_{sample_pct}_{nc}.png", False
)

# ---------------------------------------------
# Cluster the data and visualise:
# equation space and physical space
# ---------------------------------------------

# Visualize spectral clustering with 2D views of equation space
cluster_idx = model.labels_

# Plot the clusters in equation space with 2D projections
pf.plot_clustering_2d_eq_space(
    features_sc[:, :6],
    cluster_idx,
    nc,
    f"BL/SC_2D_eq_space_{sample_pct}_{nc}.png",
    False,
)

# Plot these points in physical space
pf.scatter_clustering_space(
    features_sc[:, 6],
    features_sc[:, 7],
    cluster_idx,
    nc,
    f"BL/SC_clustering_space_{sample_pct}_{nc}.png",
    False,
)

# ---------------------------------------------
# Sparse Principal Component Analysis (SPCA)
# dimensionality reduction
# ---------------------------------------------

print("----------------------------------")
print("Applying Sparse PCA")
print("----------------------------------")
# Sparse PCA to identify directions of nonzero variance in each cluster

alphas = [1e-4, 1e-3, 1e-2, 0.1, 1, 10, 100, 1e3, 1e4, 1e5]
err = np.zeros([len(alphas)])


def spca_err(alpha, cluster_idx, features, nc):
    err_ = 0

    for i in range(nc):
        # Identify points in the field corresponding to each cluster
        feature_idx = np.where(cluster_idx == i)[0]
        cluster_features = features[feature_idx, :6]

        # Conduct Sparse PCA
        spca = SparsePCA(n_components=1, alpha=alpha)
        spca.fit(cluster_features)

        # Identify active and inactive terms
        inactive_terms = np.where(spca.components_[0] == 0)[0]

        err_ += np.sqrt(np.sum((cluster_features[:, inactive_terms].ravel()) ** 2))

    return err_


err = Parallel(n_jobs=4)(
    delayed(spca_err)(alpha, cluster_idx, features, nc) for alpha in alphas
)

pf.plot_spca_residuals(
    alphas, err, f"BL/SC_spca_residuals_{sample_pct}_{nc}.png", False
)

# Now with optimal alpha, get the active terms in each cluster
alpha_opt = int(sys.argv[3])  # Optimal alpha value

spca_model = np.zeros([nc, nfeatures])  # Store the active terms for each cluster

for i in range(nc):
    feature_idx = np.nonzero(cluster_idx == i)[0]
    cluster_features = features_sc[feature_idx, :6]

    spca = SparsePCA(
        n_components=1, alpha=alpha_opt, random_state=seed
    )  # normalize_components=True
    spca.fit(cluster_features)

    active_terms = np.nonzero(spca.components_[0])[0]
    if len(active_terms) > 0:
        spca_model[i, active_terms] = 1  # Set the active terms to 1

# Plot the active terms in each cluster
pf.plot_balance_models(
    spca_model,
    labels,
    False,
    f"BL/SC_active_terms_{sample_pct}_{nc}_{alpha_opt}.png",
    False,
)


# ---------------------------------------------
# Resulting Final Balance Models
# ---------------------------------------------

print("----------------------------------")
print("Obtaining Final balance models")
print("----------------------------------")

# Identify clusters with identical balance models
balance_models, model_index = np.unique(spca_model, axis=0, return_inverse=True)
nmodels = balance_models.shape[0]

# Make new cluster_idx based on the unique SPCA balance model
balance_idx = np.array([model_index[i] for i in cluster_idx])

# Plot the balance models in a grid
pf.plot_balance_models(
    balance_models,
    labels,
    True,
    f"BL/SC_balance_models_{sample_pct}_{nc}_{alpha_opt}.png",
    False,
)

# Plot the clustering in space
pf.scatter_clustering_space(
    features_sc[:, 6],
    features_sc[:, 7],
    balance_idx,
    nmodels,
    "BL/SC_spca_clustering_space.png",
    False,
)

# Visualize the clusters in equation space with 2D projections
pf.plot_feature_space(
    features_sc[:, :6],
    balance_idx,
    f"BL/SC_feature_space_{sample_pct}_{nc}_{alpha_opt}.png",
    False,
)


# ---------------------------------------------
# Validate the balance models with some diagnostics
# ---------------------------------------------
print("----------------------------------")
print("Validating the balance models")
print("----------------------------------")

# ----- Outer Layer Scaling -----
# The length scale of the outer layer should scale with: l ~ x^(4/5)
print("----- Outer layer scaling -----")

# Define some variables
u_map = np.reshape(u_bar, (ny, nx), order="F")

x_min = 110  # Where inertial balance begins
x_turb = 500  # Where transitional region ends (based on resutls with GMM)

x_idx = np.nonzero(x > x_min)[0]
x_layer = x[x_idx]


# Next, find the 99% of free stream velocity line
delta = np.zeros(len(x))
# Loop until velocity falls past 99% freestream
for i in range(len(x)):
    j = 0
    while u_map[j, i] < 0.99:
        j += 1
    delta[i] = y[j - 1]

# Fit inertial balance to power law
power_law = lambda x, a, b: a * x**b  # noqa: E731
x_to_fit = x_layer > x_turb  # End of transitional region
gmm_fit = power_law(x_layer, 0.06832071, 0.80)  # Fit from the GMM results

# Plot the inertial sublayer scaling
pf.scatter_sublayer_scaling(
    x,
    features_sc[:, 6],
    features_sc[:, 7],
    balance_idx,
    delta,
    x_layer,
    gmm_fit,
    x_to_fit,
    f"BL/SC_sublayer_scaling_{sample_pct}_{nc}_{alpha_opt}.png",
    False,
)

# ----- Self-similarity -----
# In the near-wall region, the wall-normal profiles of velocity should be self-similar
# over all x locations. This is done from observing the resulting
# dominant balance scatter plot in the report.

# Compute friction velocity with an estimate of the wall shear stress
u_tau = np.sqrt(nu * u_y[::ny])
y_plus = np.outer(1, u_tau / nu)
print(y_plus.shape)
x_plt = [600, 700, 800, 900, x[-3]]
for i in range(5):
    x_idx = np.nonzero(x > x_plt[i])[0][0]
    print(y_plus[:, x_idx])
