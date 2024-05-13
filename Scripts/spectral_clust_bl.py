# ---------------------------------------------
# Import Modules
# ---------------------------------------------

import numpy as np
import h5py
import sys
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import SparsePCA


# adding Tools to the system path
sys.path.insert(0, "../Tools/")
import plot_funcs as pf  # noqa: E402
import preprocessing as pp  # noqa: E402

# ---------------------------------------------
# Preprocessing
# ---------------------------------------------

# Load the data from http://turbulence.pha.jhu.edu/Transition_bl.aspx
file = h5py.File("../Data/Transition_BL_Time_Averaged_Profiles.h5", "r")

# Get arrays for variables and the Reynold's averages
x = np.array(file["x_coor"])
y = np.array(file["y_coor"])
u = np.array(file["um"])
v = np.array(file["vm"])
p = np.array(file["pm"])
Ruu = np.array(file["uum"]) - u**2
Ruv = np.array(file["uvm"]) - u * v
Rvv = np.array(file["uvm"]) - v**2


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

Dx, Dy = pp.get_derivatives(nx, ny, dx, dy)

# Get double derivatives

Dxx = 2 * (Dx @ Dx)
Dyy = 2 * (Dy @ Dy)

# Flatten arrays for matrix multiplication, using FORTRAN ordering

u = u.flatten("F")
v = v.flatten("F")
p = p.flatten("F")
Ruu = Ruu.flatten("F")
Ruv = Ruv.flatten("F")

# Get derivatives of variables

ux = Dx @ u
uy = Dy @ u
vx = Dx @ v
vy = Dy @ v
px = Dx @ p
py = Dy @ p
lap_u = (Dxx + Dyy) @ u
Ruux = Dx @ Ruu
Ruvy = Dy @ Ruv

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
        [u * ux, v * uy, px, nu * lap_u, Ruvy, Ruux, 1e-3 * X_flat, 1e-3 * Y_flat]
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
nc = 6  # Number of clusters
seed = np.random.randint(2**32)
seed = 75016  # Keep a seed for debugging/plotting
np.random.seed(seed)
model = SpectralClustering(
    n_clusters=nc, affinity="nearest_neighbors", n_neighbors=8, random_state=seed
)

# Train on only a subset of the data, limited by the high memory cost of spectral clustering
sample_pct = 0.02
mask = np.random.permutation(features.shape[0])[: int(sample_pct * features.shape[0])]
features_sc = features[mask, :]
model.fit_predict(features_sc[:, :6])

# Get covariances in each cluster
covs = np.zeros((nc, nfeatures, nfeatures))
for i in range(nc):
    mask_ = model.labels_ == i
    covs[i, :, :] = np.cov(features_sc[mask_, :6].T)

# Plot the covariances
pf.plot_cov_mat(covs, nfeatures, nc, "Other", False)

# ---------------------------------------------
# Cluster the data and visualise:
# equation space and physical space
# ---------------------------------------------

# Visualize spectral clustering with 2D views of equation space
cluster_idx = model.labels_ + 1

# Plot the clusters in equation space with 2D projections
pf.plot_clustering_2d_eq_space(features_sc[:, :6], cluster_idx, nc, False)

# Plot these points in physical space
pf.scatter_clustering_space(features_sc[:, 6], features_sc[:, 7], cluster_idx, False)

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
sparsity = np.zeros([len(alphas)])
cluster_idx = cluster_idx - 1

for k in range(len(alphas)):
    for i in range(nc):
        # Identify points in the field corresponding to each cluster
        feature_idx = np.nonzero(cluster_idx == i)[0]
        cluster_features = features_sc[feature_idx, :6]

        # Conduct Sparse PCA
        spca = SparsePCA(
            n_components=1, alpha=alphas[k], random_state=seed
        )  # normalize_components=True
        spca.fit(cluster_features)

        # Identify active and terms
        active_terms = np.nonzero(spca.components_[0])[0]
        inactive_terms = [feat for feat in range(nfeatures) if feat not in active_terms]

        # Calculate the error, as the sum of the norms of the inactive terms
        err[k] += np.linalg.norm(cluster_features[:, inactive_terms])

pf.plot_spca_residuals(alphas, err, False)

# Now with optimal alpha, get the active terms in each cluster
alpha_opt = 1  # Optimal alpha value

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
pf.plot_active_terms(spca_model, labels, False)


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

# Plot a grid with active terms in each cluster
gridmap = balance_models.copy()
gridmask = gridmap == 0
gridmap = (gridmap.T * np.arange(nmodels)).T + 1
gridmap[gridmask] = 0

# Delete unused terms
grid_mask = np.nonzero(np.all(gridmap == 0, axis=0))[0]
gridmap = np.delete(gridmap, grid_mask, axis=1)
grid_labels = np.delete(labels, grid_mask)

# Plot the balance models in a grid
pf.plot_balance_models(gridmap, grid_labels, False)

# Plot the clustering in space
pf.scatter_spca_reduced_clustering(
    features_sc[:, 6], features_sc[:, 7], balance_idx, False
)

# Visualize the clusters in equation space with 2D projections
pf.plot_feature_space(features_sc[:, :6], balance_idx, False)


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
u_map = np.reshape(u, (ny, nx), order="F")

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
    False,
)

# ----- Self-similarity -----
# In the near-wall region, the wall-normal profiles of velocity should be self-similar
# over all x locations. This is done from observing the resulting
# dominant balance scatter plot in the report.

# Compute friction velocity with an estimate of the wall shear stress
u_tau = np.sqrt(nu * uy[::ny])
y_plus = np.outer(1, u_tau / nu)
print(y_plus.shape)
x_plt = [600, 700, 800, 900, x[-3]]
for i in range(5):
    x_idx = np.nonzero(x > x_plt[i])[0][0]
    print(y_plus[:, x_idx])
