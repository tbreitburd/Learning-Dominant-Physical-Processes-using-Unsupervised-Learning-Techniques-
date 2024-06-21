# ---------------------------------------------
# Import Modules and set plotting parameters
# ---------------------------------------------

import numpy as np
import pandas as pd
import h5py
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import SparsePCA
import sklearn as sk
from scipy.optimize import curve_fit, root
from scipy.integrate import odeint
import sys
from joblib import Parallel, delayed

sys.path.append("../Tools/")
import plot_funcs as pf  # noqa: E402
import blasius_solution as bs  # noqa: E402
import preprocessing as pp  # noqa: E402
from CustomGMM import CustomGMM  # noqa: E402


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

pf.plot_reynolds_stress(x, y, X, Y, u_bar, R_uv, "BL/Rey_stress_uv.png", False)

# ------- Get the derivatives --------
print("----------------------------------")
print("Getting derivatives")
print("----------------------------------")

method = sys.argv[1]
print(f"Method: {method}")

# Get space steps
dx = x[1] - x[0]
dy = y[1:] - y[:-1]

nx = len(x)  # Number of points in x
ny = len(y)  # Number of points in y

if method == "original":
    # Get the gradients using scipy's sparse matrix
    Dx, Dy = pp.get_derivatives(nx, ny, dx, dy)

    # Get double derivatives
    Dxx = 2 * (Dx @ Dx)
    Dyy = 2 * (Dy @ Dy)

    # Flatten arrays for matrix multiplication, using FORTRAN ordering
    u_bar = u_bar.flatten("F")
    v_bar = v_bar.flatten("F")
    p_bar = p_bar.flatten("F")
    R_uu = R_uu.flatten("F")
    R_uv = R_uv.flatten("F")

    # Get derivatives of variables
    u_x = Dx @ u_bar
    u_y = Dy @ u_bar
    v_x = Dx @ v_bar
    v_y = Dy @ v_bar
    p_x = Dx @ p_bar
    p_y = Dy @ p_bar
    lap_u = (Dxx + Dyy) @ u_bar
    R_uux = Dx @ R_uu
    R_uvy = Dy @ R_uv

else:
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

# ---------------------------------------------
# Visualising the RANS equation
# ---------------------------------------------

# Labels of terms in the RANS equation
labels = [
    r"$\bar{u} \bar{u}_x$",
    r"$\bar{v}\bar{u}_y$",
    r"$\rho^{-1} \bar{p}_x$",
    r"$\nu \nabla^2 \bar{u}$",
    r"$\overline{(u^\prime v^\prime)}_y$",
    r"$\overline{({u^\prime} ^2)}_x$",
]

# Plot all six terms in the RANS equation
if method == "original":
    path = "BL/RANS_terms.png"
else:
    path = "BL/custom_RANS_terms.png"

pf.plot_equation_terms_bound_lay(
    x,
    y,
    nx,
    ny,
    u_bar,
    u_x,
    u_y,
    v_bar,
    R_uvy,
    R_uux,
    p_x,
    nu,
    lap_u,
    path,
    False,
)


# ---------------------------------------------
# Cluster using Gaussian Mixture Models (GMM)
# ---------------------------------------------
print("----------------------------------")
print("Clustering with GMM")
print("----------------------------------")

# Number of clusters
nc = int(sys.argv[2])
print(f"Number of clusters: {nc}")

if method == "original":
    # Gather the terms into an array of features
    features = (
        1e3 * np.vstack([u_bar * u_x, v_bar * u_y, p_x, nu * lap_u, R_uvy, R_uux]).T
    )
    nfeatures = features.shape[1]

    # Fit Gaussian mixture model
    seed = 76016  # Set a seed for debugging/plotting
    model = GaussianMixture(n_components=nc, random_state=seed)

    # Train on only a subset (10%) of the data due to large size
    sample_pct = 0.1
    mask = np.random.permutation(features.shape[0])[
        : int(sample_pct * features.shape[0])
    ]
    model.fit(features[mask, :])

    algo = "GMM"
    path = "BL/GMM_cov_mat.png"

else:
    # Gather the terms into an array of features
    features = 1e3 * pd.DataFrame(
        {
            "uu_x": u_bar * u_x,
            "vu_y": v_bar * u_y,
            "p_x": p_x,
            "nu_lap_u": nu * lap_u,
            "R_uvy": R_uvy,
            "R_uux": R_uux,
        }
    )
    features = features.to_numpy()
    nfeatures = 6

    # Fit Gaussian mixture model
    seed = 75016  # Set a seed for debugging/plotting
    np.random.seed(seed)
    model = CustomGMM(n_components=nc, n_features=nfeatures, random_state=seed)

    # Train on only a subset (10%) of the data
    sample_pct = 0.1
    mask, _ = sk.model_selection.train_test_split(
        range(features.shape[0]), train_size=sample_pct, random_state=seed
    )
    model.fit(features[mask, :])

    algo = "CustomGMM"
    path = "BL/custom_GMM_cov_mat.png"

# Plot the covariance matrices between terms for each of the GMM cluster
pf.plot_cov_mat(model, nfeatures, nc, labels, algo, path, False)

# ---------------------------------------------
# Cluster the data and visualise:
# equation space and physical space
# ---------------------------------------------

# Predict the cluster index for each data point
cluster_idx = model.predict(features) + 1

# Plot the clusters in equation space with 2D projections
if method == "original":
    path = "BL/GMM_2D_eq_space.png"
else:
    path = "BL/custom_GMM_2D_eq_space.png"

pf.plot_clustering_2d_eq_space(features[mask, :], cluster_idx[mask], nc, path, False)

# Assign points in space to each cluster
cluster_idx = cluster_idx - 1
clustermap = np.reshape(cluster_idx, [ny, nx], order="F")

# Visualize the clustering in space
if method == "original":
    path = "BL/GMM_clustering_space.png"
else:
    path = "BL/custom_GMM_clustering_space.png"

pf.plot_clustering_space(clustermap, x, y, X, Y, nx, ny, nc, u_bar, U_inf, path, False)

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

if method == "original":
    for k in range(len(alphas)):
        for i in range(nc):
            # Identify points in the field corresponding to each cluster
            feature_idx = np.nonzero(cluster_idx == i)[0]
            cluster_features = features[feature_idx, :]

            # Conduct Sparse PCA
            spca = SparsePCA(
                n_components=1, alpha=alphas[k]
            )  # normalize_components=True
            spca.fit(cluster_features)

            # Identify active and inactive terms
            active_terms = np.nonzero(spca.components_[0])[0]
            inactive_terms = [
                feat for feat in range(nfeatures) if feat not in active_terms
            ]

            # Calculate the error, as the sum of the norms of the inactive terms
            err[k] += np.linalg.norm(cluster_features[:, inactive_terms])


else:

    def spca_err(alpha, cluster_idx, features, nc):
        err_ = 0

        for i in range(nc):
            # Identify points in the field corresponding to each cluster
            feature_idx = np.where(cluster_idx == i)[0]
            cluster_features = features[feature_idx, :]

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

if method == "original":
    path = "BL/GMM_spca_residuals.png"
else:
    path = "BL/custom_GMM_spca_residuals.png"

pf.plot_spca_residuals(alphas, err, path, False)


# Now with optimal alpha, get the active terms in each cluster
alpha_opt = float(sys.argv[3])  # Optimal alpha value
print(f"Optimal alpha: {alpha_opt}")

spca_model = np.zeros([nc, nfeatures])  # Store the active terms for each cluster

if method == "original":
    for i in range(nc):
        feature_idx = np.nonzero(cluster_idx == i)[0]
        cluster_features = features[feature_idx, :]

        spca = SparsePCA(n_components=1, alpha=alpha_opt)  # normalize_components=True
        spca.fit(cluster_features)

        active_terms = np.nonzero(spca.components_[0])[0]
        if len(active_terms) > 0:
            spca_model[i, active_terms] = 1  # Set the active terms to 1
else:
    for i in range(nc):
        feature_idx = np.where(cluster_idx == i)[0]
        cluster_features = features[feature_idx, :]

        spca = SparsePCA(n_components=1, alpha=alpha_opt)  # normalize_components=True
        spca.fit(cluster_features)

        active_terms = np.where(spca.components_[0] != 0)[0]
        if len(active_terms) > 0:
            spca_model[i, active_terms] = 1  # Set the active terms to 1

if method == "original":
    path = "BL/GMM_active_terms.png"
else:
    path = "BL/custom_GMM_active_terms.png"

pf.plot_balance_models(spca_model, labels, False, path, False)


# ---------------------------------------------
# Resulting Final Balance Models
# ---------------------------------------------

print("----------------------------------")
print("Obtaining Final balance models")
print("----------------------------------")


if method == "original":
    # Identify clusters with identical balance models
    balance_models, model_index = np.unique(spca_model, axis=0, return_inverse=True)
    nmodels = balance_models.shape[0]

    # Make new cluster_idx based on the unique SPCA balance model
    balance_idx = np.array([model_index[i] for i in cluster_idx])
    balancemap = np.reshape(balance_idx, [ny, nx], order="F")


else:
    # Convert the spca_model array to a dataframe
    spca_temp = pd.DataFrame(spca_model.copy())

    # Group the balance models by the values of all columns
    grouped_models = spca_temp.groupby(np.arange(nfeatures).tolist())
    grouped_models = grouped_models.groups.items()

    # Combine balance models that have identical active terms
    # For each balance model, the spca models that have the same active terms
    # are given the same index
    balance_models = pd.DataFrame(np.zeros((len(grouped_models), nfeatures)))
    model_idx = np.zeros(len(spca_model), dtype=int)
    for i, model in enumerate(grouped_models):
        idx = model[1].to_list()
        model_idx[idx] = i
        balance_models.loc[i] = spca_temp.loc[idx[0]].to_numpy()

    # Assign new cluster indices
    balance_idx = np.array([model_idx[i] for i in cluster_idx])
    balancemap = np.reshape(balance_idx, [ny, nx], order="F")

    # Convert the balance models to a numpy array
    balance_models = balance_models.drop_duplicates(keep="first")
    balance_models = balance_models.to_numpy()
    nmodels = balance_models.shape[0]

if method == "original":
    path = "BL/GMM_balance_models.png"
else:
    path = "BL/custom_GMM_balance_models.png"

# Plot the balance models in a grid
pf.plot_balance_models(balance_models, labels, True, path, False)

# Plot the clustering in space after SPCA
if method == "original":
    path = "BL/GMM_spca_clustering_space.png"
else:
    path = "BL/custom_GMM_spca_clustering_space.png"

pf.plot_clustering_space(balancemap, x, y, X, Y, nx, ny, nc, u_bar, U_inf, path, False)

# Visualize the clusters in equation space with 2D projections
if method == "original":
    path = "BL/GMM_feature_space.png"
else:
    path = "BL/custom_GMM_feature_space.png"

pf.plot_feature_space(features[mask, :], balance_idx[mask], path, False)

# ---------------------------------------------
# Validate the balance models with some diagnostics
# ---------------------------------------------
print("----------------------------------")
print("Validating the balance models")
print("----------------------------------")

# ----- Outer Layer Scaling -----
# The length scale of the outer layer should scale with: l ~ x^(4/5)
print("----- Outer layer scaling -----")

# Create a u_bar field:
u_map = np.reshape(u_bar, (ny, nx), order="F")

# Find which cluster is the inertial sublayer.
if method == "original":
    inert_sub_idx = (
        np.where(np.all(balance_models == [1, 0, 0, 0, 1, 0], axis=1))[0] + 1
    )
else:
    inert_sub_idx = np.where(np.all(balance_models == [1, 0, 0, 0, 1, 0], axis=1))[0]
print(inert_sub_idx)

# Define some variables
x_min = 110  # Where inertial balance begins
x_turb = 500  # Where transitional region ends

x_idx = np.nonzero(x > x_min)[0]
x_layer = x[x_idx]

# First, find the upper extent of the inertial sublayer
y_gmm = np.zeros(len(x_idx))
# Loop through wall-normal direction until the balance changes
for i in range(len(x_idx)):
    j = len(y) - 1
    while balancemap[j, x_idx[i]] == inert_sub_idx:
        j -= 1
    y_gmm[i] = y[j]  # Store upper value of inertial balance

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
p_gmm, cov = curve_fit(power_law, x_layer[x_to_fit], y_gmm[x_to_fit])
gmm_fit = power_law(x_layer, *p_gmm)
print("Fitted parameters for the power law:")
print(p_gmm)  # Print the fit parameters

# Plot the inertial sublayer scaling
if method == "original":
    path = "BL/GMM_sublayer_scaling.png"
else:
    path = "BL/custom_GMM_sublayer_scaling.png"

pf.plot_sublayer_scaling(
    x, y, balancemap, delta, x_layer, gmm_fit, p_gmm, x_to_fit, path, False
)

# ----- Self-similarity -----
# In the near-wall region, the wall-normal profiles of velocity should be self-similar
# over all x locations. Here we show that the profiles are indeed collapsible up until
# the vertical extent of the inertial sublayer.
print("----- Self-similarity test -----")

# Compute friction velocity with an estimate of the wall shear stress
u_tau = np.sqrt(nu * u_y[::ny])

# Define wall units
y_plus = np.outer(y, u_tau / nu)
u_plus = np.reshape(u_bar, [ny, nx], order="F") / u_tau

# Plot the self-similarity of the flow
print("y+ coordinates where the balance ends:")

if method == "original":
    path = "BL/GMM_self_similarity.png"
    pf.plot_self_similarity(x, 0, y_plus, u_plus, balancemap, path, show=False)
else:
    path = "BL/custom_GMM_self_similarity.png"
    pf.plot_self_similarity(x, 2, y_plus, u_plus, balancemap, path, show=False)


# ----- Blasius Solution in laminar regime -----
# There is an inflow region with negligible Reynolds stresses (left boundary),
# which suggests the flow is laminar there. Thus, we expect the Blasius Boundary
# Layer solution to be a good approximation in this region.
print("----- Blasius Solution -----")

# Solve Blasius equations numerically

# Arbitrary "infinite" upper limit for domain
eta_inf = 200
# Step size
d_eta = 0.01
eta = np.arange(0, eta_inf, d_eta)

# Initial guess for unknown initial condition
F_init = [0, 0, 0]

# Solve root-finding problem for unknown initial condition
opt_res = root(bs.bc_fn, F_init, tol=1e-4)
F0 = [0, 0, opt_res.x[2]]

# Evaluate with resulting initial conditions
f = odeint(lambda y, t: bs.blasius_rhs(y), F0, eta)

pf.plot_blasius_solution(eta, f, "BL/blasius_solution.png", False)

# Then, compare inflow profile to this Blasius Solution.
pf.plot_blasius_deviation(
    x, y, nx, ny, u_bar, eta, f, U_inf, nu, "BL/blasius_deviation.png", False
)
