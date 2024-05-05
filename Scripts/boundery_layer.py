# ---------------------------------------------
# Import Modules and set plotting parameters
# ---------------------------------------------

import numpy as np
import h5py
import sys

sys.path.append("../Tools/")
from sklearn.mixture import GaussianMixture  # noqa: E402
from sklearn.decomposition import SparsePCA  # noqa: E402
from scipy.optimize import curve_fit, root  # noqa: E402
from scipy.integrate import odeint  # noqa: E402
import plot_funcs as pf  # noqa: E402
import blasius_solution as bs  # noqa: E402
import preprocessing as pp  # noqa: E402
import matplotlib as mpl  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
import seaborn as sns  # noqa: E402


mpl.rc("text", usetex=True)
mpl.rc("font", family="serif")
mpl.rc("figure", figsize=(15, 3))
mpl.rc("xtick", labelsize=14)
mpl.rc("ytick", labelsize=14)
mpl.rc("axes", labelsize=20)
mpl.rc("axes", titlesize=20)
mpl.rcParams["figure.dpi"] = 150

# Set colormap

sns_list = sns.color_palette("deep").as_hex()
sns_list.insert(0, "#ffffff")
sns_cmap = ListedColormap(sns_list)

cm = sns_cmap

mpl_colors = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

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

pf.plot_reynolds_stress(x, y, X, Y, u, Ruv, False)

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
pf.plot_equation_terms_bound_lay(
    x, y, nx, ny, u, ux, uy, v, Ruvy, Ruux, px, nu, lap_u, False
)


# ---------------------------------------------
# Obtain the equation space representation
# of the RANS equation
# ---------------------------------------------

# Gather the terms into an array of features
features = 1e3 * np.vstack([u * ux, v * uy, px, nu * lap_u, Ruvy, Ruux]).T
nfeatures = features.shape[1]

# ---------------------------------------------
# Cluster using Gaussian Mixture Models (GMM)
# ---------------------------------------------
print("----------------------------------")
print("Clustering with GMM")
print("----------------------------------")

# Fit Gaussian mixture model
nc = 6  # Number of clusters
seed = 76016  # Set a seed for debugging/plotting
model = GaussianMixture(n_components=nc, random_state=seed)

# Train on only a subset (10%) of the data due to large size
sample_pct = 0.1
mask = np.random.permutation(features.shape[0])[: int(sample_pct * features.shape[0])]
model.fit(features[mask, :])

# Plot the covariance matrices between terms for each of the GMM cluster
pf.plot_cov_mat(model, nfeatures, nc, "GMM", False)

# ---------------------------------------------
# Cluster the data and visualise:
# equation space and physical space
# ---------------------------------------------

# Predict the cluster index for each data point
cluster_idx = model.predict(features) + 1
clim = [0.5, cm.N - 0.5]
bounds = np.arange(0.5, nc + 1.5)

# Plot the clusters in equation space with 2D projections
pf.plot_clustering_2d_eq_space(features, cluster_idx, nc, False)

# Assign points in space to each cluster
cluster_idx_space = cluster_idx - 1
clustermap = np.reshape(cluster_idx_space, [ny, nx], order="F")

# Visualize the clustering in space
pf.plot_clustering_space(clustermap, x, y, X, Y, nx, ny, nc, u, U_inf, False)

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

for k in range(len(alphas)):
    for i in range(nc):
        # Identify points in the field corresponding to each cluster
        feature_idx = np.nonzero(cluster_idx_space == i)[0]
        cluster_features = features[feature_idx, :]

        # Conduct Sparse PCA
        spca = SparsePCA(n_components=1, alpha=alphas[k])  # normalize_components=True
        spca.fit(cluster_features)

        # Identify active and terms
        active_terms = np.nonzero(spca.components_[0])[0]
        inactive_terms = [feat for feat in range(nfeatures) if feat not in active_terms]

        # Calculate the error, as the sum of the norms of the inactive terms
        err[k] += np.linalg.norm(cluster_features[:, inactive_terms])

pf.plot_spca_residuals(alphas, err, False)


# Now with optimal alpha, get the active terms in each cluster
alpha_opt = 10  # Optimal alpha value

spca_model = np.zeros([nc, nfeatures])  # Store the active terms for each cluster

for i in range(nc):
    feature_idx = np.nonzero(cluster_idx_space == i)[0]
    cluster_features = features[feature_idx, :]

    spca = SparsePCA(n_components=1, alpha=alpha_opt)  # normalize_components=True
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
balance_idx = np.array([model_index[i] for i in cluster_idx_space])
balancemap = np.reshape(balance_idx, [ny, nx], order="F")

# Plot a grid with active terms in each cluster
gridmap = balance_models.copy()
gridmask = gridmap == 0
gridmap = (gridmap.T * np.arange(nmodels)).T + 1
gridmap[gridmask] = 0

# Remove terms that are never used
grid_mask = np.nonzero(np.all(gridmap == 0, axis=0))[0]
gridmap = np.delete(gridmap, grid_mask, axis=1)
grid_labels = np.delete(labels, grid_mask)

# Plot the balance models in a grid
pf.plot_balance_models(gridmap, grid_labels, False)

# Plot the clustering in space after SPCA
pf.plot_spca_reduced_clustering(x, y, balancemap, False)

# Visualize the clusters in equation space with 2D projections
pf.plot_feature_space(features[mask, :], balance_idx[mask], False)

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
u_map = np.reshape(u, (ny, nx), order="F")  # Reshape u to 2D

# Based on the obtained balance models:
x_min = 110  # Where inertial balance begins
x_turb = 500  # Where transitional region ends

# Find the x index where the inertial balance begins
x_idx = np.nonzero(x > x_min)[0]
x_layer = x[x_idx]

# First, find the upper extent of the inertial sublayer
y_gmm = np.zeros(len(x_idx))
# Loop through wall-normal direction until the balance changes
for i in range(len(x_idx)):
    j = len(y) - 1
    while balancemap[j, x_idx[i]] == 2:
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

# Finally, fit inertial balance to power law
power_law = lambda x, a, b: a * x**b  # noqa: E731

x_to_fit = x_layer > x_turb  # End of transitional region
p_gmm, cov = curve_fit(power_law, x_layer[x_to_fit], y_gmm[x_to_fit])  # Fit power law
gmm_fit = power_law(x_layer, *p_gmm)
print("Fitted parameters for the power law:")
print(p_gmm)  # Print the fit parameters

# Plot the inertial sublayer scaling
pf.plot_sublayer_scaling(
    x, y, balancemap, delta, x_layer, gmm_fit, p_gmm, x_to_fit, False
)

# ----- Self-similarity -----
# In the near-wall region, the wall-normal profiles of velocity should be self-similar
# over all x locations. Here we show that the profiles are indeed collapsible up until
# the vertical extent of the inertial sublayer.
print("----- Self-similarity test -----")

# Compute friction velocity with an estimate of the wall shear stress
u_tau = np.sqrt(nu * uy[::ny])

# Define wall units
y_plus = np.outer(y, u_tau / nu)
u_plus = np.reshape(u, [ny, nx], order="F") / u_tau

# Plot the self-similarity of the flow
print("y+ coordinates where the balance ends:")
pf.plot_self_similarity(x, y_plus, u_plus, balancemap, False)

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

pf.plot_blasius_solution(eta, f, False)

# Then, compare inflow profile to this Blasius Solution.
pf.plot_blasius_deviation(x, y, nx, ny, u, eta, f, U_inf, nu, False)
