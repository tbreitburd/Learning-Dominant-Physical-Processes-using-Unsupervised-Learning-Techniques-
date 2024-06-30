"""!@file preprocessing.py

@brief Module to compute the derivative terms for the Turbulent Boundary Layer case.

@details This module provides functions to compute the derivative terms for
the Turbulent Boundary Layer case. And it includes 2 methods, the first one
is the original method written by Callaham et al. (2021) which uses the scipy.sparse library
and the second one is the alternate method using numpy.gradient()

@author T.Breitburd and Callaham et al."""

from scipy import sparse
import numpy as np


def get_derivatives(nx, ny, dx, dy):
    """!@brief Get the derivative operators for the Turbulent Boundary Layer case

    @param nx: int
        Number of points in the x-direction
    @param ny: int
        Number of points in the y-direction
    @param dx: float
        Step in the x-direction
    @param dy: array
        Step in the y-direction

    @return Dx: scipy.sparse.csr_matrix
        1st derivative to 2nd order accuracy in the x-direction
    @return Dy: scipy.sparse.csr_matrix
        1st derivative to 2nd order accuracy in the y-direction

    @author T.Breitburd and Callaham et al.
    """

    Dy = sparse.diags([-1, 1], [-1, 1], shape=(ny, ny)).toarray()

    # Get 2nd order accurate forward/backwards difference at the boundaries
    Dy[0, :3] = np.array([-3, 4, -1])
    Dy[-1, -3:] = np.array([1, -4, 3])

    # Divide by 2*dy to get the derivative
    for i in range(ny - 1):
        Dy[i, :] = Dy[i, :] / (2 * dy[i])
    Dy[-1, :] = Dy[-1, :] / (2 * dy[-1])

    # Repeat the matrix for each points in the x-direction
    Dy = sparse.block_diag([Dy for i in range(nx)])  # Creates a block diagonal matrix
    # with the Dy matrix on the diagonal

    # Get the 1st derivative with respect to x
    Dx = sparse.diags([-1, 1], [-ny, ny], shape=(nx * ny, nx * ny))
    Dx = sparse.lil_matrix(Dx)

    # Get 2nd order accurate forwards/backwards difference at boundaries
    for i in range(ny):
        Dx[i, i] = -3
        Dx[i, ny + i] = 4
        Dx[i, 2 * ny + i] = -1
        Dx[-(i + 1), -(i + 1)] = 3
        Dx[-(i + 1), -(ny + i + 1)] = -4
        Dx[-(i + 1), -(2 * ny + i + 1)] = 1
    Dx = Dx / (2 * dx)

    Dx = Dx.tocsr()
    Dy = Dy.tocsr()

    return Dx, Dy


def get_derivatives_numpy(nx, ny, dx, y, u, v, p, R_uu, R_uv):
    """!@brief Get the derivative operators for the Turbulent Boundary Layer case

    @param nx: int
        Number of points in the x-direction
    @param ny: int
        Number of points in the y-direction
    @param dx: float
        Step in the x-direction
    @param y: array
        Array of y values
    @param u: array
        Array of u values
    @param v: array
        Array of v values
    @param p: array
        Array of p values
    @param R_uu: array
        Array of R_uu values
    @param R_uv: array
        Array of R_uv values

    @return u_x: array
        1st derivative to 2nd order accuracy in the x-direction
    @return u_y: array
        1st derivative to 2nd order accuracy in the y-direction
    @return lap_u: array
        Laplacian of u
    @return v_y: array
        1st derivative to 2nd order accuracy in the y-direction
    @return p_x: array
        1st derivative to 2nd order accuracy in the x-direction
    @return R_uux: array
        1st derivative to 2nd order accuracy in the x-direction of R_uu
    @return R_uvy: array
        1st derivative to 2nd order accuracy in the y-direction of R_uv

    @author T.Breitburd
    """

    # Initialize the arrays
    u_x = np.zeros((ny, nx))
    u_y = np.zeros((ny, nx))
    u_yy = np.zeros((ny, nx))
    u_xx = np.zeros((ny, nx))
    v_y = np.zeros((ny, nx))
    p_x = np.zeros((ny, nx))
    R_uux = np.zeros((ny, nx))
    R_uvy = np.zeros((ny, nx))

    # Compute the x-derivatives
    u_x = np.gradient(u, dx, edge_order=2, axis=1)
    p_x = np.gradient(p, dx, edge_order=2, axis=1)
    R_uux = np.gradient(R_uu, dx, edge_order=2, axis=1)

    # Compute the 2nd derivatives
    u_xx = np.gradient(u_x, dx, edge_order=2, axis=1)

    # Compute the y-derivatives
    u_y = np.gradient(u, y, edge_order=2, axis=0)
    v_y = np.gradient(v, y, edge_order=2, axis=0)
    R_uvy = np.gradient(R_uv, y, edge_order=2, axis=0)

    # Compute the 2nd derivatives
    u_yy = np.gradient(u_y, y, edge_order=2, axis=0)

    # Compute the laplacian
    lap_u = u_xx + u_yy

    return u_x, u_y, lap_u, v_y, p_x, R_uux, R_uvy
