from scipy import sparse
import numpy as np


def get_derivatives(nx, ny, dx, dy):
    """Get the derivatives for the 2D domain

    Parameters:
    -----------
    nx: int
        Number of points in the x-direction
    ny: int
        Number of points in the y-direction
    dx: float
        Step in the x-direction
    dy: float
        Step in the y-direction

    Returns:
    --------
    Dx: sparse matrix
        1st derivative to 2nd order accuracy in the x-direction
    Dy: sparse matrix
        1st derivative to 2nd order accuracy in the y-direction
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
    """Get the derivatives for the 2D domain

    Parameters:
    -----------
    nx: int
        Number of points in the x-direction
    ny: int
        Number of points in the y-direction
    dx: float
        Step in the x-direction
    y: numpy array
        Array of y coordinates
    u: numpy array
        Array of u velocities
    v: numpy array
        Array of v velocities
    p: numpy array
        Array of pressures
    R_uu: numpy array
        Array of R_uu values
    R_uv: numpy array
        Array of R_uv values

    Returns:
    --------
    u_x: numpy array
        1st derivative to 2nd order accuracy in the x-direction
    u_y: numpy array
        1st derivative to 2nd order accuracy in the y-direction
    lap_u: numpy array
        Laplacian of u
    v_y: numpy array
        1st derivative to 2nd order accuracy in the y-direction
    p_x: numpy array
        1st derivative to 2nd order accuracy in the x-direction
    R_uux: numpy array
        1st derivative to 2nd order accuracy in the x-direction
    R_uvy: numpy array
        1st derivative to 2nd order accuracy in the y-direction
    """

    u_x = np.zeros((ny, nx))
    u_y = np.zeros((ny, nx))
    u_yy = np.zeros((ny, nx))
    u_xx = np.zeros((ny, nx))
    v_y = np.zeros((ny, nx))
    p_x = np.zeros((ny, nx))
    R_uux = np.zeros((ny, nx))
    R_uvy = np.zeros((ny, nx))

    u_x = np.gradient(u, dx, edge_order=2, axis=1)
    p_x = np.gradient(p, dx, edge_order=2, axis=1)
    R_uux = np.gradient(R_uu, dx, edge_order=2, axis=1)
    u_xx = np.gradient(u_x, dx, edge_order=2, axis=1)

    u_y = np.gradient(u, y, edge_order=2, axis=0)
    v_y = np.gradient(v, y, edge_order=2, axis=0)
    R_uvy = np.gradient(R_uv, y, edge_order=2, axis=0)
    u_yy = np.gradient(u_y, y, edge_order=2, axis=0)

    lap_u = u_xx + u_yy

    return u_x, u_y, lap_u, v_y, p_x, R_uux, R_uvy
