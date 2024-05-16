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

    # Get 2nd order forward/backwards at the boundaries
    Dy[0, :3] = np.array([-3, 4, -1])
    Dy[-1, -3:] = np.array([1, -4, 3])

    for i in range(ny - 1):
        Dy[i, :] = Dy[i, :] / (2 * dy[i])
    Dy[-1, :] = Dy[-1, :] / (2 * dy[-1])

    # Repeat this for the x-direction
    Dy = sparse.block_diag([Dy for i in range(nx)])  # Creates a block diagonal matrix
    # with the Dy matrix on the diagonal

    Dx = sparse.diags([-1, 1], [-ny, ny], shape=(nx * ny, nx * ny))
    Dx = sparse.lil_matrix(Dx)

    # Get 2nd order forwards/backwards with boundary conditions
    for i in range(ny):
        Dx[i, i] = -3
        Dx[i, ny + i] = 4
        Dx[i, 2 * ny + i] = -1
        Dx[-(i + 1), -(i + 1)] = 3
        Dx[-(i + 1), -(ny + i + 1)] = -4
        Dx[-(i + 1), -(2 * ny + i + 1)] = 1
    Dx = Dx / (2 * dx)

    Dx = sparse.csr_matrix(Dx)
    Dy = sparse.csr_matrix(Dy)

    return Dx, Dy
