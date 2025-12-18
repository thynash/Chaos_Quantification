import numpy as np

def henon_map(a, b, x0, y0, N, discard=100):
    """
    Henon map:
        x_{n+1} = 1 - a x_n^2 + y_n
        y_{n+1} = b x_n

    Parameters
    ----------
    a, b : float
        Henon parameters
    x0, y0 : float
        Initial conditions
    N : int
        Number of samples to return
    discard : int
        Number of transient iterations

    Returns
    -------
    traj : ndarray, shape (N, 2)
        Time series [x_n, y_n]
    """
    x = np.empty(N + discard)
    y = np.empty(N + discard)

    x[0], y[0] = x0, y0

    for n in range(N + discard - 1):
        x[n+1] = 1.0 - a * x[n]**2 + y[n]
        y[n+1] = b * x[n]

    return np.column_stack((x[discard:], y[discard:]))

