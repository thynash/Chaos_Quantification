import numpy as np

# -------------------------------------------------
# Logistic map Lyapunov exponent
# -------------------------------------------------

def lyapunov_logistic(x, r):
    """
    Finite-time Lyapunov exponent for logistic map.
    """
    return np.mean(np.log(np.abs(r * (1 - 2 * x))))


# -------------------------------------------------
# Henon map Lyapunov exponent (largest)
# -------------------------------------------------

def lyapunov_henon(traj, a, b):
    """
    Finite-time largest Lyapunov exponent for Henon map
    using tangent vector propagation.
    """
    v = np.array([1.0, 0.0])
    lyap_sum = 0.0

    for x, y in traj:
        J = np.array([
            [-2 * a * x, 1.0],
            [b,          0.0]
        ])
        v = J @ v
        norm = np.linalg.norm(v)
        v /= norm
        lyap_sum += np.log(norm)

    return lyap_sum / len(traj)


# -------------------------------------------------
# Coupled logistic map Lyapunov exponent (largest)
# -------------------------------------------------

def lyapunov_coupled(traj, r, eps):
    """
    Finite-time largest Lyapunov exponent for diffusively
    coupled logistic maps.
    """
    v = np.array([1.0, 0.0])
    lyap_sum = 0.0

    for x, y in traj:
        fxp = r * (1 - 2 * x)
        fyp = r * (1 - 2 * y)

        J = np.array([
            [(1 - eps) * fxp, eps * fyp],
            [eps * fxp, (1 - eps) * fyp]
        ])

        v = J @ v
        norm = np.linalg.norm(v)
        v /= norm
        lyap_sum += np.log(norm)

    return lyap_sum / len(traj)

