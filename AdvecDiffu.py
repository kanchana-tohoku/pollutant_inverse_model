import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Analytical solution of 1-D advection–diffusion eq.
# Based on Zhang & Xin (2017), Eq. (3)
# --------------------------------------------------

def pollutant_concentration(
    x, t,
    source_positions,
    source_strengths,
    l=1000.0,        # river length (m)
    u=0.1,           # flow velocity (m/s)
    Ex=5.0,          # diffusion coefficient (m^2/s)
    K=1e-5,          # decay coefficient (1/s)
    n_max=1000       # number of Fourier terms
):
    """
    Compute pollutant concentration C(x,t) using analytical solution.

    Parameters
    ----------
    x : array_like
        Spatial positions (m)
    t : float
        Time (s)
    source_positions : array_like
        Positions of pollution sources xi (m)
    source_strengths : array_like
        Source strengths Mi (g/s)
    l, u, Ex, K : float
        Physical parameters
    n_max : int
        Truncation of infinite series

    Returns
    -------
    C : ndarray
        Pollutant concentration at x and t
    """

    x = np.asarray(x)
    C = np.zeros_like(x, dtype=float)

    source_positions = np.asarray(source_positions)
    source_strengths = np.asarray(source_strengths)

    for n in range(1, n_max + 1):
        lambda_n = (n * np.pi / l)**2
        denom = u**2 / (4 * Ex) + lambda_n + K

        # Sum over pollution sources
        source_sum = 0.0
        for Mi, xi in zip(source_strengths, source_positions):
            source_sum += Mi * np.exp(-u * xi / (2 * Ex)) * np.sin(n * np.pi * xi / l)

        time_factor = np.exp(u**2 * t / (4 * Ex)) - np.exp(-denom * t)
        spatial_factor = np.sin(n * np.pi * x / l)

        C += (2 / l) * (source_sum / denom) * time_factor * spatial_factor

    # Exponential correction term
    C *= np.exp(u * x / (2 * Ex) - u**2 * t / (4 * Ex))

    return C


# --------------------------------------------------
# Example simulation (from the paper)
# --------------------------------------------------

if __name__ == "__main__":

    # Spatial grid
    x = np.linspace(0, 1000, 500)

    # Time (s)
    t = 4000

    # ---- Case 1: Single pollution source ----
    source_positions = [500]     # m
    source_strengths = [3.0]     # g/s

    C_single = pollutant_concentration(
        x, t,
        source_positions,
        source_strengths,
        n_max=1000
    )

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(x, C_single, label="Single source (x=500 m, M=3 g/s)")
    plt.xlabel("Distance along river (m)")
    plt.ylabel("Concentration (mg/L)")
    plt.title("Pollutant concentration along river (t = 4000 s)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    print(C_single)
