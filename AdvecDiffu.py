import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------
# Analytical solution of 1-D advection–diffusion eq.
# Based on Zhang & Xin (2017), Eq. (3)
# --------------------------------------------------


#==Forward Model (with release time slot)======


def pollutant_concentration(
    x, t,
    source_positions,
    source_strengths,
    release_windows,   # NEW: [(t_start, t_end), ...]
    l ,
    u,
    Ex,
    K,
    n_max
):
    """
    Compute pollutant concentration with finite release duration.

    release_windows : list of (t_start, t_end)
    """

    x = np.asarray(x)
    C_total = np.zeros_like(x, dtype=float)

    for xi, Mi, (t_start, t_end) in zip(source_positions,
                                        source_strengths,
                                        release_windows):

        # Time since release started
        t1 = t - t_start
        t2 = t - t_end

        # If observation is before release → zero contribution
        if t1 <= 0:
            continue

        # Compute continuous solution at shifted times
        C1 = _continuous_solution(x, t1, xi, Mi, l, u, Ex, K, n_max)

        if t2 > 0:
            C2 = _continuous_solution(x, t2, xi, Mi, l, u, Ex, K, n_max)
        else:
            C2 = 0

        C_total += C1 - C2
        C_total = np.maximum(C_total, 0.0)

    return C_total

#=== continuous solution for ADE =========
    
def _continuous_solution(x, t, xi, Mi, l, u, Ex, K, n_max):

    if t <= 0:
        return np.zeros_like(x)

    x = np.asarray(x)
    C = np.zeros_like(x, dtype=float)

    for n in range(1, n_max + 1):

        lam = (n * np.pi / l)**2
        beta = Ex * lam + K

        # Time factor (stable)
        decay = 1 - np.exp(-beta * t)

        # Source projection
        source_proj = Mi * np.sin(n * np.pi * xi / l)

        # Space eigenfunction
        space = np.sin(n * np.pi * x / l)

        C += (2 / l) * (source_proj / beta) * decay * space

    # Apply physical advection shift separately
    x_shift = x - u * t

    # Interpolate concentration to shifted grid
    C_advected = np.interp(x_shift, x, C, left=0, right=0)

    return np.maximum(C_advected, 0.0)

# --------------------------------------------------
# Example simulation 
# --------------------------------------------------


x_obs = np.arange(80, 5000, 1)

source_positions = [100]
source_strengths = [0.]
release_windows = [(0, 1)]

l = 5000
u = 1
Ex = 5
K = 1e-5
n_max = 5000   

time_values = np.arange(1,3600,600) #[1, 30, 60, 90, 600, 1200, 1800, 2400, 3000, 3600] #[50,100]

plt.figure(figsize=(8, 5))

for T_obs in time_values:
    
    C_model = pollutant_concentration(
        x_obs,
        T_obs,
        source_positions,
        source_strengths,
        release_windows,
        l,
        u,
        Ex,
        K,
        n_max
    )
    
    plt.plot(x_obs, C_model, label=f"t = {T_obs} s")

plt.xlabel("Distance along river (m)")
plt.ylabel("Concentration (mg/L)")
plt.title("Pollutant concentration at different times")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

