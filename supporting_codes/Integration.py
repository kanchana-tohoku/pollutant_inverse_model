import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, cumulative_trapezoid
from scipy.optimize import root_scalar

# Piecewise velocity function
def U(t):
    if 0 <= t <= 1:      # velocity = 0 between t=0 and t=1
        return 2 * t + 0.5 * np.sin(t)
    else:
        return 2.0 + 0.5 * np.sin(t)

# Known values
x1 = 1.5        # entry location
x2 = 10.0       # observed location
t2 = 8.0        # observation time

# Define equation to solve for t1
def travel_time_equation(t1):
    integral, _ = quad(U, t1, t2)
    return integral - (x2 - x1)

# Solve for t1 using root finding
sol = root_scalar(travel_time_equation, bracket=[0, t2], method='bisect')

if sol.converged:
    t1 = sol.root
    print(f"Estimated entry time t1 = {t1:.4f}")
    print(f"Check: Distance travelled = {quad(U, t1, t2)[0]:.4f}")
else:
    print("Root finding did not converge.")

# ---- Plot trajectory and velocity ----
t = np.linspace(0, t2, 500)
U_values = np.array([U(tt) for tt in t])

# Integrate velocity to get position x(t)
x = x1 + cumulative_trapezoid(U_values, t, initial=0.0)

plt.figure(figsize=(8,5))
plt.plot(t, x, label="x(t): position")
plt.plot(t, U_values, '--', label="U(t): velocity")
plt.axvline(t1, color='r', linestyle=':', label=f"t1 ≈ {t1:.2f}")
plt.scatter([t2], [x2], color='k', marker='o', label=f"Observed point (t2={t2}, x2={x2})")
plt.xlabel("Time t")
plt.ylabel("Position x(t) / Velocity U(t)")
plt.title("Trajectory from Piecewise Velocity")
plt.legend()
plt.grid(True)
plt.show()
