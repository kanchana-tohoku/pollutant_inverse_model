import numpy as np
import matplotlib.pyplot as plt

# Parameters
x = np.linspace(-10, 30, 400)
x0 = 0       # release location
u = 1.0      # velocity (m/s)
t = 10.0     # seconds
Ds = [1.0, 0.1, 0.01, 0.001]  # dispersion values

plt.figure(figsize=(8,5))

for D in Ds:
    G = (1/np.sqrt(4*np.pi*D*t)) * np.exp(-((x - x0 - u*t)**2) / (4*D*t))
    plt.plot(x, G, label=f"D={D}")

# Pure advection line (D→0) – delta moves at x = ut
plt.axvline(x0 + u*t, color="k", linestyle="--", label="Pure advection (x = ut)")

plt.title("Transition of Green's Function: Diffusion → Pure Advection")
plt.xlabel("x (space)")
plt.ylabel("G(x,t; x₀)")
plt.legend()
plt.grid(True)
plt.show()

