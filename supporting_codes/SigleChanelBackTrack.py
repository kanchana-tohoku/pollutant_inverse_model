# Back tracking the trajectory 

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# === 1. Example data setup (replace with your real data) ===
x_points = np.arange(0, 2100, 100)  # spatial grid: 0–2000 m
t_points = np.arange(0, 1001, 1)    # time grid: 0–1000 s

# Example: make up a spatially + temporally varying velocity field
U = 1.2 + 0.5 * np.sin(2*np.pi*t_points[None,:]/400) * np.exp(-x_points[:,None]/2000)
#U = 1.2 + 0.1 * np.random.rand() * t_points[None,:] + 0.1 * np.random.rand() * x_points[:,None]

# === 2. Interpolator for u(x, t) ===
interp_u = RegularGridInterpolator((x_points, t_points), U)

# === 3. Initial condition ===
x_now = 1600.0   # position where object was seen
t_now = 450.0    # time of observation
dt = 1.0         # integration step (s)
n_steps = 300    # go 300 seconds back

# === 4. Backward integration ===
xs = [x_now]
ts = [t_now]

for i in range(n_steps):
    u_now = interp_u([[xs[-1], ts[-1]]])[0]
    x_prev = xs[-1] - u_now * dt    # backward in time
    t_prev = ts[-1] - dt
    xs.append(x_prev)
    ts.append(t_prev)

xs = np.array(xs)
ts = np.array(ts)

# === 5. Plot ===
plt.figure(figsize=(7,4))
plt.plot(ts, xs, 'b-', lw=2)
plt.xlabel('Time (s)')
plt.ylabel('Position x (m)')
plt.title('Backward trajectory of floating object')
plt.grid(True)
plt.show()
