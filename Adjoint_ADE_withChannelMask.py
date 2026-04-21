import os
import pcraster as pcr

# ===================================================
# CONFIGURATION
# ===================================================
case = 3

CONFIG = {
    "clone_map": "C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/pcr_dem.map",
    "ldd_map": r"C:\Users\kanch\Research_models\data_2\input_maps\topography\LDD\WGS_LDD.map",
    "channel_mask": f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}/fluxpath",
    "final_mass_map": f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}/M0000002.000",
    "flow_prefix": r"C:/Users/kanch/Research_models/data_2/input_maps_synthetic/discharge/Discharge/Q",
    "output_dir": f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}/TimeReversal_Adjoint_Channel",
    "deltaT": 1.0,
    "deltaX": 100.0,
    "velocity": 1.8,
    "nrOfTimeSteps": 1999
}

# ===================================================
# INITIALIZATION
# ===================================================
pcr.setclone(CONFIG["clone_map"])

ldd = pcr.readmap(CONFIG["ldd_map"])
channel = pcr.readmap(CONFIG["channel_mask"])

# Ensure boolean mask
channel = pcr.ifthen(channel > 0, pcr.boolean(1))

os.makedirs(CONFIG["output_dir"], exist_ok=True)

deltaT = CONFIG["deltaT"]
deltaX = CONFIG["deltaX"]
velocity = CONFIG["velocity"]
alpha = deltaT / deltaX
base_prefix = CONFIG["flow_prefix"]

# ===================================================
# TIME-SLICE NAMING
# ===================================================
def step_to_filename(step):
    s = step - 1
    block = s // 999
    slice_no = (s % 999) + 1
    return f"{base_prefix}{block:07d}.{slice_no:03d}"

def step_to_output_filename(step, output_dir):
    s = step - 1
    block = s // 999
    slice_no = (s % 999) + 1
    filename = f"MI{block:06d}.{slice_no:03d}"
    return os.path.join(output_dir, filename)

# ===================================================
# ADJOINT STEP WITH CHANNEL PROJECTION
# ===================================================
def adjoint_step_channel(M_next, Q_map):

    # Restrict mass to channel cells
    M_next = pcr.ifthen(channel, M_next)

    # Cross-sectional area
    A = Q_map / velocity

    # Safe concentration
    C = pcr.ifthenelse(A > 0, M_next / A, 0)

    # Flux only on channel
    Flux = pcr.ifthen(channel, Q_map * C)

    # Adjoint routing
    FluxAdj = pcr.downstream(ldd, Flux)

    # Backward adjoint update
    M_prev = M_next + alpha * (FluxAdj - Flux)

    # Enforce positivity
    M_prev = pcr.ifthenelse(M_prev < 0, 0, M_prev)

    # Project again to channel network
    M_prev = pcr.ifthen(channel, M_prev)

    return M_prev

# ===================================================
# LOAD FINAL MASS MAP
# ===================================================
print("Loading final mass map...")
M_current = pcr.readmap(CONFIG["final_mass_map"])

# Restrict initial state to channel
M_current = pcr.ifthen(channel, M_current)

# ===================================================
# TIME REVERSAL LOOP
# ===================================================
print("Starting adjoint-based time reversal (channel-restricted)...")

for t in reversed(range(1, CONFIG["nrOfTimeSteps"])):

    print(f"Reconstructing t = {t}")

    Q_filename = step_to_filename(t)
    Q_map = pcr.readmap(Q_filename)

    # Restrict discharge to channel
    Q_map = pcr.ifthen(channel, Q_map)

    M_prev = adjoint_step_channel(M_current, Q_map)

    output_path = step_to_output_filename(t, CONFIG["output_dir"])
    pcr.report(M_prev, output_path)

    M_current = M_prev

print("Adjoint-based channel-restricted time reversal completed.")