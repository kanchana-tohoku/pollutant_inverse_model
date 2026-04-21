import os
import pcraster as pcr

# ===================================================
# CONFIGURATION
# ===================================================
CONFIG = {
    "clone_map": r"C:\Users\kanch\Research_models\data_2\input_maps\topography\DEM\pcr_dem.map",
    "ldd_map": r"C:\Users\kanch\Research_models\data_2\input_maps\topography\LDD\WGS_LDD.map",
    "final_mass_map": r"C:\Users\kanch\Research_models\data_2\out_ADErev6\case_2\M0000002.000",
    "flow_prefix": r"C:/Users/kanch/Research_models/data_2/input_maps_synthetic/discharge/Discharge/Q",
    "output_dir": r"C:\Users\kanch\Research_models\data_2\out_ADErev6\case_2\TimeReversal_Adjoint",
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
# TRUE ADJOINT BACKWARD STEP
# ===================================================
def adjoint_step(M_next, Q_map):

    # Cross-sectional area
    A = Q_map / velocity

    # Avoid division by zero
    C = pcr.ifthenelse(A > 0, M_next / A, 0)

    # Flux
    Flux = Q_map * C

    # Adjoint routing (transpose of upstream)
    FluxAdj = pcr.downstream(ldd, Flux)

    # Adjoint update
    M_prev = M_next + alpha * (FluxAdj - Flux)

    # Enforce positivity
    M_prev = pcr.ifthenelse(M_prev < 0, 0, M_prev)

    return M_prev

# ===================================================
# LOAD FINAL MASS MAP
# ===================================================
print("Loading final mass map...")
M_current = pcr.readmap(CONFIG["final_mass_map"])

# ===================================================
# TIME REVERSAL LOOP
# ===================================================
print("Starting adjoint-based time reversal...")

for t in reversed(range(1, CONFIG["nrOfTimeSteps"])):

    print(f"Reconstructing t = {t}")

    Q_filename = step_to_filename(t)
    Q_map = pcr.readmap(Q_filename)

    M_prev = adjoint_step(M_current, Q_map)

    output_path = step_to_output_filename(t, CONFIG["output_dir"])
    pcr.report(M_prev, output_path)

    M_current = M_prev

print("Adjoint-based time reversal completed.")