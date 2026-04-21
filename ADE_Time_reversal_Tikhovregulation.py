import os
import pcraster as pcr

# ===================================================
# CONFIGURATION
# ===================================================
case =1

CONFIG = {
    "clone_map": r"C:\Users\kanch\Research_models\data_2\input_maps\topography\DEM\pcr_dem.map",
    "ldd_map": r"C:\Users\kanch\Research_models\data_2\input_maps\topography\LDD\WGS_LDD.map",
    "final_mass_map": f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}/M0000002.000",
    "channel_mask": f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}/fluxpath",
    "flow_prefix": r"C:/Users/kanch/Research_models/data_2/input_maps_synthetic/discharge/Discharge/Q",
    "output_dir": f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}/TimeReversal_Regularized",
    "deltaT": 1.0,
    "deltaX": 100.0,
    "velocity": 1.8,
    "nrOfTimeSteps": 1999,
    "lambda_reg": 0.03,
    "max_iter": 25,
    "tolerance": 1e-6
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
lambda_reg = CONFIG["lambda_reg"]
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
# SAFE LAPLACIAN (NO MV PROPAGATION)
# ===================================================
def laplacian_safe(M):

    north = pcr.shift(M, 0, 1)
    south = pcr.shift(M, 0, -1)
    east  = pcr.shift(M, 1, 0)
    west  = pcr.shift(M, -1, 0)

    lap = (north + south + east + west - 4 * M) / (deltaX**2)

    valid = (
        pcr.defined(north) &
        pcr.defined(south) &
        pcr.defined(east)  &
        pcr.defined(west)
    )

    return pcr.ifthen(valid, lap)

# ===================================================
# REGULARIZED BACKWARD SOLVER
# ===================================================
def compute_previous_M_regularized(M_next, Q_map, domain_mask, t):

    M_prev = M_next

    for k in range(CONFIG["max_iter"]):
     
        # --- Safe division ---
        A = Q_map / velocity
        C = pcr.ifthenelse(A > 0, M_prev / A, 0)

        FluxOut = Q_map * C
        FluxIn = pcr.upstream(ldd, FluxOut)

        # --- Backward update ---
        M_new = M_next + (deltaT / deltaX) * (FluxOut - FluxIn)

        # --- Regularization ---
        smooth_term = lambda_reg * laplacian_safe(M_new)

        M_new = pcr.cover(M_new - smooth_term, M_new)

        # --- Enforce positivity ---
        M_new = pcr.ifthenelse(M_new < 0, 0, M_new)

        # --- Enforce domain mask (CRITICAL) ---
        M_new = pcr.ifthen(domain_mask, M_new)

        # --- Convergence check ---
        diff = pcr.maptotal(pcr.abs(M_new - M_prev))

        M_prev = M_new
        
        
        if float(diff) < CONFIG["tolerance"]:
            print(f"t={t} converged in {k+1} iterations")
            break

    return M_prev

# ===================================================
# LOAD FINAL MASS MAP
# ===================================================
print("Loading final mass map...")
M_current = pcr.readmap(CONFIG["final_mass_map"])

# Define computational domain once
domain_mask = pcr.defined(M_current)

# ===================================================
# TIME REVERSAL LOOP
# ===================================================
print("Starting regularized time reversal...")

for t in reversed(range(1, CONFIG["nrOfTimeSteps"])):

    print(f"Reconstructing t = {t}")

    Q_filename = step_to_filename(t)
    Q_map = pcr.readmap(Q_filename)

    M_prev = compute_previous_M_regularized(
        M_current,
        Q_map,
        domain_mask,
        t
    )

    output_path = step_to_output_filename(t, CONFIG["output_dir"])
    pcr.report(M_prev, output_path)

    M_current = M_prev

print("Regularized time reversal completed.")