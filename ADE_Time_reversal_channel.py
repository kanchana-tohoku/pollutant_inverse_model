import os
import pcraster as pcr

# ===================================================
# CONFIGURATION
# ===================================================
case =3

CONFIG = {
    "clone_map": r"C:\Users\kanch\Research_models\data_2\input_maps\topography\DEM\pcr_dem.map",
    "ldd_map": r"C:\Users\kanch\Research_models\data_2\input_maps\topography\LDD\WGS_LDD.map",
    "final_mass_map": f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}/M0000002.000",
    "channel_mask": f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}/fluxpath",
    "flow_prefix": r"C:/Users/kanch/Research_models/data_2/input_maps_synthetic/discharge/Discharge/Q",
    "output_dir": f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}/TimeReversal_channel_2",
    "deltaT": 1,
    "deltaX": 100,
    "velocity": 1.8,
    "nrOfTimeSteps": 2000,
    "max_iter": 25,
    "tolerance": 1e-8
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
base_prefix = CONFIG["flow_prefix"]

channel = pcr.readmap(CONFIG["channel_mask"])
channel = pcr.ifthen(channel > 0, pcr.boolean(1))

# ===================================================
# PCRASTER TIME-SLICE NAMING
# ===================================================
def step_to_filename(step):
    """
    Convert 1-based step index to PCRaster time-slice filename.
    """
    s = step #- 1

    block = s // 1000
    slice_no = (s % 1000) #+ 1

    return f"{base_prefix}{block:07d}.{slice_no:03d}"

def step_to_output_filename(step, output_dir):
    """
    Convert timestep to PCRaster dynamic filename with prefix 'MI'.
    Example:
        step=10   → MI000000.010
        step=1568 → MI000001.568
    """
    s = step #- 1

    block = s // 1000
    slice_no = (s % 1000) #+ 1

    filename = f"MI{block:06d}.{slice_no:03d}"
    return os.path.join(output_dir, filename)

# ===================================================
# PURE BACKWARD SOLVER (NO REGULARIZATION)
# ===================================================
def compute_previous_M(M_next, Q_map, t):

    # Initial guess
    #M_prev = pcr.ifthen(channel,M_next)
    M_prev = M_next

    for k in range(CONFIG["max_iter"]):

#        A = Q_map / velocity
#        C = pcr.ifthenelse(A > 0, M_prev / A, 0)
        
 #       FluxOut = Q_map * C
#        FluxAdj = pcr.downstream(ldd, FluxOut)
        
#        M_new = M_next + (deltaT / deltaX) * (FluxAdj - FluxOut)
        
 #       M_new = pcr.ifthenelse(M_new < 0, 0, M_new)        


        A = Q_map / velocity
        C = M_prev / A

        FluxOut = Q_map * C
        FluxIn = pcr.upstream(ldd, FluxOut)

        # Backward update
        M_new = M_next + (deltaT / deltaX) * (FluxOut - FluxIn)

        #Enforce positivity (optional but recommended)
        M_new = pcr.ifthenelse(M_new < 0, 0, M_new)

        # Convergence check
        diff = pcr.maptotal(pcr.abs(M_new - M_prev))

        M_prev = M_new

        if float(diff) < CONFIG["tolerance"]:
            print(f"t={t} converged in {k+1} iterations")
            break

    return M_prev

# ===================================================
# LOAD FINAL MASS MAP (t = 2000)
# ===================================================
print("Loading final mass map...")
M_current = pcr.readmap(CONFIG["final_mass_map"])

# ===================================================
# TIME REVERSAL LOOP
# ===================================================
print("Starting time reversal (no regularization)...")

for t in reversed(range(1, CONFIG["nrOfTimeSteps"])):

    print(f"Reconstructing t = {t}")

    Q_filename = step_to_filename(t)
    Q_map = pcr.readmap(Q_filename)

    M_prev = compute_previous_M(M_current, Q_map, t)

    output_path = step_to_output_filename(t, CONFIG["output_dir"])
    pcr.report(M_prev, output_path)

    M_current = M_prev

print("Time reversal completed.")