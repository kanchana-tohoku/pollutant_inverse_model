import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import pcraster as pcr
from pcraster.framework import DynamicModel, DynamicFramework
from scipy.interpolate import interp1d

# =====================================================
# CONFIGURATION
# =====================================================
CONFIG = {
    "output_dir": "C:/Users/kanch/Research_models/data_2/out_ADErev6/Reconstructed",
    "clone_map": "C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/pcr_dem.map",
    "ldd_map": "C:/Users/kanch/Research_models/data_2/input_maps/topography/LDD/WGS_LDD.map",
    "flow_map": "C:/Users/kanch/Research_models/data_2/input_maps_synthetic/discharge/Discharge",
    "mass_profile_csv": "C:/Users/kanch/Research_models/data_2/out_ADErev6/M_profile_t1200.csv",
    "deltaT": 1.0,
    "deltaX": 100.0,
    "velocity_scale": 1.8,
    "start_t": 1200,
    "end_t": 200,
    "interval": 200,
    }
# =======================
# Helpers
#========================

def step_to_filename(step_idx, base_prefix):
    """
    Convert 1-based step index to PCRaster-style filename:
      step=1   -> M0000000.001
      step=1000-> M0000000.999
      step=1001-> M0000001.000
      step=2000-> M0000002.000
    """
    s = step_idx - 1
    integer   = s // 1000
    frac_part = (s % 1000) + 1
    base = base_prefix
    return f"{base}{integer:07d}.{frac_part:03d}"


# =====================================================
# BACKWARD MODEL
# =====================================================
class SoluteTransportModel_Backward(DynamicModel):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        pcr.setclone(cfg["clone_map"])

    def initial(self):
        cfg = self.cfg
        os.makedirs(cfg["output_dir"], exist_ok=True)

        # Read LDD
        self.ldd = pcr.readmap(cfg["ldd_map"])

        # Load known mass distribution (CSV: distance, M)
        data = np.loadtxt(cfg["mass_profile_csv"], delimiter=",", skiprows=1) # 
        self.x = data[:, 0]
        self.M_current = data[:, 1]

        # Create log
        self.log_path = os.path.join(cfg["output_dir"], "backward_log.csv")
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["Timestep", "TotalMass_mg"])

        # Report initial (known) distribution
        self.report_mass_profile(cfg["start_t"], self.M_current)

        print(f"[BACKWARD] Initialized with known mass at t={cfg['start_t']}")

    def dynamic(self):
        cfg = self.cfg
        t = cfg["start_t"] - (self.currentTimeStep() - 1) * cfg["interval"]
        t_prev = t - cfg["interval"]

        # Stop if we reached end
        if t_prev < cfg["end_t"]:
            return
        
        #reading the map
        file_name_Q = step_to_filename(t, 'Q')
        
        q_path = os.path.join(cfg["flow_map"], f"{file_name_Q}")
        if not os.path.exists(q_path):
            print(f"[BACKWARD] ⚠ Missing discharge map: {q_path}")
            return

        Q_map = pcr.readmap(q_path)

        # Extract Q values along the main channel path
        # (Option 1) use pcr downstream path sampling
        Q_values, distances = self.extract_channel_Q(Q_map)

        # Estimate local velocity (scaled)
        v = Q_values / np.max(Q_values) * cfg["velocity_scale"]

        # Compute backward shift (Δx = v * Δt)
        x_shift = v * cfg["interval"]
        x_back = self.x + x_shift

        # Interpolate mass profile backward
        f = interp1d(x_back, self.M_current, fill_value=0.0, bounds_error=False)
        M_prev = f(self.x)
        self.M_current = M_prev.copy()

        # Report reconstructed profile
        self.report_mass_profile(t_prev, M_prev)

        print(f"[BACKWARD] Reconstructed mass at t={t_prev}")

    # =====================================================
    # Helper functions
    # =====================================================
    def extract_channel_Q(self, Q_map):
        """Extract Q along main channel cells in order of flow."""
        # Define main channel mask
        stream = pcr.streamorder(self.ldd)
        main_channel = pcr.ifthen(stream == pcr.mapmaximum(stream), stream)

        # Convert to array (raster order)
        Q_np = pcr.pcr2numpy(Q_map, np.nan)
        mask_np = pcr.pcr2numpy(main_channel, 0)
        idx = np.where(mask_np > 0)
        Q_values = Q_np[idx]
        # Generate synthetic distance (for 1D representation)
        distances = np.arange(0, len(Q_values)) * self.cfg["deltaX"]
        return Q_values, distances

    def report_mass_profile(self, t, M_profile):
        out_file = os.path.join(self.cfg["output_dir"], f"M_back_{t:04d}.csv")
        np.savetxt(out_file, np.column_stack([self.x, M_profile]),
                   delimiter=",", header="Distance(m),Mass_per_length(mg/m)", comments="")

        total_mass = np.sum(M_profile) * self.cfg["deltaX"]
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([t, total_mass])

# =====================================================
# RUN MODEL
# =====================================================
model = SoluteTransportModel_Backward(CONFIG)
dyn = DynamicFramework(model, (CONFIG["start_t"] - CONFIG["end_t"]) // CONFIG["interval"] + 1)
dyn.run()

# =====================================================
# PLOT RESULTS
# =====================================================
import glob
csv_files = sorted(glob.glob(os.path.join(CONFIG["output_dir"], "M_back_*.csv")))
plt.figure(figsize=(10,6))
for f in csv_files:
    t = int(os.path.basename(f).split("_")[-1].split(".")[0])
    data = np.loadtxt(f, delimiter=",", skiprows=1)
    plt.plot(data[:,0], data[:,1], label=f"t={t}")
plt.xlabel("Distance along channel (m)")
plt.ylabel("Mass per length (mg/m)")
plt.title("Backward Reconstructed Pollutant Mass Distributions")
plt.legend()
plt.grid(True)
plt.show()
