import os
import csv
import pcraster as pcr
from pcraster.framework import DynamicModel, DynamicFramework
import numpy as np

# ========================
# Configuration
# ========================
CONFIG = {
    "output_dir": "C:/Users/kanch/Research_models/data_2/out_ADErev6/Trial",
    "clone_map": "C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/pcr_dem.map",
    "ldd_map": "C:/Users/kanch/Research_models/data_2/input_maps/topography/LDD/WGS_LDD.map",
    "pollutant_map": "C:/Users/kanch/Research_models/data_2/input_maps/pollutants/WGS/pollution_source_WGS_200points.map",
    "flow_map": "C:/Users/kanch/Research_models/data_2/input_maps_synthetic/discharge/Discharge/Q",
    "noise": 1, 
    "log_file": "solute_mass_log.csv",
    "deltaT": 1,
    "deltaX": 100,
    "velocity": 1.8,
    "release_start": 1,
    "release_end": 2,
    "nrOfTimeSteps": 2000,
    "ocean_cell": (278, 18),
    "replacement_csv": "C:/Users/kanch/Research_models/data_2/input_maps/pollutants/WGS/Replacement_pollution_source_WGS_200points.csv",
}

#========
#replacing existing pollutants
#=========
def apply_replacements_from_csv(poll_map, csv_path):
    """
    Reads CSV file with columns: ID,row,column,value
    and replaces corresponding cells in the pollutant map.
    """
    arr = pcr.pcr2numpy(poll_map, np.nan)

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            r = int(row["row"])
            c = int(row["column"])
            val = float(row["value"])

            # Convert 1-based (PCRaster) to 0-based (NumPy)
            arr[r - 1, c - 1] = val

            print(f"Replaced cell ({r},{c}) with value {val}")

    new_map = pcr.numpy2pcr(pcr.Scalar, arr, np.nan)
    return new_map







# ===================================================
# 1. Upwind Euler model (as before)
# ===================================================
class SoluteTransportModel_1stO_UW_Euler(DynamicModel):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        pcr.setclone(cfg["clone_map"])

    def initial(self):
        self.deltaT = self.cfg["deltaT"]
        self.deltaX = self.cfg["deltaX"]
        self.ldd = pcr.readmap(self.cfg["ldd_map"])
        
        #self.E_nominal = pcr.readmap(self.cfg["pollutant_map"]) / self.deltaX
        #---------------
        poll_map = pcr.readmap(self.cfg["pollutant_map"])

        # Apply CSV-based replacements
        poll_map = apply_replacements_from_csv(
            poll_map,
            self.cfg["replacement_csv"]
            )

        self.E_nominal = poll_map / self.deltaX
        
        #-------------
        
        self.M = pcr.scalar(0.0)
        self.C = pcr.scalar(0.0)
        self.ocean_release = 0.0
        self.Theoretical_Total = 0.0
        os.makedirs(self.cfg["output_dir"], exist_ok=True)
        self.log_path = os.path.join(self.cfg["output_dir"], "solute_mass_log_Upwind.csv")
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["timestep", "Theoretical_Total", "Model_Total", "Error", "ErrorPct"])
        total_init = float(pcr.maptotal(self.E_nominal))
        print(f"[UPWIND] Total initial pollution = {total_init:.3f}")

    def apply_pollutant_release(self, t):
        s, e = self.cfg["release_start"], self.cfg["release_end"]
        return pcr.ifthenelse((t >= s) & (t <= e), self.E_nominal, pcr.scalar(0))

    def dynamic(self):
        t = self.currentTimeStep()
        self.E = self.apply_pollutant_release(t)
        Q_out = self.readmap(self.cfg["flow_map"])
        velocity = self.cfg["velocity"]
        A = Q_out / velocity
        self.C = self.M / A
        FluxOut = Q_out * self.C
        FluxIn = pcr.upstream(self.ldd, FluxOut)
        M_new = self.M - (self.deltaT / self.deltaX) * (FluxOut - FluxIn) + self.deltaT * self.E
        self.M = pcr.ifthenelse(M_new < 0, 0, M_new)
        self.C = self.M / A
        row, col = self.cfg["ocean_cell"]
        self.ocean_release += pcr.cellvalue(FluxOut, row, col)[0] * self.deltaT
        Model_Total = float(pcr.maptotal(self.M) * self.deltaX) + self.ocean_release
        self.Theoretical_Total += float(pcr.maptotal(self.E * self.deltaX)) * self.deltaT
        Err = self.Theoretical_Total - Model_Total
        ErrPct = Err / self.Theoretical_Total * 100 if self.Theoretical_Total else 0
        print(f"[UPWIND] {t}: T={self.Theoretical_Total:.2f} M={Model_Total:.2f} Err={Err:.2f} ({ErrPct:.2f}%)")
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([t, self.Theoretical_Total, Model_Total, Err, ErrPct])
            
        self.report(self.M, os.path.join(self.cfg["output_dir"], "M"))

# ===================================================
# 2. CIP (Cubic Interpolated Propagation) Scheme
# ===================================================
class SoluteTransportModel_CIP(DynamicModel):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        pcr.setclone(cfg["clone_map"])

    def initial(self):
        self.deltaT = self.cfg["deltaT"]
        self.deltaX = self.cfg["deltaX"]
        self.ldd = pcr.readmap(self.cfg["ldd_map"])
        self.E_nominal = pcr.readmap(self.cfg["pollutant_map"]) / self.deltaX
        self.M = pcr.scalar(0.0)
        self.C = pcr.scalar(0.0)
        self.dCdx = pcr.scalar(0.0)  # spatial gradient of concentration
        self.ocean_release = 0.0
        self.Theoretical_Total = 0.0
        os.makedirs(self.cfg["output_dir"], exist_ok=True)
        self.log_path = os.path.join(self.cfg["output_dir"], "solute_mass_log_CIP.csv")
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(["timestep", "Theoretical_Total", "Model_Total", "Error", "ErrorPct"])
        print("[CIP] Initialized model.")

    def apply_pollutant_release(self, t):
        s, e = self.cfg["release_start"], self.cfg["release_end"]
        return pcr.ifthenelse((t >= s) & (t <= e), self.E_nominal, pcr.scalar(0))

    def dynamic(self):
        t = self.currentTimeStep()
        self.E = self.apply_pollutant_release(t)
        Q_out = self.readmap(self.cfg["flow_map"])
        u = self.cfg["velocity"]
        A = Q_out / u
        self.C = self.M / A

        # Compute spatial gradient (central diff approx)
        dCdx_temp = pcr.upstream(self.ldd, self.C) - self.C
        self.dCdx = dCdx_temp / self.deltaX

        # Cubic interpolation propagation (semi-Lagrangian)
        udt = -u * self.deltaT
        a = ((self.dCdx) + pcr.upstream(self.ldd, self.dCdx)) / (self.deltaX**2) \
            - 2 * (self.C - pcr.upstream(self.ldd, self.C)) / (self.deltaX**3)
        b = (2 * self.dCdx + pcr.upstream(self.ldd, self.dCdx)) / self.deltaX \
            - 3 * (self.C - pcr.upstream(self.ldd, self.C)) / (self.deltaX**2)
        c = self.dCdx
        d = self.C
        C_new = a * udt**3 + b * udt**2 + c * udt + d
        dCdx_new = 3 * a * udt**2 + 2 * b * udt + c

        # Convert back to mass, add emissions
        M_new = C_new * A + self.deltaT * self.E
        self.M = pcr.ifthenelse(M_new < 0, 0, M_new)
        self.C = self.M / A
        self.dCdx = dCdx_new

        # Ocean release and mass balance
        row, col = self.cfg["ocean_cell"]
        self.ocean_release += pcr.cellvalue(Q_out * self.C, row, col)[0] * self.deltaT
        Model_Total = float(pcr.maptotal(self.M) * self.deltaX) + self.ocean_release
        self.Theoretical_Total += float(pcr.maptotal(self.E * self.deltaX)) * self.deltaT
        Err = self.Theoretical_Total - Model_Total
        ErrPct = Err / self.Theoretical_Total * 100 if self.Theoretical_Total else 0
        print(f"[CIP] {t}: T={self.Theoretical_Total:.2f} M={Model_Total:.2f} Err={Err:.2f} ({ErrPct:.2f}%)")
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow([t, self.Theoretical_Total, Model_Total, Err, ErrPct])
            
        self.report(self.M, os.path.join(self.cfg["output_dir"], "M"))
# ===================================================
# Run both simulations
# ===================================================
if __name__ == "__main__":
    print("=== Running Upwind Euler Scheme ===")
    model_up = SoluteTransportModel_1stO_UW_Euler(CONFIG)
    dyn_up = DynamicFramework(model_up, lastTimeStep=CONFIG["nrOfTimeSteps"], firstTimestep=1)
    dyn_up.run()

    print("\n=== Running CIP Scheme ===")
    model_cip = SoluteTransportModel_CIP(CONFIG)
    dyn_cip = DynamicFramework(model_cip, lastTimeStep=CONFIG["nrOfTimeSteps"], firstTimestep=1)
    #dyn_cip.run()
