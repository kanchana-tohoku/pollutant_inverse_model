import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pcraster as pcr
from pcraster.framework import DynamicModel, DynamicFramework


# ========================
# Configuration
# ========================
#case1 = Steady Flow + random variation
case1 = {
    # Paths
    "output_dir": "C:/Users/kanch/Research_models/data_2/out_ADErev6/Tial",
    "clone_map": "C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/pcr_dem.map",
    "ldd_map": "C:/Users/kanch/Research_models/data_2/input_maps/topography/LDD/WGS_LDD.map",
    "pollutant_map": "C:/Users/kanch/Research_models/data_2/input_maps/pollutants/WGS/pollution_source_WGS_200points.map",
    "flow_map": "C:/Users/kanch/Research_models/data_2/input_maps_synthetic/discharge/Discharge/steady",
    "noise" : 1,
    "log_file": "solute_mass_log.csv",

    # Simulation settings
    "deltaT": 1,           # time step [s]
    "deltaX": 100,         # cell length [m]
    "velocity": 1.8,       # assumed constant flow velocity [m/s]
    "release_start": 1,    # pollutant release start step
    "release_end": 2,      # pollutant release end step
    "nrOfTimeSteps": 2000,  # total simulation time steps

    # Ocean release point (cell coordinates)
    "ocean_cell": (278, 18),   # (row, col) where mass leaves to ocean
}

CONFIG = case1

# ========================
# Model
# ========================
class SoluteTransportModel_1stO_UW_Euler(DynamicModel):# 1st order Up-wind scheme explicit Euler
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        pcr.setclone(cfg["clone_map"])

    def initial(self):
        self.deltaT = self.cfg["deltaT"]
        self.deltaX = self.cfg["deltaX"]

        # Load maps
        self.ldd = pcr.readmap(self.cfg["ldd_map"])
        self.E_nominal = pcr.readmap(self.cfg["pollutant_map"]) / self.deltaX
        #self.Q_out = pcr.readmap(self.cfg["flow_map"]) + pcr.scalar(1e-7)

        # State variables
        self.M = pcr.scalar(0.0)
        self.C = pcr.scalar(0.0)
        self.ocean_release = 0.0
        self.Theoretical_Total = 0.0

        # Prepare output directory
        os.makedirs(self.cfg["output_dir"], exist_ok=True)
        self.log_path = os.path.join(self.cfg["output_dir"], self.cfg["log_file"])

        # Initialize log file
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "Theoretical_Total", "Model_Total", "Error", "ErrorPct"])

        # Report initial flux path (optional check)
        fluxpath = pcr.accuflux(self.ldd, self.E_nominal)
        pcr.report(fluxpath, os.path.join(self.cfg["output_dir"], "fluxpath"))

        # Mass balance initial parameters
        total_init = float(pcr.maptotal(self.E_nominal))
        print(f"Total initial pollution = {total_init}")
        
        # Plotting initial parameters
        
    def apply_pollutant_release(self, t):
        start, end = self.cfg["release_start"], self.cfg["release_end"]
        return pcr.ifthenelse((t >= start) & (t <= end), self.E_nominal, pcr.scalar(0))

    def dynamic(self):
        t = self.currentTimeStep()

        # Pollutant release control
        self.E = self.apply_pollutant_release(t)

        # Flow
        #Q_out = self.Q_out + self.Q_out * pcr.mapnormal() * self.cfg["noise"]
        #Q_out = pcr.ifthenelse(Q_out <= 0, 1e-9, Q_out)
        #self.report(Q_out, os.path.join(self.cfg["output_dir"], "Q"))
        
        #Dynamic Flow
        Q_out = self.readmap("C:/Users/kanch/Research_models/data_2/input_maps_synthetic/discharge/Discharge/Q")
        
        # Cross-sectional area
        velocity = self.cfg["velocity"]
        AvgCrossSec = Q_out / velocity

        # Update concentration
        self.C = self.M / AvgCrossSec

        # Fluxes (simplified upwind scheme)
        FluxOut = Q_out * self.C
        FluxIn = pcr.upstream(self.ldd, FluxOut)

        # Update pollutant mass
        M_new = self.M - (self.deltaT / self.deltaX) * (FluxOut - FluxIn) + self.deltaT * self.E
        self.M = pcr.ifthenelse(M_new < 0, 0, M_new)
        self.C = self.M / AvgCrossSec
        self.report(self.M, os.path.join(self.cfg["output_dir"], "M"))
        self.report(self.C, os.path.join(self.cfg["output_dir"], "C"))
        
        # Ocean release
        row, col = self.cfg["ocean_cell"]
        self.ocean_release += pcr.cellvalue(FluxOut, row, col)[0] * self.deltaT

        # Mass balance
        Model_Total = float(pcr.maptotal(self.M) * self.deltaX) + self.ocean_release
        self.Theoretical_Total += float(pcr.maptotal(self.E * self.deltaX)) * self.deltaT

        Error = self.Theoretical_Total - Model_Total
        ErrorPct = Error / self.Theoretical_Total * 100 if self.Theoretical_Total else 0

        print(f"{t}: T={self.Theoretical_Total:.2f} M={Model_Total:.2f} "
              f"Error={Error:.2f} ({ErrorPct:.2f}%) Ocean={self.ocean_release:.3f}")

        # Log results
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([t, self.Theoretical_Total, Model_Total, Error, ErrorPct])


class SoluteTransportModel_CIP_SemiLag(DynamicModel):# CIP (Cubic Interpolated Propagation) Semi-Lagrangian method
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        pcr.setclone(cfg["clone_map"])

    def initial(self):
        self.deltaT = self.cfg["deltaT"]
        self.deltaX = self.cfg["deltaX"]

        # Load maps
        self.ldd = pcr.readmap(self.cfg["ldd_map"])
        self.E_nominal = pcr.readmap(self.cfg["pollutant_map"]) / self.deltaX
        #self.Q_out = pcr.readmap(self.cfg["flow_map"]) + pcr.scalar(1e-7)

        # State variables
        self.M = pcr.scalar(0.0)
        self.C = pcr.scalar(0.0)
        self.ocean_release = 0.0
        self.Theoretical_Total = 0.0

        # Prepare output directory
        os.makedirs(self.cfg["output_dir"], exist_ok=True)
        self.log_path = os.path.join(self.cfg["output_dir"], self.cfg["log_file"])

        # Initialize log file
        with open(self.log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestep", "Theoretical_Total", "Model_Total", "Error", "ErrorPct"])

        # Report initial flux path (optional check)
        fluxpath = pcr.accuflux(self.ldd, self.E_nominal)
        pcr.report(fluxpath, os.path.join(self.cfg["output_dir"], "fluxpath"))

        # Mass balance initial parameters
        total_init = float(pcr.maptotal(self.E_nominal))
        print(f"Total initial pollution = {total_init}")
        
        # Plotting initial parameters
        
    def apply_pollutant_release(self, t):
        start, end = self.cfg["release_start"], self.cfg["release_end"]
        return pcr.ifthenelse((t >= start) & (t <= end), self.E_nominal, pcr.scalar(0))

    def dynamic(self):
        t = self.currentTimeStep()

        # Pollutant release control
        self.E = self.apply_pollutant_release(t)

        # Flow
        #Q_out = self.Q_out + self.Q_out * pcr.mapnormal() * self.cfg["noise"]
        #Q_out = pcr.ifthenelse(Q_out <= 0, 1e-9, Q_out)
        #self.report(Q_out, os.path.join(self.cfg["output_dir"], "Q"))
        
        #Dynamic Flow
        Q_out = self.readmap("C:/Users/kanch/Research_models/data_2/input_maps_synthetic/discharge/Discharge/Q")
        
        # Cross-sectional area
        velocity = self.cfg["velocity"]
        AvgCrossSec = Q_out / velocity

        # Update concentration
        self.C = self.M / AvgCrossSec

        # Fluxes (simplified upwind scheme)
        FluxOut = Q_out * self.C
        FluxIn = pcr.upstream(self.ldd, FluxOut)

        # Update pollutant mass
        M_new = self.M - (self.deltaT / self.deltaX) * (FluxOut - FluxIn) + self.deltaT * self.E
        self.M = pcr.ifthenelse(M_new < 0, 0, M_new)
        self.C = self.M / AvgCrossSec
        self.report(self.M, os.path.join(self.cfg["output_dir"], "M"))
        self.report(self.C, os.path.join(self.cfg["output_dir"], "C"))
        
        # Ocean release
        row, col = self.cfg["ocean_cell"]
        self.ocean_release += pcr.cellvalue(FluxOut, row, col)[0] * self.deltaT

        # Mass balance
        Model_Total = float(pcr.maptotal(self.M) * self.deltaX) + self.ocean_release
        self.Theoretical_Total += float(pcr.maptotal(self.E * self.deltaX)) * self.deltaT

        Error = self.Theoretical_Total - Model_Total
        ErrorPct = Error / self.Theoretical_Total * 100 if self.Theoretical_Total else 0

        print(f"{t}: T={self.Theoretical_Total:.2f} M={Model_Total:.2f} "
              f"Error={Error:.2f} ({ErrorPct:.2f}%) Ocean={self.ocean_release:.3f}")

        # Log results
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([t, self.Theoretical_Total, Model_Total, Error, ErrorPct])


# ========================
# Run Simulation
# ========================
if __name__ == "__main__":
    model1 = SoluteTransportModel_1stO_UW_Euler(CONFIG)
    dynamic_model1 = DynamicFramework(model1,
                                     lastTimeStep=CONFIG["nrOfTimeSteps"],
                                     firstTimestep=1)
    
    model2 = SoluteTransportModel_CIP_SemiLag(CONFIG)
    dynamic_model2 = DynamicFramework(model2,
                                     lastTimeStep=CONFIG["nrOfTimeSteps"],
                                     firstTimestep=1)
    
    dynamic_model1.run() # 1st order Up-wind scheme explicit Euler
    #dynamic_model2.run() # CIP (Cubic Interpolated Propagation) Semi-Lagrangian method

# ======================
# Plotting output
# ======================

# ========================
# Comments on the results
# ========================
# This model asuumes instance mixing
# Errors related to First-order upwind scheme occurs