

#======== Model Selection==========
Running_Upwind_Euler_Scheme  = 1
CIP_Scheme  = 0

#===== Model Parameters ==============
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
}



#=========Plotting configuration=========
CONFIG = {
    "output_dir": "C:/Users/kanch/Research_models/data_2/out_ADErev6/Trial",
    "graph_output_dir": "C:/Users/kanch/Research_models/data_2/out_ADErev6/Graphs",
    "nrOfTimeSteps": 100,     # total time steps
    "timeStepInterval": 10,   # interval to sample maps
    "clone_map": "C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/pcr_dem.map"
}

