from pcraster import *

# Set up environment
setclone("C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/ReshapedDEM_UTM100x100.map")  # Use the mask or DEM for spatial reference

# Read your DEM
dem = readmap("C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/ReshapedDEM_UTM100x100.map")

# Recreate LDD with sink-filling
ldd_fixed = lddcreate(
    dem,         # DEM
    10000,       # slope length (max flow path length allowed)
    0.01,        # slope threshold (minimum slope)
    1e20,        # threshold for defining pits (use high value to avoid real pits being excluded)
    True         # True = fill pits
)

report(ldd_fixed, "fixed_ldd.map")
