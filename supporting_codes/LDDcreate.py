from pcraster import *

setclone("C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/ReshapedDEM_UTM100x100.map")
dem = readmap("C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/ReshapedDEM_UTM100x100.map")

# Fill pits and get hydrologically correct LDD
ldd = lddcreate(dem, 1e31, 1e31, 1e31, 1e31)

# Optionally write it to disk
report(ldd, "filled_LDD.map")
