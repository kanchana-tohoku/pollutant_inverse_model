from pcraster import *
from pcraster.framework import *

class OverlandFlowModel(DynamicModel):
    def __init__(self):
        # Set the clone map here before initializing the parent class
        setclone("C:/Users/kanch/Research_models/data_2/input_maps/general_details/PCRaster/KRB_mask.map")
        DynamicModel.__init__(self)

    def initial(self):
        # Read Digital Elevation Model (DEM)
        self.dem = readmap("C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/ReshapedDEM_UTM100x100.map")

        # Compute Local Drain Direction (LDD)
        self.ldd = lddcreate(self.dem, 1e31, 1e31, 1e31, 1e31)

        # Initial discharge is zero everywhere
        self.discharge = scalar(0)
        
    def dynamic(self):
        # Get the current timestep
        timestep = self.currentTimeStep()

        # Read dynamic rainfall input for the current timestep
        #self.rainfall = readmap(f"C:\\Users\\kanch\\Research_models\\rainfall_{timestep}.map")
        self.rainfall = scalar(.80)
        
        # Compute accumulated flow using `accuflux`
        self.discharge = self.discharge + accuflux(self.ldd, self.rainfall)

        # Save discharge output for the current timestep
        report(self.discharge, f"C:\\Users\\kanch\\Research_models\\DischargeMaps\\discharge_{timestep}.map")

# Run the model for 10 timesteps
nrOfTimeSteps = 50
model = OverlandFlowModel()
dynamicModel = DynamicFramework(model, nrOfTimeSteps)
dynamicModel.run()