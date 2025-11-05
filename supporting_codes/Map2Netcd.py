import os
import glob
import numpy as np
import xarray as xr
import pcraster as pcr

# Define input directory containing .map files and output file
input_dir = r"C:\Users\kanch\Research_models\DischargeMaps"  # Make sure this path is correct
output_nc = r"C:\Users\kanch\Research_models\DischargeMaps\output.nc"

# Get list of .map files
map_files = sorted(glob.glob(os.path.join(input_dir, "*.map")))

# Check if any .map files exist
if not map_files:
    raise FileNotFoundError(f"No .map files found in {input_dir}")

# Read first file to get dimensions
first_map = pcr.readmap(map_files[0])
first_array = pcr.pcr2numpy(first_map, -9999)  # Convert to NumPy
rows, cols = first_array.shape  # Get shape

# Create an empty data array
data_stack = np.zeros((len(map_files), rows, cols))

# Read all .map files
for i, file in enumerate(map_files):
    data_stack[i, :, :] = pcr.pcr2numpy(pcr.readmap(file), -9999)  # Using -9999 as NoData

# Create xarray DataArray
time_dim = np.arange(len(map_files))  # Dummy time index
da = xr.DataArray(data_stack, dims=("time", "y", "x"), coords={"time": time_dim})

# Convert to Dataset and save as NetCDF
ds = xr.Dataset({"data": da})
ds.to_netcdf(output_nc)

print(f"NetCDF file saved: {output_nc}")
