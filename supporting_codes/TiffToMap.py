import os
import numpy as np
from osgeo import gdal
import pcraster

# === INPUTS ===
input_tif = "C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/FilledReshaped.tif"
output_map = "C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/DEM_filled.map"
clone = "C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/ReshapedDEM_UTM100x100.map"

# === Step 1: Read GeoTIFF using GDAL ===
gdal.UseExceptions()
ds = gdal.Open(input_tif)
if ds is None:
    raise FileNotFoundError(f"Cannot open {input_tif}")

array = ds.ReadAsArray().astype(np.float32)

# === Step 2: Set PCRaster clone using your trusted .map ===
pcraster.setclone(clone)

# === Step 3: Handle NoData values ===
nodata_value = ds.GetRasterBand(1).GetNoDataValue()
if nodata_value is None:
    nodata_value = -9999  # fallback
array = np.where(np.isclose(array, nodata_value), pcraster.missingValue, array)

# === Step 4: Convert to PCRaster map and write ===
map_data = pcraster.numpy2pcr(pcraster.Scalar, array, pcraster.missingValue)
pcraster.report(map_data, output_map)

print(f"✅ TIFF converted successfully to: {output_map}")
