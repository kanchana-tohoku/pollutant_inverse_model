import processing

# -------------------------------
# Paths
# -------------------------------
shapefile = r'C:/Users/kanch/Research_models/data_2/input_maps/pollutants/WGS/pollutants_at_waterways_1.shp'
rasterized_map = r'C:/Users/kanch/Research_models/data_2/input_maps/pollutants/pollutions_at_waterway_1.map'
zero_map = r'C:/Users/kanch/Research_models/data_2/input_maps/general_details/PCRaster/scalar0.map'
final_covered_map = r'C:/Users/kanch/Research_models/data_2/input_maps/pollutants/WGS/pollution_source_WGS_200points.map'

# -------------------------------
# Step 1: Rasterize shapefile using GDAL
# -------------------------------
processing.run("gdal:rasterize", {
    'INPUT': shapefile,
    'FIELD': 'pol_dis_kg',
    'BURN': 0,
    'USE_Z': False,
    'UNITS': 1,  # Pixels in map units
    'WIDTH': 0.0009051204872720414,
    'HEIGHT': 0.0009051204872720414,
    'EXTENT': '79.855359957,80.778582854,6.744777086,7.230826788 [EPSG:4326]',
    'NODATA': 0,
    'OPTIONS': None,
    'DATA_TYPE': 5,  # Float32
    'INIT': None,
    'INVERT': False,
    'EXTRA': '-co PCRASTER_VALUESCALE=VS_SCALAR\n',
    'OUTPUT': rasterized_map
})

# -------------------------------
# Step 2: Apply PCRaster's cover operator to fill no-data with zero
# -------------------------------
processing.run("pcraster:cover", {
    'INPUT': rasterized_map,
    'INPUT2': [zero_map],
    'OUTPUT': final_covered_map
})

print("✅ Rasterization and cover operation completed successfully.")
