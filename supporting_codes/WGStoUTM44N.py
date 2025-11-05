import sys
import os

from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsProcessingFeedback
)
import processing
from qgis.analysis import QgsNativeAlgorithms

# === SETUP QGIS APPLICATION ===
QGIS_PATH = "C:/Program Files/QGIS 3.34.12/bin/qgis-ltr-bin.exe"  # ✅ Change to match your QGIS installation
QgsApplication.setPrefixPath(QGIS_PATH, True)
qgs = QgsApplication([], False)
qgs.initQgis()

# === LOAD PROCESSING FRAMEWORK ===
import processing
from processing.core.Processing import Processing
Processing.initialize()

# === LOAD GDAL TOOLS ===
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

# === INPUT/OUTPUT ===
input_path = 'C:/Users/kanch/Research_models/data_2/input_maps/meteo/Precipitation_PCRaster/pr000000.002'
output_path = 'C:/Users/kanch/Research_models/data_2/input_maps/meteo/Precip_UTM44N/pr000000.002.map'

source_crs = QgsCoordinateReferenceSystem('EPSG:4326')   # WGS84
target_crs = QgsCoordinateReferenceSystem('EPSG:32644')  # UTM 44N

# === Create feedback (optional) ===
feedback = QgsProcessingFeedback()

# === Run the reprojection ===
processing.run(
    "gdal:warpreproject",
    {
        'INPUT': input_path,
        'SOURCE_CRS': source_crs,
        'TARGET_CRS': target_crs,
        'RESAMPLING': 1,  # Bilinear
        'NODATA': None,
        'TARGET_RESOLUTION': None,
        'OPTIONS': None,
        'DATA_TYPE': 0,
        'TARGET_EXTENT': None,
        'TARGET_EXTENT_CRS': None,
        'MULTITHREADING': False,
        'EXTRA': '-co "PCRASTER_VALUESCALE=VS_SCALAR"',
        'OUTPUT': output_path
    },
    feedback=feedback
)

# === Close the QGIS app ===
qgs.exitQgis()
