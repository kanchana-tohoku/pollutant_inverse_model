#processing.run("pcraster:resample", {'INPUT':['C:/Users/kanch/Research_models/data_2/input_maps/meteo/Precip_UTM44N/pr000000.001.map'],'INPUT1':'C:/Users/kanch/Research_models/data_2/input_maps/general_details/PCRaster/KRB_mask.map','OUTPUT':'C:/Users/kanch/Research_models/data_2/input_maps/meteo/Precip_100x100_UTM44N/pr000000001.map'})
import os
from qgis.core import (
    QgsApplication,
    QgsProcessingFeedback,
)
from qgis.analysis import QgsNativeAlgorithms
import processing
from processing.core.Processing import Processing

# === QGIS SETUP ===
QGIS_PREFIX_PATH = "C:/Program Files/QGIS 3.34.12/bin/qgis-ltr-bin.exe"  # Adjust this if your QGIS is installed elsewhere
QgsApplication.setPrefixPath(QGIS_PREFIX_PATH, True)
qgs = QgsApplication([], False)
qgs.initQgis()

# === INITIALIZE PROCESSING ===
Processing.initialize()
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

# === INPUT/OUTPUT PATHS ===
input_map = "C:/Users/kanch/Research_models/data_2/input_maps/meteo/Precip_UTM44N/pr000000.001.map"
clone_map = "C:/Users/kanch/Research_models/data_2/input_maps/general_details/PCRaster/KRB_mask.map"
output_map = "C:/Users/kanch/Research_models/data_2/input_maps/meteo/Precip_100x100_UTM44N/pr000000001.map"

# === FEEDBACK OBJECT ===
feedback = QgsProcessingFeedback()

# === PARAMETERS ===
params = {
    'INPUT': [input_map],
    'INPUT1': clone_map,
    'OUTPUT': output_map
}

# === RUN RESAMPLE ===
try:
    print(f"🌀 Resampling: {os.path.basename(input_map)}")
    processing.run("pcraster:resample", params, feedback=feedback)
    print(f"✅ Output saved to: {output_map}")
except Exception as e:
    print(f"❌ Error: {e}")

# === CLEAN UP ===
qgs.exitQgis()
