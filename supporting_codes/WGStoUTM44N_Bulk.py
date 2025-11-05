import os
from qgis.core import (
    QgsApplication,
    QgsCoordinateReferenceSystem,
    QgsProcessingFeedback
)
from qgis.analysis import QgsNativeAlgorithms
import processing
from processing.core.Processing import Processing

# === SETUP QGIS APPLICATION ===
QGIS_PREFIX_PATH = "C:/Program Files/QGIS 3.34.12"  # Correct path to QGIS root
QgsApplication.setPrefixPath(QGIS_PREFIX_PATH, True)
qgs = QgsApplication([], False)
qgs.initQgis()

# === LOAD PROCESSING TOOLS ===
Processing.initialize()
QgsApplication.processingRegistry().addProvider(QgsNativeAlgorithms())

# === CRS SETTINGS ===
source_crs = QgsCoordinateReferenceSystem('EPSG:4326')   # WGS84
target_crs = QgsCoordinateReferenceSystem('EPSG:32644')  # UTM 44N

# === INPUT / OUTPUT FOLDERS ===
input_folder = 'C:/Users/kanch/Research_models/data_2/input_maps/discharge/Discharge_WGS'
output_folder = 'C:/Users/kanch/Research_models/data_2/input_maps/discharge/Discharge_UTM44N'

# === Create feedback object ===
feedback = QgsProcessingFeedback()

# === Loop through file names like pr000000.001 to pr000014.789 ===
for i in range(0, 1453):  # up to pr000014.789 (14 * 1000 + 789)
    int_part = i // 1000
    decimal_part = i % 1000
    file_name = f"pr{int_part:06d}.{decimal_part:03d}.map"
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, f"{file_name}.map")

    print(f"🌀 Reprojecting: {file_name} → {output_path}")

    if not os.path.exists(input_path):
        print(f"⚠️  Skipped: {file_name} not found.")
        continue

    try:
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
        print(f"✅ Done: {file_name}")
    except Exception as e:
        print(f"❌ Failed to convert {file_name}: {e}")

# === Close QGIS app ===
qgs.exitQgis()

# === POST-PROCESS: Remove ".map" extension from output files ===
print("🔧 Removing '.map' extension from output files...")

for file in os.listdir(output_folder):
    if file.endswith(".map"):
        full_path = os.path.join(output_folder, file)
        new_name = os.path.splitext(file)[0]  # removes .map
        new_path = os.path.join(output_folder, new_name)
        os.rename(full_path, new_path)
        print(f"🔁 Renamed: {file} → {new_name}")

print("✅ Extension removal completed.")
