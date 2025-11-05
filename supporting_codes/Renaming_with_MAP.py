import os

# === Set your folder path here ===
folder_path = 'C:/Users/kanch/Research_models/data_2/input_maps/discharge/Discharge_WGS'

# === Loop through all files in the folder ===
for file in os.listdir(folder_path):
    if not file.endswith('.map'):
        old_path = os.path.join(folder_path, file)
        new_path = os.path.join(folder_path, file + '.map')
        os.rename(old_path, new_path)
        print(f"🔁 Renamed: {file} → {file}.map")

print("✅ All non-.map files have been renamed.")
