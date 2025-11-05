import os

# Path to your folder
folder = r"C:\Users\kanch\Research_models\data_2\input_maps\discharge\Discharge_WGS"

# Loop through all files in the folder
for filename in os.listdir(folder):
    if filename.endswith(".map"):
        old_path = os.path.join(folder, filename)
        new_filename = filename[:-4]  # Remove last 4 characters (".map")
        new_path = os.path.join(folder, new_filename)
        os.rename(old_path, new_path)
        print(f"Renamed: {filename} → {new_filename}")
