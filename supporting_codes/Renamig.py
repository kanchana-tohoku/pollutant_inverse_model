import os

# Target directory
folder_path = r"C:\Users\kanch\Research_models\data_2\out_ADErev6\case_2_analytical\Zn"

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)
    
    # Skip if not a file
    if not os.path.isfile(old_path):
        continue
    
    # Ensure filename is long enough
    if len(filename) < 4:
        continue
    
    # Remove 2nd to 4th characters (index 1 to 3)
    d = "_Cd"
    new_filename = filename[0:4] + filename[7:]
    
    new_path = os.path.join(folder_path, new_filename)
    
    # Rename file
    os.rename(old_path, new_path)
    
    print(f"Renamed: {filename} -> {new_filename}")

print("Done!")