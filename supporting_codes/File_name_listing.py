import os

def list_folders(path, indent=0):
    # Get all files and directories in the current path
    try:
        items = sorted(os.listdir(path))
    except PermissionError:
        print(" " * indent + "[Permission Denied]")
        return

    for item in items:
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):   # ✅ Only process directories
            print(" " * indent + f"-{item}/")
            list_folders(item_path, indent + 4)

# Set your target path here
target_path = "C:/Users/kanch/Research_models/data_2/input_maps"
print(f"{os.path.basename(target_path)}/")
list_folders(target_path, indent=4)

