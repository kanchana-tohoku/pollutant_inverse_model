import os
import pcraster as pcr

# ==============================
# Settings
# ==============================
case = 2

clone_map = "C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/pcr_dem.map"
ldd_map = "C:/Users/kanch/Research_models/data_2/input_maps/topography/LDD/WGS_LDD.map"
source_map = f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}/new_poll_map"

output_dir = f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}"
os.makedirs(output_dir, exist_ok=True)

pcr.setclone(clone_map)

# ==============================
# Read maps
# ==============================
ldd = pcr.readmap(ldd_map)
source = pcr.readmap(source_map) / 0.238

# Convert to boolean source
#source = pcr.boolean(source_raw > 0)

# ==============================
# Initialize
# ==============================

order_map = pcr.scalar(0)


print("Starting fast downstream traversal...")

counter = 0
downstream_cell = pcr.upstream(ldd, source)

while counter < 150:

    # Move one step upstream and accumulate
    downstream_cell = pcr.upstream(ldd, downstream_cell) + downstream_cell

    # Boundary treatment (remove negatives / MV)
    #downstream_cell = pcr.ifthenelse(downstream_cell > 0, source, 0)

    counter += 1
    print(counter)



print(f"Finished. Total downstream steps: {counter}")

# ==============================
# Save result
# ==============================
output_path = os.path.join(output_dir, "downstream_sequential.map")
pcr.report(downstream_cell, output_path)

print(f"✅ Saved to:\n{output_path}")