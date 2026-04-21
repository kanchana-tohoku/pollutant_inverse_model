import numpy as np
import pandas as pd
import pcraster as pcr
import os

# --------------------------------------------------
# USER INPUT
# --------------------------------------------------
map_path = r"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_2/fluxpath"

row_min, row_max = 350, 470
col_min, col_max = 200, 300

output_dir = r"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_2"
output_csv = os.path.join(output_dir, "extracted_cells.csv")

# --------------------------------------------------
# READ MAP
# --------------------------------------------------
raster = pcr.readmap(map_path)
array = pcr.pcr2numpy(raster, np.nan)

nrows, ncols = array.shape
print("Raster size:", nrows, ncols)

# --------------------------------------------------
# EXTRACT DOMAIN
# --------------------------------------------------
records = []

for i in range(row_min, row_max + 1):
    for j in range(col_min, col_max + 1):
        
        if 0 <= i < nrows and 0 <= j < ncols:
            
            value = array[i, j]
            
            if not np.isnan(value) and value != 0:
                records.append([value, i, j])

# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------
df = pd.DataFrame(records, columns=["value", "row", "column"])

df.to_csv(output_csv, index=False)

print("Extraction complete.")
print("Total cells found:", len(df))
print("Saved to:", output_csv)