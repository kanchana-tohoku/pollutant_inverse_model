import numpy as np

# Define the start and end coordinates
start_row, start_col = 218, 45
end_row, end_col = 279, 120

# Create arrays of row and column indices
rows = np.arange(start_row, end_row + 1)
cols = np.arange(start_col, end_col + 1)

# Create a grid of all row-col pairs
row_grid, col_grid = np.meshgrid(rows, cols, indexing='ij')

# Stack them as (row, col) pairs
cell_indices = np.column_stack((row_grid.ravel(), col_grid.ravel()))

# Save as .npy file
np.save("C:/Users/kanch/Research_models/data_2/out_ADErev6/clipped/cell_indices.npy", cell_indices)

# Optional: save as CSV too
np.savetxt("cell_indices.csv", cell_indices, delimiter=",", fmt="%d")

print("Saved as 'cell_indices.npy' and 'cell_indices.csv'.")
