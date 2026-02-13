import os
import csv
import pcraster as pcr

# ========================
# Define cells to track
# ========================
single_channel_coordinates = [
    (254,116),(254,117),(255,117),(256,116),(257,115),(258,115),(259,114),
    (260,114),(261,115),(262,114),(263,114),(264,114),(265,113),
    (266,113),(266,112),(267,111),(267,110),(268,109),(269,108),
    (270,107),(271,106),(272,106),(271,105),(271,104),(272,103),
    (272,102),(272,101),(272,100),(273,99),(273,98),(273,97),
    (273,96),(274,95),(275,94),(275,93),(275,92),(274,91),
    (273,90),(272,89),(271,88),(270,87)
]

# Define confluence cells with readable names
confluence_cells = {
    "A": (225,69),
    "B": (225,75),
    "A′": (228,70),
    "C": (229,71),
    "B′": (229,72)
}

cells_to_track = single_channel_coordinates #or confluence_cells or any other set of cells

#Defining the out put CSV format

#==== time step - fixed; varying x(cell) value======#
time_step = 1200 #the time step value that should be considered
column_1 = x_value
column_2 = M #M value at given time step

#===== varying timestep; fixed cell value=======#
cell_value = # obtain from the above given set of coordinates


