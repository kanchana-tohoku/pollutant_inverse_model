import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pcraster as pcr
from pcraster.framework import DynamicModel, DynamicFramework

# ========================
# Configuration
# ========================
CONFIG = {
    "output_dir": "C:/Users/kanch/Research_models/data_2/out_ADErev6/",
    "graph_output_dir": "C:/Users/kanch/Research_models/data_2/out_ADErev6/Graphs",
    "nrOfTimeSteps": 2000,     # total time steps
    "timeStepInterval": 200,   # interval to sample maps
    "clone_map": "C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/pcr_dem.map"
}

# ========================
# Define cells to track
# ========================
cells_to_track = [
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


# Synthetic distance-like x-labels
distance_labels = np.arange(-100, -100 + len(cells_to_track) * 100, 100)

# ========================
# Dynamic Model Definition
# ========================
class TrackConcentration(DynamicModel):
    def __init__(self, config, main_cells, confl_cells, dist_labels):
        DynamicModel.__init__(self)
        self.config = config
        self.main_cells = main_cells
        self.confl_cells = confl_cells
        self.dist_labels = dist_labels

        # Dictionaries to hold results
        self.data_main = {f"({r},{c})": [] for (r, c) in self.main_cells}
        self.data_confl = {f"({r},{c})": [] for (r, c) in self.confl_cells.values()}

        self.timesteps = []

        # Set clone
        pcr.setclone(self.config["clone_map"])

    def initial(self):
        print("Starting dynamic tracking...")

    def dynamic(self):
        t = self.currentTimeStep()

        # Process only t=1 and every interval (150, 300, 450, ...)
        if (t != 1) and (t % self.config["timeStepInterval"] != 0):
            return

        M_map = self.readmap(os.path.join(self.config["output_dir"], "M"))
        self.timesteps.append(t)

        # --- Track main channel cells ---
        for (r, c) in self.main_cells:
            try:
                val = pcr.cellvalue(M_map, r, c)[0]
            except Exception:
                val = np.nan
            self.data_main[f"({r},{c})"].append(val)

        # --- Track confluence cells ---
        for (r, c) in self.confl_cells.values():

            try:
                val = pcr.cellvalue(M_map, r, c)[0]
            except Exception:
                val = np.nan
            self.data_confl[f"({r},{c})"].append(val)

    def postprocess_Single_channel(self):
        graph_output_dir = self.config["graph_output_dir"]
        os.makedirs(graph_output_dir, exist_ok=True)

        # ---- Save and Plot: Main Channel ----
        df_main = pd.DataFrame(self.data_main, index=self.timesteps)
        df_main.to_csv(os.path.join(graph_output_dir, "tracked_cells_Masses.csv"))
        print(f"\n✅ Saved main channel cell data")

        df_main_T = df_main.T
        x = self.dist_labels

        plt.figure(figsize=(10, 6))
        for t in df_main_T.columns:
            plt.plot(x, df_main_T[t], label=f"t={t}", linewidth=1.5)

        plt.xlabel("Distance along channel (m)", fontsize=14)
        plt.ylabel("Mass per length (mg/m)", fontsize=14)
        plt.title("Pollutant Mass per length along the main channel for Cr", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14, ncol=3)
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()
 
    def postprocess_confluence(self):
        graph_output_dir = self.config["graph_output_dir"]
        os.makedirs(graph_output_dir, exist_ok=True)        
        
        # ---- Save and Plot: Confluence Cells ----
        df_confl = pd.DataFrame(self.data_confl, index=self.timesteps)
        df_confl.to_csv(os.path.join(graph_output_dir, "confluence_cells_Masses.csv"))
        print(f"✅ Saved confluence cell data")

        plt.figure(figsize=(8, 8))

        # Use the name labels from the dictionary
        for label, (r, c) in self.confl_cells.items():
            col = f"({r},{c})"
            if col in df_confl.columns:
                plt.plot(df_confl.index, df_confl[col], label=f"{label}", linewidth=2)

        plt.xlabel("Time step", fontsize=14)
        plt.ylabel("Mass per length (mg/m)", fontsize=14)
        plt.title("Pollutant Mass per length at confluence cells", fontsize=14)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.legend(fontsize=14, ncol=2, title="Confluence cells")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.show()

        # Log M values for all confluence cells
        log_file = os.path.join(graph_output_dir, "confluence_cells_log.csv")
        
        # Write header only once (first run)
        write_header = not os.path.exists(log_file)
        
        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["TimeStep"] + list(self.confl_cells.keys()))
        
            for i, t in enumerate(self.timesteps):
                row = [t]
                for label, (r, c) in self.confl_cells.items():
                    val_list = self.data_confl.get(f"({r},{c})", [])
                    val = val_list[i] if i < len(val_list) else np.nan
                    row.append(val)
                writer.writerow(row)



# ========================
# Run Framework
# ========================
if __name__ == "__main__":
    model = TrackConcentration(CONFIG, cells_to_track, confluence_cells, distance_labels)
    dynamic_model = DynamicFramework(model,
                                     lastTimeStep=CONFIG["nrOfTimeSteps"],
                                     firstTimestep=1)
    dynamic_model.run()
    model.postprocess_Single_channel()
    model.postprocess_confluence()
