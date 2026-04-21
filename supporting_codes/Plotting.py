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
case = 1

CONFIG = {
    "output_dir": f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}/TimeReversal_NoReg_2",
    "graph_output_dir": f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}/TimeReversal_NoReg_2",
    "nrOfTimeSteps": 1999,     # total time steps
    "timeStepInterval": 1999,   # interval to sample maps
    "clone_map": "C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/pcr_dem.map"
}

# ========================
# Define cells to track
# ========================
cells_to_track = [
(464,224),(464,225),(464,226),(463,227),(463,228),(463,229),
(464,230),(464,231),(465,232),(465,233),(465,234),(465,235),(465,236),(464,237),
(463,238),(462,239),(462,240),(462,241),(461,242),(459,243),(460,243),(459,244),
(459,245),(460,246),(460,247),(459,248),(458,248),(457,248),(456,249),(455,249),
(454,248),(453,249),(452,249),(451,249),(450,250),(449,251),(448,252),(447,252),
(446,252),(445,253),(444,253),(443,253),(442,253),(441,253),(440,253),(439,253),
(438,253),(437,253),(436,252),(435,252),(434,251),(433,250),(432,250),(431,250),
(430,250),(429,249),(428,248),(427,248),(426,247),(425,247),(424,247),(423,246),
(422,246),(421,246),(420,245),(419,245),(418,246),(417,246),(416,246),(415,246),
(414,246),(413,245),(412,244),(411,243),(410,242),(409,243),(408,244),(407,244),
(406,243),(405,242),(404,242),(403,243),(402,244),(401,243),(400,243),(399,242),
(398,242),(397,242),(396,242),(395,241),(394,240),(393,239),(392,238),(391,238),
(390,238),(389,238),(388,239),(387,239),(386,239),(385,239),(384,239),(383,238),
(382,237),(381,236),(380,236),(379,235),(378,234),(377,234),(376,235),(375,234),
(374,233),(373,233),(372,234),(371,235),(370,235),(369,234),(368,234),(367,234),
]

# Define confluence cells with readable names
confluence_cells = {
    "A": (464,224),
    "B": (464,225),
    "A′": (464,226),
    "C": (463,227),
    "B′": (463,229)
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

        M_map = self.readmap(os.path.join(self.config["output_dir"], "MI"))
        self.timesteps.append(t)

        # --- Track main channel cells ---
        for (r, c) in self.main_cells:
            try:
                val = pcr.cellvalue(M_map, r+1, c+1)[0]
            except Exception:
                val = np.nan
            self.data_main[f"({r},{c})"].append(val)

        # --- Track confluence cells ---
        for (r, c) in self.confl_cells.values():

            try:
                val = pcr.cellvalue(M_map, r+1, c+1)[0]
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

        plt.xlabel("Distance along channel (m)", fontsize=24)
        plt.ylabel("Mass per length (mg/m)", fontsize=24)
        plt.ylim(0, 0.0025)   # <-- set your desired max value here
        plt.title("Pollutant Mass per length along the main channel", fontsize=24)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(fontsize=24, ncol=2)
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
