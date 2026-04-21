"""
=============================================================================
ADE_FVD_PCR_multimetal.py
=============================================================================
Multi-metal forward pollutant transport model.

Extends ADE_FVD_PCR.py to simulate each heavy metal species independently,
using a pollutant signature CSV to decompose total source mass into
per-metal emission maps.

KEY IDEA
--------
Conservative transport is linear:
    M_metal(t+1) = M_metal(t) - (dT/dX)*(FluxOut - FluxIn) + dT * E_metal
Each metal sees the same Q(x,t) and LDD, only E_metal differs.
→ Run one transport instance per metal, same framework.

NEW INPUT: Signature CSV
------------------------
Required columns:
    source_id, row, col, total_mass, Cd, Cr, Pb, Cu, Zn
    (or any metal column names — model reads them dynamically)

    - source_id : unique identifier for the source cell
    - row, col  : 1-based PCRaster cell coordinates
    - total_mass: total pollutant mass released by this source [mg]
    - Cd … Zn   : fraction of total_mass for each metal (each row sums to 1.0)

Example row:
    SRC_01, 45, 12, 500.0, 0.16, 0.21, 0.18, 0.11, 0.34

OUTPUTS (per metal, per timestep)
----------------------------------
    <output_dir>/<METAL>/M_<metal>XXXXXXX.XXX   — mass map [mg/m]
    <output_dir>/<METAL>/conc_<metal>XXXXXX.XXX  — concentration map [mg/L]
    <output_dir>/mass_balance_log.csv             — timestep mass balance for all metals
    <output_dir>/source_metal_maps/               — initial E maps per metal (diagnostic)

=============================================================================
"""

import os
import csv
import pcraster as pcr
from pcraster.framework import DynamicModel, DynamicFramework
import numpy as np

# =============================================================================
# CONFIGURATION
# =============================================================================

case = 1

CONFIG = {
    # --- output ---
    "output_dir":   f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}_multimetal",

    # --- base maps ---
    "clone_map":    "C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/pcr_dem.map",
    "ldd_map":      "C:/Users/kanch/Research_models/data_2/input_maps/topography/LDD/WGS_LDD.map",
    "pollutant_map":"C:/Users/kanch/Research_models/data_2/input_maps/pollutants/WGS/pollution_source_WGS_200points.map",

    # --- discharge time series prefix ---
    "flow_map":     "C:/Users/kanch/Research_models/data_2/input_maps_synthetic/discharge/Discharge/Q",

    # --- NEW: signature CSV (replaces old replacement_csv) ---
    # Must contain: source_id, row, col, total_mass, <metal1>, <metal2>, ...
    "signature_csv": f"C:/Users/kanch/Research_models/data_2/input_maps/pollutants/WGS/case{case}/source_signatures.csv",

    # --- numerics ---
    "deltaT":         1,      # seconds
    "deltaX":         100,    # metres
    "velocity":       1.8,    # m/s  (A = Q / velocity)
    "release_start":  1,      # first timestep of pollutant release
    "release_end":    2,      # last  timestep of pollutant release
    "nrOfTimeSteps":  2000,

    # --- ocean/outlet cell (1-based row, col) ---
    "ocean_cell": (278, 18),

    # --- reporting options ---
    "report_concentration": True,   # also save C maps (mg/L) alongside M maps
}


# =============================================================================
# SIGNATURE CSV LOADER
# =============================================================================

def load_signature_csv(csv_path):
    """
    Read the signature CSV and return:
        sources  : list of dicts, one per source row
        metals   : list of metal column names (e.g. ['Cd','Cr','Pb','Cu','Zn'])

    Each dict in `sources` has keys:
        'source_id', 'row', 'col', 'total_mass', and one key per metal.

    Validation
    ----------
    - Fraction columns must sum to 1.0 per row (tolerance 1e-3).
    - Warns if any fraction is negative.
    """
    sources = []
    metals  = []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames

        # Everything after 'total_mass' is treated as a metal column
        fixed_cols = {"source_id", "row", "col", "total_mass"}
        metals = [c for c in fieldnames if c not in fixed_cols]

        if not metals:
            raise ValueError(
                "Signature CSV has no metal columns. "
                "Expected columns beyond: source_id, row, col, total_mass."
            )

        for i, row in enumerate(reader, start=2):   # i=2 → first data row
            source = {
                "source_id":  row["source_id"],
                "row":        int(row["row"]),
                "col":        int(row["col"]),
                "total_mass": float(row["total_mass"]),
            }
            frac_sum = 0.0
            for m in metals:
                frac = float(row[m])
                if frac < 0:
                    print(f"  [WARNING] Negative fraction for metal '{m}' "
                          f"at source {row['source_id']} (CSV row {i}) → set to 0")
                    frac = 0.0
                source[m] = frac
                frac_sum += frac

            # Normalise if fractions don't quite sum to 1
            if abs(frac_sum - 1.0) > 1e-3:
                print(f"  [WARNING] Source {row['source_id']} fractions sum to "
                      f"{frac_sum:.4f} ≠ 1.0 → normalising")
                for m in metals:
                    source[m] /= frac_sum

            sources.append(source)

    print(f"[Signatures] Loaded {len(sources)} sources, "
          f"{len(metals)} metals: {metals}")
    return sources, metals


# =============================================================================
# BUILD PER-METAL EMISSION MAPS
# =============================================================================

def build_metal_emission_maps(sources, metals, base_poll_map, deltaX, output_dir):
    """
    For each metal, build a PCRaster scalar map of source emission rate:
        E_metal[cell] = total_mass[cell] * fraction[metal] / deltaX

    The base_poll_map defines the spatial grid (MV pattern, NoData cells).
    Source cells NOT in base_poll_map (i.e. newly defined in the CSV) are
    added at their (row, col) positions.

    Parameters
    ----------
    sources      : list of source dicts from load_signature_csv()
    metals       : list of metal name strings
    base_poll_map: PCRaster map — used to initialise spatial structure
    deltaX       : cell size [m]
    output_dir   : directory where diagnostic E maps are saved

    Returns
    -------
    E_maps : dict  { metal_name : PCRaster scalar map of E [mg / (m·s)] }
    """
    diag_dir = os.path.join(output_dir, "source_metal_maps")
    os.makedirs(diag_dir, exist_ok=True)

    # Start with a zero base array, same shape as the clone
    base_arr = pcr.pcr2numpy(base_poll_map, np.nan)
    nrows, ncols = base_arr.shape

    E_maps = {}

    for metal in metals:
        # Initialise to zero everywhere (NaN → 0 emission)
        arr = np.zeros_like(base_arr)

        for src in sources:
            r, c = src["row"] - 1, src["col"] - 1   # 0-based NumPy indexing
            if r < 0 or r >= nrows or c < 0 or c >= ncols:
                print(f"  [WARNING] Source {src['source_id']} at "
                      f"({src['row']},{src['col']}) is outside the grid — skipped")
                continue

            metal_mass = src["total_mass"] * src[metal]   # [mg]
            arr[r, c] += metal_mass / deltaX               # [mg/m]

        pcr_map = pcr.numpy2pcr(pcr.Scalar, arr, np.nan)
        E_maps[metal] = pcr_map

        # Save diagnostic map
        diag_path = os.path.join(diag_dir, f"E_nominal_{metal}")
        pcr.report(pcr_map, diag_path)
        total = float(pcr.maptotal(pcr_map))
        print(f"  [E_map] {metal:>6s}  total emission = {total * deltaX:.4f} mg")

    return E_maps


# =============================================================================
# MULTI-METAL TRANSPORT MODEL
# =============================================================================

class MultiMetalTransportModel(DynamicModel):
    """
    Runs the 1D upwind finite-volume ADE for every metal simultaneously.

    Each metal has its own:
        self.M[metal]            — mass per unit length map [mg/m]
        self.C[metal]            — concentration map [mg/L = mg/m³×1e-3]
        self.ocean_release[metal]— cumulative mass leaving at ocean cell
        self.Theoretical_Total[metal]

    All metals share the same Q(x,t) and LDD, loaded once per timestep.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        pcr.setclone(cfg["clone_map"])

    # ------------------------------------------------------------------
    def initial(self):
        self.deltaT  = self.cfg["deltaT"]
        self.deltaX  = self.cfg["deltaX"]
        self.ldd     = pcr.readmap(self.cfg["ldd_map"])

        os.makedirs(self.cfg["output_dir"], exist_ok=True)

        # --- Load signatures and build per-metal E maps ---
        base_poll_map = pcr.readmap(self.cfg["pollutant_map"])
        sources, self.metals = load_signature_csv(self.cfg["signature_csv"])

        print(f"\n[Model] Building per-metal emission maps …")
        self.E_nominal = build_metal_emission_maps(
            sources, self.metals, base_poll_map,
            self.deltaX, self.cfg["output_dir"]
        )

        # --- Create per-metal output subdirectories ---
        self.metal_dirs = {}
        for metal in self.metals:
            d = os.path.join(self.cfg["output_dir"], metal)
            os.makedirs(d, exist_ok=True)
            self.metal_dirs[metal] = d

        # --- Initialise state variables per metal ---
        self.M                = {m: pcr.scalar(0.0) for m in self.metals}
        self.C                = {m: pcr.scalar(0.0) for m in self.metals}
        self.ocean_release    = {m: 0.0             for m in self.metals}
        self.Theoretical_Total= {m: 0.0             for m in self.metals}

        # --- Mass balance log (all metals in one CSV) ---
        self.log_path = os.path.join(self.cfg["output_dir"], "mass_balance_log.csv")
        with open(self.log_path, "w", newline="") as f:
            header = ["timestep", "metal",
                      "Theoretical_Total", "Model_Total", "Error", "ErrorPct"]
            csv.writer(f).writerow(header)

        # --- Diagnostic: report accuflux for each metal ---
        for metal in self.metals:
            fluxpath = pcr.accuflux(self.ldd, self.E_nominal[metal])
            pcr.report(fluxpath,
                       os.path.join(self.metal_dirs[metal], f"fluxpath_{metal}"))

        print(f"\n[Model] Initialised for metals: {self.metals}\n")

    # ------------------------------------------------------------------
    def _apply_release(self, metal, t):
        """Return emission map for metal at timestep t (zero outside window)."""
        s, e = self.cfg["release_start"], self.cfg["release_end"]
        return pcr.ifthenelse(
            (t >= s) & (t <= e),
            self.E_nominal[metal],
            pcr.scalar(0.0)
        )

    # ------------------------------------------------------------------
    def dynamic(self):
        t = self.currentTimeStep()

        # Load discharge once — shared by all metals
        Q = self.readmap(self.cfg["flow_map"])
        A = Q / self.cfg["velocity"]          # cross-section [m²]

        row_oc, col_oc = self.cfg["ocean_cell"]

        for metal in self.metals:
            E = self._apply_release(metal, t)

            # ---- Upwind finite-volume ADE step ----
            self.C[metal] = pcr.ifthenelse(
                A > 0, self.M[metal] / A, pcr.scalar(0.0)
            )
            FluxOut = Q * self.C[metal]
            FluxIn  = pcr.upstream(self.ldd, FluxOut)

            M_new = (self.M[metal]
                     - (self.deltaT / self.deltaX) * (FluxOut - FluxIn)
                     + self.deltaT * E)
            self.M[metal] = pcr.ifthenelse(M_new < 0, pcr.scalar(0.0), M_new)
            self.C[metal] = pcr.ifthenelse(
                A > 0, self.M[metal] / A, pcr.scalar(0.0)
            )

            # ---- Ocean/outlet accounting ----
            flux_ocean = pcr.cellvalue(FluxOut, row_oc, col_oc)[0]
            self.ocean_release[metal] += flux_ocean * self.deltaT

            # ---- Mass balance ----
            Model_Total = (float(pcr.maptotal(self.M[metal])) * self.deltaX
                           + self.ocean_release[metal])
            self.Theoretical_Total[metal] += (
                float(pcr.maptotal(E * self.deltaX)) * self.deltaT
            )
            TT  = self.Theoretical_Total[metal]
            Err = TT - Model_Total
            ErrPct = (Err / TT * 100) if TT else 0.0

            # ---- Console output (every 50 steps to reduce noise) ----
            if t % 50 == 0 or t <= 5:
                print(f"  [{metal}] t={t:5d} | "
                      f"Theory={TT:.4f}  Model={Model_Total:.4f}  "
                      f"Err={Err:.4f} ({ErrPct:.2f}%)")

            # ---- Write mass balance log ----
            with open(self.log_path, "a", newline="") as f:
                csv.writer(f).writerow(
                    [t, metal, TT, Model_Total, Err, f"{ErrPct:.4f}"]
                )

            # ---- Save mass map ----
            # Naming: <metal>/M_<metal>XXXXXXX.XXX
            # e.g.   Cd/M_Cd0000001.001
            m_path = self._map_path(metal, t, prefix="M_")
            self.report(self.M[metal], m_path)

            # ---- Optionally save concentration map ----
            if self.cfg.get("report_concentration", False):
                c_path = self._map_path(metal, t, prefix="C_")
                self.report(self.C[metal], c_path)

    # ------------------------------------------------------------------
    def _map_path(self, metal, step, prefix="M_"):
        """
        Build a PCRaster time-series output filename for a given metal.
        Pattern:  <metal_dir>/<prefix><metal>XXXXXXX.XXX
        Example:  Cd/M_Cd0000001.001   (step=1000)
        """
        s        = step - 1
        block    = s // 999
        slice_no = (s % 999) + 1
        filename = f"{prefix}{metal}{block:07d}.{slice_no:03d}"
        return os.path.join(self.metal_dirs[metal], filename)


# =============================================================================
# SIGNATURE CSV EXAMPLE GENERATOR
# =============================================================================

def write_example_signature_csv(path):
    """
    Write a minimal example signature CSV so you can see the expected format.
    Edit this to match your actual source locations and signatures.

    Columns
    -------
    source_id  : arbitrary unique label
    row        : 1-based PCRaster row
    col        : 1-based PCRaster column
    total_mass : total pollutant mass released [mg]
    Cd, Cr, Pb, Cu, Zn : metal fractions (must sum to 1.0 per row)
    """
    rows = [
        ["source_id", "row", "col", "total_mass", "Cd", "Cr", "Pb", "Cu", "Zn"],
        ["SRC_01",  45,  12, 500.0, 0.16, 0.21, 0.18, 0.11, 0.34],
        ["SRC_02",  60,  30, 800.0, 0.16, 0.34, 0.25, 0.24, 0.01],
        ["SRC_03",  80,  55, 300.0, 0.32, 0.24, 0.01, 0.26, 0.17],
        ["SRC_04", 100,  70, 650.0, 0.23, 0.13, 0.29, 0.18, 0.17],
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        csv.writer(f).writerows(rows)
    print(f"[Example CSV] Written to: {path}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    # ---- Optionally generate an example signature CSV ----
    # Uncomment the line below the first time you run this script to
    # create a template you can fill in with your real source data.
    #
    # write_example_signature_csv(CONFIG["signature_csv"])

    print("=" * 60)
    print("Multi-metal upwind ADE forward model")
    print(f"Case  : {case}")
    print(f"Output: {CONFIG['output_dir']}")
    print("=" * 60)

    model = MultiMetalTransportModel(CONFIG)
    framework = DynamicFramework(
        model,
        lastTimeStep  = CONFIG["nrOfTimeSteps"],
        firstTimestep = 1
    )
    framework.run()

    print("\nSimulation complete.")
    print(f"Mass balance log : {CONFIG['output_dir']}/mass_balance_log.csv")
    print(f"Metal map dirs   : {CONFIG['output_dir']}/<METAL>/")