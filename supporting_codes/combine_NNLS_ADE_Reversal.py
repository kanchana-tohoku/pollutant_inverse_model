"""
=============================================================================
Inverse_Pollutant_Tracer.py
=============================================================================
Merged Inverse Modelling Framework for Pollutant Source Identification
in River Networks.

CONCEPT (from flowchart):
  Step 1 – Detect pollutant at observation point (x_obs, t_obs)
  Step 2 – Single-channel backward time-reversal to backtrack mass from the
            observation point to the nearest upstream confluence, finding the
            mass distribution just upstream of that confluence at the arrival
            timestep (x_confluence, t_confluence).
  Step 3 – NNLS bifurcation solver: apportion the mass at the confluence
            to each incoming branch using pollutant chemical signatures.
  Repeat – Move up to the next confluence for each branch and repeat
            Steps 2 → 3 until all confluences in the network are resolved.

INPUTS REQUIRED:
  - PCRaster clone map and LDD map
  - Forward model mass output maps (M*.***) — produced by ADE_FVD_PCR.py
  - Time-varying discharge maps (Q*.***) 
  - Confluence cells map or list of confluence (row, col) locations
  - Pollutant source signature matrix [n_metals x n_sources]
  - Observed downstream concentration vector [n_metals] at (x_obs, t_obs)

OUTPUTS:
  - Reconstructed mass maps at each timestep per branch segment
  - Contribution fractions (x_k) from each upstream branch at each confluence
  - Estimated pollutant release timing and magnitude per source

Author  : Based on research by W.M. Kanchana D. Wijeratna
          Hydro-Environment System Laboratory, Tohoku University
=============================================================================
"""

import os
import csv
import numpy as np
import pcraster as pcr
from scipy.optimize import nnls

# =============================================================================
# CONFIGURATION
# =============================================================================

case = 2

CONFIG = {
    # --- Map inputs ---
    "clone_map":    r"C:\Users\kanch\Research_models\data_2\input_maps\topography\DEM\pcr_dem.map",
    "ldd_map":      r"C:\Users\kanch\Research_models\data_2\input_maps\topography\LDD\WGS_LDD.map",
    "channel_mask": f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}/fluxpath",

    # --- Forward model mass maps (produced by ADE_FVD_PCR.py) ---
    # Used as the starting point for backward tracking (mass at t_obs)
    "forward_mass_dir":  f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}/",
    "forward_mass_prefix": "M",   # filenames: M0000000.001, M0000001.000, etc.

    # --- Discharge maps ---
    "flow_prefix":  r"C:/Users/kanch/Research_models/data_2/input_maps_synthetic/discharge/Discharge/Q",

    # --- Output ---
    "output_dir":   f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}/InverseTracer",

    # --- Numerical parameters ---
    "deltaT":    1,       # seconds
    "deltaX":    100,     # metres
    "velocity":  1.8,     # m/s (used to compute cross-section A = Q/u)
    "max_iter":  25,      # max fixed-point iterations per timestep
    "tolerance": 1e-8,    # convergence criterion (maptotal of |dM|)

    # --- Observation (Step 1) ---
    # Row/col are 1-based PCRaster convention
    "obs_row":  278,      # row of downstream observation sensor
    "obs_col":  18,       # col of downstream observation sensor
    "t_obs":    2000,     # timestep at which pollutant was detected

    # --- Confluence cells ---
    # List of (row, col) tuples for all confluence cells in the network,
    # ordered from downstream to upstream (the loop works bottom-up).
    # Replace with your actual confluence locations.
    "confluence_cells": [
        (240, 20),   # confluence 1 – nearest upstream of observation
        (200, 25),   # confluence 2 – next level up
        (160, 30),   # confluence 3 – and so on …
        # add more as needed
    ],

    # --- Pollutant source signatures ---
    # Rows = metal species, Cols = candidate sources
    # Each column is the signature vector of one source (normalised or raw mg/L).
    # Shape: (n_metals, n_sources)
    "source_signatures": np.array([
        # Metal1  Metal2  Metal3  Metal4  Metal5
        [0.16,   0.16,   0.32,   0.23],   # Source S1-S4
        [0.21,   0.34,   0.24,   0.13],
        [0.18,   0.25,   0.01,   0.29],
        [0.11,   0.24,   0.26,   0.18],
        [0.34,   0.01,   0.17,   0.17],
    ], dtype=float),  # shape (5 metals, 4 sources)

    # Labels for each source column (for output logging)
    "source_labels": ["S1", "S2", "S3", "S4"],
}

# =============================================================================
# INITIALISATION
# =============================================================================

pcr.setclone(CONFIG["clone_map"])
ldd     = pcr.readmap(CONFIG["ldd_map"])
channel = pcr.readmap(CONFIG["channel_mask"])
channel = pcr.ifthen(channel > 0, pcr.boolean(1))

os.makedirs(CONFIG["output_dir"], exist_ok=True)

deltaT   = CONFIG["deltaT"]
deltaX   = CONFIG["deltaX"]
velocity = CONFIG["velocity"]

# =============================================================================
# HELPER – PCRaster file-naming conventions
# =============================================================================

def mass_map_filename(step, directory, prefix):
    """
    Return the PCRaster time-series filename for a forward mass map.
    step is 1-based.  e.g. step=1    → M0000000.001
                            step=1000 → M0000001.001
    """
    s        = step - 1
    block    = s // 999
    slice_no = (s % 999) + 1
    return os.path.join(directory, f"{prefix}{block:07d}.{slice_no:03d}")


def flow_map_filename(step):
    """Return discharge map filename for a given timestep (1-based)."""
    s        = step - 1
    block    = s // 999
    slice_no = (s % 999) + 1
    return f"{CONFIG['flow_prefix']}{block:07d}.{slice_no:03d}"


def output_map_filename(step, subdir, prefix="MI"):
    """Return output filename for a reconstructed backward mass map."""
    s        = step - 1
    block    = s // 999
    slice_no = (s % 999) + 1
    return os.path.join(subdir, f"{prefix}{block:06d}.{slice_no:03d}")


# =============================================================================
# STEP 2 – Single-channel backward time-reversal
# =============================================================================

def backward_one_step(M_current, Q_map):
    """
    Reconstruct mass distribution one timestep earlier using fixed-point
    iteration of the reversed finite-volume upwind scheme.

    Forward scheme:   M(t+1) = M(t) - (dT/dX)*(FluxOut - FluxIn) + dT*S
    Backward inversion (no source term assumed during backtracking):
                      M(t)   = M(t+1) + (dT/dX)*(FluxOut - FluxIn)
    where FluxOut and FluxIn are evaluated at the unknown M(t) → iterate.

    Returns M_prev (the reconstructed mass map at t-1).
    """
    M_prev = M_current   # initial guess

    for k in range(CONFIG["max_iter"]):
        A       = Q_map / velocity
        C       = pcr.ifthenelse(A > 0, M_prev / A, pcr.scalar(0))
        FluxOut = Q_map * C
        FluxIn  = pcr.upstream(ldd, FluxOut)

        M_new = M_current + (deltaT / deltaX) * (FluxOut - FluxIn)
        M_new = pcr.ifthenelse(M_new < 0, pcr.scalar(0), M_new)

        diff = float(pcr.maptotal(pcr.abs(M_new - M_prev)))
        M_prev = M_new

        if diff < CONFIG["tolerance"]:
            break

    return M_prev


def backtrack_to_confluence(M_start, t_start, t_stop, segment_label, output_subdir):
    """
    Run the backward solver from t_start down to t_stop (exclusive),
    saving reconstructed mass maps along the way.

    Parameters
    ----------
    M_start       : PCRaster map – mass distribution at t_start
    t_start       : int – timestep to start backtracking from (inclusive)
    t_stop        : int – timestep to stop at (the confluence arrival time)
    segment_label : str – used for output subfolder naming
    output_subdir : str – directory to save per-step mass maps

    Returns
    -------
    M_at_confluence : PCRaster map – reconstructed mass just upstream of
                      the confluence at timestep t_stop
    """
    os.makedirs(output_subdir, exist_ok=True)
    M_current = M_start

    print(f"\n[Backtrack] Segment '{segment_label}': t={t_start} → t={t_stop}")

    for t in range(t_start, t_stop, -1):
        Q_map  = pcr.readmap(flow_map_filename(t))
        M_prev = backward_one_step(M_current, Q_map)

        out_path = output_map_filename(t, output_subdir)
        pcr.report(M_prev, out_path)

        print(f"  t={t:5d}  |dM|={float(pcr.maptotal(pcr.abs(M_prev - M_current))):.4e}")
        M_current = M_prev

    return M_current   # mass map reconstructed at t_stop


# =============================================================================
# STEP 3 – NNLS bifurcation apportionment
# =============================================================================

def extract_mass_at_cell(M_map, row, col):
    """
    Read the scalar mass value from a PCRaster map at (row, col).
    row, col are 1-based (PCRaster convention).
    """
    val, valid = pcr.cellvalue(M_map, row, col)
    return float(val) if valid else 0.0


def nnls_apportion(observation_vector, signature_matrix):
    """
    Solve the NNLS system:  min ||S·x - d||²  subject to x >= 0

    Parameters
    ----------
    observation_vector : np.ndarray, shape (n_metals,)
        Observed pollutant concentrations (or mass) at the confluence,
        one entry per metal species.
    signature_matrix   : np.ndarray, shape (n_metals, n_sources)
        Each column is the chemical signature of one upstream source.

    Returns
    -------
    contributions : np.ndarray, shape (n_sources,)
        Estimated volume/mass contribution from each source.
    residual      : float
        Residual norm of the NNLS fit.
    rel_error     : float
        Relative error: ||S·x - d|| / ||d||
    """
    contributions, residual_sq = nnls(signature_matrix, observation_vector)
    residual   = np.sqrt(residual_sq)
    norm_d     = np.linalg.norm(observation_vector)
    rel_error  = residual / norm_d if norm_d > 0 else np.nan
    return contributions, residual, rel_error


def build_observation_vector_from_map(M_map, confluence_row, confluence_col,
                                      n_metals=1):
    """
    Build the multi-metal observation vector 'd' at a confluence cell.

    In the current single-metal forward model, this returns a 1-element
    vector.  When you extend to multi-metal transport (Cd, Cr, Pb, Cu, Zn),
    pass the per-metal mass maps as a list and stack them here.

    For now we use the total mass at the confluence cell scaled by deltaX
    as a proxy for the observed mass.

    Parameters
    ----------
    M_map            : PCRaster map (or list of maps for multi-metal)
    confluence_row   : int (1-based)
    confluence_col   : int (1-based)
    n_metals         : int – number of metal species

    Returns
    -------
    d : np.ndarray, shape (n_metals,)
    """
    if isinstance(M_map, list):
        # Multi-metal case: one map per metal
        d = np.array([extract_mass_at_cell(m, confluence_row, confluence_col)
                      for m in M_map])
    else:
        # Single-metal case – replicate or use total mass in channel
        # as a placeholder until multi-metal maps are available
        total_mass = extract_mass_at_cell(M_map, confluence_row, confluence_col)
        d = np.full(n_metals, total_mass / n_metals)
    return d


def log_apportionment(log_path, confluence_id, t_confluence,
                      contributions, rel_error, source_labels):
    """Append one apportionment result row to a CSV log file."""
    write_header = not os.path.exists(log_path)
    with open(log_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            header = ["confluence_id", "t_confluence", "rel_error"] + source_labels
            writer.writerow(header)
        row = [confluence_id, t_confluence, f"{rel_error:.6f}"] + \
              [f"{x:.6f}" for x in contributions]
        writer.writerow(row)


# =============================================================================
# MAIN INVERSE MODELLING PIPELINE
# =============================================================================

def run_inverse_tracer():
    """
    Master routine implementing the flowchart:

    Step 1  : Load mass at observation point (t_obs).
    Loop over confluences from downstream → upstream:
      Step 2  : Backtrack from current observation to next confluence.
      Step 3  : NNLS apportionment at confluence.
      Advance : Use the backtracked mass as the new starting point for
                each upstream branch (recursive / iterative expansion).
    """

    sig_matrix   = CONFIG["source_signatures"]    # (n_metals, n_sources)
    src_labels   = CONFIG["source_labels"]
    n_metals     = sig_matrix.shape[0]
    log_path     = os.path.join(CONFIG["output_dir"], "apportionment_log.csv")

    # ------------------------------------------------------------------
    # STEP 1 – Load mass map at observation time
    # ------------------------------------------------------------------
    t_obs = CONFIG["t_obs"]
    obs_mass_path = mass_map_filename(t_obs,
                                      CONFIG["forward_mass_dir"],
                                      CONFIG["forward_mass_prefix"])
    print(f"\n{'='*60}")
    print(f"STEP 1 – Loading observation mass map at t={t_obs}")
    print(f"         File: {obs_mass_path}")
    print(f"{'='*60}")

    M_obs = pcr.readmap(obs_mass_path)

    # Observation vector at sensor location
    d_obs = build_observation_vector_from_map(M_obs,
                                              CONFIG["obs_row"],
                                              CONFIG["obs_col"],
                                              n_metals=n_metals)
    print(f"  Observed mass vector at sensor: {d_obs}")

    # ------------------------------------------------------------------
    # Iterate through confluences (downstream → upstream)
    # ------------------------------------------------------------------
    # State: current mass map and the timestep it corresponds to
    M_current = M_obs
    t_current = t_obs

    confluence_cells = CONFIG["confluence_cells"]

    for idx, (conf_row, conf_col) in enumerate(confluence_cells):
        confluence_id = f"C{idx+1}_r{conf_row}_c{conf_col}"
        print(f"\n{'='*60}")
        print(f"Processing confluence {idx+1}/{len(confluence_cells)}: "
              f"({conf_row}, {conf_col})  [id={confluence_id}]")
        print(f"{'='*60}")

        # ---------------------------------------------------------------
        # STEP 2 – Backtrack to this confluence
        # ---------------------------------------------------------------
        # We backtrack from t_current until the pollutant signal is
        # concentrated near the confluence cell.
        #
        # Heuristic for t_stop: travel time from sensor to confluence.
        # travel_time = distance_in_cells * deltaX / velocity (approx.)
        # For a full implementation, t_stop should be determined by
        # finding when maptotal(M) in the segment above the confluence
        # peaks – here we use a simple fixed-point approach.
        #
        # For now t_stop is estimated as t_current minus the approximate
        # number of steps needed for the signal to travel between the
        # previous point and this confluence.  Replace with your own
        # travel-time calculation based on the LDD path length.

        # --- Compute approximate travel time to this confluence ---
        # (placeholder – use actual LDD path length / velocity in practice)
        t_stop = max(1, t_current - 200)   # <<< replace with real path length

        segment_subdir = os.path.join(CONFIG["output_dir"],
                                      f"segment_{confluence_id}")

        M_at_confluence = backtrack_to_confluence(
            M_current,
            t_start       = t_current,
            t_stop        = t_stop,
            segment_label = confluence_id,
            output_subdir = segment_subdir
        )

        t_confluence = t_stop
        print(f"\n[Step 2 done] Mass reconstructed at confluence "
              f"({conf_row},{conf_col}) at t={t_confluence}")

        # ---------------------------------------------------------------
        # STEP 3 – NNLS apportionment at this confluence
        # ---------------------------------------------------------------
        d_confluence = build_observation_vector_from_map(
            M_at_confluence, conf_row, conf_col, n_metals=n_metals
        )
        print(f"\n[Step 3] NNLS apportionment at confluence {confluence_id}")
        print(f"  Observation vector d = {d_confluence}")

        contributions, residual, rel_error = nnls_apportion(
            d_confluence, sig_matrix
        )

        print(f"  Contributions (x_k) : {dict(zip(src_labels, contributions))}")
        print(f"  Residual            : {residual:.6e}")
        print(f"  Relative error      : {rel_error*100:.2f}%")

        log_apportionment(log_path, confluence_id, t_confluence,
                          contributions, rel_error, src_labels)

        # ---------------------------------------------------------------
        # ADVANCE – The reconstructed confluence mass becomes the new
        # starting state for the next (further upstream) confluence.
        # In a full branching network you would spawn one backtracker
        # per incoming branch, weighted by the NNLS contribution x_k.
        # ---------------------------------------------------------------
        M_current = M_at_confluence
        t_current = t_confluence

    print(f"\n{'='*60}")
    print("Inverse tracing complete.")
    print(f"Apportionment log saved to: {log_path}")
    print(f"{'='*60}")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    run_inverse_tracer()