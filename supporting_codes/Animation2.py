#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PCRaster Pollutant Animation Script
------------------------------------
Creates GIF/MP4 animation from ADE output maps.

Author: Kanchana
Optimized & documented version 22 Feb 2026
"""

# ======================
# REQUIREMENTS
# ======================
# pip install pcraster matplotlib imageio
# (Optional for MP4) install ffmpeg

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.colors import LogNorm, PowerNorm, Normalize
import matplotlib.ticker as mticker
import pcraster as pcr


# ==========================================================
# ======================== CONFIG ==========================
# ==========================================================
case = 3

#data_dir      = Path("C:/Users/kanch/Research_models/data_2/out_ADErev6/case_2")
                     
data_dir =      Path(f"C:/Users/kanch/Research_models/data_2/out_ADErev6/case_{case}/TimeReversal_NoReg_2")
                    
# ----------------------------------------------------------
# Create animation output directory inside data_dir
# ----------------------------------------------------------
anim_output_dir = data_dir / "animation"
anim_output_dir.mkdir(parents=True, exist_ok=True)

out_gif = anim_output_dir / "pollutant_animation.gif"
out_mp4 = None   # or anim_output_dir / "pollutant_animation.mp4"


base_prefix   = "MI" # for reversal "MI"
start_step    = 1
end_step      = 2000
ref_clone     = "C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/pcr_dem.map"

# ROI options
roi_by_window = True
row_min, row_max = 380, 500
col_min, col_max = 200, 300
mask_raster_path = None   # use only if roi_by_window=False

# Optional channel mask
channel_mask_path = "C:/Users/kanch/Research_models/data_2/input_maps_synthetic/pollutants/200Sources/Channels.map"

# Rendering
fps         = 10
cmap_name   = "turbo"
scale_mode  = "log"      # "linear" | "log" | "power"
power_gamma = 0.5

eps_fraction_of_vmax = 0.01
absolute_eps         = 1e-10


# ==========================================================
# ====================== UTILITIES =========================
# ==========================================================

def step_to_filename(step):
    """Convert 1-based step index to PCRaster time-slice filename."""
    s = step - 1
    return f"{base_prefix}{s//1000:06d}.{(s%1000)+1:03d}"   # 07d for M


def read_map(path):
    """Read PCRaster map into NumPy array."""
    return pcr.pcr2numpy(pcr.readmap(str(path)), np.nan).astype(float)


def build_norm(vmin, vmax):
    """
    Build safe normalization.
    Prevents LogNorm crashes.
    """

    if not np.isfinite(vmax) or vmax <= 0:
        raise RuntimeError("All values are zero or invalid in selected region.")

    if scale_mode == "log":
        vmin = max(vmin, vmax * 1e-6)
        if vmin >= vmax:
            vmin = vmax * 1e-6
        return LogNorm(vmin=vmin, vmax=vmax, clip=True)

    if scale_mode == "power":
        if vmin >= vmax:
            vmin = vmax * 0.01
        return PowerNorm(gamma=power_gamma, vmin=vmin, vmax=vmax, clip=True)

    return Normalize(vmin=vmin, vmax=vmax, clip=True)


# ==========================================================
# ========================= MAIN ===========================
# ==========================================================

def main():

    # ------------------------------------------------------
    # 1) Initialize PCRaster clone
    # ------------------------------------------------------
    if not os.path.exists(ref_clone):
        raise FileNotFoundError("Clone map not found.")
    pcr.setclone(ref_clone)

    # ------------------------------------------------------
    # 2) Load first map (determine shape)
    # ------------------------------------------------------
    first_path = data_dir / step_to_filename(start_step)
    if not first_path.exists():
        raise FileNotFoundError(f"Missing first frame: {first_path}")

    base_arr = read_map(first_path)
    nrows, ncols = base_arr.shape

    # ------------------------------------------------------
    # 3) Define ROI slicer (fast approach)
    # ------------------------------------------------------
    if roi_by_window:
        r0, r1 = max(0,row_min), min(nrows,row_max)
        c0, c1 = max(0,col_min), min(ncols,col_max)
        slicer = np.s_[r0:r1, c0:c1]
    else:
        roi_mask = read_map(mask_raster_path) > 0.5
        slicer = None

    # ------------------------------------------------------
    # 4) Load optional channel mask (once only)
    # ------------------------------------------------------
    if channel_mask_path:
        ch_mask = read_map(channel_mask_path) > 0.5
    else:
        ch_mask = None

    # ------------------------------------------------------
    # 5) Scan once for global min/max
    # ------------------------------------------------------
    vmin, vmax = np.inf, -np.inf
    valid_steps = []

    for step in range(start_step, end_step+1):
        f = data_dir / step_to_filename(step)
        if not f.exists():
            continue

        arr = read_map(f)

        if slicer is not None:
            arr = arr[slicer]
        else:
            arr[~roi_mask] = np.nan

        if ch_mask is not None:
            arr[~ch_mask[slicer] if slicer else ~ch_mask] = np.nan

        if np.isfinite(arr).any():
            vmin = min(vmin, np.nanmin(arr))
            vmax = max(vmax, np.nanmax(arr))
            valid_steps.append(step)

    if not valid_steps:
        raise RuntimeError("No valid frames found.")

    # ------------------------------------------------------
    # 6) Threshold small values (visual clarity)
    # ------------------------------------------------------
    eps = max(absolute_eps, eps_fraction_of_vmax * vmax)
    plot_vmin = eps
    plot_vmax = vmax

    norm = build_norm(plot_vmin, plot_vmax)

    # ------------------------------------------------------
    # 7) Setup plot
    # ------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7,6))

    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad((1,1,1,0))   # transparent NaNs

    # --- Plot channel background first ---
    if ch_mask is not None:
        if slicer is not None:
            ch_display = ch_mask[slicer]
        else:
            ch_display = ch_mask
    
        ax.imshow(ch_display,
                  cmap="Greys",
                  alpha=0.4,
                  origin="upper")
    
    # --- Then pollutant layer on top ---
    im = ax.imshow(np.zeros_like(base_arr[slicer] if slicer else base_arr),
                   cmap=cmap,
                   norm=norm,
                   origin="upper",
                   alpha=0.9)

    cb = fig.colorbar(im, ax=ax)
    cb.set_label("Pollutant concentration (mg/L)")

    if scale_mode == "log":
        cb.ax.yaxis.set_major_locator(mticker.LogLocator(base=10))
    else:
        ticks = np.linspace(plot_vmin, plot_vmax, 8)
        cb.set_ticks(ticks)

    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # ------------------------------------------------------
    # 8) Animation update function
    # ------------------------------------------------------
    def update(i):
        step = valid_steps[i]
        arr = read_map(data_dir / step_to_filename(step))

        if slicer is not None:
            arr = arr[slicer]
        else:
            arr[~roi_mask] = np.nan

 #       if ch_mask is not None:
    #        arr[~ch_mask[slicer] if slicer else ~ch_mask] = np.nan

        arr[arr <= eps] = np.nan
        im.set_data(arr)

        ax.set_title(f"Step {step}")
        return [im]

    anim = FuncAnimation(fig, update,
                         frames=len(valid_steps),
                         interval=1000/fps)

    # ------------------------------------------------------
    # 9) Save outputs
    # ------------------------------------------------------
    if out_gif:
        print(f"Writing GIF to {out_gif} ...")
        anim.save(str(out_gif), writer=PillowWriter(fps=fps))

    if out_mp4:
        print(f"Writing MP4 to {out_mp4} ...")
        anim.save(str(out_mp4), writer=FFMpegWriter(fps=fps))

    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()