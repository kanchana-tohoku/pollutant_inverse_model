#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 14:45:44 2025

@author: kumudu
"""

# ---- REQUIREMENTS ----
# pip install pcraster matplotlib imageio
# (FFMPEG optional for MP4: sudo apt-get install ffmpeg)

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from matplotlib.colors import LogNorm, PowerNorm, Normalize
import pcraster as pcr

# ======================
# CONFIG (edit these)
# ======================
data_dir         = Path("C:/Users/kanch/Research_models/data_2/out_ADErev6")   # folder with M0000000.001 ... etc.
base_prefix      = "M"                                   # the leading letter(s) in your filenames
n_total_steps    = 2000                                  # total frames available
start_step       = 1                                     # 1-based index (1..n_total_steps)
end_step         = 2000                                  # inclusive
ref_clone        = "C:/Users/kanch/Research_models/data_2/input_maps/topography/DEM/pcr_dem.map"  # any map with correct georeference/shape

# Region of interest selection
roi_by_window    = True                                  # True: use row/col window; False: use mask_raster
# If roi_by_window=True: provide row/col window (0-based, Python slicing semantics; end is exclusive)
row_min, row_max = 218, 279
col_min, col_max = 45, 120

# If roi_by_window=False: provide a Boolean mask raster (1 inside ROI, 0 outside)
mask_raster_path = "/path/to/roi_mask_boolean.map"

# Optional channel mask (1=channel cells, 0=else); set to None to skip
channel_mask_path = "C:/Users/kanch/Research_models/data_2/input_maps_synthetic/pollutants/200Sources/Channels.map"  # or None

# Rendering / output
out_gif           = "pollutant_animation2.gif"
out_mp4           = None            # e.g., "pollutant_animation.mp4" (requires ffmpeg)
fps               = 10              # frames per second
cmap_name         = "turbo"         # e.g., "plasma", "magma", "turbo", "viridis"
nan_color         = (1, 1, 1, 0)    # RGBA for NaN (transparent)

# === COLOR/SCALING OPTIONS ===
scale_mode        = "log"           # "linear" | "log" | "power"
power_gamma       = 0.5             # only used if scale_mode == "power" (0.5 brightens lows)
# Values <= eps will be treated as 'no visible pollutant' (set to NaN => transparent)
eps_fraction_of_vmax = 0.01         # e.g., 1% of vmax
absolute_eps          = 1e-10       # floor in data units
# Optional gray base for channel cells so the plume sits above it
draw_channel_underlay  = True
underlay_gray          = 0.90       # 0=black .. 1=white

# Optional timestamp decoration (if you know start time & Δt)
add_timestamp     = False
start_datetime_str= "2014-01-01 00:00:00"
dt_hours          = 6

# ======================
# HELPERS
# ======================
def step_to_filename(step_idx, base=base_prefix):
    """
    Convert 1-based step index to PCRaster-style filename:
      step=1   -> M0000000.001
      step=1000-> M0000000.999
      step=1001-> M0000001.000
      step=2000-> M0000002.000
    """
    s = step_idx - 1
    integer   = s // 1000
    frac_part = (s % 1000) + 1
    return f"{base}{integer:07d}.{frac_part:03d}"

def read_map_to_array(path):
    m = pcr.readmap(str(path))
    return pcr.pcr2numpy(m, np.nan).astype(float)

def load_mask(path, shape):
    if path is None:
        return np.ones(shape, dtype=bool)
    arr = read_map_to_array(path)
    return np.isfinite(arr) & (arr > 0.5)

def apply_roi(arr, roi_mask=None, window=None):
    if window is not None:
        r0, r1, c0, c1 = window
        out = arr[r0:r1, c0:c1]
        return out
    elif roi_mask is not None:
        out = arr.copy()
        out[~roi_mask] = np.nan
        return out
    else:
        return arr

def make_norm(vmin, vmax, mode="linear", gamma=1.0):
    if mode == "log":
        vmin_eff = max(vmin, np.nextafter(0, 1))  # strictly > 0
        return LogNorm(vmin=vmin_eff, vmax=vmax, clip=True)
    elif mode == "power":
        return PowerNorm(gamma=gamma, vmin=vmin, vmax=vmax, clip=True)
    else:
        return Normalize(vmin=vmin, vmax=vmax, clip=True)

# ======================
# MAIN
# ======================
def main():
    # 1) Set clone so PCRaster knows grid/geo
    if not (ref_clone and os.path.exists(ref_clone)):
        raise FileNotFoundError("Could not set clone; check ref_clone path.")
    pcr.setclone(ref_clone)

    # 2) Determine window or mask
    roi_mask = None
    window = None
    # peek a map to learn shape
    first_map_path = data_dir / step_to_filename(start_step)
    if not first_map_path.exists():
        raise FileNotFoundError(f"First frame not found: {first_map_path}")
    first_arr = read_map_to_array(first_map_path)
    nrows, ncols = first_arr.shape

    if roi_by_window:
        # sanitize bounds
        r0 = max(0, min(row_min, nrows))
        r1 = max(0, min(row_max, nrows))
        c0 = max(0, min(col_min, ncols))
        c1 = max(0, min(col_max, ncols))
        window = (r0, r1, c0, c1)
        # quick crop preview to set shape
        first_arr = first_arr[r0:r1, c0:c1]
    else:
        roi_mask = load_mask(mask_raster_path, (nrows, ncols))
        first_arr = apply_roi(first_arr, roi_mask=roi_mask)

    # Optional: mask to channels only
    if channel_mask_path:
        ch_mask = load_mask(channel_mask_path, (nrows, ncols))
        if window:
            r0, r1, c0, c1 = window
            ch_mask = ch_mask[r0:r1, c0:c1]
        # zero-out non-channel cells
        first_arr[~ch_mask] = np.nan
    else:
        ch_mask = None

    # 3) Scan once to get global vmin/vmax in the chosen subset for consistent colors
    vmin, vmax = np.inf, -np.inf
    for step in range(start_step, end_step + 1):
        fn = data_dir / step_to_filename(step)
        if not fn.exists():
            continue
        a = read_map_to_array(fn)
        if roi_by_window:
            a = a[window[0]:window[1], window[2]:window[3]]
        elif roi_mask is not None:
            a = apply_roi(a, roi_mask=roi_mask)
        if ch_mask is not None:
            a[~ch_mask] = np.nan

        if np.isfinite(a).any():
            vmin = min(vmin, np.nanmin(a))
            vmax = max(vmax, np.nanmax(a))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise RuntimeError("Could not determine vmin/vmax; all values are NaN in the chosen subset?")

    # Define threshold for visibility: everything <= eps becomes transparent
    eps = max(absolute_eps, eps_fraction_of_vmax * vmax)

    # For color mapping, start at eps (so the first “visible” value is bright enough)
    plot_vmin = eps
    plot_vmax = vmax

    norm = make_norm(plot_vmin, plot_vmax, mode=scale_mode, gamma=power_gamma)

    # 4) Set up figure
    plt.rcParams["figure.figsize"] = (7, 6)
    fig, ax = plt.subplots()

    # Base underlay (gray) so channels are visible even when plume is transparent
    base_im = None
    if draw_channel_underlay and (ch_mask is not None):
        base = np.full_like(first_arr, np.nan, dtype=float)
        base[ch_mask] = underlay_gray  # gray only on channels
        base_im = ax.imshow(base, cmap="gray", vmin=0, vmax=1, origin="upper", interpolation="nearest")

    # Pollutant overlay colormap
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=nan_color)  # NaNs -> transparent (or your color)

    # Mask tiny values so they render as transparent (bad)
    first_arr = first_arr.copy()
    first_arr[first_arr <= eps] = np.nan

    im = ax.imshow(first_arr, cmap=cmap, norm=norm, origin="upper", interpolation="nearest")
    
    
    cb = fig.colorbar(im, ax=ax, shrink=0.85, label="Pollutant concentration (mg/L)")
    # Set more frequent tick marks (linear or log)
    if scale_mode == "log":
        import matplotlib.ticker as mticker
        # Major ticks at 1e-6, 1e-5, 1e-4, etc. Adjust to your value range
        cb.ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0,), numticks=10))
        # Optional minor ticks between major powers of 10
        cb.ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))
        cb.ax.yaxis.set_minor_formatter(mticker.NullFormatter())  # hide labels on minor ticks
    else:
        # Example: 10 evenly spaced labels
        ticks = np.linspace(plot_vmin, plot_vmax, 10)
        cb.set_ticks(ticks)
        cb.ax.set_yticklabels([f"{t:.2e}" if t < 0.01 else f"{t:.2f}" for t in ticks])

    # Optional: increase label font size
    cb.ax.tick_params(labelsize=9)
    cb.set_label("Pollutant concentration (mg/L)", fontsize=10)
    
    ax.set_title(f"Pollutant transport (steps {start_step}–{end_step})")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Optional timestamp label
    if add_timestamp:
        from datetime import datetime
        t0 = datetime.fromisoformat(start_datetime_str)
        text_ts = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                          bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))
    else:
        text_ts = None

    # Log-colorbar ticks if needed
    if scale_mode == "log":
        import matplotlib.ticker as mticker
        cb.ax.yaxis.set_major_locator(mticker.LogLocator(base=10.0))
        cb.ax.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * .1))

    # 5) Frame generator
    step_list = [s for s in range(start_step, end_step + 1)
                 if (data_dir / step_to_filename(s)).exists()]

    def step_idx_to_name(step_idx):
        return step_to_filename(step_idx)

    def update(frame_idx):
        step = step_list[frame_idx]
        fn = data_dir / step_to_filename(step)
        a = read_map_to_array(fn)
        if roi_by_window:
            a = a[window[0]:window[1], window[2]:window[3]]
        elif roi_mask is not None:
            a = apply_roi(a, roi_mask=roi_mask)
        if ch_mask is not None:
            a[~ch_mask] = np.nan

        # Make tiny concentrations transparent
        a = a.copy()
        a[a <= eps] = np.nan

        im.set_data(a)
        if add_timestamp:
            from datetime import timedelta
            elapsed_steps = step - start_step
            text_ts.set_text((t0 + timedelta(hours=elapsed_steps * dt_hours)).strftime("%Y-%m-%d %H:%M"))
            ax.set_title(f"Pollutant transport | step {step} ({step_idx_to_name(step)})")
            return [im, text_ts]
        else:
            ax.set_title(f"Pollutant transport | step {step} ({step_idx_to_name(step)})")
            return [im]

    anim = FuncAnimation(fig, update, frames=len(step_list), interval=1000/fps, blit=False)

    # 6) Save outputs
    if out_gif:
        print(f"Writing GIF: {out_gif}")
        anim.save(out_gif, writer=PillowWriter(fps=fps))

    if out_mp4:
        print(f"Writing MP4: {out_mp4}")
        try:
            anim.save(out_mp4, writer=FFMpegWriter(fps=fps, bitrate=3000))
        except Exception as e:
            print("MP4 export failed (is ffmpeg installed?). Error:", e)

    # Show the animation window (last frame will be visible)
    plt.show()
    print("Done.")

if __name__ == "__main__":
    main()
