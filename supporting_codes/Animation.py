# ---- REQUIREMENTS ----
# pip install pcraster matplotlib imageio
# (FFMPEG optional for MP4: sudo apt-get install ffmpeg)

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

import pcraster as pcr

# ======================
# CONFIG (edit these)
# ======================
data_dir         = Path("/kaiganY/kumudu/LF_ModelRun/LF_ETRS89/Kanchana/new_M/")   # folder with M0000000.001 ... etc.
base_prefix      = "M"                                   # the leading letter(s) in your filenames
n_total_steps    = 2000                                  # total frames available
start_step       = 1                                     # 1-based index (1..n_total_steps)
end_step         = 2000                                  # inclusive
ref_clone        = "/kaiganY/kumudu/LF_ModelRun/LF_ETRS89/Kanchana/pcr_dem.map"        # any map with the correct georeference/shape
roi_by_window    = True                                  # True: use row/col window; False: use mask_raster
# If roi_by_window=True: provide row/col window (0-based, Python slicing semantics; end is exclusive)
row_min, row_max = 218, 279
col_min, col_max = 45, 120

# If roi_by_window=False: provide a Boolean mask raster (1 inside ROI, 0 outside)
mask_raster_path = "/path/to/roi_mask_boolean.map"

# Optional channel mask (1=channel cells, 0=else); set to None to skip
channel_mask_path = "/kaiganY/kumudu/LF_ModelRun/LF_ETRS89/Kanchana/Channels.map" # e.g., "/path/to/river_mask.map" ele None

# Rendering / output
out_gif           = "pollutant_animation.gif"
out_mp4           = None            # e.g., "pollutant_animation.mp4" (requires ffmpeg)
fps               = 10              # frames per second
cmap_name         = "turbo"       # e.g., "plasma", "magma", "turbo", etc.
nan_color         = (1, 1, 1, 0)    # RGBA for NaN (transparent white)

# Optional timestamp decoration (if you know start time & Δt)
add_timestamp     = False
start_datetime_str= "2014-01-01 00:00:00"
dt_hours          = 6

# ======================
# HELPERS
# ======================
def step_to_filename(step_idx, base=base_prefix):
    """
    Your naming scheme increments the 'fractional' 3-digit part from 001..999,
    then rolls over the 7-digit integer part.
    step_idx is 1-based (1..N).
    Example:
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
    #print(path)
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
        # Keep shape, but put NaN outside ROI so they plot as transparent
        out = arr.copy()
        out[~roi_mask] = np.nan
        return out
    else:
        return arr

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
            # you can choose to break or continue; here we skip non-existing frames
            continue
        a = read_map_to_array(fn)
        if roi_by_window:
            a = a[window[0]:window[1], window[2]:window[3]]
        elif roi_mask is not None:
            a = apply_roi(a, roi_mask=roi_mask)

        if ch_mask is not None:
            a[~ch_mask] = np.nan

        # update min/max ignoring NaNs
        if np.isfinite(a).any():
            vmin = min(vmin, np.nanmin(a))
            vmax = max(vmax, np.nanmax(a))

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        raise RuntimeError("Could not determine vmin/vmax; all values are NaN in the chosen subset?")

    # 4) Set up figure
    plt.rcParams["figure.figsize"] = (7, 6)
    fig, ax = plt.subplots()
    cmap = plt.get_cmap(cmap_name).copy()
    cmap.set_bad(color=nan_color)  # NaNs -> transparent (or your color)
    im = ax.imshow(first_arr, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper", interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, label="Concentration")
    ax.set_title(f"Pollutant transport (steps {start_step}–{end_step})")
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")

    # Optional timestamp label
    if add_timestamp:
        from datetime import datetime, timedelta
        t0 = datetime.fromisoformat(start_datetime_str)
        text_ts = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top", ha="left",
                          bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"))

    # 5) Frame generator
    step_list = [s for s in range(start_step, end_step + 1)
                 if (data_dir / step_to_filename(s)).exists()]

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

        im.set_data(a)
        if add_timestamp:
            # step is 1-based; compute elapsed hours from global step 1
            from datetime import timedelta
            elapsed_steps = step - 1
            text_ts.set_text((t0 + timedelta(hours=elapsed_steps * dt_hours)).strftime("%Y-%m-%d %H:%M"))
        ax.set_title(f"Pollutant transport | step {step} ({step_idx_to_name(step)})")
        return [im] if not add_timestamp else [im, text_ts]

    # Helper just for a pretty label
    def step_idx_to_name(step_idx):
        return step_to_filename(step_idx)

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
            

    
    plt.show()
    #plt.close(fig)
    print("Done.")

if __name__ == "__main__":
    main()
