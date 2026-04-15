#!/usr/bin/env python
import os

import astropy.units as u
import matplotlib
from astropy.wcs.utils import proj_plane_pixel_scales
from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator

matplotlib.use("TkAgg")
import matplotlib.patches as patches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import interpolate, ndimage
from scipy.integrate import simpson

image_scale = 0.262 / 60.0  # Legacy Survey Scale arcmin/px
wcs_obj = None  # set when a FITS is loaded
overlay_points = []
contour_artists = []  # lines we draw, so we can clear them later
sx_am = image_scale
sy_am = image_scale
points = []
contrast_factor = 1.0  # Initial contrast level

image_data = None
fits_file = None

overlay_visible = False
move_mode = False  # False = add points, True = move existing points
dragging_idx = None  # index of point currently being dragged
spline_visible = False  # once Draw Spline is called, keep showing it
show_control_points = True
pick_radius_px = 10

# Create blank figure and axes
import tkinter as tk
from tkinter import filedialog, messagebox

fig, ax = plt.subplots()
startup_text = ax.text(
    0.5,
    0.5,
    "Press 'Ctrl+o' to load a FITS file",
    color="gray",
    ha="center",
    va="center",
    fontsize=20,
    transform=ax.transAxes,
)
#
# # Maximize window to full screen
manager = plt.get_current_fig_manager()

ax.set_xscale("linear")
ax.set_yscale("linear")
manager.set_window_title("FITS Viewer - No file loaded")

points = []
contrast_factor = 1.0  # Initial contrast level


def _apply_axes_formatting():
    if image_data is None:
        return

    set_axes_in_arcmin(ax, sx_am, sy_am, centered=True, img_shape=image_data.shape)
    set_centered_arcmin_ticks(
        ax, sx_am, sy_am, image_data.shape, major_am=2.0, minor_am=0.5
    )
    ax.xaxis.set_minor_locator(MultipleLocator(0.5 / sx_am))
    ax.yaxis.set_minor_locator(MultipleLocator(0.5 / sy_am))
    ax.tick_params(which="minor", length=3, width=0.8)
    ax.tick_params(which="major", length=6, width=1.2)


def _current_display_image():
    if image_data is None:
        return None

    if np.isclose(contrast_factor, 1.0):
        return image_data

    adjusted = 128 + contrast_factor * (image_data.astype(np.float32) - 128)
    adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
    return adjusted


def _compute_spline_from_points(pts):
    """
    Returns x_spline, y_spline, length_arcmin
    or (None, None, None) if not enough valid points.
    """
    if len(pts) < 2:
        return None, None, None

    x = np.array([p[0] for p in pts], dtype=float)
    y = np.array([p[1] for p in pts], dtype=float)

    # Remove consecutive duplicate points
    keep = np.ones(len(x), dtype=bool)
    if len(x) > 1:
        keep[1:] = np.hypot(np.diff(x), np.diff(y)) > 1e-9
    x = x[keep]
    y = y[keep]

    if len(x) < 2:
        return None, None, None

    ds = np.hypot(np.diff(x), np.diff(y))
    t = np.concatenate(([0.0], np.cumsum(ds)))

    t_fine = np.linspace(t[0], t[-1], 1000)

    if len(x) == 2:
        x_spline = np.interp(t_fine, t, x)
        y_spline = np.interp(t_fine, t, y)
    else:
        spline_x = interpolate.CubicSpline(t, x, bc_type="natural")
        spline_y = interpolate.CubicSpline(t, y, bc_type="natural")
        x_spline = spline_x(t_fine)
        y_spline = spline_y(t_fine)

    dx = np.diff(x_spline) * sx_am
    dy = np.diff(y_spline) * sy_am
    length_arcmin = np.sum(np.hypot(dx, dy))

    return x_spline, y_spline, length_arcmin


def _compute_spline():
    return _compute_spline_from_points(points)


def update_window_title():
    fname = os.path.basename(fits_file) if fits_file else "No file loaded"
    mode = "MOVE" if move_mode else "ADD"
    pts = "PTS ON" if show_control_points else "PTS OFF"
    manager.set_window_title(f"FITS Viewer - {fname} [{mode} mode | {pts}]")


def _draw_spline_from_points(
    pts,
    *,
    line_color,
    line_style,
    halo_color,
    label_prefix,
    label_offset=(15, 0),
    marker=None,
    markevery=None,
):
    x_spline, y_spline, length_arcmin = _compute_spline_from_points(pts)
    if x_spline is None:
        return

    (line,) = ax.plot(
        x_spline,
        y_spline,
        color=line_color,
        lw=2.3,
        linestyle=line_style,
        zorder=4,
        marker=marker,
        markevery=markevery,
        ms=4,
        mfc="none",
        mec=line_color,
    )
    line.set_path_effects(
        [pe.Stroke(linewidth=4.8, foreground=halo_color, alpha=0.85), pe.Normal()]
    )

    dx_lab, dy_lab = label_offset
    label = ax.text(
        x_spline[-1] + dx_lab,
        y_spline[-1] + dy_lab,
        f"{label_prefix}: {length_arcmin:.2f}'",
        color=line_color,
        fontsize=13,
        weight="bold",
        zorder=6,
    )
    label.set_path_effects(
        [pe.withStroke(linewidth=6.0, foreground="black", alpha=0.75), pe.Normal()]
    )


def render_scene(preserve_limits=True, highlight_idx=None):
    """
    Redraw image, points, spline(s), and length label(s).
    """
    if image_data is None:
        return

    xlim = ax.get_xlim() if preserve_limits and ax.has_data() else None
    ylim = ax.get_ylim() if preserve_limits and ax.has_data() else None

    ax.clear()
    ax.imshow(
        _current_display_image(), origin="lower", cmap="gray", interpolation="nearest"
    )
    _apply_axes_formatting()

    # Primary spline: editable/current
    if spline_visible and len(points) > 1:
        _draw_spline_from_points(
            points,
            line_color="deepskyblue",
            line_style="-",
            halo_color="black",
            label_prefix="S1",
            label_offset=(15, 0),
            marker=None,
            markevery=None,
        )

    # Secondary spline: comparison overlay
    if overlay_visible and len(overlay_points) > 1:
        _draw_spline_from_points(
            overlay_points,
            line_color="gold",
            line_style=(0, (8, 3, 1.5, 3)),
            halo_color="black",
            label_prefix="S2",
            label_offset=(15, -18),
            marker="s",
            markevery=55,
        )

    # Draw control points only if enabled
    if show_control_points and len(points) > 0:
        xp = [p[0] for p in points]
        yp = [p[1] for p in points]
        ax.plot(
            xp,
            yp,
            marker="o",
            ms=6,
            linestyle="None",
            mfc="none",
            mec="deepskyblue",
            mew=1.5,
            zorder=7,
        )

        if highlight_idx is not None and 0 <= highlight_idx < len(points):
            hx, hy = points[highlight_idx]
            ax.plot(hx, hy, marker="o", ms=10, mfc="none", mec="cyan", mew=2, zorder=8)

    if show_control_points and len(overlay_points) > 0:
        xo = [p[0] for p in overlay_points]
        yo = [p[1] for p in overlay_points]
        ax.plot(
            xo,
            yo,
            marker="s",
            ms=6,
            linestyle="None",
            mfc="none",
            mec="gold",
            mew=1.5,
            zorder=7,
        )

    if xlim is not None and ylim is not None:
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    fig.canvas.draw_idle()


def nearest_point_index(event, radius_px=10):
    """
    Pick nearest existing point using screen-pixel distance.
    """
    if len(points) == 0 or event.x is None or event.y is None:
        return None

    pts_disp = ax.transData.transform(np.asarray(points))
    d = np.hypot(pts_disp[:, 0] - event.x, pts_disp[:, 1] - event.y)
    idx = int(np.argmin(d))
    if d[idx] <= radius_px:
        return idx
    return None


def set_move_mode(enabled):
    global move_mode, dragging_idx
    move_mode = bool(enabled)
    dragging_idx = None
    update_window_title()
    render_scene()


def toggle_move_mode():
    set_move_mode(not move_mode)


def toggle_control_points():
    global show_control_points
    show_control_points = not show_control_points
    update_window_title()
    render_scene()


def set_centered_arcmin_ticks(
    ax, sx_am_per_pix, sy_am_per_pix, img_shape, major_am=2.0, minor_am=0.5
):
    """
    Force ticks every 'major_am' arcmin, with a tick at the image center (0 arcmin).
    Labels remain in arcmin using your formatter.

    sx_am_per_pix, sy_am_per_pix: arcmin/pixel along X and Y
    img_shape: (ny, nx)
    """
    ny, nx = img_shape
    cx = (nx - 1) / 2.0
    cy = (ny - 1) / 2.0

    # Convert arcmin steps to pixels
    step_x = major_am / sx_am_per_pix
    step_y = major_am / sy_am_per_pix

    # Build tick positions so one lands exactly at center
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    start_x = cx + np.ceil((xmin - cx) / step_x) * step_x
    xs = start_x + step_x * np.arange(0, int(np.floor((xmax - start_x) / step_x)) + 1)

    start_y = cy + np.ceil((ymin - cy) / step_y) * step_y
    ys = start_y + step_y * np.arange(0, int(np.floor((ymax - start_y) / step_y)) + 1)

    ax.xaxis.set_major_locator(FixedLocator(xs))
    ax.yaxis.set_major_locator(FixedLocator(ys))

    # Minor ticks (optional)
    if minor_am is not None and minor_am > 0:
        ax.xaxis.set_minor_locator(MultipleLocator(minor_am / sx_am_per_pix))
        ax.yaxis.set_minor_locator(MultipleLocator(minor_am / sy_am_per_pix))

    # Formatter: pixels → arcmin, centered at image center
    def fx(x, pos):
        return f"{(x - cx) * sx_am_per_pix:.2f}"

    def fy(y, pos):
        return f"{(y - cy) * sy_am_per_pix:.2f}"

    ax.xaxis.set_major_formatter(FuncFormatter(fx))
    ax.yaxis.set_major_formatter(FuncFormatter(fy))
    ax.set_xlabel("X [arcmin]")
    ax.set_ylabel("Y [arcmin]")

    ax.figure.canvas.draw_idle()


def _pixel_scales_arcmin(header):
    """
    Return (sx_arcmin_per_pix, sy_arcmin_per_pix).
    Uses WCS if present; otherwise tries A_SEC2 (area in arcsec^2).
    """
    # Try WCS first (handles CD/PC etc.)
    try:
        w = WCS(header)
        if w.pixel_n_dim >= 2:
            scales_deg = (
                proj_plane_pixel_scales(w) * u.deg
            )  # per-pixel angles along axes
            sx = scales_deg[0].to(u.arcmin).value
            sy = scales_deg[1].to(u.arcmin).value
            if sx > 0 and sy > 0:
                return float(sx), float(sy)
    except Exception:
        pass

    # Fallback: A_SEC2 = pixel area [arcsec^2]
    asec2 = float(header.get("A_SEC2", 0.0))
    if asec2 > 0:
        s_arcsec = asec2**0.5
        s_arcmin = s_arcsec / 60.0
        return float(s_arcmin), float(s_arcmin)

    # Last resort: assume Legacy native scale 0.262"/pix
    s_arcmin = 0.262 / 60.0
    return s_arcmin, s_arcmin


def set_axes_in_arcmin(
    ax, sx_am_per_pix, sy_am_per_pix, *, centered=False, img_shape=None
):
    """Format ticks in arcmin (labels only)."""
    if centered and img_shape is not None:
        ny, nx = img_shape
        cx = (nx - 1) / 2.0
        cy = (ny - 1) / 2.0
    else:
        cx = cy = 0.0

    def fx(x, pos):  # pixels -> arcmin (X)
        return f"{(x - cx) * sx_am_per_pix:.2f}"

    def fy(y, pos):  # pixels -> arcmin (Y)
        return f"{(y - cy) * sy_am_per_pix:.2f}"

    ax.xaxis.set_major_formatter(FuncFormatter(fx))
    ax.yaxis.set_major_formatter(FuncFormatter(fy))
    ax.set_xlabel("X [arcmin]")
    ax.set_ylabel("Y [arcmin]")


def force_ticks_every_2_arcmin(
    ax, sx_am_per_pix, sy_am_per_pix, *, centered=False, img_shape=None
):
    """
    Put major ticks every 2 arcmin on each axis.
    sx_am_per_pix, sy_am_per_pix: arcmin per pixel (linear pixel scales).
    """
    # convert 2 arcmin to pixels on each axis
    step_x_pix = 2.0 / sx_am_per_pix
    step_y_pix = 2.0 / sy_am_per_pix

    # choose offsets so a tick lands at 0 arcmin
    if centered and img_shape is not None:
        ny, nx = img_shape
        cx = (nx - 1) / 2.0
        cy = (ny - 1) / 2.0
        off_x = cx % step_x_pix
        off_y = cy % step_y_pix
        ax.xaxis.set_major_locator(MultipleLocator(base=step_x_pix, offset=cx - off_x))
        ax.yaxis.set_major_locator(MultipleLocator(base=step_y_pix, offset=cy - off_y))
    else:
        ax.xaxis.set_major_locator(MultipleLocator(base=step_x_pix))
        ax.yaxis.set_major_locator(MultipleLocator(base=step_y_pix))

    ax.figure.canvas.draw_idle()


def save_png_same_size():
    """
    Save a PNG of the current view (image + overlays) with the same pixel
    dimensions as the FITS image array. Non-destructive: restores the window.
    """
    if "image_data" not in globals() or image_data is None:
        return

    ny, nx = image_data.shape

    # Ask user for path
    root = tk.Toplevel()
    root.withdraw()
    outfile = filedialog.asksaveasfilename(
        parent=root,
        defaultextension=".png",
        filetypes=[("PNG image", "*.png")],
        title="Save PNG (same size as FITS)",
    )
    root.destroy()
    if not outfile:
        return

    # --- record current figure/axes state
    old_size = fig.get_size_inches().copy()
    old_dpi = fig.dpi
    old_pos = ax.get_position().frozen()
    # Are axes/ticks visible?
    axes_were_on = ax.xaxis.get_visible() or ax.yaxis.get_visible()

    # --- make the figure canvas exactly nx×ny pixels
    # Pick any dpi; set inches to match target pixels
    dpi = 100.0
    fig.set_dpi(dpi)
    fig.set_size_inches(nx / dpi, ny / dpi, forward=True)

    # Fill the canvas with the axes area only (no margins)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_position([0, 0, 1, 1])
    ax.set_axis_off()  # hide ticks/labels/border

    # Ensure no resampling that would blur pixel mapping
    # (only affects image draw; your overlays are vector)
    # If you want exact pixel mapping, keep what you used in imshow:
    # ax.imshow(..., interpolation='nearest')
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

    # --- write PNG
    # facecolor/transparent are your call; leaving opaque is typical for figures
    fig.savefig(outfile, dpi=dpi, bbox_inches=None, pad_inches=0)

    # --- restore prior state
    ax.set_position(old_pos)
    if axes_were_on:
        ax.set_axis_on()
    else:
        ax.set_axis_off()
    fig.set_dpi(old_dpi)
    fig.set_size_inches(old_size, forward=True)
    fig.canvas.draw_idle()
    print(f"Saved PNG: {outfile}  ({nx}×{ny} pixels)")


def _plot_image_contour(ax, x, y, color="lime", lw=1.5, alpha=0.9, max_jump=10):
    """x,y are DS9 image coords (1-based). Break at large jumps."""
    if len(x) < 2:
        return
    # convert to 0-based pixel coords
    xx = np.asarray(x, float) - 1.0
    yy = np.asarray(y, float) - 1.0

    # find big jumps between consecutive vertices
    dx = np.diff(xx)
    dy = np.diff(yy)
    jumps = np.hypot(dx, dy) > max_jump

    # indices to split on
    split_idx = np.where(jumps)[0] + 1
    parts = np.split(np.arange(len(xx)), split_idx)

    for idx in parts:
        if idx.size >= 2:
            ax.plot(xx[idx], yy[idx], color=color, lw=lw, alpha=alpha)


def load_ds9_contours_txt(path):
    """
    Parse a DS9 'Contour -> Save...' text file.
    Supports:
      - 'image' coords: lines with x y (1-based)
      - 'fk5' coords: lines with ra dec (deg or sexagesimal RA/Dec)
      - multiple contours separated by blank lines or 'contour' lines
      - ignores directive lines like 'global ...', 'levels ...', 'color=...', etc.
    Returns: {'system': 'image'|'fk5', 'contours': [(x[], y[]), ...]}
    """
    import re

    import numpy as _np

    # regex for a float (includes scientific notation)
    float_re = re.compile(
        r"""
        [+-]?                                    # sign
        (?:
          (?:\d+\.\d*|\.\d+|\d+)                 # number
          (?:[eE][+-]?\d+)?                      # optional exponent
        )
    """,
        re.VERBOSE,
    )

    def _is_directive(s):
        low = s.lower()
        if not low:
            return True
        # lines starting with these keywords are not data vertices
        starts = (
            "global",
            "levels",
            "level",
            "color",
            "width",
            "dash",
            "font",
            "system",
            "image",
            "fk5",
            "contour",
            "coords",
            "wcs",
            "projection",
            "file",
            "name",
        )
        return low.startswith(starts) or "=" in low

    system = "image"
    contours, curx, cury = [], [], []

    with open(path) as f:
        for raw in f:
            s = raw.strip()
            if not s:
                if curx:
                    contours.append((_np.array(curx, float), _np.array(cury, float)))
                    curx, cury = [], []
                continue
            low = s.lower()
            if low.startswith("image"):
                system = "image"
                continue
            if _is_directive(s):
                # skip directive lines
                continue
            nums = float_re.findall(s)
            if len(nums) >= 2:
                x, y = float(nums[0]), float(nums[1])
                curx.append(x)
                cury.append(y)
                continue
            # otherwise ignore the line

    if curx:
        contours.append((np.array(curx, float), np.array(cury, float)))
    if not contours:
        raise RuntimeError("No contour vertices parsed (file format not recognized).")
    return {"system": system, "contours": contours}


def _clear_contours():
    global contour_artists
    for art in contour_artists:
        try:
            art.remove()
        except Exception:
            pass
    contour_artists = []
    fig.canvas.draw_idle()


def load_and_draw_ds9_contours():
    """Open a file dialog, parse DS9 contours, project to pixels, and plot."""
    global contour_artists, wcs_obj

    # pick file
    root = tk.Toplevel()
    root.withdraw()
    path = filedialog.askopenfilename(
        parent=root,
        title="Load DS9 Contours",
        filetypes=[
            ("DS9 contours", "*.txt *.ctr *.con *.dat *.cnt *.reg"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not path:
        return

    cd = load_ds9_contours_txt(path)

    # convert to pixel coordinates if needed
    curves_pix = []
    if cd["system"] == "image":
        for x, y in cd["contours"]:
            _plot_image_contour(
                ax, x, y, color="deepskyblue", lw=1.0, alpha=0.9, max_jump=10
            )
        fig.canvas.draw_idle()
    else:
        tk.messagebox.showerror("Contours", f"Unknown coord system {cd['system']}")
        return

    # draw on current axes
    for x, y in curves_pix:
        (line,) = ax.plot(x, y, color="deepskyblue", lw=1.0, alpha=0.9)
        contour_artists.append(line)
    fig.canvas.draw_idle()


# Function to open a FITS file
def open_file():
    root = tk.Toplevel()
    root.withdraw()

    selected_file = filedialog.askopenfilename(
        parent=root,
        title="Select a FITS file",
        filetypes=[("FITS files", "*.fits"), ("All files", "*.")],
    )

    root.destroy()

    if selected_file:
        load_fits(selected_file)


def _read_points_csv(csv_file):
    import csv

    new_points = []

    with open(csv_file, "r", newline="") as csvfile:
        reader = csv.reader(csvfile)

        first_row = next(reader, None)
        if first_row is None:
            raise RuntimeError("CSV file is empty.")

        def parse_xy(row):
            if len(row) < 2:
                raise RuntimeError(f"Bad row: {row}")
            return float(row[0]), float(row[1])

        try:
            new_points.append(parse_xy(first_row))
        except Exception:
            h0 = first_row[0].strip().lower() if len(first_row) > 0 else ""
            h1 = first_row[1].strip().lower() if len(first_row) > 1 else ""
            if not ((h0 == "x" and h1 == "y") or (h0 == "x_pix" and h1 == "y_pix")):
                raise RuntimeError(
                    "First row is neither numeric data nor a valid x,y header."
                )

        for row in reader:
            if not row or all(str(v).strip() == "" for v in row):
                continue
            new_points.append(parse_xy(row))

    if len(new_points) == 0:
        raise RuntimeError("No valid spline points found in CSV.")

    return new_points


def _choose_spline_csv(title):
    root = tk.Toplevel()
    root.withdraw()
    csv_file = filedialog.askopenfilename(
        parent=root,
        title=title,
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
    )
    root.destroy()

    if not csv_file:
        return None, None

    return _read_points_csv(csv_file), csv_file


def load_points():
    global points, spline_visible, dragging_idx

    if image_data is None:
        messagebox.showerror("Load Spline 1 CSV", "Load a FITS image first.")
        return

    try:
        new_points, csv_file = _choose_spline_csv("Load spline 1 CSV")
        if new_points is None:
            return

        points[:] = new_points
        dragging_idx = None
        spline_visible = len(points) > 1
        render_scene()

        messagebox.showinfo(
            "Load Spline 1 CSV",
            f"Loaded {len(points)} control points from:\n{os.path.basename(csv_file)}",
        )
    except Exception as e:
        messagebox.showerror("Load Spline 1 CSV", f"Failed to load spline:\n{e}")


def load_overlay_points():
    global overlay_points, overlay_visible

    if image_data is None:
        messagebox.showerror("Load Spline 2 CSV", "Load a FITS image first.")
        return

    try:
        new_points, csv_file = _choose_spline_csv("Load spline 2 CSV")
        if new_points is None:
            return

        overlay_points[:] = new_points
        overlay_visible = len(overlay_points) > 1
        render_scene()

        messagebox.showinfo(
            "Load Spline 2 CSV",
            f"Loaded {len(overlay_points)} control points from:\n{os.path.basename(csv_file)}",
        )
    except Exception as e:
        messagebox.showerror("Load Spline 2 CSV", f"Failed to load spline:\n{e}")


# Function to save points to CSV
def save_points():
    import csv

    if len(points) == 0:
        return

    base_name = os.path.splitext(os.path.basename(fits_file))[0]
    csv_filename = f"{base_name}_spline.csv"
    with open(csv_filename, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["x", "y"])
        for px, py in points:
            csvwriter.writerow([px, py])

    def show_temp_message(title, message, timeout=2000):
        temp_root = tk.Tk()
        temp_root.withdraw()
        top = tk.Toplevel(temp_root)
        top.title(title)
        tk.Label(top, text=message, padx=20, pady=10).pack()
        top.after(timeout, top.destroy)
        temp_root.after(timeout + 100, temp_root.destroy)
        temp_root.mainloop()

    show_temp_message(
        "Save Successful", f"Clicked points saved to {csv_filename}", timeout=2000
    )


def clear_points():
    global dragging_idx, spline_visible, overlay_visible

    points.clear()
    overlay_points.clear()
    dragging_idx = None
    spline_visible = False
    overlay_visible = False
    render_scene()


def draw_spline():
    global spline_visible
    spline_visible = True
    render_scene()


# Function to load the FITS file
import tkinter.ttk as ttk  # Make sure this is imported at the top


def load_fits(selected_file):
    import json

    global galaxy_group, points, contrast_factor, fits_file, image_data, image_scale, startup_text, sx_am, sy_am, spline_visible, overlay_points, overlay_visible
    contrast_factor = 1.0
    points.clear()
    overlay_points.clear()
    spline_visible = False
    overlay_visible = False

    ax.clear()

    # Show 'Loading...' text
    startup_text = ax.text(
        0.5,
        0.5,
        "Loading...",
        color="gray",
        ha="center",
        va="center",
        fontsize=20,
        transform=ax.transAxes,
    )
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Now load the FITS file
    fits_file = selected_file
    hdr = fits.getheader(fits_file, extname="MU_OBS")
    image_data = fits.getdata(fits_file, extname="MU_OBS")
    # compute pixel scales in arcmin
    sx_am, sy_am = _pixel_scales_arcmin(hdr)
    image_scale = sx_am  # arcmin per pixel along X
    # NEW: set WCS and clear previously drawn contours
    global wcs_obj
    try:
        wcs_obj = WCS(hdr)
    except Exception:
        wcs_obj = None
    _clear_contours()
    m = np.isfinite(image_data) & (image_data > 28.5)
    image_data[m] = 40.0
    points.clear()
    render_scene(preserve_limits=False)
    update_window_title()


# Create a menu bar
menu_bar = tk.Menu(manager.window)
file_menu = tk.Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Open", accelerator="Ctrl+O", command=open_file)
file_menu.add_command(
    label="Save PNG (overlays)", accelerator="Ctrl+P", command=save_png_same_size
)
file_menu.add_separator()
file_menu.add_command(label="Quit", accelerator="Ctrl+Q", command=manager.window.quit)
menu_bar.add_cascade(label="File", menu=file_menu)
# Create a Spline menu
spline_menu = tk.Menu(menu_bar, tearoff=0)
spline_menu.add_command(label="Draw Spline", accelerator="Ctrl+D", command=draw_spline)
spline_menu.add_command(
    label="Load Spline 1 CSV", accelerator="Ctrl+I", command=load_points
)
spline_menu.add_command(label="Load Spline 2 CSV", command=load_overlay_points)
spline_menu.add_command(
    label="Save Spline 1 CSV", accelerator="Ctrl+S", command=save_points
)
spline_menu.add_command(
    label="Clear All Splines", accelerator="Ctrl+C", command=clear_points
)
spline_menu.add_separator()
spline_menu.add_command(
    label="Add Point Mode", accelerator="A", command=lambda: set_move_mode(False)
)
spline_menu.add_command(
    label="Move Point Mode", accelerator="M", command=lambda: set_move_mode(True)
)
spline_menu.add_command(
    label="Toggle Control Points", accelerator="T", command=toggle_control_points
)
menu_bar.add_cascade(label="Spline", menu=spline_menu)

# Overlays menu
over_menu = tk.Menu(menu_bar, tearoff=0)
over_menu.add_command(
    label="Load DS9 Contours…", accelerator="Ctrl+L", command=load_and_draw_ds9_contours
)
over_menu.add_command(label="Clear Contours", command=_clear_contours)
menu_bar.add_cascade(label="Overlays", menu=over_menu)

# Create a View menu
view_menu = tk.Menu(menu_bar, tearoff=0)
view_menu.add_command(
    label="Reset Contrast", accelerator="Ctrl+R", command=lambda: reset_contrast()
)
view_menu.add_command(
    label="Zoom to Points", accelerator="Ctrl+Shift+Z", command=lambda: zoom_to_points()
)
menu_bar.add_cascade(label="View", menu=view_menu)

manager.window.config(menu=menu_bar)

# Bind Ctrl+O
manager.window.bind_all("<Control-o>", lambda event: open_file())
# Bind Ctrl+L
manager.window.bind_all("<Control-l>", lambda event: load_points())
# Bind Ctrl+S
manager.window.bind_all("<Control-s>", lambda event: save_points())
# Bind Ctrl+Q
manager.window.bind_all("<Control-q>", lambda event: manager.window.quit())
# Bind Ctrl+P
manager.window.bind_all("<Control-p>", lambda event: save_png_same_size())
# Bind Ctrl+D
manager.window.bind_all("<Control-d>", lambda event: draw_spline())
# Bind Ctrl+C
manager.window.bind_all("<Control-c>", lambda event: clear_points())
# Bind Ctrl+R
manager.window.bind_all("<Control-r>", lambda event: reset_contrast())
# Bind Ctrl+T
manager.window.bind_all("<Control-t>", lambda event: load_and_draw_ds9_contours())
# Bind Ctrl+Shift+Z
manager.window.bind_all("<Control-Shift-Z>", lambda event: zoom_to_points())


def reset_contrast():
    global contrast_factor
    contrast_factor = 1.0
    render_scene()


def update_contrast():
    render_scene()


def on_button_press(event):
    global dragging_idx

    if image_data is None:
        return
    if event.inaxes != ax:
        return
    if fig.canvas.manager.toolbar.mode != "":
        return
    if event.button != 1:
        return
    if event.xdata is None or event.ydata is None:
        return

    if move_mode:
        idx = nearest_point_index(event, radius_px=pick_radius_px)
        if idx is not None:
            dragging_idx = idx
            render_scene(highlight_idx=dragging_idx)
    else:
        points.append((float(event.xdata), float(event.ydata)))
        render_scene()


def on_mouse_move(event):
    global dragging_idx

    if dragging_idx is None:
        return
    if event.inaxes != ax:
        return
    if event.xdata is None or event.ydata is None:
        return

    points[dragging_idx] = (float(event.xdata), float(event.ydata))
    render_scene(highlight_idx=dragging_idx)


def on_button_release(event):
    global dragging_idx

    if dragging_idx is None:
        return

    if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
        points[dragging_idx] = (float(event.xdata), float(event.ydata))

    dragging_idx = None
    render_scene()


def zoom_to_points():
    if len(points) > 0:
        x_vals, y_vals = zip(*points)
        ax.set_xlim(min(x_vals) - 10, max(x_vals) + 10)
        ax.set_ylim(min(y_vals) - 10, max(y_vals) + 10)
        fig.canvas.draw_idle()


# Event handler for key press
def onkey(event):
    global points, contrast_factor, fits_file, image_data
    if event.key == "+":
        contrast_factor *= 1.1
        update_contrast()

    elif event.key == "-":
        contrast_factor /= 1.1
        update_contrast()

    elif event.key == "r":
        contrast_factor = 1.0
        update_contrast()
    elif event.key == "m":
        set_move_mode(True)

    elif event.key == "a":
        set_move_mode(False)

    elif event.key == "c":
        clear_points()

    elif event.key == "t":
        toggle_control_points()


def _on_xlim_changed(_ax):
    set_centered_arcmin_ticks(ax, sx_am, sy_am, image_data.shape, 2.0, 0.5)


def _on_ylim_changed(_ax):
    set_centered_arcmin_ticks(ax, sx_am, sy_am, image_data.shape, 2.0, 0.5)


cid_press = fig.canvas.mpl_connect("button_press_event", on_button_press)
cid_move = fig.canvas.mpl_connect("motion_notify_event", on_mouse_move)
cid_release = fig.canvas.mpl_connect("button_release_event", on_button_release)
cid_key = fig.canvas.mpl_connect("key_press_event", onkey)

plt.show()
