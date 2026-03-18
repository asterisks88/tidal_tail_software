#!/usr/bin/env python3
import argparse
import json
import math
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import astropy.units as u
import h5py
import numpy as np
import pyvista as pv
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.cosmology import FlatLambdaCDM, Planck18
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from vtkmodules.vtkInteractionWidgets import (
    vtkDistanceRepresentation3D,
    vtkDistanceWidget,
)

paused = False
running = True
current_frame = 0
start_frame = 0
end_frame = 0
update_frame = False
frame_step = 1  # used by key handlers

# Constants
M_SUN_R = 4.64  # Absolute magnitude of Sun in SDSS r-band (approximate)
ARCSEC_IN_DEG = 3600.0
LEGACY_PIXEL_WIDTH = 0.262  # arcsec
LEGACY_PIXEL_AREA = LEGACY_PIXEL_WIDTH**2

# Define colors and labels for PartTypes
type_map = {
    "gas": (0, "blue", "Gas"),
    "dm": (1, "grey", "Dark Matter"),
    "disk": (2, "black", "Disk"),
    "bulge": (3, "orange", "Bulge"),
    "stars": (4, "magenta", "Stars"),
    "bh": (5, "cyan", "Black Holes"),
}


# ----------------------------
# Camera save/load (JSON)
# ----------------------------
def _camera_to_dict(p: pv.Plotter) -> dict:
    # Camera position (pos, focal, view-up) is the most stable representation
    pos, focal, viewup = p.camera_position
    cam = p.camera

    d = {
        "camera_position": [list(pos), list(focal), list(viewup)],
    }

    # Persist a few projection params to improve reproducibility
    try:
        d["clipping_range"] = list(cam.clipping_range)
    except Exception:
        try:
            d["clipping_range"] = list(cam.GetClippingRange())
        except Exception:
            pass

    try:
        d["parallel_projection"] = bool(cam.parallel_projection)
        d["parallel_scale"] = float(cam.parallel_scale)
    except Exception:
        try:
            d["parallel_projection"] = bool(cam.GetParallelProjection())
            d["parallel_scale"] = float(cam.GetParallelScale())
        except Exception:
            pass

    try:
        d["view_angle"] = float(cam.view_angle)
    except Exception:
        try:
            d["view_angle"] = float(cam.GetViewAngle())
        except Exception:
            pass

    return d


def _apply_camera_dict(p: pv.Plotter, d: dict) -> None:
    cam = p.camera

    if "camera_position" in d:
        p.camera_position = d["camera_position"]

    if "clipping_range" in d:
        try:
            cam.clipping_range = tuple(d["clipping_range"])
        except Exception:
            try:
                cam.SetClippingRange(*d["clipping_range"])
            except Exception:
                pass

    if "parallel_projection" in d:
        try:
            cam.parallel_projection = bool(d["parallel_projection"])
        except Exception:
            try:
                cam.SetParallelProjection(1 if d["parallel_projection"] else 0)
            except Exception:
                pass

    if "parallel_scale" in d:
        try:
            cam.parallel_scale = float(d["parallel_scale"])
        except Exception:
            try:
                cam.SetParallelScale(float(d["parallel_scale"]))
            except Exception:
                pass

    if "view_angle" in d:
        try:
            cam.view_angle = float(d["view_angle"])
        except Exception:
            try:
                cam.SetViewAngle(float(d["view_angle"]))
            except Exception:
                pass

    # Ensure renderer updates
    try:
        p.render()
    except Exception:
        pass


def save_camera_json(p: pv.Plotter, path: str) -> None:
    out = Path(path).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_camera_to_dict(p), indent=2, sort_keys=True))
    print(f"Saved camera to: {out}")


def load_camera_json(p: pv.Plotter, path: str) -> None:
    inp = Path(path).expanduser()
    d = json.loads(inp.read_text())
    _apply_camera_dict(p, d)
    print(f"Loaded camera from: {inp}")


def make_ruler_widget(interactor):
    """Create a fresh, fully‐styled 3D ruler widget (off by default)."""
    # 1) Create the 3D representation
    rep = vtkDistanceRepresentation3D()
    rep.SetLabelFormat("%-#6.2f")  # two decimals

    # 2) Hook it up to a DistanceWidget
    widget = vtkDistanceWidget()
    widget.SetInteractor(interactor)
    widget.SetRepresentation(rep)

    # 3) Build handles, line, and text geometry
    widget.CreateDefaultRepresentation()

    # 4) Style the handles
    rep.GetPoint1Representation().SetHandleSize(0.02)
    rep.GetPoint2Representation().SetHandleSize(0.02)

    # 5) Style the line
    rep.GetLineProperty().SetColor(0.0, 1.0, 0.0)
    rep.GetLineProperty().SetLineWidth(4)

    # 6) Style the handle spheres
    rep.GetPoint1Representation().GetProperty().SetColor(0.0, 1.0, 0.0)
    rep.GetPoint2Representation().GetProperty().SetColor(0.0, 1.0, 0.0)

    # 7) Grab the vtkFollower that draws the distance text
    label_follower = rep.GetLabelActor()

    # 8) Tint it red
    label_follower.GetProperty().SetColor(0.0, 0.0, 0.0)

    label_follower.SetScale(0.003, 0.003, 0.003)

    # 10) Hide everything to start
    widget.Off()
    rep.VisibilityOff()

    return widget, rep


def load_json_arg(s: str) -> dict:
    """
    Accept either a path to a JSON file or an inline JSON string.
    """
    p = Path(s).expanduser()
    if p.exists() and p.is_file():
        return json.loads(p.read_text())
    return json.loads(s)


def read_brick_wcs_shape(brick_file: Path, brick_hdu: int = 1):
    """
    Return (ny, nx), header, WCS, pixel_scale_arcsec (mean of axes).
    """
    with fits.open(brick_file, memmap=True) as hdul:
        data = hdul[brick_hdu].data
        hdr = hdul[brick_hdu].header.copy()

    if data is None:
        raise RuntimeError(f"Brick HDU {brick_hdu} has no image data.")

    ny, nx = data.shape
    wcs = WCS(hdr)

    # deg/pix for each axis; take mean (Legacy coadds are essentially square pixels)
    scales_deg = proj_plane_pixel_scales(wcs)  # array-like length 2
    pixscale_arcsec = float(np.mean(scales_deg) * ARCSEC_IN_DEG)

    return (ny, nx), hdr, wcs, pixscale_arcsec


def make_host_centered_wcs_header(
    *,
    brick_wcs: WCS,
    brick_hdr: fits.Header,
    host_image_size: tuple[int, int],  # (W, H)
    host_ra_deg: float,
    host_dec_deg: float,
) -> fits.Header:
    """
    Build a FITS header for a host-centered synthetic image of size (W,H)
    using the *same* pixel scale/orientation as the brick WCS, but with
    CRVAL at the host RA/Dec and CRPIX at the image center.

    Notes:
      - FITS CRPIX is 1-based. Center pixel is (W+1)/2, (H+1)/2.
      - We start from the brick header to preserve CD/PC/CDELT/CTYPE/CUNIT
        and any SIP/distortion keywords if present.
    """
    W, H = map(int, host_image_size)

    # Start from brick header (preserves WCS + any distortion keywords better than to_header())
    hdr = brick_hdr.copy()

    # Update the output image dimensions (not strictly required, but good practice)
    hdr["NAXIS"] = 2
    hdr["NAXIS1"] = W
    hdr["NAXIS2"] = H

    # Force WCS to be host-centered
    hdr["CRVAL1"] = float(host_ra_deg)
    hdr["CRVAL2"] = float(host_dec_deg)
    hdr["CRPIX1"] = (W + 1.0) / 2.0
    hdr["CRPIX2"] = (H + 1.0) / 2.0

    # Remove IRAF-style linear translation keywords if present (can introduce offsets)
    for k in ("LTV1", "LTV2", "LTM1_1", "LTM2_2", "LTM1_2", "LTM2_1", "WCSDIM"):
        if k in hdr:
            del hdr[k]

    # Ensure WCSAXES is consistent if present
    hdr["WCSAXES"] = 2

    return hdr


def _read_coords_masses_pids(h5f, part_idx):
    """
    Returns coords(N,3), masses(N), pids(N) or (None,None,None) if absent.
    Masses are scaled by 1e10 like your existing helper.
    """
    grp_name = f"PartType{part_idx}"
    if grp_name not in h5f:
        return None, None, None
    grp = h5f[grp_name]
    if "Coordinates" not in grp:
        return None, None, None

    coords = np.asarray(grp["Coordinates"][()], dtype=np.float64)
    if coords is None:
        coords = np.empty((0, 3), dtype=np.float64)

    if "Masses" in grp:
        masses = 1e10 * np.asarray(grp["Masses"][()], dtype=np.float64)
    elif "Mass" in grp:
        masses = 1e10 * np.asarray(grp["Mass"][()], dtype=np.float64)
    else:
        masses = np.ones(coords.shape[0], dtype=np.float64)

    pids = None
    if "ParticleIDs" in grp:
        pids = np.asarray(grp["ParticleIDs"][()], dtype=np.int64)

    return coords, masses, pids


def _decode_header_str_array(arr) -> list[str]:
    """
    h5py sometimes gives bytes, fixed-width bytes, or numpy objects. Normalize to list[str].
    """
    if arr is None:
        return []
    out = []
    for v in np.asarray(arr):
        if isinstance(v, (bytes, np.bytes_)):
            out.append(v.decode("utf-8", errors="ignore"))
        else:
            out.append(str(v))
    return out


def _get_galaxy_id_ranges(h5f, galaxyid_map_path=None):
    """
    Uses your header attrs:
      GalaxyID_Name, GalaxyID_Infile, GalaxyID_IDOffset, GalaxyID_IDMin/Max, GalaxyID_IDStart/End

    Returns:
      starts_s, ends_s, names_s, order_s
    where arrays are sorted by starts (increasing).
    """
    hdr = h5f["Header"].attrs
    if "GalaxyID_IDStart" in hdr and "GalaxyID_IDEnd" in hdr:
        starts = np.asarray(hdr["GalaxyID_IDStart"], dtype=np.int64)
        ends = np.asarray(hdr["GalaxyID_IDEnd"], dtype=np.int64)
        names = _decode_header_str_array(hdr.get("GalaxyID_Name", []))
        return starts, ends, names
    else:
        if galaxyid_map_path is None:
            raise RuntimeError(
                "Snapshot missing GalaxyID_* header attrs and no --galaxyid-map provided."
            )

        d = json.loads(Path(galaxyid_map_path).read_text())
        starts = np.asarray(d["GalaxyID_IDStart"], dtype=np.int64)
        ends = np.asarray(d["GalaxyID_IDEnd"], dtype=np.int64)
        names = list(d.get("GalaxyID_Name", []))

        # Optional but useful for debugging / sanity checks
        names = _decode_header_str_array(hdr.get("GalaxyID_Name", None))

    if starts.ndim != 1 or ends.ndim != 1 or starts.size != ends.size:
        raise RuntimeError(
            "GalaxyID_IDStart/End malformed (expected 1D arrays of same length)."
        )

    n = starts.size
    if n < 2:
        raise RuntimeError(
            "Need at least 2 galaxies in GalaxyID_* arrays (host + sat)."
        )

    # Sort by start to make searchsorted mapping robust
    order = np.argsort(starts)
    starts_s = starts[order]
    ends_s = ends[order]
    names_s = [names[i] if i < len(names) else f"gal{i}" for i in order]

    # Basic sanity: non-overlap / monotonic ends not strictly required, but helps
    if np.any(ends_s < starts_s):
        raise RuntimeError("Found GalaxyID range where IDEnd < IDStart.")

    return starts_s, ends_s, names_s, order


def _galaxy_index_from_pid(
    pids: np.ndarray, starts: np.ndarray, ends: np.ndarray
) -> np.ndarray:
    """
    Map ParticleIDs -> galaxy index using inclusive [start,end] per galaxy.
    starts/ends must be sorted by starts ascending (as returned by _get_galaxy_id_ranges).

    Returns:
      gid: int16 array, values in [0..N-1] or -1
    """
    pids = np.asarray(pids, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)

    gid = np.full(pids.shape, -1, dtype=np.int16)

    # Candidate galaxy is the first whose end >= pid
    idx = np.searchsorted(ends, pids, side="left")
    ok = (idx >= 0) & (idx < ends.size)
    ok &= pids >= starts[idx]
    ok &= pids <= ends[idx]
    gid[ok] = idx[ok].astype(np.int16)

    return gid


def _counts_from_masses(
    masses: np.ndarray, ml: np.ndarray, zp: float, redshift: float, cosmo
) -> np.ndarray:
    """
    Convert stellar mass -> apparent counts (nanomaggies) per particle.
    """
    masses = np.asarray(masses, dtype=np.float64)
    ml = np.asarray(ml, dtype=np.float64)

    # L = M / (M/L) in solar L
    L = np.full_like(masses, np.nan, dtype=np.float64)
    ok = (masses > 0) & (ml > 0) & np.isfinite(masses) & np.isfinite(ml)
    L[ok] = masses[ok] / ml[ok]

    abs_mag = np.full_like(masses, np.inf, dtype=np.float64)
    okL = np.isfinite(L) & (L > 0)
    abs_mag[okL] = M_SUN_R - 2.5 * np.log10(L[okL])

    dl_pc = cosmo.luminosity_distance(redshift).to(u.pc).value
    dm = 5.0 * np.log10(dl_pc / 10.0)
    app_mag = abs_mag + dm

    counts = np.zeros(masses.shape[0], dtype=np.float64)
    okm = np.isfinite(app_mag)
    counts[okm] = 10.0 ** (-0.4 * (app_mag[okm] - float(zp)))
    return counts


def _host_core_centroid(
    px,
    py,
    counts,
    galaxy_id,
    width,
    height,
    pixel_scale_arcsec,
    psf_fwhm_arcsec=0.0,
    core_radius_pix=40,
):
    sel = (
        (galaxy_id == 0)
        & np.isfinite(px)
        & np.isfinite(py)
        & np.isfinite(counts)
        & (counts > 0)
    )
    if np.count_nonzero(sel) < 10:
        return None

    # Bin host light into an image using pixel-center conventions
    img = np.histogram2d(
        py[sel],
        px[sel],
        bins=[height, width],
        range=[[-0.5, height - 0.5], [-0.5, width - 0.5]],
        weights=counts[sel],
    )[0].astype(np.float64)

    if psf_fwhm_arcsec and psf_fwhm_arcsec > 0:
        sigma_pix = (float(psf_fwhm_arcsec) / float(pixel_scale_arcsec)) / (
            2.0 * np.sqrt(2.0 * np.log(2.0))
        )
        img = convolve(
            img,
            Gaussian2DKernel(sigma_pix),
            boundary="extend",
            normalize_kernel=True,
        )

    y_peak, x_peak = np.unravel_index(np.nanargmax(img), img.shape)

    yy, xx = np.indices(img.shape)
    r2 = (xx - x_peak) ** 2 + (yy - y_peak) ** 2
    w = np.where(r2 <= core_radius_pix**2, img, 0.0)

    sw = np.sum(w)
    if sw <= 0:
        return float(x_peak), float(y_peak)

    cx = float(np.sum(xx * w) / sw)
    cy = float(np.sum(yy * w) / sw)
    return cx, cy


def compute_surface_brightness_image(
    particles,
    counts,
    galaxy_id,
    camera,
    image_size,
    zp,
    redshift,
    cosmo,
    pixel_scale_arcsec,
    psf_fwhm_arcsec,
    noise_std,
    output_magnitude=False,
    host_target_xy=None,
    sat_target_xy=None,
    masses=None,
    tail_bbox=None,
    m_sat_total_overall=None,
    m_host_total_overall=None,
):
    """
    particles: (N,3) kpc
    counts:    (N,) nanomaggies per particle (already includes M/L and distance modulus)
    galaxy_id: (N,) with 0=host, 1=sat (required for 2-anchor alignment)
    image_size: (width, height) in pixels (should match brick)
    host_target_xy / sat_target_xy: target pixels in brick coordinates (0-based floats)

    Returns:
      img, pixel_scale_arcsec
    """

    width, height = image_size
    da = cosmo.angular_diameter_distance(redshift).to(u.kpc).value  # kpc

    # Camera basis
    cam_pos = np.array(camera.GetPosition(), dtype=np.float64)
    focal = np.array(camera.GetFocalPoint(), dtype=np.float64)
    view_up = np.array(camera.GetViewUp(), dtype=np.float64)

    forward = focal - cam_pos
    forward /= np.linalg.norm(forward) + 1e-30

    # Make "up" be the camera's up direction, orthogonalized to forward
    up = view_up - np.dot(view_up, forward) * forward
    up /= np.linalg.norm(up) + 1e-30

    # Right-handed screen basis
    right = np.cross(forward, up)
    right /= np.linalg.norm(right) + 1e-30

    # FOV derived from pixel scale (independent axes; critical for non-square images)
    hfov_deg = width * float(pixel_scale_arcsec) / ARCSEC_IN_DEG
    vfov_deg = height * float(pixel_scale_arcsec) / ARCSEC_IN_DEG
    hfov_rad = np.deg2rad(hfov_deg)
    vfov_rad = np.deg2rad(vfov_deg)

    # Project particles
    delta = particles - focal
    parallel_comp = np.dot(delta, forward)[:, None] * forward[None, :]
    transverse = delta - parallel_comp

    x_phys = np.dot(transverse, right)
    y_phys = np.dot(transverse, up)

    angle_x = x_phys / da
    angle_y = y_phys / da

    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    sx = (width - 1) / 2.0
    sy = (height - 1) / 2.0
    px = cx + (angle_x / (hfov_rad / 2.0)) * sx
    py = cy + (angle_y / (vfov_rad / 2.0)) * sy

    valid = (
        (px >= -0.5)
        & (px < width - 0.5)
        & (py >= -0.5)
        & (py < height - 0.5)
        & np.isfinite(px)
        & np.isfinite(py)
        & np.isfinite(counts)
    )
    # px = (angle_x / (hfov_rad / 2.0)) * (width / 2.0) + (width / 2.0)
    # py = (angle_y / (vfov_rad / 2.0)) * (height / 2.0) + (height / 2.0)

    #    valid = (
    #        (px >= 0)
    #        & (px < width)
    #        & (py >= 0)
    #        & (py < height)
    #        & np.isfinite(px)
    #        & np.isfinite(py)
    #        & np.isfinite(counts)
    #    )
    if not np.any(valid):
        raise RuntimeError("No particles project inside the field of view.")

    # --- Alignment in pixel space ---
    # If both anchors are available, do a 2-anchor similarity transform (with optional flips).
    # Otherwise, if only host_target_xy is available, do a host-only translation to center host.
    if (host_target_xy is not None) and (galaxy_id is not None):
        gid = np.asarray(galaxy_id, dtype=np.int16)

        # def _weighted_centroid(sel):
        #    sel = sel & valid & (counts > 0)
        print(f"COUNTS {counts}")

        def _weighted_centroid(sel):
            sel = (
                sel
                & np.isfinite(px)
                & np.isfinite(py)
                & np.isfinite(counts)
                & (counts > 0)
            )

            if np.count_nonzero(sel) < 10:
                print("NOT ENOUGH")
                return None

            w = counts[sel]
            sw = np.sum(w)
            if sw <= 0:
                print("SW < 0")
                return None
            cx = float(np.sum(px[sel] * w) / sw)
            cy = float(np.sum(py[sel] * w) / sw)
            return cx, cy

        host_c = _host_core_centroid(
            px=px,
            py=py,
            counts=counts,
            galaxy_id=gid,
            width=width,
            height=height,
            pixel_scale_arcsec=pixel_scale_arcsec,
            psf_fwhm_arcsec=psf_fwhm_arcsec,
            core_radius_pix=40,
        )
        # host_c = _weighted_centroid(gid == 0)
        # sat_c = _weighted_centroid(gid == 1) if (sat_target_xy is not None) else None
        if host_c is not None:
            hx, hy = host_c
            hb_x, hb_y = map(float, host_target_xy)
            print(f"Doing host translation {hx} {hy} to {hb_x, hb_y}")

            px = px + (hb_x - hx)
            py = py + (hb_y - hy)
            sel = (
                (gid == 0)
                & np.isfinite(px)
                & np.isfinite(py)
                & np.isfinite(counts)
                & (counts > 0)
            )
            w = counts[sel]
            hx2 = float(np.sum(px[sel] * w) / np.sum(w))
            hy2 = float(np.sum(py[sel] * w) / np.sum(w))
            print(
                f"post-translation host centroid = ({hx2:.3f}, {hy2:.3f}), "
                f"target = ({hb_x:.3f}, {hb_y:.3f})"
            )
            valid = (
                (px >= 0)
                & (px < width)
                & (py >= 0)
                & (py < height)
                & np.isfinite(px)
                & np.isfinite(py)
                & np.isfinite(counts)
            )
    tail_report = None
    if (tail_bbox is not None) and (masses is not None):
        masses = np.asarray(masses, dtype=np.float64)
        if masses.shape[0] != counts.shape[0]:
            raise RuntimeError(
                "masses and counts must have same length for tail report."
            )
        tail_report = _tail_bbox_report(
            px=px,
            py=py,
            valid=valid,
            counts=counts,
            masses=masses,
            galaxy_id=galaxy_id,
            bbox=tail_bbox,
            width=width,
            height=height,
            m_sat_total_overall=m_sat_total_overall,
            m_host_total_overall=m_host_total_overall,
        )

    # Histogram to image (nanomaggies/pixel if output_magnitude=False)
    H = np.histogram2d(
        py[valid],
        px[valid],
        bins=[height, width],
        range=[[-0.5, height - 0.5], [-0.5, width - 0.5]],
        weights=counts[valid],
    )[0].astype(np.float64)
    #    H = np.histogram2d(
    #        py[valid],
    #        px[valid],
    #        bins=[height, width],
    #        range=[[0, height], [0, width]],
    #        weights=counts[valid],
    #    )[0].astype(np.float64)
    img = H  # nanomaggies/pixel

    # Add Gaussian noise (assumed same units as img)
    if noise_std and noise_std > 0:
        img = img + np.random.normal(0.0, float(noise_std), size=img.shape)

    # PSF convolution (Gaussian)
    if psf_fwhm_arcsec and psf_fwhm_arcsec > 0:
        sigma_pix = (float(psf_fwhm_arcsec) / float(pixel_scale_arcsec)) / (
            2.0 * np.sqrt(2.0 * np.log(2.0))
        )
        kernel = Gaussian2DKernel(sigma_pix)
        img = convolve(img, kernel, boundary="extend", normalize_kernel=True)

    if output_magnitude:
        # Convert to mag/arcsec^2 using pixel area
        pixel_area_arcsec2 = float(pixel_scale_arcsec) ** 2
        intensity_arcsec2 = img / pixel_area_arcsec2
        with np.errstate(divide="ignore", invalid="ignore"):
            mu = zp - 2.5 * np.log10(intensity_arcsec2)
        mu[~np.isfinite(mu)] = 35.0
        mu += 10.0 * np.log10(1.0 + redshift)  # dimming
        img = mu

    return img, float(pixel_scale_arcsec), tail_report


# ----------------------------
# Event handlers
# ----------------------------
def toggle_pause():
    global paused
    paused = not paused
    print("Paused" if paused else "Playing")


def step_forward():
    global update_frame, current_frame, end_frame, frame_step
    if paused and current_frame + frame_step <= end_frame:
        current_frame += frame_step
        update_frame = True


def step_backward():
    global update_frame, current_frame, start_frame, frame_step
    if paused and current_frame - frame_step >= start_frame:
        current_frame -= frame_step
        update_frame = True


def on_window_close(obj=None, ev=None):
    """Triggered when the user clicks the window’s X."""
    global running
    running = False  # stop main loop


def _read_coords_and_masses(h5f, part_idx):
    grp_name = f"PartType{part_idx}"
    if grp_name not in h5f:
        return None
    grp = h5f[grp_name]
    if "Coordinates" not in grp:
        return None
    coords = grp["Coordinates"][()]
    if coords is None:
        coords = np.empty((0, 3), dtype=np.float64)

    if "Masses" in grp:
        masses = 1e10 * grp["Masses"][()]
    elif "Mass" in grp:
        masses = 1e10 * grp["Mass"][()]
    else:
        print("Warning: Fallback used for mass")
        # fallback: if there's a header mass for this type? assume unit mass
        masses = np.ones(coords.shape[0], dtype=float)

    return coords, masses


def _tail_bbox_report(
    px,
    py,
    valid,
    counts,
    masses,
    galaxy_id,
    bbox,
    width,
    height,
    m_sat_total_overall=None,
    m_host_total_overall=None,
):
    """
    Compute flux+mass in bbox, split by host/sat, using per-particle projected pixels.
    bbox: (x0,x1,y0,y1) in 0-based pixel coordinates, Python slicing convention.
    """
    x0, x1, y0, y1 = map(float, bbox)

    # Clip bbox to image bounds
    x0c = max(0.0, min(float(width), x0))
    x1c = max(0.0, min(float(width), x1))
    y0c = max(0.0, min(float(height), y0))
    y1c = max(0.0, min(float(height), y1))

    if not (x1c > x0c and y1c > y0c):
        return {
            "bbox_clipped": (x0c, x1c, y0c, y1c),
            "ok": False,
            "reason": "bbox has no overlap with image after clipping",
        }

    in_bbox = valid & (px >= x0c) & (px < x1c) & (py >= y0c) & (py < y1c)

    gid = np.asarray(galaxy_id, dtype=np.int16) if galaxy_id is not None else None

    out = {
        "ok": True,
        "bbox_clipped": (x0c, x1c, y0c, y1c),
    }

    if gid is None:
        out["ok"] = False
        out["reason"] = "galaxy_id is required to compute satellite-only tail fraction"
        return out

    sat_in_bbox = in_bbox & (gid == 1)
    host_in_bbox = in_bbox & (gid == 0)

    M_sat_bbox = float(np.nansum(masses[sat_in_bbox]))
    M_host_bbox = float(np.nansum(masses[host_in_bbox]))

    out["M_sat_bbox_Msun"] = M_sat_bbox
    out["M_host_bbox_Msun"] = M_host_bbox

    # "Overall" denominators (independent of FOV)
    out["M_sat_total_overall_Msun"] = (
        float(m_sat_total_overall) if m_sat_total_overall is not None else math.nan
    )
    out["M_host_total_overall_Msun"] = (
        float(m_host_total_overall) if m_host_total_overall is not None else math.nan
    )

    if (
        (m_sat_total_overall is not None)
        and np.isfinite(m_sat_total_overall)
        and (m_sat_total_overall > 0)
    ):
        out["tail_mass_fraction_sat_overall"] = M_sat_bbox / float(m_sat_total_overall)
    else:
        out["tail_mass_fraction_sat_overall"] = math.nan

    # Keep the "in-FOV" versions as diagnostics (optional but recommended)
    sat_valid = valid & (gid == 1)
    M_sat_in_fov = float(np.nansum(masses[sat_valid]))
    out["M_sat_in_fov_Msun"] = M_sat_in_fov
    out["tail_mass_fraction_sat_in_fov"] = (
        (M_sat_bbox / M_sat_in_fov)
        if (np.isfinite(M_sat_in_fov) and M_sat_in_fov > 0)
        else math.nan
    )

    return out


def gadget_hdf5_movie(
    snap_format,
    step_size,
    time_delay,
    keep_looping,
    point_size,
    plot_options,
    zp=None,
    psf_fwhm=None,
    noise_std=None,
    mag_out=None,
    mp4_path=None,
    host_info=None,
    sat_info=None,
    galaxyid_map=None,
    host_image_size=(3000, 3000),
    brick_hdr=None,
    brick_wcs=None,
    brick_shape=None,  # (ny, nx)
    brick_pixscale=None,  # arcsec/pix
    fps=None,
    show_window=True,
    window_size=(1280, 720),
    camera_load=None,
    camera_save=None,
    tail_bbox=None,
    debug_photometry=False,
):
    global paused, running, update_frame, current_frame, start_frame, end_frame, frame_step
    frame_step = max(1, int(step_size))

    recording = mp4_path is not None
    if recording and keep_looping:
        print("Note: --loop is ignored when recording an MP4.")
        keep_looping = False

    if recording and fps is None:
        fps = int(round(1.0 / time_delay)) if time_delay > 0 else 30

    # Camera setup (fallback/default)
    camera_angle = np.deg2rad(0)
    camera_pos = [
        (0, 0, 1000),
        (0, 0, 0),
        (np.cos(camera_angle), 0, np.sin(camera_angle)),
    ]

    # Decide which families to plot
    if not any(plot_options.values()):
        families = list(type_map.keys())
    else:
        families = [k for k, v in plot_options.items() if v]

    # Cosmology + parameters
    cosmo = Planck18
    if host_info is None or sat_info is None:
        raise RuntimeError(
            "host_info and sat_info are required for synthetic-image generation."
        )

    z_host = float(host_info["inputs"]["z"])
    z_sat = float(sat_info["inputs"]["z"])
    if abs(z_host - z_sat) > 1e-6:
        print(f"Warning: host z={z_host} differs from sat z={z_sat}; using host z.")
    redshift = z_host

    ml_host = float(host_info["stellar_mass"]["M_over_L_r"])
    ml_sat = float(sat_info["stellar_mass"]["M_over_L_r"])

    host_ra = float(host_info["inputs"]["ra"])
    host_dec = float(host_info["inputs"]["dec"])
    sat_ra = float(sat_info["inputs"]["ra"])
    sat_dec = float(sat_info["inputs"]["dec"])

    if (
        brick_wcs is None
        or brick_hdr is None
        or brick_shape is None
        or brick_pixscale is None
    ):
        raise RuntimeError("--brick-file is required (WCS+shape needed).")

    ny_b, nx_b = brick_shape
    width_b, height_b = nx_b, ny_b

    # Brick target pixels (0-based)
    host_xb, host_yb = brick_wcs.all_world2pix(host_ra, host_dec, 0)
    sat_xb, sat_yb = brick_wcs.all_world2pix(sat_ra, sat_dec, 0)
    host_target_xy = (float(host_xb), float(host_yb))
    sat_target_xy = (float(sat_xb), float(sat_yb))
    print(f"Target cords in brick image:  host {host_target_xy} sat {sat_target_xy}")

    off_screen = recording and (not show_window)
    p = pv.Plotter(
        off_screen=off_screen,
        title="Gadget HDF5 Snapshot Movie Viewer",
        window_size=window_size,
    )

    p.add_axes()
    p.set_background("white")
    p.camera_position = camera_pos

    if show_window:
        p.add_key_event("space", toggle_pause)
        p.add_key_event("Next", step_forward)
        p.add_key_event("Prior", step_backward)
        p.iren.add_observer("WindowCloseEvent", on_window_close)
        p.iren.add_observer("ExitEvent", on_window_close)

    # Open the first frame to initialize datasets + galaxy ID ranges
    first_file = snap_format.format(start_frame)
    with h5py.File(first_file, "r") as f0:
        t_code0 = f0["Header"].attrs.get("Time", 0.0)
        time_myr0 = float(t_code0) * 1e3

        id_starts, id_ends, gal_names, gal_order = _get_galaxy_id_ranges(
            f0, galaxyid_map
        )
        if id_starts is None:
            raise RuntimeError(
                "Missing GalaxyID_IDStart/GalaxyID_IDEnd; cannot separate host vs sat."
            )

        # By your convention:
        # - first galaxy in list is host
        # - second is satellite
        # After sorting, that *should* still be index 0 and 1 if your ID ranges are monotonically increasing,
        # but do not rely on that. Instead, enforce host/sat indices from the ORIGINAL order.

        # gal_order[k] = original index of the k-th sorted range
        # We want the sorted index where original index == 0 (host) and == 1 (sat)
        host_gid = int(np.where(gal_order == 0)[0][0])
        sat_gid = int(np.where(gal_order == 1)[0][0])

        clouds = {}
        actors = {}

        for key in families:
            idx, color, label = type_map[key]
            coords, masses = _read_coords_and_masses(f0, idx)
            if coords is None:
                coords = np.empty((0, 3), dtype=np.float64)

            op = 0.4 if key in ("dm", "bulge") else 1.0
            cloud = pv.PolyData(coords)
            clouds[key] = cloud
            actors[key] = p.add_mesh(
                cloud,
                render_points_as_spheres=True,
                point_size=point_size,
                color=color,
                opacity=op,
                label=label,
            )

    p.add_legend()
    text_actor = p.add_text(
        f"Frame: {start_frame} Time: {time_myr0:.3f} Myr",
        position="lower_left",
        font_size=12,
        color="blue",
    )

    iren = p.iren.interactor
    ruler, rep = make_ruler_widget(iren)
    label_actor = None

    def toggle_ruler():
        nonlocal ruler, rep, label_actor
        if ruler.GetEnabled():
            ruler.Off()
            rep.VisibilityOff()
            if label_actor:
                p.remove_actor(label_actor)
                label_actor = None
        else:
            ruler, rep = make_ruler_widget(iren)
            rep.VisibilityOn()
            ruler.On()
            label_actor = p.add_text(
                "RULER",
                position="lower_right",
                font_size=16,
                color="green",
                name="ruler_label",
            )
        p.render()

    p.add_key_event("l", toggle_ruler)

    def _load_luminous_frame(frame_num: int):
        """
        Load disk+bulge+stars for this frame, return coords, counts, galaxy_id.
        galaxy_id: 0=host, 1=sat
        """
        frame_file = snap_format.format(frame_num)
        coords_list = []
        counts_list = []
        gid_list = []
        masses_list = []

        with h5py.File(frame_file, "r") as f:
            id_starts, id_ends, gal_names, gal_order = _get_galaxy_id_ranges(
                f, galaxyid_map
            )
            if id_starts is None:
                raise RuntimeError("GalaxyID ranges missing in this snapshot header.")

            for key in ("disk", "bulge", "stars"):
                part_idx = type_map[key][0]
                coords, masses, pids = _read_coords_masses_pids(f, part_idx)
                if (
                    coords is None
                    or masses is None
                    or pids is None
                    or coords.shape[0] == 0
                ):
                    continue

                gid = _galaxy_index_from_pid(pids, id_starts, id_ends)

                ok = (gid == host_gid) | (gid == sat_gid)
                if not np.any(ok):
                    continue

                coords = coords[ok]
                masses = masses[ok]
                gid = gid[ok]

                # Map to M/L using your host/sat identity
                ml = np.where(gid == host_gid, ml_host, ml_sat).astype(np.float64)
                cnt = _counts_from_masses(
                    masses, ml, zp=float(zp), redshift=redshift, cosmo=cosmo
                )

                # IMPORTANT: normalize galaxy_id to 0=host, 1=sat for the alignment code
                gid01 = np.where(gid == host_gid, 0, 1).astype(np.int16)

                coords_list.append(coords)
                counts_list.append(cnt)
                gid_list.append(gid01)
                masses_list.append(masses.astype(np.float64))

        if not coords_list:
            raise RuntimeError("No luminous particles found for this frame.")
        coords_all = np.vstack(coords_list).astype(np.float64)
        counts_all = np.concatenate(counts_list).astype(np.float64)
        gid_all = np.concatenate(gid_list).astype(np.int16)
        masses_all = np.asarray(np.concatenate(masses_list), dtype=np.float64)

        # "Overall" (independent of FOV): total luminous stellar mass per galaxy from the selected particles
        m_host_total = float(np.nansum(masses_all[gid_all == 0]))
        m_sat_total = float(np.nansum(masses_all[gid_all == 1]))

        return coords_all, counts_all, gid_all, masses_all, m_host_total, m_sat_total

    # 's' key handler: synthesize image aligned to brick and save fits
    def save_surface_brightness():
        global current_frame
        try:
            coords, counts, gid, masses, m_host_total, m_sat_total = (
                _load_luminous_frame(current_frame)
            )
        except Exception as e:
            import traceback

            print(traceback.format_exc())
            print(f"[Error] Failed to load luminous particles: {e}", file=sys.stderr)
            return

        if debug_photometry:

            def _mag_from_nmgy(f_nmgy: float) -> float:
                f = float(f_nmgy)
                if not np.isfinite(f) or f <= 0:
                    return float("nan")
                return float(zp) - 2.5 * np.log10(f)

            # Integrated flux per galaxy from particle counts (nmgy)
            f_host = float(np.sum(counts[gid == 0]))
            f_sat = float(np.sum(counts[gid == 1]))

            m_host_model = _mag_from_nmgy(f_host)
            m_sat_model = _mag_from_nmgy(f_sat)

            m_host_obs = float(host_info["inputs"]["r"])
            m_sat_obs = float(sat_info["inputs"]["r"])

            dm_host = m_host_model - m_host_obs
            dm_sat = m_sat_model - m_sat_obs

            # flux ratio model/obs implied by dm (10^{-0.4 dm})
            fr_host = 10.0 ** (-0.4 * dm_host) if np.isfinite(dm_host) else float("nan")
            fr_sat = 10.0 ** (-0.4 * dm_sat) if np.isfinite(dm_sat) else float("nan")

            print(
                "[DebugPhot] Integrated apparent magnitudes from particle counts (AB):"
            )
            print(
                f"[DebugPhot]   Host: F={f_host:.6e} nmgy  m_model={m_host_model:.4f}  m_obs(r)={m_host_obs:.4f}  dm={dm_host:+.4f}  (F_model/F_obs={fr_host:.3f})"
            )
            print(
                f"[DebugPhot]   Sat : F={f_sat:.6e} nmgy  m_model={m_sat_model:.4f}  m_obs(r)={m_sat_obs:.4f}  dm={dm_sat:+.4f}  (F_model/F_obs={fr_sat:.3f})"
            )

        try:
            camera = p.camera
            img, pixscale, tail_report = compute_surface_brightness_image(
                coords,
                counts,
                gid,
                camera,
                (width_b, height_b),
                zp=float(zp),
                redshift=redshift,
                cosmo=cosmo,
                pixel_scale_arcsec=float(brick_pixscale),
                psf_fwhm_arcsec=float(psf_fwhm) if psf_fwhm is not None else 0.0,
                noise_std=float(noise_std) if noise_std is not None else 0.0,
                output_magnitude=bool(mag_out),
                host_target_xy=host_target_xy,
                sat_target_xy=sat_target_xy,
                masses=masses,
                tail_bbox=tail_bbox,
                m_sat_total_overall=m_sat_total,
                m_host_total_overall=m_host_total,
            )
        except Exception as e:
            print(f"[Error] Failed to compute synthetic image: {e}", file=sys.stderr)
            return

        if debug_photometry:
            print(
                f"[DebugPhot] Flux rendered in image (sum over pixels, nmgy): {np.nansum(img):.6e}"
            )

        if tail_report is not None and tail_report.get("ok", False):
            x0c, x1c, y0c, y1c = tail_report["bbox_clipped"]
            print(
                f"[TailBBox] bbox (clipped) x=[{x0c:.1f},{x1c:.1f})  y=[{y0c:.1f},{y1c:.1f})"
            )

            M_sat_bbox = tail_report.get("M_sat_bbox_Msun", math.nan)
            M_sat_tot = tail_report.get("M_sat_total_overall_Msun", math.nan)
            f_tail = tail_report.get("tail_mass_fraction_sat_overall", math.nan)

            print(
                "[TailBBox] Tail mass fraction (satellite, OVERALL denom): "
                f"M_sat_bbox={M_sat_bbox:.6e} Msun  "
                f"M_sat_total={M_sat_tot:.6e} Msun  "
                f"f_tail={f_tail:.6f}"
            )

            # Optional diagnostics
            M_sat_fov = tail_report.get("M_sat_in_fov_Msun", math.nan)
            f_tail_fov = tail_report.get("tail_mass_fraction_sat_in_fov", math.nan)
            print(
                "[TailBBox] Diagnostic (satellite, IN-FOV denom): "
                f"M_sat_in_fov={M_sat_fov:.6e} Msun  f_tail_fov={f_tail_fov:.6f}"
            )

        # Prompt for filename via Tkinter
        filename = None
        try:
            root = tk.Tk()
            root.withdraw()
            fn = filedialog.asksaveasfilename(
                title="Save simulated FITS (aligned to brick)",
                defaultextension=".fits",
                filetypes=[("FITS files", "*.fits"), ("All files", "*.*")],
            )
            root.destroy()
            if fn:
                filename = fn
        except Exception:
            filename = input("Enter output FITS filename (e.g., image.fits): ").strip()

        if not filename:
            print("Save cancelled.", file=sys.stderr)
            return

        # Output header: copy brick header so WCS matches exactly
        hdr = brick_hdr.copy()
        hdr["ZP"] = (float(zp), "Zero point magnitude used (nanomaggies convention)")
        hdr["REDSHFT"] = (float(redshift), "Redshift used for distance modulus")
        hdr["MLHOST"] = (float(ml_host), "Host M/L_r (solar units)")
        hdr["MLSAT"] = (float(ml_sat), "Satellite M/L_r (solar units)")
        hdr["HRA"] = (float(host_ra), "Host RA (deg)")
        hdr["HDEC"] = (float(host_dec), "Host Dec (deg)")
        hdr["SRA"] = (float(sat_ra), "Satellite RA (deg)")
        hdr["SDEC"] = (float(sat_dec), "Satellite Dec (deg)")
        hdr["PSF_FWHM"] = (
            float(psf_fwhm) if psf_fwhm is not None else 0.0,
            "PSF FWHM applied (arcsec)",
        )
        hdr["NOISESTD"] = (
            float(noise_std) if noise_std is not None else 0.0,
            "Gaussian noise sigma added (output units)",
        )
        hdr["PIXSCL"] = (float(pixscale), "Pixel scale used in render (arcsec/pix)")
        hdr["FRAME"] = (int(current_frame), "Snapshot frame rendered")
        hdr["GIDH"] = (int(host_gid), "Sorted GalaxyID index used as host")
        hdr["GIDS"] = (int(sat_gid), "Sorted GalaxyID index used as satellite")
        hdr["GNAMEH"] = (
            str(gal_names[host_gid]),
            "GalaxyID_Name for host (from snapshot)",
        )
        hdr["GNAMES"] = (
            str(gal_names[sat_gid]),
            "GalaxyID_Name for satellite (from snapshot)",
        )

        if bool(mag_out):
            hdr["COMMENT"] = "Synthetic image in mag/arcsec^2 aligned to brick WCS."
        else:
            hdr["COMMENT"] = (
                "Synthetic image in nanomaggies/pixel aligned to brick WCS."
            )

        hdu = fits.PrimaryHDU(np.asarray(img, dtype=np.float32), header=hdr)
        try:
            hdu.writeto(filename, overwrite=True)
            print(f"Saved aligned synthetic FITS to: {filename}")
        except Exception as e:
            print(f"[Error] Failed to write FITS file: {e}", file=sys.stderr)

    p.add_key_event("s", save_surface_brightness)

    host_w, host_h = map(int, host_image_size)
    host_center_xy = ((host_w - 1) / 2.0, (host_h - 1) / 2.0)

    # Use the brick-derived sat offset if it fits in the host-centered frame; otherwise omit sat anchor
    sat_dx = float(sat_xb - host_xb)
    sat_dy = float(sat_yb - host_yb)
    sat_target_host = (host_center_xy[0] + sat_dx, host_center_xy[1] + sat_dy)
    if not (0 <= sat_target_host[0] < host_w and 0 <= sat_target_host[1] < host_h):
        sat_target_host = None

    def save_host_centered_surface_brightness():
        global current_frame
        try:
            coords, counts, gid, masses, m_host_total, m_sat_total = (
                _load_luminous_frame(current_frame)
            )
        except Exception as e:
            print(f"[Error] Failed to load luminous particles: {e}", file=sys.stderr)
            return

        if debug_photometry:

            def _mag_from_nmgy(f_nmgy: float) -> float:
                f = float(f_nmgy)
                if not np.isfinite(f) or f <= 0:
                    return float("nan")
                return float(zp) - 2.5 * np.log10(f)

            # Integrated flux per galaxy from particle counts (nmgy)
            f_host = float(np.sum(counts[gid == 0]))
            f_sat = float(np.sum(counts[gid == 1]))

            m_host_model = _mag_from_nmgy(f_host)
            m_sat_model = _mag_from_nmgy(f_sat)

            m_host_obs = float(host_info["inputs"]["r"])
            m_sat_obs = float(sat_info["inputs"]["r"])

            dm_host = m_host_model - m_host_obs
            dm_sat = m_sat_model - m_sat_obs

            # flux ratio model/obs implied by dm (10^{-0.4 dm})
            fr_host = 10.0 ** (-0.4 * dm_host) if np.isfinite(dm_host) else float("nan")
            fr_sat = 10.0 ** (-0.4 * dm_sat) if np.isfinite(dm_sat) else float("nan")

            print(
                "[DebugPhot] Integrated apparent magnitudes from particle counts (AB):"
            )
            print(
                f"[DebugPhot]   Host: F={f_host:.6e} nmgy  m_model={m_host_model:.4f}  m_obs(r)={m_host_obs:.4f}  dm={dm_host:+.4f}  (F_model/F_obs={fr_host:.3f})"
            )
            print(
                f"[DebugPhot]   Sat : F={f_sat:.6e} nmgy  m_model={m_sat_model:.4f}  m_obs(r)={m_sat_obs:.4f}  dm={dm_sat:+.4f}  (F_model/F_obs={fr_sat:.3f})"
            )

        try:
            camera = p.camera
            img, pixscale, tail_report = compute_surface_brightness_image(
                coords,
                counts,
                gid,
                camera,
                (host_w, host_h),
                zp=float(zp),
                redshift=redshift,
                cosmo=cosmo,
                pixel_scale_arcsec=float(brick_pixscale),
                psf_fwhm_arcsec=float(psf_fwhm) if psf_fwhm is not None else 0.0,
                noise_std=float(noise_std) if noise_std is not None else 0.0,
                output_magnitude=bool(mag_out),
                host_target_xy=host_center_xy,
                sat_target_xy=sat_target_host,
            )
        except Exception as e:
            print(
                f"[Error] Failed to compute host-centered synthetic image: {e}",
                file=sys.stderr,
            )
            return

        if debug_photometry:
            print(
                f"[DebugPhot] Flux rendered in image (sum over pixels, nmgy): {np.nansum(img):.6e}"
            )

        # Prompt for filename via Tkinter
        filename = None
        try:
            root = tk.Tk()
            root.withdraw()
            fn = filedialog.asksaveasfilename(
                title="Save simulated FITS (host-centered)",
                defaultextension=".fits",
                filetypes=[("FITS files", "*.fits"), ("All files", "*.*")],
            )
            root.destroy()
            if fn:
                filename = fn
        except Exception:
            filename = input(
                "Enter output FITS filename (e.g., host_centered.fits): "
            ).strip()

        if not filename:
            print("Save cancelled.", file=sys.stderr)
            return

        # Build WCS centered on the host, same orientation/scale as the brick
        hdr = make_host_centered_wcs_header(
            brick_wcs=brick_wcs,
            brick_hdr=brick_hdr,
            host_image_size=host_image_size,  # (3000, 3000) by default
            host_ra_deg=host_ra,
            host_dec_deg=host_dec,
        )

        # Add the same provenance keywords you already write for brick-aligned images
        hdr["ZP"] = (float(zp), "Zero point magnitude used (nanomaggies convention)")
        hdr["REDSHFT"] = (float(redshift), "Redshift used for distance modulus")
        hdr["MLHOST"] = (float(ml_host), "Host M/L_r (solar units)")
        hdr["MLSAT"] = (float(ml_sat), "Satellite M/L_r (solar units)")
        hdr["HRA"] = (float(host_ra), "Host RA (deg)")
        hdr["HDEC"] = (float(host_dec), "Host Dec (deg)")
        hdr["SRA"] = (float(sat_ra), "Satellite RA (deg)")
        hdr["SDEC"] = (float(sat_dec), "Satellite Dec (deg)")
        hdr["PSF_FWHM"] = (
            float(psf_fwhm) if psf_fwhm is not None else 0.0,
            "PSF FWHM applied (arcsec)",
        )
        hdr["NOISESTD"] = (
            float(noise_std) if noise_std is not None else 0.0,
            "Gaussian noise sigma added (output units)",
        )
        hdr["PIXSCL"] = (float(pixscale), "Pixel scale used in render (arcsec/pix)")
        hdr["FRAME"] = (int(current_frame), "Snapshot frame rendered")
        hdr["GIDH"] = (int(host_gid), "Sorted GalaxyID index used as host")
        hdr["GIDS"] = (int(sat_gid), "Sorted GalaxyID index used as satellite")

        hdr["IMTYPE"] = ("HOSTCTR", "Host-centered synthetic render")
        hdr["IMW"] = (int(host_w), "Image width (pixels)")
        hdr["IMH"] = (int(host_h), "Image height (pixels)")

        if bool(mag_out):
            hdr["COMMENT"] = "Synthetic image in mag/arcsec^2, host-centered WCS."
        else:
            hdr["COMMENT"] = "Synthetic image in nanomaggies/pixel, host-centered WCS."
        hdu = fits.PrimaryHDU(np.asarray(img, dtype=np.float32), header=hdr)
        try:
            hdu.writeto(filename, overwrite=True)
            print(f"Saved host aligned synthetic FITS to: {filename}")
        except Exception as e:
            print(f"[Error] Failed to write FITS file: {e}", file=sys.stderr)

    p.add_key_event("h", save_host_centered_surface_brightness)

    # Initialize render window
    p.show(auto_close=False, interactive_update=show_window)

    # Apply requested camera after show()
    if camera_load:
        load_camera_json(p, camera_load)

    if show_window:

        def _save_cam_key():
            save_path = camera_save or "camera.json"
            save_camera_json(p, save_path)

        p.add_key_event("c", _save_cam_key)

    if recording:
        p.open_movie(mp4_path, framerate=fps)

    def _update_scene(frame_num: int):
        nonlocal text_actor
        frame_file = snap_format.format(frame_num)
        with h5py.File(frame_file, "r") as f:
            t_code = f["Header"].attrs.get("Time", 0.0)
            time_myr = float(t_code) * 1e3

            for key in families:
                idx, _, _ = type_map[key]
                coords, masses = _read_coords_and_masses(f, idx)
                if coords is None:
                    coords = np.empty((0, 3), dtype=np.float64)
                clouds[key].points = coords

        try:
            text_actor.SetInput(f"Frame: {frame_num} Time: {time_myr:.3f} Myr")
        except Exception:
            p.remove_actor(text_actor)
            text_actor = p.add_text(
                f"Frame: {frame_num} Time: {time_myr:.3f} Myr",
                position="lower_left",
                font_size=12,
                color="blue",
            )

    current_frame = start_frame

    if recording and (not show_window):
        try:
            for fr in range(start_frame, end_frame + 1, frame_step):
                _update_scene(fr)
                p.render()
                p.write_frame()
        finally:
            p.close()
        return

    try:
        while running:
            while running and current_frame <= end_frame:
                if (not paused) or update_frame:
                    try:
                        _update_scene(current_frame)

                        if show_window:
                            p.update()
                        else:
                            p.render()

                        if recording:
                            p.write_frame()

                        if show_window:
                            time.sleep(time_delay)

                        if not update_frame:
                            current_frame += frame_step
                        else:
                            update_frame = False

                    except Exception as e:
                        print(f"Error on frame {current_frame}: {e}")
                        keep_looping = False
                        break
                else:
                    p.update()
                    time.sleep(time_delay)

            if not keep_looping:
                paused = True
                current_frame = end_frame
            else:
                current_frame = start_frame

    except KeyboardInterrupt:
        print("Ctrl+c Interrupt")
    finally:
        p.close()


def main():
    global start_frame, end_frame

    parser = argparse.ArgumentParser(
        description="Visualize Gadget-4 HDF5 snapshots with PyVista; press 's' to save aligned synthetic image."
    )
    parser.add_argument(
        "pattern", type=str, help="Filename pattern, e.g., output_{:06d}.hdf5"
    )
    parser.add_argument("end_frame", type=int, help="End frame (inclusive)")
    parser.add_argument("-b", "--start_frame", type=int, default=0, help="Start frame")

    parser.add_argument(
        "--point-size", type=float, default=1.0, help="Point glyph size"
    )
    parser.add_argument("-s", "--step", type=int, default=1, help="Frame step")
    parser.add_argument(
        "-d", "--delay", type=float, default=0.1, help="Delay between frames (sec)"
    )
    parser.add_argument("-l", "--loop", action="store_true", help="Loop continuously")

    parser.add_argument(
        "--mp4", type=str, default=None, help="Write MP4 (requires imageio[ffmpeg])"
    )
    parser.add_argument(
        "--fps", type=float, default=None, help="MP4 FPS (default ~1/delay else 30)"
    )
    parser.add_argument(
        "--show", action="store_true", help="Show window while recording MP4"
    )
    parser.add_argument(
        "--window-size", type=int, nargs=2, metavar=("W", "H"), default=(1280, 720)
    )

    parser.add_argument(
        "--camera-load", type=str, default=None, help="Load camera JSON at startup"
    )
    parser.add_argument(
        "--camera-save",
        type=str,
        default="camera_save.json",
        help="Default camera JSON path",
    )

    # New: alignment + per-galaxy photometry inputs
    parser.add_argument(
        "--host-params",
        type=str,
        required=True,
        help="Host params JSON (file path or inline JSON string). Must include inputs.ra/dec/z and stellar_mass.M_over_L_r",
    )
    parser.add_argument(
        "--sat-params",
        type=str,
        required=True,
        help="Satellite params JSON (file path or inline JSON string). Must include inputs.ra/dec/z and stellar_mass.M_over_L_r",
    )
    parser.add_argument(
        "--galaxyid-map",
        type=Path,
        required=True,
        help="ID map to tell what particles in the model belong to the host and to the satellite.",
    )
    parser.add_argument(
        "--host-image-size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=(3000, 3000),
        help="Host-centered synthetic image size in pixels (W H). Default: 3000 300",
    )
    parser.add_argument(
        "--brick-file",
        type=Path,
        required=True,
        help="Legacy brick/coadd FITS for the real image (used for WCS + shape).",
    )
    parser.add_argument(
        "--brick-hdu", type=int, default=1, help="Brick image HDU (default 1)"
    )

    parser.add_argument(
        "--zp",
        type=float,
        default=22.5,
        help="Zero point magnitude (Legacy nanomaggies convention)",
    )
    parser.add_argument(
        "--noise",
        type=float,
        default=0.01,
        help="Gaussian noise sigma in 1 pixel (Legacy Survey is 0.01)(same units as output)",
    )
    parser.add_argument(
        "--psf", type=float, default=0.0, help="PSF FWHM to convolve (arcsec)"
    )
    parser.add_argument(
        "--mag",
        action="store_true",
        help="Output mag/arcsec^2 (otherwise nanomaggies/pixel)",
    )
    parser.add_argument(
        "--tail-bbox",
        type=float,
        nargs=4,
        metavar=("X0", "X1", "Y0", "Y1"),
        default=None,
        help=(
            "Tail bounding box in OUTPUT IMAGE PIXELS (after alignment), 0-based, "
            "Python slicing convention: x0<=x<x1, y0<=y<y1. "
            "Coordinates are in image array space: x increases right, y increases down."
        ),
    )
    parser.add_argument(
        "--tail-bbox-1based",
        action="store_true",
        help=(
            "Interpret --tail-bbox coordinates as 1-based and convert to 0-based "
            "by subtracting 1 from X0,X1,Y0,Y1."
        ),
    )
    parser.add_argument(
        "--debug-photometry",
        action="store_true",
        help="Print integrated apparent magnitudes (from particle counts) and compare to host_info/sat_info inputs.r",
    )

    parser.add_argument("--gas", action="store_true", help="Plot gas particles")
    parser.add_argument("--dm", action="store_true", help="Plot dark matter halo")
    parser.add_argument("--disk", action="store_true", help="Plot disk particles")
    parser.add_argument("--bulge", action="store_true", help="Plot bulge particles")
    parser.add_argument("--stars", action="store_true", help="Plot star particles")
    parser.add_argument("--bh", action="store_true", help="Plot black-hole particles")

    args = parser.parse_args()

    tail_bbox = None
    if args.tail_bbox is not None:
        x0, x1, y0, y1 = map(float, args.tail_bbox)
        if args.tail_bbox_1based:
            x0 -= 1.0
            x1 -= 1.0
            y0 -= 1.0
            y1 -= 1.0
        tail_bbox = (x0, x1, y0, y1)

    plot_opts = {
        "gas": args.gas,
        "dm": args.dm,
        "disk": args.disk,
        "bulge": args.bulge,
        "stars": args.stars,
        "bh": args.bh,
    }
    if all(v is False for v in plot_opts.values()):
        print("Must specify items to plot")
        sys.exit(0)

    host_info = load_json_arg(args.host_params)
    sat_info = load_json_arg(args.sat_params)

    brick_shape, brick_hdr, brick_wcs, brick_pixscale = read_brick_wcs_shape(
        args.brick_file, brick_hdu=int(args.brick_hdu)
    )

    start_frame = args.start_frame
    end_frame = args.end_frame
    show_window = True if args.mp4 is None else args.show

    gadget_hdf5_movie(
        args.pattern,
        args.step,
        args.delay,
        args.loop,
        args.point_size,
        plot_opts,
        zp=args.zp,
        psf_fwhm=args.psf,
        noise_std=args.noise,
        mag_out=args.mag,
        mp4_path=args.mp4,
        host_info=host_info,
        sat_info=sat_info,
        galaxyid_map=args.galaxyid_map,
        host_image_size=tuple(args.host_image_size),
        brick_hdr=brick_hdr,
        brick_wcs=brick_wcs,
        brick_shape=brick_shape,
        brick_pixscale=brick_pixscale,
        fps=args.fps,
        show_window=show_window,
        window_size=tuple(args.window_size),
        camera_load=args.camera_load,
        camera_save=args.camera_save,
        tail_bbox=tail_bbox,
        debug_photometry=bool(args.debug_photometry),
    )


if __name__ == "__main__":
    main()
