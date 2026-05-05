#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.interpolate import splev, splprep
from scipy.ndimage import binary_closing, gaussian_filter, label


def read_fits_2d(path: str, hdu_spec="0"):
    """
    Read a FITS HDU by integer index or EXTNAME and return (data, header).
    """
    with fits.open(path, memmap=True) as hdul:
        try:
            hdu_index = int(hdu_spec)
            data = hdul[hdu_index].data
            hdr = hdul[hdu_index].header.copy()
        except ValueError:
            found = None
            for hdu in hdul:
                if hdu.header.get("EXTNAME", "") == hdu_spec:
                    found = hdu
                    break
            if found is None:
                raise ValueError(f"HDU '{hdu_spec}' not found in {path}")
            data = found.data
            hdr = found.header.copy()

    if data is None:
        raise RuntimeError(f"No image data found in HDU {hdu_spec} of {path}")

    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise RuntimeError(f"Expected 2D image, got shape={data.shape}")
    return data, hdr


def read_mask_fits(path: str, hdu_spec="0"):
    """
    Read a FITS mask image and return boolean mask (True = masked).
    Any finite nonzero value is treated as masked.
    """
    data, _ = read_fits_2d(path, hdu_spec)
    return np.isfinite(data) & (data != 0)


def get_wcs_from_header(hdr):
    """
    Return a celestial WCS if possible, else None.
    """
    try:
        w = WCS(hdr).celestial
        if w.pixel_n_dim < 2 or w.world_n_dim < 2:
            return None
        return w
    except Exception:
        return None


def header_satellite_seed_xy(hdr, wcs):
    """
    Try to get default satellite seed from SRA/SDEC header keywords.
    Returns (x, y) in 0-based pixel coordinates, or None.
    """
    if wcs is None:
        return None
    if ("SRA" not in hdr) or ("SDEC" not in hdr):
        return None

    try:
        sra = float(hdr["SRA"])
        sdec = float(hdr["SDEC"])
        x, y = wcs.world_to_pixel_values(sra, sdec)
        if np.isfinite(x) and np.isfinite(y):
            return float(x), float(y)
    except Exception:
        return None
    return None


def read_seed_points_csv(path):
    """
    Read ordered seed points from CSV.
    Accepts columns x,y or x_pix,y_pix or plain 2-column CSV.
    Returns array of shape (N,2).
    """
    path = Path(path)

    try:
        arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
        if arr.dtype.names is not None:
            names = [n.lower() for n in arr.dtype.names]

            def find_col(cands):
                for c in cands:
                    if c in names:
                        return arr.dtype.names[names.index(c)]
                return None

            xname = find_col(["x", "x_pix", "xpix"])
            yname = find_col(["y", "y_pix", "ypix"])
            if xname is not None and yname is not None:
                x = np.asarray(arr[xname], dtype=float)
                y = np.asarray(arr[yname], dtype=float)
                pts = np.column_stack([x, y])
                if pts.ndim == 2 and pts.shape[1] == 2:
                    return pts
    except Exception:
        pass

    arr = np.loadtxt(path, delimiter=",", dtype=float)
    arr = np.atleast_2d(arr)
    if arr.shape[1] < 2:
        raise RuntimeError(f"Seed CSV must contain at least two columns: {path}")
    return np.asarray(arr[:, :2], dtype=float)


def mu_to_relative_intensity(mu):
    """
    Convert mag/arcsec^2 to arbitrary relative linear intensity.
    Absolute scaling does not matter for centroiding/tracing.
    """
    mu = np.asarray(mu, dtype=np.float64)
    finite = np.isfinite(mu)
    if not np.any(finite):
        raise RuntimeError("No finite pixels in mu image.")

    mu_ref = np.nanmin(mu[finite])  # brightest finite pixel
    I = np.zeros_like(mu, dtype=np.float64)
    I[finite] = 10.0 ** (-0.4 * (mu[finite] - mu_ref))
    return I, mu_ref


def nearest_true_pixel(mask, x0, y0, max_radius=50):
    """
    Find nearest True pixel in mask to floating-point (x0,y0).
    Returns integer (x, y) or None.
    """
    h, w = mask.shape
    xi = int(round(x0))
    yi = int(round(y0))

    if 0 <= xi < w and 0 <= yi < h and mask[yi, xi]:
        return xi, yi

    yy, xx = np.nonzero(mask)
    if xx.size == 0:
        return None

    d2 = (xx - x0) ** 2 + (yy - y0) ** 2
    k = np.argmin(d2)
    if np.sqrt(d2[k]) <= max_radius:
        return int(xx[k]), int(yy[k])
    return None


def weighted_centroid_in_radius(I, mask, x0, y0, radius):
    yy, xx = np.indices(I.shape)
    rr2 = (xx - x0) ** 2 + (yy - y0) ** 2
    sel = mask & (rr2 <= radius**2) & np.isfinite(I) & (I > 0)

    if np.count_nonzero(sel) < 3:
        return float(x0), float(y0)

    w = I[sel]
    sw = np.sum(w)
    if sw <= 0:
        return float(x0), float(y0)

    cx = float(np.sum(xx[sel] * w) / sw)
    cy = float(np.sum(yy[sel] * w) / sw)
    return cx, cy


def component_from_seed(mask, x_seed, y_seed):
    """
    Return connected component containing the seed pixel.
    """
    labels, nlab = label(mask)
    if nlab == 0:
        raise RuntimeError("No connected components found in mask.")

    lab = labels[y_seed, x_seed]
    if lab == 0:
        raise RuntimeError("Seed pixel is not inside any connected component.")

    return labels == lab


def estimate_initial_direction(component, I, cx, cy, core_exclusion, direction_radius):
    """
    Estimate the outward tail direction from weighted PCA in an annulus.
    Returns unit vector [vx, vy].
    """
    yy, xx = np.indices(I.shape)
    dx = xx - cx
    dy = yy - cy
    rr = np.hypot(dx, dy)

    sel = (
        component
        & np.isfinite(I)
        & (I > 0)
        & (rr >= core_exclusion)
        & (rr <= direction_radius)
    )

    if np.count_nonzero(sel) < 10:
        sel = component & np.isfinite(I) & (I > 0) & (rr >= core_exclusion)

    if np.count_nonzero(sel) < 5:
        raise RuntimeError("Not enough pixels to estimate initial tail direction.")

    x = dx[sel]
    y = dy[sel]
    w = I[sel]
    sw = np.sum(w)
    if sw <= 0:
        raise RuntimeError("Zero total weight when estimating initial direction.")

    xbar = np.sum(w * x) / sw
    ybar = np.sum(w * y) / sw
    x0 = x - xbar
    y0 = y - ybar

    cxx = np.sum(w * x0 * x0) / sw
    cxy = np.sum(w * x0 * y0) / sw
    cyy = np.sum(w * y0 * y0) / sw
    C = np.array([[cxx, cxy], [cxy, cyy]], dtype=np.float64)

    evals, evecs = np.linalg.eigh(C)
    v = evecs[:, np.argmax(evals)]
    v = v / (np.linalg.norm(v) + 1e-30)

    proj = x * v[0] + y * v[1]
    pos_score = np.sum(w[proj > 0] * proj[proj > 0]) if np.any(proj > 0) else 0.0
    neg_score = np.sum(w[proj < 0] * (-proj[proj < 0])) if np.any(proj < 0) else 0.0
    if neg_score > pos_score:
        v = -v

    return v


def segment_tube_score(
    I_eff,
    usable,
    p0,
    p1,
    sigma_perp,
    sigma_parallel,
):
    """
    Score the line integral of I_eff in a Gaussian tube around segment p0->p1.
    """
    yy, xx = np.indices(I_eff.shape)

    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    dv = p1 - p0
    L = np.linalg.norm(dv)
    if L <= 0:
        return -np.inf

    v = dv / L

    dx = xx - p0[0]
    dy = yy - p0[1]

    proj = dx * v[0] + dy * v[1]
    perp = dx * v[1] - dy * v[0]

    sel = (
        usable
        & (proj >= -2.0 * sigma_parallel)
        & (proj <= L + 2.0 * sigma_parallel)
        & (np.abs(perp) <= 3.0 * sigma_perp)
    )

    if not np.any(sel):
        return -np.inf

    w = (
        I_eff[sel]
        * np.exp(-0.5 * (perp[sel] / sigma_perp) ** 2)
        * np.exp(-0.5 * ((proj[sel] - 0.5 * L) / sigma_parallel) ** 2)
    )
    return float(np.sum(w))


def segment_tube_score_robust(
    I_eff,
    usable,
    p0,
    p1,
    sigma_perp,
    sigma_parallel,
    n_long_bins=8,
    min_bin_frac=0.35,
    mean_weight=0.30,
    std_penalty=0.35,
):
    """
    Score a candidate segment p0->p1 using continuity of signal along the tube,
    not just total integrated brightness.

    The segment is divided into longitudinal bins. A good segment should have
    signal spread across many bins, not one very bright hotspot.

    Score = Q25(fbin) + mean_weight*mean(fbin) - std_penalty*std(fbin)

    where fbin is the Gaussian cross-tube weighted flux in each longitudinal bin.
    """
    yy, xx = np.indices(I_eff.shape)

    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    dv = p1 - p0
    L = np.linalg.norm(dv)
    if L <= 0:
        return -np.inf

    v = dv / L

    dx = xx - p0[0]
    dy = yy - p0[1]

    proj = dx * v[0] + dy * v[1]
    perp = dx * v[1] - dy * v[0]

    sel = usable & (proj >= 0.0) & (proj <= L) & (np.abs(perp) <= 3.0 * sigma_perp)
    if not np.any(sel):
        return -np.inf

    # Cross-tube Gaussian weighting only. Longitudinal robustness is handled by binning.
    w = I_eff[sel] * np.exp(-0.5 * (perp[sel] / sigma_perp) ** 2)
    p = proj[sel]

    edges = np.linspace(0.0, L, int(n_long_bins) + 1)
    fbin = np.zeros(int(n_long_bins), dtype=np.float64)

    for i in range(int(n_long_bins)):
        if i < int(n_long_bins) - 1:
            m = (p >= edges[i]) & (p < edges[i + 1])
        else:
            m = (p >= edges[i]) & (p <= edges[i + 1])
        if np.any(m):
            fbin[i] = np.sum(w[m])

    # Require support over a decent fraction of the segment
    n_support = np.count_nonzero(fbin > 0)
    if n_support < max(2, int(np.ceil(min_bin_frac * n_long_bins))):
        return -np.inf

    q25 = np.percentile(fbin, 25.0)
    mu = np.mean(fbin)
    sig = np.std(fbin)

    return float(q25 + mean_weight * mu - std_penalty * sig)


def trace_tail_points_beam(
    component,
    I,
    cx,
    cy,
    direction,
    step_pix=10.0,
    max_steps=200,
    seed_points=None,
    star_mask=None,
    intensity_power=0.5,
    cap_percentile=99.0,
    sigma_perp_factor=0.45,
    sigma_parallel_factor=0.6,
    beam_width=7,
    lookahead=4,
    angle_offsets_deg=(-36, -24, -16, -8, 0, 8, 16, 24, 36),
    step_scales=(0.85, 1.0, 1.15),
    direction_relax=0.10,
    min_move_pix=1.5,
    turn_penalty=2,
    curvature_penalty=1,
    n_long_bins=8,
    min_bin_frac=0.35,
    mean_weight=0.30,
    std_penalty=0.35,
):
    """
    Multi-hypothesis tail tracer using beam search and a robust segment score
    based on longitudinal continuity along each candidate tube.
    """
    if seed_points is not None:
        seed_points = np.asarray(seed_points, dtype=np.float64)
        if seed_points.ndim != 2 or seed_points.shape[1] != 2:
            raise RuntimeError("seed_points must have shape (N,2)")
        if seed_points.shape[0] == 0:
            seed_points = None

    if seed_points is not None:
        pts = [(float(x), float(y)) for x, y in seed_points]
        cur = np.array(seed_points[-1], dtype=np.float64)
        if seed_points.shape[0] >= 2:
            v = np.array(seed_points[-1] - seed_points[-2], dtype=np.float64)
            nv = np.linalg.norm(v)
            if nv <= 0:
                v = np.array(direction, dtype=np.float64)
            else:
                v /= nv
        else:
            v = np.array(direction, dtype=np.float64)
            v /= np.linalg.norm(v) + 1e-30
    else:
        pts = [(float(cx), float(cy))]
        cur = np.array([float(cx), float(cy)], dtype=np.float64)
        v = np.array(direction, dtype=np.float64)
        v /= np.linalg.norm(v) + 1e-30

    usable = component & np.isfinite(I) & (I > 0)
    if star_mask is not None:
        usable &= ~np.asarray(star_mask, dtype=bool)

    if not np.any(usable):
        return np.asarray(pts, dtype=np.float64)

    h, w = I.shape

    def segment_mostly_in_component(p0, p1, min_frac=0.85, n_samples=12):
        """
        Require most of the straight segment from p0 to p1 to stay inside component.
        """
        p0 = np.asarray(p0, dtype=np.float64)
        p1 = np.asarray(p1, dtype=np.float64)

        xs = np.linspace(p0[0], p1[0], int(n_samples))
        ys = np.linspace(p0[1], p1[1], int(n_samples))

        inside = 0
        total = 0

        h, w = component.shape
        for x, y in zip(xs, ys):
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < w and 0 <= yi < h:
                total += 1
                if component[yi, xi]:
                    inside += 1

        if total == 0:
            return False

        return (inside / total) >= float(min_frac)

    def endpoint_in_component(p):
        """
        Strict endpoint test: rounded endpoint pixel must lie inside component.
        """
        x, y = float(p[0]), float(p[1])

        if not np.isfinite(x) or not np.isfinite(y):
            return False

        xi = int(round(x))
        yi = int(round(y))

        if xi < 0 or xi >= component.shape[1] or yi < 0 or yi >= component.shape[0]:
            return False

        return bool(component[yi, xi])

    I_eff = np.zeros_like(I, dtype=np.float64)
    I_eff[usable] = np.power(I[usable], float(intensity_power))

    if cap_percentile is not None:
        vals = I_eff[usable]
        if vals.size > 0:
            cap = np.percentile(vals, float(cap_percentile))
            I_eff[usable] = np.minimum(I_eff[usable], cap)

    sigma_perp = max(1e-6, sigma_perp_factor * step_pix)
    sigma_parallel = max(1e-6, sigma_parallel_factor * step_pix)

    angle_offsets = np.deg2rad(np.asarray(angle_offsets_deg, dtype=np.float64))

    def rotate(vec, ang):
        c = np.cos(ang)
        s = np.sin(ang)
        return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])

    def path_score_from_state(p_start, v_start, depth):
        beams = [
            {
                "p": np.array(p_start, dtype=np.float64),
                "v": np.array(v_start, dtype=np.float64),
                "score": 0.0,
                "pts": [],
                "last_turn": 0.0,
            }
        ]

        for _ in range(depth):
            new_beams = []
            for b in beams:
                for da in angle_offsets:
                    v_try = rotate(b["v"], da)
                    nv = np.linalg.norm(v_try)
                    if nv <= 0:
                        continue
                    v_try /= nv

                    for sf in step_scales:
                        L = float(step_pix) * float(sf)
                        p_new = b["p"] + L * v_try

                        if not endpoint_in_component(p_new):
                            continue

                        if not segment_mostly_in_component(
                            b["p"], p_new, min_frac=0.85, n_samples=12
                        ):
                            continue

                        seg_score = segment_tube_score_robust(
                            I_eff=I_eff,
                            usable=usable,
                            p0=b["p"],
                            p1=p_new,
                            sigma_perp=sigma_perp,
                            sigma_parallel=sigma_parallel,
                            n_long_bins=n_long_bins,
                            min_bin_frac=min_bin_frac,
                            mean_weight=mean_weight,
                            std_penalty=std_penalty,
                        )
                        if not np.isfinite(seg_score):
                            continue

                        turn_cost = turn_penalty * (da**2)
                        curv_cost = curvature_penalty * ((da - b["last_turn"]) ** 2)

                        score_new = b["score"] + seg_score - turn_cost - curv_cost

                        new_beams.append(
                            {
                                "p": p_new,
                                "v": v_try,
                                "score": score_new,
                                "pts": b["pts"] + [p_new.copy()],
                                "last_turn": da,
                            }
                        )

            if not new_beams:
                break

            new_beams.sort(key=lambda q: q["score"], reverse=True)
            beams = new_beams[:beam_width]

        if not beams:
            return [], -np.inf

        best = max(beams, key=lambda q: q["score"])
        return best["pts"], float(best["score"])

    for _ in range(max_steps):
        best_future_pts, best_score = path_score_from_state(cur, v, lookahead)
        if len(best_future_pts) == 0 or not np.isfinite(best_score):
            break

        nxt = np.array(best_future_pts[0], dtype=np.float64)

        dv = nxt - cur
        d = np.linalg.norm(dv)
        if d < min_move_pix:
            break
        dv /= d

        v = (1.0 - direction_relax) * v + direction_relax * dv
        v /= np.linalg.norm(v) + 1e-30

        pts.append((float(nxt[0]), float(nxt[1])))
        cur = nxt

    return np.asarray(pts, dtype=np.float64)


def fit_cubic_spline(points, smoothing=3.0, n_samples=400):
    """
    Fit parametric cubic spline x(s), y(s) using cumulative arc length.
    Returns sampled spline points and the tck tuple.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape[0] < 2:
        raise RuntimeError("Need at least 2 points to fit a spline.")

    ds = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    if s[-1] <= 0:
        raise RuntimeError("Degenerate traced point set.")

    u = s / s[-1]
    k = min(3, points.shape[0] - 1)

    tck, _ = splprep(
        [points[:, 0], points[:, 1]],
        u=u,
        s=float(smoothing),
        k=k,
    )
    us = np.linspace(0.0, 1.0, int(n_samples))
    xs, ys = splev(us, tck)
    spline = np.column_stack([xs, ys])
    return spline, tck


def save_points_csv(path, arr, header="x_pix,y_pix"):
    arr = np.asarray(arr, dtype=np.float64)
    np.savetxt(path, arr, delimiter=",", header=header, comments="")


def save_points_world_csv(path, pix_points, wcs):
    """
    Save x_pix, y_pix, ra_deg, dec_deg.
    """
    pix_points = np.asarray(pix_points, dtype=np.float64)
    ra_deg, dec_deg = wcs.pixel_to_world_values(pix_points[:, 0], pix_points[:, 1])
    out = np.column_stack([pix_points[:, 0], pix_points[:, 1], ra_deg, dec_deg])
    np.savetxt(
        path,
        out,
        delimiter=",",
        header="x_pix,y_pix,ra_deg,dec_deg",
        comments="",
    )


def save_ds9_region_image(path, points, color="green"):
    """
    DS9 region in image coordinates.
    Uses 'segment' because DS9 'line' only accepts exactly two points.
    """
    points = np.asarray(points, dtype=np.float64)
    with open(path, "w") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write(f"global color={color} width=2\n")
        f.write("image\n")
        xy = ",".join(f"{x+1:.3f},{y+1:.3f}" for x, y in points)  # DS9 image is 1-based
        f.write(f"segment({xy})\n")


def save_ds9_region_fk5(path, points_pix, wcs, color="green"):
    """
    DS9 region in FK5 coordinates.
    """
    points_pix = np.asarray(points_pix, dtype=np.float64)
    ra_deg, dec_deg = wcs.pixel_to_world_values(points_pix[:, 0], points_pix[:, 1])

    with open(path, "w") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write(f"global color={color} width=2\n")
        f.write("fk5\n")
        xy = ",".join(f"{ra:.8f},{dec:.8f}" for ra, dec in zip(ra_deg, dec_deg))
        f.write(f"segment({xy})\n")


def append_summary_csv(path, row):
    """
    Append one summary row to a CSV file.
    Writes the header if the file does not already exist.
    """
    path = Path(path)
    write_header = (not path.exists()) or (path.stat().st_size == 0)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    p = argparse.ArgumentParser(
        description="Trace a satellite tidal tail centerline and fit a cubic spline."
    )
    p.add_argument("fits", help="Input mu_from_legacy FITS image (mag/arcsec^2)")
    p.add_argument("--hdu", default="0", help="HDU index or EXTNAME (default: 0)")
    p.add_argument(
        "--mu-max",
        type=float,
        default=29.0,
        help="Faintest surface brightness to include in tail mask (default: 29.0)",
    )
    p.add_argument(
        "--smooth-sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma in pixels for relative intensity map",
    )
    p.add_argument(
        "--close-iters",
        type=int,
        default=1,
        help="Binary closing iterations to bridge small mask gaps",
    )
    p.add_argument(
        "--x0",
        type=float,
        default=None,
        help="Optional approximate satellite x position (0-based pixel)",
    )
    p.add_argument(
        "--y0",
        type=float,
        default=None,
        help="Optional approximate satellite y position (0-based pixel)",
    )
    p.add_argument(
        "--seed-search-radius",
        type=float,
        default=60.0,
        help="Max distance to nearest mask pixel from supplied seed",
    )
    p.add_argument(
        "--centroid-radius",
        type=float,
        default=12.0,
        help="Radius in pixels for centroiding the satellite core",
    )
    p.add_argument(
        "--core-exclusion",
        type=float,
        default=10.0,
        help="Exclude inner core (pixels) when estimating initial tail direction",
    )
    p.add_argument(
        "--direction-radius",
        type=float,
        default=40.0,
        help="Outer radius (pixels) for initial direction estimate",
    )
    p.add_argument(
        "--theta0",
        type=float,
        default=None,
        help="Optional manual initial tail direction in degrees CCW from +x",
    )
    p.add_argument(
        "--seed-points-csv",
        type=str,
        default=None,
        help=(
            "CSV of ordered initial seed points (x,y or x_pix,y_pix) in 0-based pixels. "
            "The tracer starts from the last seed point."
        ),
    )
    p.add_argument(
        "--seed-point",
        type=float,
        nargs=2,
        action="append",
        default=None,
        metavar=("X", "Y"),
        help="Add an ordered manual seed point in 0-based pixels. May be used multiple times.",
    )
    p.add_argument(
        "--star-mask-fits",
        type=str,
        default=None,
        help="Optional FITS file containing a star mask (nonzero = masked).",
    )
    p.add_argument(
        "--star-mask-hdu",
        type=str,
        default="0",
        help="HDU index or EXTNAME for the star mask FITS (default: 0)",
    )
    p.add_argument(
        "--step",
        type=float,
        default=10.0,
        help="Nominal tracing step size in pixels",
    )
    p.add_argument(
        "--half-width",
        type=float,
        default=8.0,
        help="Half-width scale for cross-tail tracing kernel in pixels",
    )
    p.add_argument(
        "--forward-min",
        type=float,
        default=0.4,
        help="Minimum forward distance in units of step size",
    )
    p.add_argument(
        "--forward-max",
        type=float,
        default=1.8,
        help="Maximum forward distance in units of step size",
    )
    p.add_argument(
        "--direction-relax",
        type=float,
        default=0.20,
        help="How strongly to update direction each step (0..1)",
    )
    p.add_argument(
        "--min-weight-pixels",
        type=int,
        default=6,
        help="Minimum number of weighted pixels needed to continue tracing",
    )
    p.add_argument(
        "--min-move",
        type=float,
        default=1.5,
        help="Minimum centroid movement per step in pixels",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="Maximum number of tracing steps",
    )
    p.add_argument(
        "--intensity-power",
        type=float,
        default=0.5,
        help="Use I**power for tracing weights (default: 0.5 = sqrt(I))",
    )
    p.add_argument(
        "--cap-percentile",
        type=float,
        default=99.0,
        help="Cap tracing weights at this percentile within usable pixels",
    )
    p.add_argument(
        "--sigma-parallel-factor",
        type=float,
        default=0.6,
        help="Gaussian sigma along forward direction in units of step size",
    )
    p.add_argument(
        "--sigma-perp-factor",
        type=float,
        default=0.5,
        help="Gaussian sigma across the tail in units of half-width",
    )
    p.add_argument(
        "--spline-s",
        type=float,
        default=3.0,
        help="Spline smoothing parameter passed to splprep",
    )
    p.add_argument(
        "--spline-samples",
        type=int,
        default=400,
        help="Number of sampled points on the fitted spline",
    )
    p.add_argument(
        "--beam-width",
        type=int,
        default=7,
        help="Beam width for multi-hypothesis tail tracing",
    )
    p.add_argument(
        "--lookahead",
        type=int,
        default=4,
        help="Lookahead depth for beam-search tracing",
    )
    p.add_argument(
        "--n-long-bins",
        type=int,
        default=8,
        help="Number of longitudinal bins per candidate segment",
    )
    p.add_argument(
        "--min-bin-frac",
        type=float,
        default=0.35,
        help="Minimum fraction of longitudinal bins with support",
    )
    p.add_argument(
        "--mean-weight",
        type=float,
        default=0.30,
        help="Weight of mean longitudinal bin flux in robust segment score",
    )
    p.add_argument(
        "--std-penalty",
        type=float,
        default=0.35,
        help="Penalty on longitudinal bin scatter in robust segment score",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default="tail_trace",
        help="Output prefix for CSV/PNG/region products",
    )
    p.add_argument(
        "--summary-csv",
        type=str,
        default=None,
        help="Optional CSV file to append a one-line summary of the trace results",
    )
    args = p.parse_args()

    mu, hdr = read_fits_2d(args.fits, args.hdu)
    wcs = get_wcs_from_header(hdr)

    I, mu_ref = mu_to_relative_intensity(mu)
    I_s = gaussian_filter(I, sigma=float(args.smooth_sigma))

    I_thresh = 10.0 ** (-0.4 * (float(args.mu_max) - mu_ref))
    mask = np.isfinite(mu) & (I_s >= I_thresh)

    if args.close_iters > 0:
        mask = binary_closing(mask, iterations=int(args.close_iters))

    if not np.any(mask):
        raise RuntimeError(
            "Threshold mask is empty. Try a fainter --mu-max or smaller --smooth-sigma."
        )

    manual_seed_points = None
    if args.seed_points_csv is not None:
        manual_seed_points = read_seed_points_csv(args.seed_points_csv)

    if args.seed_point is not None:
        pts_cli = np.asarray(args.seed_point, dtype=float)
        if manual_seed_points is None:
            manual_seed_points = pts_cli
        else:
            manual_seed_points = np.vstack([manual_seed_points, pts_cli])

    if manual_seed_points is not None and manual_seed_points.shape[0] < 1:
        manual_seed_points = None

    star_mask = None
    if args.star_mask_fits is not None:
        star_mask = read_mask_fits(args.star_mask_fits, args.star_mask_hdu)
        if star_mask.shape != mask.shape:
            raise RuntimeError(
                f"star mask shape {star_mask.shape} does not match image shape {mask.shape}"
            )

    seed_source = None
    if manual_seed_points is not None:
        seed = nearest_true_pixel(
            mask,
            manual_seed_points[0, 0],
            manual_seed_points[0, 1],
            max_radius=float(args.seed_search_radius),
        )
        if seed is None:
            raise RuntimeError(
                "Could not find a mask pixel near the first manual seed point."
            )
        x_seed, y_seed = seed
        seed_source = "manual seed points"

    elif args.x0 is not None and args.y0 is not None:
        seed = nearest_true_pixel(
            mask, args.x0, args.y0, max_radius=float(args.seed_search_radius)
        )
        if seed is None:
            raise RuntimeError(
                "Could not find a mask pixel near the supplied --x0/--y0 position."
            )
        x_seed, y_seed = seed
        seed_source = "user x0/y0"

    else:
        hdr_seed = header_satellite_seed_xy(hdr, wcs)
        if hdr_seed is not None:
            seed = nearest_true_pixel(
                mask,
                hdr_seed[0],
                hdr_seed[1],
                max_radius=float(args.seed_search_radius),
            )
            if seed is not None:
                x_seed, y_seed = seed
                seed_source = "header SRA/SDEC"
            else:
                yy_peak, xx_peak = np.unravel_index(np.nanargmax(I_s), I_s.shape)
                x_seed, y_seed = int(xx_peak), int(yy_peak)
                seed_source = "brightest smoothed pixel (header seed not on mask)"
        else:
            yy_peak, xx_peak = np.unravel_index(np.nanargmax(I_s), I_s.shape)
            x_seed, y_seed = int(xx_peak), int(yy_peak)
            seed_source = "brightest smoothed pixel"

    component = component_from_seed(mask, x_seed, y_seed)

    if manual_seed_points is not None:
        cx = float(manual_seed_points[0, 0])
        cy = float(manual_seed_points[0, 1])

        if manual_seed_points.shape[0] >= 2:
            direction = np.array(
                manual_seed_points[-1] - manual_seed_points[-2],
                dtype=np.float64,
            )
            nd = np.linalg.norm(direction)
            if nd <= 0:
                raise RuntimeError("Last two manual seed points are identical.")
            direction /= nd
        elif args.theta0 is not None:
            th = np.deg2rad(float(args.theta0))
            direction = np.array([np.cos(th), np.sin(th)], dtype=np.float64)
        else:
            direction = estimate_initial_direction(
                component=component,
                I=I_s,
                cx=cx,
                cy=cy,
                core_exclusion=float(args.core_exclusion),
                direction_radius=float(args.direction_radius),
            )
    else:
        cx, cy = weighted_centroid_in_radius(
            I_s, component, x_seed, y_seed, radius=float(args.centroid_radius)
        )

        if args.theta0 is not None:
            th = np.deg2rad(float(args.theta0))
            direction = np.array([np.cos(th), np.sin(th)], dtype=np.float64)
        else:
            direction = estimate_initial_direction(
                component=component,
                I=I_s,
                cx=cx,
                cy=cy,
                core_exclusion=float(args.core_exclusion),
                direction_radius=float(args.direction_radius),
            )

    pts = trace_tail_points_beam(
        component=component,
        I=I_s,
        cx=cx,
        cy=cy,
        direction=direction,
        step_pix=float(args.step),
        max_steps=int(args.max_steps),
        seed_points=manual_seed_points,
        star_mask=star_mask,
        intensity_power=float(args.intensity_power),
        cap_percentile=(
            float(args.cap_percentile) if args.cap_percentile is not None else None
        ),
        sigma_perp_factor=float(args.sigma_perp_factor),
        sigma_parallel_factor=float(args.sigma_parallel_factor),
        beam_width=int(args.beam_width),
        lookahead=int(args.lookahead),
        direction_relax=float(args.direction_relax),
        n_long_bins=int(args.n_long_bins),
        min_bin_frac=float(args.min_bin_frac),
        mean_weight=float(args.mean_weight),
        std_penalty=float(args.std_penalty),
    )

    usable = component & np.isfinite(I) & (I > 0)
    if star_mask is not None:
        usable &= ~star_mask

    if pts.shape[0] < 4:
        raise RuntimeError(
            f"Only traced {pts.shape[0]} points. "
            "Try lowering --mu-max, increasing --half-width, providing manual seed points, "
            "or supplying --theta0."
        )

    spline, _ = fit_cubic_spline(
        pts,
        smoothing=float(args.spline_s),
        n_samples=int(args.spline_samples),
    )

    prefix = Path(args.out_prefix)

    raw_pix_csv = prefix.with_suffix(".raw_points_pix.csv")
    spline_pix_csv = prefix.with_suffix(".spline_points_pix.csv")
    raw_world_csv = prefix.with_suffix(".raw_points_world.csv")
    spline_world_csv = prefix.with_suffix(".spline_points_world.csv")

    reg_raw_img = prefix.with_suffix(".raw_points_image.reg")
    reg_spline_img = prefix.with_suffix(".spline_image.reg")
    reg_raw_fk5 = prefix.with_suffix(".raw_points_fk5.reg")
    reg_spline_fk5 = prefix.with_suffix(".spline_fk5.reg")

    png_out = prefix.with_suffix(".png")

    save_points_csv(raw_pix_csv, pts, header="x_pix,y_pix")
    save_points_csv(spline_pix_csv, spline, header="x_pix,y_pix")

    save_ds9_region_image(reg_raw_img, pts, color="yellow")
    save_ds9_region_image(reg_spline_img, spline, color="green")

    wrote_world = False
    wrote_fk5_reg = False
    if wcs is not None:
        try:
            save_points_world_csv(raw_world_csv, pts, wcs)
            save_points_world_csv(spline_world_csv, spline, wcs)
            wrote_world = True
        except Exception as e:
            print(f"[Warn] Could not write world-coordinate CSVs: {e}")

        try:
            save_ds9_region_fk5(reg_raw_fk5, pts, wcs, color="yellow")
            save_ds9_region_fk5(reg_spline_fk5, spline, wcs, color="green")
            wrote_fk5_reg = True
        except Exception as e:
            print(f"[Warn] Could not write FK5 region files: {e}")

    finite = np.isfinite(mu)
    vmin = np.nanpercentile(mu[finite], 2)
    vmax = np.nanpercentile(mu[finite], 98)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(mu, origin="lower", cmap="gray_r", vmin=vmin, vmax=vmax)
    # ax.contour(component.astype(float), levels=[0.5], colors="cyan", linewidths=0.7)

    if manual_seed_points is not None:
        ax.plot(
            manual_seed_points[:, 0],
            manual_seed_points[:, 1],
            "o-",
            ms=4,
            lw=1.2,
            color="orange",
            label="manual seed pts",
        )

    if usable is not None:
        ax.contour(
            usable.astype(float),
            levels=[0.5],
            colors="cyan",
            linewidths=0.7,
            alpha=0.8,
        )

    if star_mask is not None:
        ax.contour(
            star_mask.astype(float),
            levels=[0.5],
            colors="red",
            linewidths=0.5,
            alpha=0.7,
        )

    ax.plot(pts[:, 0], pts[:, 1], "o-", ms=3, lw=1.2, color="yellow", label="trace pts")
    ax.plot(
        spline[:, 0],
        spline[:, 1],
        "-",
        lw=2.0,
        color="lime",
        label="cubic spline",
    )
    ax.plot(cx, cy, "rx", ms=8, mew=2, label="start/centroid")
    ax.set_title("Satellite tail spline trace")
    ax.set_xlabel("x [pix]")
    ax.set_ylabel("y [pix]")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(png_out, bbox_inches="tight")
    plt.close(fig)

    seg = np.hypot(np.diff(spline[:, 0]), np.diff(spline[:, 1]))
    length_pix = float(np.sum(seg))
    length_arcmin = length_pix * 0.262 * 4 / 60.0
    # This script does not currently estimate a length uncertainty.
    # Leave placeholders for later Monte Carlo / noise-realization errors.
    length_err_pix = np.nan
    length_err_arcmin = np.nan

    summary_row = {
        "input_fits": str(args.fits),
        "hdu": str(args.hdu),
        "out_prefix": str(args.out_prefix),
        "seed_source": str(seed_source),
        "x_seed_pix": int(x_seed),
        "y_seed_pix": int(y_seed),
        "start_x_pix": float(cx),
        "start_y_pix": float(cy),
        "traced_points": int(pts.shape[0]),
        "spline_samples": int(spline.shape[0]),
        "mu_max": float(args.mu_max),
        "smooth_sigma_pix": float(args.smooth_sigma),
        "close_iters": int(args.close_iters),
        "seed_search_radius_pix": float(args.seed_search_radius),
        "centroid_radius_pix": float(args.centroid_radius),
        "core_exclusion_pix": float(args.core_exclusion),
        "direction_radius_pix": float(args.direction_radius),
        "theta0_deg": float(args.theta0) if args.theta0 is not None else np.nan,
        "step_pix": float(args.step),
        "direction_relax": float(args.direction_relax),
        "min_move_pix": float(args.min_move),
        "max_steps": int(args.max_steps),
        "intensity_power": float(args.intensity_power),
        "cap_percentile": (
            float(args.cap_percentile) if args.cap_percentile is not None else np.nan
        ),
        "sigma_parallel_factor": float(args.sigma_parallel_factor),
        "sigma_perp_factor": float(args.sigma_perp_factor),
        "spline_s": float(args.spline_s),
        "beam_width": int(args.beam_width),
        "lookahead": int(args.lookahead),
        "n_long_bins": int(args.n_long_bins),
        "min_bin_frac": float(args.min_bin_frac),
        "mean_weight": float(args.mean_weight),
        "std_penalty": float(args.std_penalty),
        "length_pix": float(length_pix),
        "length_arcmin": float(length_arcmin),
        "length_err_pix": float(length_err_pix),
        "length_err_arcmin": float(length_err_arcmin),
        "raw_pix_csv": str(raw_pix_csv),
        "spline_pix_csv": str(spline_pix_csv),
        "raw_world_csv": str(raw_world_csv) if wrote_world else "",
        "spline_world_csv": str(spline_world_csv) if wrote_world else "",
        "raw_image_reg": str(reg_raw_img),
        "spline_image_reg": str(reg_spline_img),
        "raw_fk5_reg": str(reg_raw_fk5) if wrote_fk5_reg else "",
        "spline_fk5_reg": str(reg_spline_fk5) if wrote_fk5_reg else "",
        "overlay_png": str(png_out),
    }

    if args.summary_csv is not None:
        append_summary_csv(args.summary_csv, summary_row)

    print(f"Seed source: {seed_source}")
    print(f"Seed pixel used for component: ({x_seed}, {y_seed})")
    print(f"Start point: ({cx:.3f}, {cy:.3f})")
    print(f"Traced points: {pts.shape[0]}")
    print(f"Spline samples: {spline.shape[0]}")
    print(f"Spline length: {length_pix:.3f} pix")
    print(f"Spline length: {length_arcmin:.3f} arcmin")
    print(f"Saved raw pixel CSV:      {raw_pix_csv}")
    print(f"Saved spline pixel CSV:   {spline_pix_csv}")
    if wrote_world:
        print(f"Saved raw world CSV:      {raw_world_csv}")
        print(f"Saved spline world CSV:   {spline_world_csv}")
    else:
        print("World-coordinate CSVs not written (missing/invalid celestial WCS).")
    print(f"Saved raw image region:   {reg_raw_img}")
    print(f"Saved spline image region:{reg_spline_img}")
    if wrote_fk5_reg:
        print(f"Saved raw FK5 region:     {reg_raw_fk5}")
        print(f"Saved spline FK5 region:  {reg_spline_fk5}")
    else:
        print("FK5 region files not written (missing/invalid celestial WCS).")
    print(f"Saved overlay PNG:        {png_out}")
    if args.summary_csv is not None:
        print(f"Saved summary CSV:        {args.summary_csv}")


if __name__ == "__main__":
    main()
