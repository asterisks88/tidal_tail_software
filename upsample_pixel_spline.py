#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
from scipy.interpolate import splprep, splev

from astropy.io import fits
from astropy.wcs import WCS


def read_xy_csv(path):
    """
    Read CSV containing at least two columns of x,y pixel coordinates.
    Accepts either:
      - headerless two-column CSV
      - headered CSV with columns like x,y or x_pix,y_pix
    Returns Nx2 float array.
    """
    path = Path(path)

    # First try with header names
    try:
        arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
        if arr.dtype.names is not None:
            names = [n.lower() for n in arr.dtype.names]

            def find_col(candidates):
                for c in candidates:
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

    # Fallback: plain 2-column CSV
    arr = np.loadtxt(path, delimiter=",", dtype=float)
    arr = np.atleast_2d(arr)
    if arr.shape[1] < 2:
        raise RuntimeError(f"CSV must contain at least 2 columns: {path}")
    return np.asarray(arr[:, :2], dtype=float)


def fit_cubic_spline(points, smoothing=0.0, n_samples=400):
    """
    Fit a parametric spline x(u), y(u) using cumulative arc length parameterization.
    Returns sampled spline points and the tck tuple.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise RuntimeError("points must have shape (N,2)")
    if points.shape[0] < 2:
        raise RuntimeError("Need at least 2 points to fit a spline.")

    ds = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    s = np.concatenate([[0.0], np.cumsum(ds)])
    if s[-1] <= 0:
        raise RuntimeError("Degenerate input points.")

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


def save_ds9_region_image(path, points, color="green"):
    """
    Save DS9 region in IMAGE coordinates.
    Uses 'segment', which DS9 accepts for a polyline/path.
    DS9 image coordinates are 1-based.
    """
    points = np.asarray(points, dtype=np.float64)
    with open(path, "w") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write(f"global color={color} width=2\n")
        f.write("image\n")
        xy = ",".join(f"{x+1:.3f},{y+1:.3f}" for x, y in points)
        f.write(f"segment({xy})\n")


def get_wcs_from_fits(path, hdu_spec="0"):
    with fits.open(path, memmap=True) as hdul:
        try:
            hdu_index = int(hdu_spec)
            hdr = hdul[hdu_index].header.copy()
        except ValueError:
            found = None
            for hdu in hdul:
                if hdu.header.get("EXTNAME", "") == hdu_spec:
                    found = hdu
                    break
            if found is None:
                raise ValueError(f"HDU '{hdu_spec}' not found in {path}")
            hdr = found.header.copy()

    try:
        w = WCS(hdr).celestial
        if w.pixel_n_dim < 2 or w.world_n_dim < 2:
            raise RuntimeError("FITS does not contain a usable celestial WCS.")
        return w
    except Exception as e:
        raise RuntimeError(f"Could not build celestial WCS from FITS: {e}")


def save_ds9_region_fk5(path, points_pix, wcs, color="green"):
    """
    Save DS9 region in FK5 coordinates.
    """
    points_pix = np.asarray(points_pix, dtype=np.float64)
    ra_deg, dec_deg = wcs.pixel_to_world_values(points_pix[:, 0], points_pix[:, 1])

    with open(path, "w") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write(f"global color={color} width=2\n")
        f.write("fk5\n")
        xy = ",".join(f"{ra:.8f},{dec:.8f}" for ra, dec in zip(ra_deg, dec_deg))
        f.write(f"segment({xy})\n")


def save_world_csv(path, points_pix, wcs):
    points_pix = np.asarray(points_pix, dtype=np.float64)
    ra_deg, dec_deg = wcs.pixel_to_world_values(points_pix[:, 0], points_pix[:, 1])
    out = np.column_stack([points_pix[:, 0], points_pix[:, 1], ra_deg, dec_deg])
    np.savetxt(
        path,
        out,
        delimiter=",",
        header="x_pix,y_pix,ra_deg,dec_deg",
        comments="",
    )


def main():
    p = argparse.ArgumentParser(
        description="Upsample a sparse pixel spline/control-point CSV into a densely sampled spline and save DS9 region files."
    )
    p.add_argument("csv", help="Input CSV of sparse pixel points")
    p.add_argument(
        "--samples",
        type=int,
        default=400,
        help="Number of sampled spline points to output (default: 400)",
    )
    p.add_argument(
        "--spline-s",
        type=float,
        default=0.0,
        help="Spline smoothing parameter passed to splprep (default: 0 = interpolating spline)",
    )
    p.add_argument(
        "--color",
        type=str,
        default="green",
        help="DS9 region color (default: green)",
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default="upsampled_spline",
        help="Output prefix (default: upsampled_spline)",
    )
    p.add_argument(
        "--fits",
        type=str,
        default=None,
        help="Optional FITS file with celestial WCS for FK5 region/world CSV output",
    )
    p.add_argument(
        "--hdu",
        type=str,
        default="0",
        help="FITS HDU index or EXTNAME for WCS (default: 0)",
    )
    args = p.parse_args()

    pts = read_xy_csv(args.csv)
    if pts.shape[0] < 2:
        raise RuntimeError("Need at least 2 input points.")

    spline, _ = fit_cubic_spline(
        pts,
        smoothing=float(args.spline_s),
        n_samples=int(args.samples),
    )

    prefix = Path(args.out_prefix)

    raw_csv = prefix.with_suffix(".input_points.csv")
    spline_csv = prefix.with_suffix(".sampled_points.csv")
    reg_img = prefix.with_suffix(".image.reg")

    save_points_csv(raw_csv, pts, header="x_pix,y_pix")
    save_points_csv(spline_csv, spline, header="x_pix,y_pix")
    save_ds9_region_image(reg_img, spline, color=args.color)

    print(f"Read input points:        {pts.shape[0]}")
    print(f"Saved input point CSV:    {raw_csv}")
    print(f"Saved sampled point CSV:  {spline_csv}")
    print(f"Saved DS9 image region:   {reg_img}")

    if args.fits is not None:
        wcs = get_wcs_from_fits(args.fits, args.hdu)
        reg_fk5 = prefix.with_suffix(".fk5.reg")
        world_csv = prefix.with_suffix(".sampled_points_world.csv")

        save_ds9_region_fk5(reg_fk5, spline, wcs, color=args.color)
        save_world_csv(world_csv, spline, wcs)

        print(f"Saved DS9 FK5 region:     {reg_fk5}")
        print(f"Saved sampled world CSV:  {world_csv}")


if __name__ == "__main__":
    main()
