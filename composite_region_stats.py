#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from regions import PixCoord, Regions, SkyRegion


def resolve_hdu(hdul, hdu_spec):
    """
    Resolve HDU by integer index or EXTNAME string.
    """
    s = str(hdu_spec)
    if s.lstrip("-").isdigit():
        return hdul[int(s)]
    return hdul[s]


def read_fits_2d(path, hdu_spec="0"):
    with fits.open(path, memmap=True) as hdul:
        hdu = resolve_hdu(hdul, hdu_spec)
        data = np.asarray(hdu.data)
        header = hdu.header.copy()

    data = np.squeeze(data)
    if data.ndim != 2:
        raise ValueError(
            f"Selected HDU is not 2D after squeeze; got shape {data.shape}"
        )
    return data, header


def region_to_pixel(region, wcs):
    if isinstance(region, SkyRegion):
        return region.to_pixel(wcs)
    return region


def build_union_mask(data_shape, regions_list, wcs):
    """
    Build a boolean mask from a list of DS9 regions using pixel-center inclusion.

    Included regions are OR-ed together.
    Excluded regions remove pixels from the accumulated mask.
    """
    ny, nx = data_shape
    mask = np.zeros((ny, nx), dtype=bool)

    for reg in regions_list:
        preg = region_to_pixel(reg, wcs)

        include = bool(preg.meta.get("include", True))

        # Bounding box for efficiency
        bbox = preg.bounding_box
        xmin = max(0, int(np.floor(bbox.ixmin)))
        xmax = min(nx, int(np.ceil(bbox.ixmax)))
        ymin = max(0, int(np.floor(bbox.iymin)))
        ymax = min(ny, int(np.ceil(bbox.iymax)))

        if xmin >= xmax or ymin >= ymax:
            continue

        yy, xx = np.mgrid[ymin:ymax, xmin:xmax]

        # In astropy/regions pixel coordinates, pixel centers are at integer values
        coords = PixCoord(x=xx, y=yy)
        sub = preg.contains(coords)

        if include:
            mask[ymin:ymax, xmin:xmax] |= sub
        else:
            mask[ymin:ymax, xmin:xmax] &= ~sub

    return mask


def pixel_area_arcsec2(wcs):
    try:
        if not wcs.has_celestial:
            return None
        scales_deg = proj_plane_pixel_scales(wcs.celestial)
        return float(scales_deg[0] * scales_deg[1] * 3600.0 * 3600.0)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(
        description="Compute statistics over the union of DS9 regions on a FITS image."
    )
    ap.add_argument("fits_image", help="Input FITS image")
    ap.add_argument("region_file", help="DS9 region file")
    ap.add_argument(
        "--hdu",
        default="0",
        help="HDU index or EXTNAME to read from the FITS file (default: 0)",
    )
    ap.add_argument(
        "--keep-nan",
        action="store_true",
        help="Do not discard NaN/Inf values before computing stats",
    )
    args = ap.parse_args()

    fits_path = Path(args.fits_image)
    reg_path = Path(args.region_file)

    if not fits_path.exists():
        sys.exit(f"ERROR: FITS file not found: {fits_path}")
    if not reg_path.exists():
        sys.exit(f"ERROR: region file not found: {reg_path}")

    data, header = read_fits_2d(fits_path, args.hdu)
    wcs = WCS(header)

    try:
        regs = Regions.read(reg_path, format="ds9")
    except Exception as e:
        sys.exit(
            "ERROR: Could not parse DS9 region file.\n"
            "If this is a DS9 composite region, try dissolving the composite in DS9,\n"
            "save the member regions, and rerun.\n"
            f"Parser message: {e}"
        )

    mask = build_union_mask(data.shape, regs, wcs)

    if not np.any(mask):
        sys.exit("ERROR: Region mask is empty on this image.")

    vals = np.asarray(data[mask], dtype=float)

    if not args.keep_nan:
        vals = vals[np.isfinite(vals)]

    if vals.size == 0:
        sys.exit("ERROR: No finite pixels remain inside the region mask.")

    n_pix = vals.size
    pix_area = pixel_area_arcsec2(wcs)
    area_arcsec2 = None if pix_area is None else n_pix * pix_area

    print(f"Image          : {fits_path}")
    print(f"Region file    : {reg_path}")
    print(f"HDU            : {args.hdu}")
    print(f"Pixels in union: {n_pix}")

    if area_arcsec2 is not None:
        print(f"Area           : {area_arcsec2:.6f} arcsec^2")

    print(f"Sum            : {np.sum(vals):.10g}")
    print(f"Mean           : {np.mean(vals):.10g}")
    print(f"Median         : {np.median(vals):.10g}")
    print(f"Std            : {np.std(vals, ddof=0):.10g}")
    print(f"Min            : {np.min(vals):.10g}")
    print(f"Max            : {np.max(vals):.10g}")


if __name__ == "__main__":
    main()
