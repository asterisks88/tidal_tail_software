#!/usr/bin/env python3
"""
Query Gaia DR3 sources over the footprint of a Legacy Survey image and
save a local FITS table for use in auto_mask_from_tractor.py.

Output columns include:
  source_id, ra, dec, phot_g_mean_mag, phot_bp_mean_mag, phot_rp_mean_mag,
  parallax, pmra, pmdec

Example:
  ./get_gaia_for_legacy_image.py \
    --image legacysurvey-0470m607-image-r.fits.fz \
    --image-hdu 1 \
    --maglim 21.0 \
    --padding-arcmin 2.0 \
    --out gaia_dr3_0470m607_g21.fits
"""

from __future__ import annotations

import argparse
from pathlib import Path

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from astroquery.gaia import Gaia


def image_center_and_radius(image: Path, image_hdu: int, padding_arcmin: float):
    with fits.open(image, memmap=True) as hdul:
        data = hdul[image_hdu].data
        hdr = hdul[image_hdu].header

    if data is None:
        raise RuntimeError(f"No image data in {image} HDU {image_hdu}")

    ny, nx = data.shape
    wcs = WCS(hdr)

    # Pixel coordinates, 0-based, matching Astropy WCS convention.
    xc = 0.5 * (nx - 1)
    yc = 0.5 * (ny - 1)

    center = wcs.pixel_to_world(xc, yc)

    corners_x = [0, nx - 1, nx - 1, 0]
    corners_y = [0, 0, ny - 1, ny - 1]
    corners = wcs.pixel_to_world(corners_x, corners_y)

    radius = center.separation(corners).max()
    radius = radius + padding_arcmin * u.arcmin

    return center, radius, nx, ny


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--image-hdu", type=int, default=1)
    ap.add_argument("--maglim", type=float, default=21.0)
    ap.add_argument(
        "--padding-arcmin",
        type=float,
        default=2.0,
        help="Extra radius around image footprint.",
    )
    ap.add_argument(
        "--table",
        default="gaiadr3.gaia_source_lite",
        help="Gaia table to query. Use gaiadr3.gaia_source for full table.",
    )
    ap.add_argument("--out", type=Path, required=True)

    args = ap.parse_args()

    center, radius, nx, ny = image_center_and_radius(
        args.image,
        args.image_hdu,
        args.padding_arcmin,
    )

    ra0 = center.ra.deg
    dec0 = center.dec.deg
    radius_deg = radius.to_value(u.deg)

    print(f"Image shape: nx={nx}, ny={ny}")
    print(f"Query center: RA={ra0:.8f} deg, Dec={dec0:.8f} deg")
    print(f"Query radius: {radius_deg:.6f} deg")
    print(f"Gaia G limit: {args.maglim:.2f}")

    query = f"""
    SELECT
        source_id,
        ra,
        dec,
        phot_g_mean_mag,
        phot_bp_mean_mag,
        phot_rp_mean_mag,
        parallax,
        pmra,
        pmdec
    FROM {args.table}
    WHERE
        phot_g_mean_mag <= {args.maglim}
        AND 1 = CONTAINS(
            POINT('ICRS', ra, dec),
            CIRCLE('ICRS', {ra0}, {dec0}, {radius_deg})
        )
    """

    print("Submitting Gaia query...")
    job = Gaia.launch_job_async(query)
    tab = job.get_results()

    print(f"Retrieved {len(tab)} Gaia sources")

    # Convert to normal Astropy Table before writing.
    outtab = Table(tab)
    outtab.write(args.out, overwrite=True)

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
