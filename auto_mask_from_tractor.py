#!/usr/bin/env python3
"""
auto_mask_from_tractor.py

Generate a simple maskbits-like image (0=good, nonzero=masked) aligned to a Legacy Survey coadd image.

- Stars (type == PSF): circular masks
- Galaxies (type != PSF): elliptical masks derived from (shape_r, shape_e1, shape_e2)

Designed for DR10 Tractor catalogs:
  type: PSF/REX/DEV/EXP/SER ...
  shape_r: arcsec (half-light radius)
  shape_e1, shape_e2: ellipticity components

References:
- DR10 tractor catalog columns (type, shape_r, shape_e1/e2) are documented on the Legacy Survey site.

Typical use:
  ./auto_mask_from_tractor.py \
    --image legacysurvey-0470m607-image-r.fits.fz --image-hdu 1 \
    --tractor tractor-0470m607.fits --tractor-hdu 1 \
    --psfsize legacysurvey-0470m607-psfsize-r.fits.fz --psfsize-hdu 1 \
    --pixscale-arcsec 0.262 \
    --maglim 24.5 \
    --star-kfwhm 3.0 \
    --gal-kre 4.0 \
    --keep-circle 3200,1550,220 \
    --out-maskbits maskbits_auto.fits

If you will pass --bbox to make_residual_images.py, DO NOT crop this mask.
Let make_residual_images.py crop it internally.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS


def read_image_and_wcs(path: Path, hdu: int) -> tuple[np.ndarray, fits.Header, WCS]:
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[hdu].data, dtype=np.float32)
        hdr = hdul[hdu].header
    wcs = WCS(hdr)
    return data, hdr, wcs


def read_table(path: Path, hdu: int) -> np.ndarray:
    with fits.open(path, memmap=True) as hdul:
        tab = hdul[hdu].data
        if tab is None:
            raise RuntimeError(f"No table in {path} HDU {hdu}")
        return tab


def nanomaggies_to_abmag(flux_nmgy: np.ndarray) -> np.ndarray:
    # Legacy nanomaggies: mag = 22.5 - 2.5 log10(flux)
    f = np.asarray(flux_nmgy, dtype=np.float64)
    out = np.full_like(f, np.inf, dtype=np.float64)
    m = f > 0
    out[m] = 22.5 - 2.5 * np.log10(f[m])
    return out


def parse_keep_circle(s: str) -> Tuple[float, float, float]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("--keep-circle must be 'x,y,r' in PIXELS")
    return float(parts[0]), float(parts[1]), float(parts[2])


def ellipse_params_from_e1e2(e1: float, e2: float) -> Tuple[float, float]:
    """
    Convert (e1,e2) to:
      - axis ratio q = b/a  (clamped)
      - position angle theta (radians), measured CCW from +x
    Using the common definition e = (a-b)/(a+b), with components e1=e cos2θ, e2=e sin2θ.
    """
    e = float(np.hypot(e1, e2))
    e = min(e, 0.95)
    q = (1.0 - e) / (1.0 + e)
    q = float(np.clip(q, 0.05, 1.0))
    theta = 0.5 * float(np.arctan2(e2, e1))
    return q, theta


def draw_disk(mask: np.ndarray, xc: float, yc: float, r: float, value: int) -> None:
    ny, nx = mask.shape
    x0 = max(int(np.floor(xc - r)), 0)
    x1 = min(int(np.ceil(xc + r)) + 1, nx)
    y0 = max(int(np.floor(yc - r)), 0)
    y1 = min(int(np.ceil(yc + r)) + 1, ny)
    if x1 <= x0 or y1 <= y0:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    rr2 = (xx - xc) ** 2 + (yy - yc) ** 2
    mask[y0:y1, x0:x1][rr2 <= r * r] = value


def draw_ellipse(
    mask: np.ndarray,
    xc: float,
    yc: float,
    a: float,
    b: float,
    theta: float,
    value: int,
) -> None:
    ny, nx = mask.shape
    # Bounding box radius conservative
    r = max(a, b)
    x0 = max(int(np.floor(xc - r)), 0)
    x1 = min(int(np.ceil(xc + r)) + 1, nx)
    y0 = max(int(np.floor(yc - r)), 0)
    y1 = min(int(np.ceil(yc + r)) + 1, ny)
    if x1 <= x0 or y1 <= y0:
        return

    yy, xx = np.mgrid[y0:y1, x0:x1]
    dx = xx - xc
    dy = yy - yc

    ct = np.cos(theta)
    st = np.sin(theta)
    xp = ct * dx + st * dy
    yp = -st * dx + ct * dy

    val = (xp * xp) / (a * a + 1e-30) + (yp * yp) / (b * b + 1e-30)
    mask[y0:y1, x0:x1][val <= 1.0] = value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--image-hdu", type=int, default=1)

    ap.add_argument("--tractor", type=Path, required=True)
    ap.add_argument("--tractor-hdu", type=int, default=1)

    ap.add_argument(
        "--psfsize",
        type=Path,
        default=None,
        help="Optional psfsize coadd image (arcsec) to set per-star radii.",
    )
    ap.add_argument("--psfsize-hdu", type=int, default=1)

    ap.add_argument("--pixscale-arcsec", type=float, default=0.262)

    ap.add_argument(
        "--band",
        choices=["g", "r", "i", "z"],
        default="r",
        help="Band used for flux->mag cut; expects tractor flux_<band>.",
    )
    ap.add_argument(
        "--maglim",
        type=float,
        default=24.5,
        help="Mask objects brighter than this (smaller number = brighter).",
    )

    ap.add_argument(
        "--star-kfwhm",
        type=float,
        default=3.0,
        help="Star mask radius = star_kfwhm * FWHM_pix (minimum applies too).",
    )
    ap.add_argument("--star-rmin-pix", type=float, default=3.0)

    ap.add_argument(
        "--gal-kre",
        type=float,
        default=4.0,
        help="Galaxy ellipse semi-major a = gal_kre * (shape_r/pixscale) in pixels.",
    )
    ap.add_argument("--gal-amin-pix", type=float, default=4.0)

    ap.add_argument(
        "--mask-value",
        type=int,
        default=1,
        help="Nonzero value written into masked pixels (0 remains unmasked).",
    )

    only_grp = ap.add_mutually_exclusive_group()
    only_grp.add_argument(
        "--stars-only",
        action="store_true",
        help="Mask only PSF (stellar) sources; do not mask extended sources (galaxies).",
    )
    only_grp.add_argument(
        "--galaxies-only",
        action="store_true",
        help="Mask only extended sources (galaxies); do not mask PSF (stellar) sources.",
    )

    ap.add_argument(
        "--keep-circle",
        action="append",
        default=[],
        help="Do NOT mask inside this circle (pixels): 'x,y,r'. Can repeat.",
    )
    ap.add_argument("--out-maskbits", type=Path, required=True)

    args = ap.parse_args()

    img, hdr, wcs = read_image_and_wcs(args.image, args.image_hdu)
    ny, nx = img.shape

    psf_map = None
    if args.psfsize is not None:
        psf_map, _, _ = read_image_and_wcs(args.psfsize, args.psfsize_hdu)
        if psf_map.shape != img.shape:
            raise RuntimeError(
                f"psfsize shape {psf_map.shape} != image shape {img.shape}"
            )

    tab = read_table(args.tractor, args.tractor_hdu)

    # Required columns
    for col in ["ra", "dec", "type"]:
        if col not in tab.names:
            raise RuntimeError(f"Tractor table missing column '{col}'")

    flux_col = f"flux_{args.band}"
    if flux_col not in tab.names:
        raise RuntimeError(f"Tractor table missing column '{flux_col}'")

    mag = nanomaggies_to_abmag(tab[flux_col])
    use = np.isfinite(mag) & (mag <= float(args.maglim))

    # World -> pixel (0-index) in the image WCS
    ra = np.asarray(tab["ra"][use], dtype=np.float64)
    dec = np.asarray(tab["dec"][use], dtype=np.float64)
    x, y = wcs.all_world2pix(ra, dec, 0)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    in_im = (x >= -5) & (x < nx + 5) & (y >= -5) & (y < ny + 5)
    x = x[in_im]
    y = y[in_im]
    t = tab["type"][use][in_im]
    mag_use = mag[use][in_im]

    # Normalize type strings
    def _tostr(v) -> str:
        if isinstance(v, (bytes, np.bytes_)):
            return v.decode("utf-8", errors="ignore")
        return str(v)

    tstr = np.array([_tostr(v) for v in t], dtype=object)

    mask = np.zeros((ny, nx), dtype=np.uint8)

    # Pre-parse keep circles
    keep_circles: List[Tuple[float, float, float]] = [
        parse_keep_circle(s) for s in args.keep_circle
    ]

    # Mask sources
    n_star = 0
    n_gal = 0
    pixscale = float(args.pixscale_arcsec)

    for xi, yi, ti, mi in zip(x, y, tstr, mag_use):
        # skip if wildly off-image
        if xi < -50 or xi > nx + 50 or yi < -50 or yi > ny + 50:
            continue
        if ti == "PSF":
            if args.galaxies_only:
                continue
            # Star radius from psfsize map if provided; else assume median-ish FWHM=1.3"
            if psf_map is not None:
                xip = int(np.clip(round(xi), 0, nx - 1))
                yip = int(np.clip(round(yi), 0, ny - 1))
                fwhm_arcsec = float(psf_map[yip, xip])
                if not np.isfinite(fwhm_arcsec) or fwhm_arcsec <= 0:
                    fwhm_arcsec = 1.3
            else:
                fwhm_arcsec = 1.3

            fwhm_pix = fwhm_arcsec / pixscale
            r = max(float(args.star_rmin_pix), float(args.star_kfwhm) * fwhm_pix)

            draw_disk(mask, xi, yi, r, int(args.mask_value))
            n_star += 1
        else:
            # Ellipse from galaxy shape if available; otherwise fall back to a modest circle.
            if (
                ("shape_r" in tab.names)
                and ("shape_e1" in tab.names)
                and ("shape_e2" in tab.names)
            ):
                # Need the same row values; easiest is to locate via ra/dec match is expensive.
                # Instead, we approximate by re-reading from the filtered table indices:
                # For robust use, run with full table indexing; kept simple here by using extra columns in-place.
                # If these columns are missing for a given ti, they may be 0.
                pass

            # We can only use shape columns if present in the *filtered* table slice; so rebuild arrays once.
            # (This is cheap compared to drawing.)
            # NOTE: The loop will be re-run; keep simple by using precomputed arrays outside the loop in a real pipeline.
            n_gal += 1

    # Second pass: mask extended sources (unless --stars-only)
    if not args.stars_only:
        nonpsf = tstr != "PSF"
        xg = x[nonpsf]
        yg = y[nonpsf]
        n_gal = int(np.count_nonzero(nonpsf))

        have_shapes = (
            ("shape_r" in tab.names)
            and ("shape_e1" in tab.names)
            and ("shape_e2" in tab.names)
        )
        if have_shapes:
            shape_r = np.asarray(tab["shape_r"][use][in_im], dtype=np.float64)[nonpsf]
            shape_e1 = np.asarray(tab["shape_e1"][use][in_im], dtype=np.float64)[nonpsf]
            shape_e2 = np.asarray(tab["shape_e2"][use][in_im], dtype=np.float64)[nonpsf]

            for xi, yi, rr, e1, e2 in zip(xg, yg, shape_r, shape_e1, shape_e2):
                if not np.isfinite(rr) or rr <= 0:
                    draw_disk(
                        mask, xi, yi, float(args.gal_amin_pix), int(args.mask_value)
                    )
                    continue
                q, theta = ellipse_params_from_e1e2(e1, e2)
                a = max(float(args.gal_amin_pix), float(args.gal_kre) * (rr / pixscale))
                b = max(1.0, a * q)
                draw_ellipse(mask, xi, yi, a, b, theta, int(args.mask_value))
        else:
            # No shape columns: fall back to circles for all extended sources
            rgal = float(args.gal_amin_pix)
            for xi, yi in zip(xg, yg):
                draw_disk(mask, xi, yi, rgal, int(args.mask_value))

    # Apply keep circles: set those pixels back to 0
    for xc, yc, rc in keep_circles:
        yy, xx = np.mgrid[0:ny, 0:nx]
        sel = (xx - xc) ** 2 + (yy - yc) ** 2 <= rc * rc
        mask[sel] = 0

    # Write FITS
    hdu = fits.PrimaryHDU(data=mask.astype(np.uint8), header=hdr)
    hdu.writeto(args.out_maskbits, overwrite=True)

    print(f"Wrote {args.out_maskbits}")
    print(
        f"Masked pixels: {np.count_nonzero(mask)} / {mask.size} ({100*np.count_nonzero(mask)/mask.size:.2f}%)"
    )
    if args.stars_only:
        print(f"Stars masked: {n_star} (galaxies skipped)")
    if args.galaxies_only:
        print(f"Galaxies masked: {n_gal} (stars skipped)")
    print("Note: output is a simple 0/1-style mask; treat any nonzero as masked.")


if __name__ == "__main__":
    main()
