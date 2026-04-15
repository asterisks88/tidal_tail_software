#!/usr/bin/env python3
"""
get_host_psf_from_psfsize.py

Read a Legacy Survey psfsize FITS file and a host-parameters JSON file,
sample the psfsize map at the host RA/Dec, and print the PSF FWHM in arcsec.

Example
-------
./get_host_psf_from_psfsize.py \
    --psfsize-file legacysurvey-0470m607-psfsize-r.fits.fz \
    --host-params host_params.json

Optional:
  --hdu 1           FITS HDU to read (default: 1)
  --order 1         interpolation order: 0=nearest, 1=bilinear (default: 1)
  --verbose         print extra info
"""

import argparse
import json
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy import ndimage


def load_json_arg(s: str) -> dict:
    """
    Accept either a path to a JSON file or an inline JSON string.
    """
    p = Path(s).expanduser()
    if p.exists() and p.is_file():
        return json.loads(p.read_text())
    return json.loads(s)


def read_fits_2d(path: Path, hdu: int = 1):
    """
    Read a 2D FITS image and return (data, header).
    """
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[hdu].data, dtype=np.float64)
        hdr = hdul[hdu].header.copy()

    if data.ndim != 2:
        raise RuntimeError(f"{path} HDU {hdu} is not 2D; got shape {data.shape}")
    return data, hdr


def sample_map_at_radec(map_data, map_wcs, ra_deg, dec_deg, order=1):
    """
    Sample a 2D map at sky coordinates (RA, Dec).

    Parameters
    ----------
    map_data : 2D ndarray
    map_wcs  : astropy.wcs.WCS
    ra_deg, dec_deg : float
    order : int
        Interpolation order:
          0 = nearest neighbor
          1 = bilinear

    Returns
    -------
    value : float
        Sampled map value.
    x, y : float
        Pixel coordinates in 0-based convention.
    """
    x, y = map_wcs.all_world2pix(float(ra_deg), float(dec_deg), 0)

    if not np.isfinite(x) or not np.isfinite(y):
        raise RuntimeError(
            f"Non-finite pixel coordinates from WCS for RA={ra_deg}, Dec={dec_deg}"
        )

    ny, nx = map_data.shape
    if x < 0 or x > (nx - 1) or y < 0 or y > (ny - 1):
        raise RuntimeError(
            f"Requested sky position lies outside psfsize image: "
            f"x={x:.3f}, y={y:.3f}, shape={(ny, nx)}"
        )

    val = ndimage.map_coordinates(
        map_data,
        [[y], [x]],
        order=int(order),
        mode="nearest",
        prefilter=(int(order) > 1),
    )[0]

    if not np.isfinite(val):
        raise RuntimeError(
            f"Sampled non-finite psfsize value at RA={ra_deg}, Dec={dec_deg}"
        )

    return float(val), float(x), float(y)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sample a Legacy Survey psfsize map at the host RA/Dec from a "
            "host-parameters JSON file and print the PSF FWHM in arcsec."
        )
    )
    parser.add_argument(
        "--psfsize-file",
        type=Path,
        required=True,
        help="Legacy Survey psfsize FITS file",
    )
    parser.add_argument(
        "--host-params",
        type=str,
        required=True,
        help=(
            "Host params JSON file or inline JSON. "
            "Must contain inputs.ra and inputs.dec"
        ),
    )
    parser.add_argument(
        "--hdu",
        type=int,
        default=1,
        help="FITS HDU for the psfsize image (default: 1)",
    )
    parser.add_argument(
        "--order",
        type=int,
        choices=[0, 1],
        default=1,
        help="Interpolation order: 0=nearest, 1=bilinear (default: 1)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra information",
    )

    args = parser.parse_args()

    host_info = load_json_arg(args.host_params)

    try:
        ra_deg = float(host_info["inputs"]["ra"])
        dec_deg = float(host_info["inputs"]["dec"])
    except Exception as e:
        raise RuntimeError(
            "Host params JSON must contain host_info['inputs']['ra'] and "
            "host_info['inputs']['dec']"
        ) from e

    psfsize_data, psfsize_hdr = read_fits_2d(args.psfsize_file, hdu=int(args.hdu))
    psfsize_wcs = WCS(psfsize_hdr)

    psf_fwhm_arcsec, xpix, ypix = sample_map_at_radec(
        psfsize_data,
        psfsize_wcs,
        ra_deg,
        dec_deg,
        order=int(args.order),
    )

    if args.verbose:
        print(f"psfsize file : {args.psfsize_file}")
        print(f"HDU          : {args.hdu}")
        print(f"Host RA, Dec : {ra_deg:.8f}, {dec_deg:.8f} deg")
        print(f"Pixel (0-based): x={xpix:.3f}, y={ypix:.3f}")
        print(f"Interpolation : {'bilinear' if args.order == 1 else 'nearest'}")

    print(f"{psf_fwhm_arcsec:.6f}")


if __name__ == "__main__":
    main()
