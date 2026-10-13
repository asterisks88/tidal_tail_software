#!/usr/bin/env python3

import argparse
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy.units import UnitsWarning


def mag_from_nmgy(flux):
    if flux <= 0 or not np.isfinite(flux):
        return np.nan
    return 22.5 - 2.5 * np.log10(flux)


def column_as_float(tab, name):
    """
    Return plain numeric values from an Astropy table column, stripping any unit.
    This avoids accidentally turning deg into deg^2 by multiplying twice.
    """
    col = tab[name]

    if hasattr(col, "quantity"):
        return np.asarray(col.quantity.value, dtype=float)

    return np.asarray(col, dtype=float)


def scalar_value(x):
    """
    Convert Astropy scalar, numpy scalar, or length-1 array-like value to float.
    """
    return float(np.ravel(np.asarray(x))[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tractor_fits", help="tractor-<brick>.fits")
    ap.add_argument("--ra", type=float, required=True)
    ap.add_argument("--dec", type=float, required=True)
    ap.add_argument("--max-sep-arcsec", type=float, default=5.0)
    args = ap.parse_args()

    # Suppress harmless FITS unit warnings from some Tractor catalog columns.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnitsWarning)
        tab = Table.read(args.tractor_fits, hdu=1)

    target = SkyCoord(
        ra=float(args.ra) * u.deg,
        dec=float(args.dec) * u.deg,
        frame="icrs",
    )

    ra = column_as_float(tab, "ra")
    dec = column_as_float(tab, "dec")

    coords = SkyCoord(
        ra=ra * u.deg,
        dec=dec * u.deg,
        frame="icrs",
    )

    idx, sep2d, _ = target.match_to_catalog_sky(coords)

    idx0 = int(np.ravel(np.asarray(idx))[0])
    sep_arcsec = float(np.ravel(sep2d.to_value(u.arcsec))[0])

    if sep_arcsec > args.max_sep_arcsec:
        raise RuntimeError(
            f"No Tractor source found within {args.max_sep_arcsec} arcsec. "
            f"Nearest source is {sep_arcsec:.3f} arcsec away."
        )

    row = tab[idx0]

    print(f"Nearest source separation: {sep_arcsec:.3f} arcsec")

    for col in ["release", "brickid", "objid"]:
        if col in tab.colnames:
            print(f"{col}: {row[col]}")

    print(f"type: {row['type']}")
    print(f"ra, dec: {row['ra']}, {row['dec']}")

    if "bx" in tab.colnames and "by" in tab.colnames:
        print(f"brick x, y: {row['bx']:.3f}, {row['by']:.3f}")

    print()
    print("Fluxes:")

    for band in ["g", "r", "i", "z", "W1", "W2", "W3", "W4"]:
        col = f"flux_{band}"
        if col in tab.colnames:
            flux = float(row[col])
            mag = mag_from_nmgy(flux)
            print(f"  {col:8s}: {flux:12.6g} nMgy   mag_{band}: {mag:.4f}")

    print()
    print("Shape/model parameters:")

    shape_r = np.nan
    if "shape_r" in tab.colnames:
        shape_r = float(row["shape_r"])
        print(f"shape_r:  {shape_r:.6f} arcsec")
        print("          # Tractor semi-major-axis model half-light radius")

    q = np.nan
    phi_deg = np.nan

    if "shape_e1" in tab.colnames and "shape_e2" in tab.colnames:
        e1 = float(row["shape_e1"])
        e2 = float(row["shape_e2"])
        eps = np.hypot(e1, e2)

        if np.isfinite(eps) and eps < 1:
            q = (1.0 - eps) / (1.0 + eps)
            phi_deg = 0.5 * np.degrees(np.arctan2(e2, e1))
            phi_deg = phi_deg % 180.0
        else:
            q = np.nan
            phi_deg = np.nan

        print(f"shape_e1: {e1:.6f}")
        print(f"shape_e2: {e2:.6f}")
        print(f"epsilon:  {eps:.6f}")
        print(f"q=b/a:    {q:.6f}")
        print(f"phi:      {phi_deg:.3f} deg")

        if np.isfinite(shape_r) and np.isfinite(q):
            shape_r_circ = shape_r * np.sqrt(q)
            print(f"shape_r_circularized: {shape_r_circ:.6f} arcsec")
            print("                      # shape_r * sqrt(q)")
    else:
        print("No shape_e1/shape_e2 columns found.")
        print()
        print("Shape/model parameters:")

    if "sersic" in tab.colnames:
        print(f"sersic:   {float(row['sersic']):.6f}")

    if "dchisq" in tab.colnames:
        print()
        print("dchisq:")
        print("  Usually ordered approximately as [PSF, REX, DEV, EXP, SER]")
        print(f"  {np.asarray(row['dchisq'])}")

    print()
    print("Useful catalog flags/fit diagnostics:")

    for col in [
        "fitbits",
        "maskbits",
        "fracmasked_g",
        "fracmasked_r",
        "fracmasked_i",
        "fracmasked_z",
        "fracin_g",
        "fracin_r",
        "fracin_i",
        "fracin_z",
        "rchisq_g",
        "rchisq_r",
        "rchisq_i",
        "rchisq_z",
    ]:
        if col in tab.colnames:
            print(f"{col}: {row[col]}")


if __name__ == "__main__":
    main()
