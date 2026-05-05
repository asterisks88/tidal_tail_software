#!/usr/bin/env python3
"""
angsize_to_phys.py

Given a redshift z and an angular size in arcminutes, print:
  1) the angular-diameter distance
  2) the projected proper physical size of the feature

Usage:
  python angsize_to_phys.py 0.25 3.4
  python angsize_to_phys.py --z 0.25 --theta-arcmin 3.4
"""

import argparse

import astropy.units as u
from astropy.cosmology import Planck18


def physical_size_from_angular_arcmin(z, theta_arcmin, cosmology=Planck18):
    """
    Compute angular-diameter distance and projected proper physical size.

    Parameters
    ----------
    z : float
        Redshift of the feature.
    theta_arcmin : float
        Angular size in arcminutes.
    cosmology : astropy.cosmology.Cosmology, optional
        Cosmology to use. Default is Planck18.

    Returns
    -------
    DA : astropy.units.Quantity
        Angular-diameter distance.
    size : astropy.units.Quantity
        Projected proper physical size.
    """
    theta = theta_arcmin * u.arcmin
    DA = cosmology.angular_diameter_distance(z)
    size = (theta.to(u.rad).value * DA).to(u.kpc)
    return DA, size


def main():
    parser = argparse.ArgumentParser(
        description="Convert angular size (arcmin) at redshift z into projected proper physical size using Astropy Planck18."
    )
    parser.add_argument("z", nargs="?", type=float, help="Redshift")
    parser.add_argument("theta_arcmin", nargs="?", type=float, help="Angular size in arcminutes")
    parser.add_argument("--z", dest="z_kw", type=float, help="Redshift")
    parser.add_argument("--theta-arcmin", dest="theta_kw", type=float, help="Angular size in arcminutes")
    args = parser.parse_args()

    z = args.z_kw if args.z_kw is not None else args.z
    theta_arcmin = args.theta_kw if args.theta_kw is not None else args.theta_arcmin

    if z is None or theta_arcmin is None:
        parser.error("you must provide z and theta_arcmin, either positionally or with --z and --theta-arcmin")

    DA, size = physical_size_from_angular_arcmin(z, theta_arcmin)

    # Optional cross-check using Astropy's built-in transverse scale helper
    scale = Planck18.kpc_proper_per_arcmin(z)
    size_check = (theta_arcmin * u.arcmin * scale).to(u.kpc)

    print(f"Cosmology: {Planck18.name}")
    print(f"Redshift z: {z}")
    print(f"Angular size: {theta_arcmin} arcmin")
    print(f"Angular-diameter distance: {DA:.6f}")
    print(f"Projected proper physical size: {size:.6f}")
    print(f"Scale at z: {scale:.6f}")
    print(f"Cross-check size from kpc/arcmin scale: {size_check:.6f}")


if __name__ == "__main__":
    main()
