#!/usr/bin/env python3
"""
legacy_ellipse_profile.py

Measure an elliptical curve of growth from a Legacy Survey / mu_from_legacy_background.py
output image and a small target dictionary containing at least RA and Dec.

Outputs:
  1. A CSV profile with progressively larger elliptical apertures.
  2. A JSON geometry/summary file with PA, axis ratio, Reff, fluxes, and errors.
  3. Optional profile and diagnostic plots.
  4. A compact g/r Sérsic-total magnitude and circularized Reff JSON when --band is g or r.

Flux convention:
  - Image data are assumed to be nanomaggies per pixel.
  - The preferred input HDU is SKY_SUB_NMGY.

Error convention:
  - If --invvar-fits is supplied, aperture flux errors are computed by summing
    1/invvar over unmasked pixels.
  - If no invvar is supplied, the script estimates a fallback per-pixel sky RMS
    from an outer elliptical annulus, or uses --sky-sigma-nmgy-pix if supplied.
  - Optional systematic estimates are stored separately in geometry JSON. Their
    nuisance-parameter scales must be supplied from external checks; they are
    not inferred from the per-pixel inverse variance.

Examples:

  python legacy_ellipse_profile.py measure \
      legacysurvey-0470m607-mu.fits target.py LEDA175897_profile.csv \
      --max-sma-arcsec 80 --step-arcsec 1 \
      --plot-profile LEDA175897_profile.png \
      --diagnostic-png LEDA175897_apertures.png

  python legacy_ellipse_profile.py measure \
      legacysurvey-0470m607-mu.fits target.py LEDA175897_profile.csv \
      --invvar-fits legacysurvey-0470m607-invvar-r.fits.fz \
      --noise-corr-factor 1.5

  python legacy_ellipse_profile.py plot LEDA175897_profile.csv \
      --output LEDA175897_profile.png
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import os
import platform
import sys
import tempfile
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.units import UnitsWarning
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area

AB_ZP_NMGY = 22.5


def mag_from_nmgy(flux_nmgy: np.ndarray | float) -> np.ndarray | float:
    flux = np.asarray(flux_nmgy, dtype=float)
    out = np.full_like(flux, np.nan, dtype=float)
    good = np.isfinite(flux) & (flux > 0)
    out[good] = AB_ZP_NMGY - 2.5 * np.log10(flux[good])
    if np.ndim(flux_nmgy) == 0:
        return float(out)
    return out


def mu_from_nmgy_per_arcsec2(sb_nmgy_arcsec2: np.ndarray | float) -> np.ndarray | float:
    return mag_from_nmgy(sb_nmgy_arcsec2)


def mag_err_from_flux_err(flux_nmgy: float, flux_err_nmgy: float) -> float:
    if not np.isfinite(flux_nmgy) or not np.isfinite(flux_err_nmgy) or flux_nmgy <= 0:
        return np.nan
    return float(2.5 / np.log(10.0) * flux_err_nmgy / flux_nmgy)


def aperture_flux_error(
    aperture: np.ndarray,
    invvar: Optional[np.ndarray] = None,
    sky_sigma_nmgy_pix: Optional[float] = None,
    noise_corr_factor: float = 1.0,
) -> float:
    """
    Estimate uncertainty on summed nanomaggy flux in an aperture.

    Preferred mode: sum 1/invvar over aperture pixels.
    Fallback mode: sqrt(Npix) * sky_sigma_nmgy_pix.
    """
    n_pix = int(np.count_nonzero(aperture))
    if n_pix <= 0:
        return np.nan

    if invvar is not None:
        iv = np.asarray(invvar[aperture], dtype=float)
        good = np.isfinite(iv) & (iv > 0)
        if np.any(good):
            var = float(np.sum(1.0 / iv[good]))
            return float(noise_corr_factor * np.sqrt(var))

    if (
        sky_sigma_nmgy_pix is not None
        and np.isfinite(sky_sigma_nmgy_pix)
        and sky_sigma_nmgy_pix > 0
    ):
        return float(noise_corr_factor * sky_sigma_nmgy_pix * np.sqrt(n_pix))

    return np.nan


def mag_extinction_corrected(mag: float, extinction_mag: Optional[float]) -> float:
    if (
        extinction_mag is None
        or not np.isfinite(extinction_mag)
        or not np.isfinite(mag)
    ):
        return np.nan
    return float(mag - extinction_mag)


def flux_extinction_corrected(
    flux_nmgy: float, mw_transmission: Optional[float]
) -> float:
    if (
        mw_transmission is None
        or not np.isfinite(mw_transmission)
        or mw_transmission <= 0
    ):
        return np.nan
    if not np.isfinite(flux_nmgy):
        return np.nan
    return float(flux_nmgy / mw_transmission)


def column_as_float(tab: Table, name: str) -> np.ndarray:
    col = tab[name]
    if hasattr(col, "quantity"):
        return np.asarray(col.quantity.value, dtype=float)
    return np.asarray(col, dtype=float)


def get_nearest_tractor_row(
    tractor_fits: str | Path,
    ra_deg: float,
    dec_deg: float,
    hdu: str = "1",
    max_sep_arcsec: float = 5.0,
) -> tuple[Table, int, float, str]:
    """Read a Tractor table and return the nearest row index to RA/Dec."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnitsWarning)
        tab = Table.read(tractor_fits, hdu=int(hdu) if str(hdu).isdigit() else hdu)

    target = SkyCoord(
        ra=float(ra_deg) * u.deg, dec=float(dec_deg) * u.deg, frame="icrs"
    )
    coords = SkyCoord(
        ra=column_as_float(tab, "ra") * u.deg,
        dec=column_as_float(tab, "dec") * u.deg,
        frame="icrs",
    )
    idx, sep2d, _ = target.match_to_catalog_sky(coords)
    idx0 = int(np.ravel(np.asarray(idx))[0])
    sep_arcsec = float(np.ravel(sep2d.to_value(u.arcsec))[0])

    if sep_arcsec > max_sep_arcsec:
        raise ValueError(
            f"Nearest Tractor source is {sep_arcsec:.3f} arcsec away, larger than "
            f"--tractor-max-sep-arcsec={max_sep_arcsec}."
        )

    row = tab[idx0]
    source_id_parts = []
    for key in ["release", "brickid", "objid"]:
        if key in tab.colnames:
            source_id_parts.append(f"{key}={row[key]}")
    source_id = ", ".join(source_id_parts) if source_id_parts else f"row={idx0}"
    return tab, idx0, sep_arcsec, source_id


def read_mw_transmission_from_tractor(
    tractor_fits: str | Path,
    ra_deg: float,
    dec_deg: float,
    band: str,
    hdu: str = "1",
    max_sep_arcsec: float = 5.0,
) -> tuple[float, float, str, str]:
    """
    Return mw_transmission_<band> from the nearest Tractor row.

    Returns: transmission, separation_arcsec, column_name, source_id_string.
    """
    band = band.lower()
    colname = f"mw_transmission_{band}"
    tab, idx0, sep_arcsec, source_id = get_nearest_tractor_row(
        tractor_fits=tractor_fits,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        hdu=hdu,
        max_sep_arcsec=max_sep_arcsec,
    )

    if colname not in tab.colnames:
        raise ValueError(f"Column {colname!r} not found in {tractor_fits}")

    transmission = float(tab[colname][idx0])
    if not np.isfinite(transmission) or transmission <= 0:
        raise ValueError(f"Invalid {colname}={transmission} for nearest Tractor source")

    return transmission, sep_arcsec, colname, source_id


def sersic_b_n(n: float) -> float:
    """Approximate b_n such that R_e encloses half the Sérsic model light."""
    if n <= 0:
        return np.nan
    return float(2.0 * n - 1.0 / 3.0 + 4.0 / (405.0 * n) + 46.0 / (25515.0 * n * n))


def sersic_radius_for_light_fraction(re_arcsec: float, n: float, frac: float) -> float:
    """
    Semi-major-axis radius enclosing a chosen fraction of a Sérsic model.

    Uses gammaincinv when scipy is available. Falls back to common practical
    multiples if scipy is unavailable.
    """
    if not np.isfinite(re_arcsec) or re_arcsec <= 0:
        return np.nan
    if not np.isfinite(n) or n <= 0:
        return np.nan
    if not (0 < frac < 1):
        raise ValueError("Sérsic light fraction must be between 0 and 1")

    try:
        from scipy.special import gammaincinv

        bn = sersic_b_n(n)
        x = gammaincinv(2.0 * n, frac)
        return float(re_arcsec * (x / bn) ** n)
    except Exception:
        # Conservative fallback if scipy is not installed.
        fallback = {0.80: 3.0, 0.90: 4.0, 0.95: 5.0, 0.98: 7.0}
        nearest = min(fallback, key=lambda k: abs(k - frac))
        return float(re_arcsec * fallback[nearest])


def estimate_max_sma_from_tractor(
    tractor_fits: str | Path,
    ra_deg: float,
    dec_deg: float,
    hdu: str,
    max_sep_arcsec: float,
    method: str,
    factor: float,
    light_frac: float,
) -> tuple[float, str, float, str]:
    """
    Estimate a practical max SMA from the nearest Tractor model.

    Returns: max_sma_arcsec, description, match_sep_arcsec, source_id.
    """
    tab, idx0, sep_arcsec, source_id = get_nearest_tractor_row(
        tractor_fits=tractor_fits,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        hdu=hdu,
        max_sep_arcsec=max_sep_arcsec,
    )
    row = tab[idx0]

    if "shape_r" not in tab.colnames:
        raise ValueError("Tractor table does not contain shape_r")
    shape_r = float(row["shape_r"])
    if not np.isfinite(shape_r) or shape_r <= 0:
        raise ValueError(f"Invalid Tractor shape_r={shape_r}")

    source_type = str(row["type"]).strip() if "type" in tab.colnames else "UNKNOWN"

    if method == "factor":
        if not np.isfinite(factor) or factor <= 0:
            raise ValueError("--tractor-max-sma-factor must be positive")
        radius = float(factor * shape_r)
        desc = f"tractor_factor:{factor:g}*shape_r; type={source_type}; shape_r={shape_r:.6g} arcsec"
    elif method == "sersic-fraction":
        if source_type.upper() == "EXP":
            n = 1.0
        elif source_type.upper() == "DEV":
            n = 4.0
        elif source_type.upper() == "REX":
            n = 1.0
        elif (
            "sersic" in tab.colnames
            and np.isfinite(float(row["sersic"]))
            and float(row["sersic"]) > 0
        ):
            n = float(row["sersic"])
        else:
            raise ValueError(
                "Sérsic-fraction max-SMA requires type EXP/DEV/REX or a valid sersic column"
            )
        radius = sersic_radius_for_light_fraction(shape_r, n, light_frac)
        desc = f"tractor_sersic_fraction:{light_frac:g}; type={source_type}; n={n:.4g}; shape_r={shape_r:.6g} arcsec"
    else:
        raise ValueError(f"Unknown --tractor-max-sma-method={method!r}")

    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Failed to derive a positive max SMA from the Tractor row")
    return radius, desc, sep_arcsec, source_id


def estimate_geometry_from_tractor(
    tractor_fits: str | Path,
    ra_deg: float,
    dec_deg: float,
    hdu: str,
    max_sep_arcsec: float,
) -> tuple[float, float, str, float, str]:
    """
    Estimate q=b/a and sky PA from the nearest Tractor source.

    Uses Legacy/Tractor ellipticity components:
        epsilon = sqrt(e1^2 + e2^2)
        q = (1 - epsilon) / (1 + epsilon)
        PA = 0.5 atan2(e2, e1)  [mod 180 deg]

    Returns: q, pa_deg_east_of_north, description, match_sep_arcsec, source_id.
    """
    tab, idx0, sep_arcsec, source_id = get_nearest_tractor_row(
        tractor_fits=tractor_fits,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        hdu=hdu,
        max_sep_arcsec=max_sep_arcsec,
    )
    row = tab[idx0]

    for col in ["shape_e1", "shape_e2"]:
        if col not in tab.colnames:
            raise ValueError(f"Tractor table does not contain {col}")

    e1 = float(row["shape_e1"])
    e2 = float(row["shape_e2"])
    eps = float(np.hypot(e1, e2))
    if not np.isfinite(eps) or eps >= 1.0:
        raise ValueError(
            f"Invalid Tractor ellipticity from shape_e1={e1}, shape_e2={e2}; epsilon={eps}"
        )

    q = float((1.0 - eps) / (1.0 + eps))
    q = float(np.clip(q, 0.05, 1.0))

    if eps < 1.0e-8:
        pa_deg = 0.0
        pa_note = "PA arbitrary for nearly round source"
    else:
        pa_deg = float((0.5 * np.degrees(np.arctan2(e2, e1))) % 180.0)
        pa_note = "PA from shape_e1/shape_e2"

    source_type = str(row["type"]).strip() if "type" in tab.colnames else "UNKNOWN"
    shape_r_txt = ""
    if "shape_r" in tab.colnames:
        shape_r_txt = f"; shape_r={float(row['shape_r']):.6g} arcsec"
    desc = (
        f"tractor_shape; type={source_type}; e1={e1:.6g}; e2={e2:.6g}; "
        f"epsilon={eps:.6g}; q={q:.6g}; PA={pa_deg:.6g} deg; {pa_note}{shape_r_txt}"
    )
    return q, pa_deg, desc, sep_arcsec, source_id


def read_target_dict(path: str | Path) -> dict[str, Any]:
    """
    Read a target dictionary from JSON or from a Python file containing a dict literal.
    This does not execute the file.
    """
    text = Path(path).read_text()

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    tree = ast.parse(text, filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            obj = ast.literal_eval(node)
            if isinstance(obj, dict):
                return obj

    raise ValueError(f"Could not find a dictionary in {path}")


def get_hdu_data_header(
    hdul: fits.HDUList, hdu_name_or_index: str
) -> tuple[np.ndarray, fits.Header, str]:
    try:
        hdu = hdul[hdu_name_or_index]
        if hdu.data is not None:
            return np.asarray(hdu.data, dtype=float), hdu.header, str(hdu.name)
    except Exception:
        pass

    try:
        idx = int(hdu_name_or_index)
        hdu = hdul[idx]
        if hdu.data is not None:
            return np.asarray(hdu.data, dtype=float), hdu.header, str(idx)
    except Exception:
        pass

    for name in ["SKY_SUB_NMGY", "SKY_SUB", "DATA_NMGY", "IMAGE", "PRIMARY"]:
        try:
            hdu = hdul[name]
            if hdu.data is not None:
                return np.asarray(hdu.data, dtype=float), hdu.header, str(hdu.name)
        except Exception:
            continue

    for i, hdu in enumerate(hdul):
        if hdu.data is not None and np.asarray(hdu.data).ndim == 2:
            return np.asarray(hdu.data, dtype=float), hdu.header, str(i)

    raise ValueError("No 2-D image HDU found in FITS file")


def make_wcs(image_header: fits.Header, primary_header: fits.Header) -> WCS:
    w = WCS(image_header)
    if w.has_celestial:
        return w.celestial
    w = WCS(primary_header)
    if w.has_celestial:
        return w.celestial
    raise ValueError("No celestial WCS found in selected HDU or primary header")


def pixel_area_arcsec2_from_wcs(
    wcs: WCS, fallback_pixscale: Optional[float] = None
) -> float:
    try:
        area_deg2 = float(proj_plane_pixel_area(wcs))
        if np.isfinite(area_deg2) and area_deg2 > 0:
            return area_deg2 * 3600.0**2
    except Exception:
        pass
    if fallback_pixscale is not None:
        return float(fallback_pixscale) ** 2
    raise ValueError("Could not infer pixel area from WCS; provide --pixscale-arcsec")


def load_mask(
    mask_fits: Optional[str], mask_hdu: str, expected_shape: tuple[int, int]
) -> Optional[np.ndarray]:
    if not mask_fits:
        return None
    with fits.open(mask_fits) as hdul:
        data, _, _ = get_hdu_data_header(hdul, mask_hdu)
    if data.shape != expected_shape:
        raise ValueError(
            f"Mask shape {data.shape} does not match image shape {expected_shape}"
        )
    return np.asarray(data != 0, dtype=bool)


def load_invvar(
    invvar_fits: Optional[str], invvar_hdu: str, expected_shape: tuple[int, int]
) -> Optional[np.ndarray]:
    if not invvar_fits:
        return None
    with fits.open(invvar_fits) as hdul:
        data, _, _ = get_hdu_data_header(hdul, invvar_hdu)
    if data.shape != expected_shape:
        raise ValueError(
            f"Invvar shape {data.shape} does not match image shape {expected_shape}"
        )
    data = np.asarray(data, dtype=float)
    data[~np.isfinite(data)] = 0.0
    data[data < 0] = 0.0
    return data


def crop_slices_around(
    x0: float, y0: float, radius_pix: float, shape: tuple[int, int]
) -> tuple[slice, slice]:
    ny, nx = shape
    pad = int(np.ceil(radius_pix)) + 8
    xmin = max(0, int(np.floor(x0)) - pad)
    xmax = min(nx, int(np.floor(x0)) + pad + 1)
    ymin = max(0, int(np.floor(y0)) - pad)
    ymax = min(ny, int(np.floor(y0)) + pad + 1)
    return slice(ymin, ymax), slice(xmin, xmax)


def robust_sky_sigma(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 10:
        return 0.0, 0.0, np.nan
    mean, median, std = sigma_clipped_stats(values, sigma=3.0, maxiters=5)
    return float(mean), float(median), float(std)


@dataclass
class Geometry:
    target_name: str
    ra_deg: float
    dec_deg: float
    x0_pix_initial: float
    y0_pix_initial: float
    x0_pix_used: float
    y0_pix_used: float
    hdu_used: str
    pixscale_arcsec: float
    pixel_area_arcsec2: float
    theta_image_deg: float
    pa_major_deg_east_of_north: float
    pa_minor_deg_east_of_north: float
    q_b_over_a: float
    ellipticity: float
    moment_radius_arcsec: float
    moment_threshold_sigma: float
    error_method: str
    sky_sigma_nmgy_pix_for_errors: Optional[float] = None
    noise_corr_factor: float = 1.0
    band: Optional[str] = None
    input_extinction_corrected: bool = False
    extinction_source: Optional[str] = None
    extinction_mag_band: Optional[float] = None
    mw_transmission_band: Optional[float] = None
    tractor_extinction_column: Optional[str] = None
    tractor_extinction_sep_arcsec: Optional[float] = None
    tractor_extinction_source_id: Optional[str] = None
    tractor_max_sma_description: Optional[str] = None
    tractor_max_sma_sep_arcsec: Optional[float] = None
    tractor_max_sma_source_id: Optional[str] = None
    tractor_geometry_description: Optional[str] = None
    tractor_geometry_sep_arcsec: Optional[float] = None
    tractor_geometry_source_id: Optional[str] = None
    pa_source: Optional[str] = None
    q_source: Optional[str] = None
    reff_elliptical_sma_arcsec: Optional[float] = None
    reff_circularized_arcsec: Optional[float] = None
    reff_elliptical_sma_kpc: Optional[float] = None
    reff_circularized_kpc: Optional[float] = None
    reff_sersiccorr_source: Optional[str] = None
    reff_sersiccorr_light_frac: Optional[float] = None
    reff_sersiccorr_target_fraction_of_aperture_flux: Optional[float] = None
    reff_sersiccorr_elliptical_sma_arcsec: Optional[float] = None
    reff_sersiccorr_circularized_arcsec: Optional[float] = None
    reff_sersiccorr_elliptical_sma_kpc: Optional[float] = None
    reff_sersiccorr_circularized_kpc: Optional[float] = None
    flux_nmgy_within_reff_sersiccorr: Optional[float] = None
    flux_extcorr_nmgy_within_reff_sersiccorr: Optional[float] = None
    mag_ab_within_reff_sersiccorr: Optional[float] = None
    mag_extcorr_ab_within_reff_sersiccorr: Optional[float] = None
    total_flux_nmgy_used_for_reff: Optional[float] = None
    total_flux_err_nmgy_used_for_reff: Optional[float] = None
    total_flux_extcorr_nmgy_used_for_reff: Optional[float] = None
    total_flux_extcorr_err_nmgy_used_for_reff: Optional[float] = None
    sersic_total_correction_source: Optional[str] = None
    sersic_total_correction_light_frac: Optional[float] = None
    total_flux_sersiccorr_nmgy: Optional[float] = None
    total_flux_sersiccorr_err_nmgy: Optional[float] = None
    total_mag_sersiccorr_ab: Optional[float] = None
    total_mag_sersiccorr_err_ab: Optional[float] = None
    total_flux_sersiccorr_extcorr_nmgy: Optional[float] = None
    total_flux_sersiccorr_extcorr_err_nmgy: Optional[float] = None
    total_mag_sersiccorr_extcorr_ab: Optional[float] = None
    total_mag_used_for_reff: Optional[float] = None
    total_mag_err_used_for_reff: Optional[float] = None
    total_mag_extcorr_used_for_reff: Optional[float] = None
    flux_nmgy_within_reff: Optional[float] = None
    flux_err_nmgy_within_reff: Optional[float] = None
    flux_extcorr_nmgy_within_reff: Optional[float] = None
    flux_extcorr_err_nmgy_within_reff: Optional[float] = None
    mag_ab_within_reff: Optional[float] = None
    mag_err_ab_within_reff: Optional[float] = None
    mag_extcorr_ab_within_reff: Optional[float] = None
    max_sma_arcsec: Optional[float] = None
    max_sma_source: Optional[str] = None
    bulge_radius_source: Optional[str] = None
    bulge_radius_sma_arcsec: Optional[float] = None
    bulge_radius_circularized_arcsec: Optional[float] = None
    bulge_radius_sma_kpc: Optional[float] = None
    bulge_radius_circularized_kpc: Optional[float] = None
    bulge_flux_nmgy: Optional[float] = None
    bulge_flux_err_nmgy: Optional[float] = None
    bulge_flux_extcorr_nmgy: Optional[float] = None
    bulge_flux_extcorr_err_nmgy: Optional[float] = None
    bulge_mag_ab: Optional[float] = None
    bulge_mag_err_ab: Optional[float] = None
    bulge_mag_extcorr_ab: Optional[float] = None
    bulge_disk_fit_mu0: Optional[float] = None
    bulge_disk_fit_slope_mag_per_arcsec: Optional[float] = None
    bulge_excess_threshold_mag: Optional[float] = None
    input_parameters: Optional[dict[str, Any]] = None
    notes: Optional[str] = None
    systematic_uncertainties: Optional[dict[str, Any]] = None


def estimate_moment_geometry(
    data: np.ndarray,
    valid: np.ndarray,
    x0: float,
    y0: float,
    pixscale_arcsec: float,
    moment_radius_arcsec: float,
    threshold_sigma: float,
    recenter: bool = True,
) -> tuple[float, float, float, float]:
    radius_pix = moment_radius_arcsec / pixscale_arcsec
    yy, xx = np.indices(data.shape, dtype=float)
    rr = np.hypot(xx - x0, yy - y0)
    aperture = (rr <= radius_pix) & valid & np.isfinite(data)

    if np.count_nonzero(aperture) < 20:
        raise ValueError(
            "Too few valid pixels in moment aperture; increase --moment-radius-arcsec or check mask"
        )

    _, sky_med, sky_std = robust_sky_sigma(data[aperture])
    if not np.isfinite(sky_std) or sky_std <= 0:
        sky_std = np.nanstd(data[aperture])
    if not np.isfinite(sky_std) or sky_std <= 0:
        sky_std = 1.0e-12

    signal = data - sky_med
    weights = np.clip(signal - threshold_sigma * sky_std, 0.0, None)
    gaussian = np.exp(-0.5 * (rr / max(radius_pix / 2.0, 1.0)) ** 2)
    weights *= gaussian
    weights[~aperture] = 0.0

    if np.sum(weights) <= 0:
        weights = np.clip(signal, 0.0, None) * gaussian
        weights[~aperture] = 0.0

    if np.sum(weights) <= 0:
        raise ValueError("Could not obtain positive weights for moment geometry")

    if recenter:
        xc = float(np.sum(weights * xx) / np.sum(weights))
        yc = float(np.sum(weights * yy) / np.sum(weights))
    else:
        xc, yc = float(x0), float(y0)

    dx = xx - xc
    dy = yy - yc
    wsum = np.sum(weights)
    cov = np.array(
        [
            [
                float(np.sum(weights * dx * dx) / wsum),
                float(np.sum(weights * dx * dy) / wsum),
            ],
            [
                float(np.sum(weights * dx * dy) / wsum),
                float(np.sum(weights * dy * dy) / wsum),
            ],
        ]
    )

    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evec = evecs[:, order[0]]

    theta = float(np.arctan2(evec[1], evec[0]))
    if theta < 0:
        theta += np.pi

    if evals[0] <= 0 or evals[1] <= 0:
        q = 1.0
    else:
        q = float(np.sqrt(evals[1] / evals[0]))
    q = float(np.clip(q, 0.05, 1.0))

    return xc, yc, theta, q


def pa_from_pixel_theta(
    wcs: WCS, x0: float, y0: float, theta_rad: float, step_pix: float = 20.0
) -> float:
    c0 = wcs.pixel_to_world(x0, y0)
    x1 = x0 + step_pix * np.cos(theta_rad)
    y1 = y0 + step_pix * np.sin(theta_rad)
    c1 = wcs.pixel_to_world(x1, y1)
    pa = c0.position_angle(c1).to_value(u.deg)
    return float(pa % 180.0)


def pixel_theta_from_pa(
    wcs: WCS, x0: float, y0: float, pa_deg: float, offset_arcsec: float = 20.0
) -> float:
    c0 = wcs.pixel_to_world(x0, y0)
    c1 = c0.directional_offset_by(pa_deg * u.deg, offset_arcsec * u.arcsec)
    x1, y1 = wcs.world_to_pixel(c1)
    theta = float(np.arctan2(y1 - y0, x1 - x0))
    if theta < 0:
        theta += np.pi
    if theta >= np.pi:
        theta -= np.pi
    return theta


def elliptical_radius_pix(
    xx: np.ndarray, yy: np.ndarray, x0: float, y0: float, theta: float, q: float
) -> np.ndarray:
    dx = xx - x0
    dy = yy - y0
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    x_major = dx * cos_t + dy * sin_t
    y_minor = -dx * sin_t + dy * cos_t
    return np.sqrt(x_major**2 + (y_minor / q) ** 2)


def estimate_auto_max_sma_arcsec(
    data: np.ndarray,
    valid: np.ndarray,
    invvar: Optional[np.ndarray],
    x0: float,
    y0: float,
    theta: float,
    q: float,
    pixscale_arcsec: float,
    step_arcsec: float,
    search_max_arcsec: float,
    min_sma_arcsec: float,
    snr_threshold: float,
    frac_growth_threshold: float,
    consecutive: int,
    noise_corr_factor: float = 1.0,
) -> float:
    """
    Estimate a reasonable maximum semi-major-axis radius for a curve of growth.

    The stopping condition is deliberately conservative: after min_sma_arcsec,
    require `consecutive` annuli for which the annular flux is low-SNR and the
    cumulative curve of growth changes by less than frac_growth_threshold.
    """
    if search_max_arcsec <= step_arcsec:
        raise ValueError("Auto max-SMA search radius is too small")

    search_max_pix = search_max_arcsec / pixscale_arcsec
    crop_y, crop_x = crop_slices_around(x0, y0, search_max_pix, data.shape)
    sub_data = data[crop_y, crop_x]
    sub_valid = valid[crop_y, crop_x]
    sub_invvar = invvar[crop_y, crop_x] if invvar is not None else None

    yy, xx = np.indices(sub_data.shape, dtype=float)
    x_full = xx + crop_x.start
    y_full = yy + crop_y.start
    er_pix = elliptical_radius_pix(x_full, y_full, x0, y0, theta, q)

    # Fallback sky RMS for the auto decision if no invvar exists.
    sky_sigma = None
    if sub_invvar is None:
        outer = (
            (er_pix >= 0.75 * search_max_pix)
            & (er_pix <= search_max_pix)
            & sub_valid
            & np.isfinite(sub_data)
        )
        if np.count_nonzero(outer) >= 20:
            _, _, sky_sigma = robust_sky_sigma(sub_data[outer])

    radii = np.arange(step_arcsec, search_max_arcsec + 0.5 * step_arcsec, step_arcsec)
    prev_ap = np.zeros(sub_data.shape, dtype=bool)
    cumulative_flux = 0.0
    quiet_count = 0
    last_good_radius = float(radii[-1])

    for radius_arcsec in radii:
        radius_pix = radius_arcsec / pixscale_arcsec
        ap = (er_pix <= radius_pix) & sub_valid
        ann = ap & ~prev_ap
        n_ann = int(np.count_nonzero(ann))
        if n_ann <= 0:
            prev_ap = ap
            continue

        ann_flux = float(np.nansum(sub_data[ann]))
        cumulative_before = cumulative_flux
        cumulative_flux += ann_flux

        ann_err = aperture_flux_error(
            ann,
            invvar=sub_invvar,
            sky_sigma_nmgy_pix=sky_sigma,
            noise_corr_factor=noise_corr_factor,
        )
        ann_snr = (
            abs(ann_flux) / ann_err if np.isfinite(ann_err) and ann_err > 0 else np.inf
        )

        if np.isfinite(cumulative_before) and abs(cumulative_before) > 0:
            frac_growth = abs(ann_flux) / abs(cumulative_before)
        else:
            frac_growth = np.inf

        if (
            radius_arcsec >= min_sma_arcsec
            and ann_snr < snr_threshold
            and frac_growth < frac_growth_threshold
        ):
            quiet_count += 1
        else:
            quiet_count = 0
            if radius_arcsec >= min_sma_arcsec:
                last_good_radius = float(radius_arcsec)

        if quiet_count >= consecutive:
            return max(
                float(min_sma_arcsec), float(radius_arcsec - consecutive * step_arcsec)
            )

        prev_ap = ap

    return last_good_radius


def interpolate_half_light_radius(
    r: np.ndarray, cum_flux: np.ndarray, total_flux: float
) -> float:
    r = np.asarray(r, dtype=float)
    f = np.asarray(cum_flux, dtype=float)
    good = np.isfinite(r) & np.isfinite(f)
    r = r[good]
    f = f[good]
    if r.size < 2 or not np.isfinite(total_flux) or total_flux <= 0:
        return np.nan
    fmono = np.maximum.accumulate(f)
    half = 0.5 * total_flux
    if half < fmono[0] or half > fmono[-1]:
        return np.nan
    return float(np.interp(half, fmono, r))


def interpolate_radius_at_flux(
    r: np.ndarray,
    cum_flux: np.ndarray,
    target_flux: float,
) -> float:
    """
    Return the semi-major-axis radius where the cumulative curve of growth
    reaches target_flux.

    Uses a monotonicized version of the curve of growth to reduce sensitivity
    to small sky-noise wiggles.
    """
    r = np.asarray(r, dtype=float)
    f = np.asarray(cum_flux, dtype=float)

    good = np.isfinite(r) & np.isfinite(f)
    r = r[good]
    f = f[good]

    if r.size < 2 or not np.isfinite(target_flux) or target_flux <= 0:
        return np.nan

    fmono = np.maximum.accumulate(f)

    if target_flux < fmono[0] or target_flux > fmono[-1]:
        return np.nan

    return float(np.interp(target_flux, fmono, r))


def estimate_bulge_radius_from_disk_excess(
    sma_arcsec: np.ndarray,
    ann_mu: np.ndarray,
    q: float,
    disk_fit_min_frac: float,
    excess_threshold_mag: float,
    min_fit_points: int = 8,
) -> dict[str, float | None]:
    r = np.asarray(sma_arcsec, dtype=float)
    mu = np.asarray(ann_mu, dtype=float)
    good = np.isfinite(r) & np.isfinite(mu)
    if np.count_nonzero(good) < min_fit_points + 3:
        return {
            "bulge_radius_sma_arcsec": None,
            "bulge_radius_circularized_arcsec": None,
            "disk_mu0": None,
            "disk_slope": None,
        }

    rmax = np.nanmax(r[good])
    fit_min = disk_fit_min_frac * rmax
    fit = good & (r >= fit_min)
    if np.count_nonzero(fit) < min_fit_points:
        return {
            "bulge_radius_sma_arcsec": None,
            "bulge_radius_circularized_arcsec": None,
            "disk_mu0": None,
            "disk_slope": None,
        }

    rr = r[fit]
    mm = mu[fit]
    p = np.polyfit(rr, mm, 1)
    model = np.polyval(p, rr)
    resid = mm - model
    _, med, std = robust_sky_sigma(resid)
    keep = np.isfinite(resid)
    if np.isfinite(std) and std > 0:
        keep &= np.abs(resid - med) < 2.5 * std
    if np.count_nonzero(keep) >= min_fit_points:
        p = np.polyfit(rr[keep], mm[keep], 1)

    slope, mu0 = float(p[0]), float(p[1])
    if slope <= 0:
        return {
            "bulge_radius_sma_arcsec": None,
            "bulge_radius_circularized_arcsec": None,
            "disk_mu0": mu0,
            "disk_slope": slope,
        }

    disk_mu = slope * r + mu0
    excess = disk_mu - mu
    inner = good & (r < fit_min) & np.isfinite(excess)
    candidates = r[inner & (excess >= excess_threshold_mag)]
    if candidates.size == 0:
        br = None
        br_circ = None
    else:
        br = float(np.nanmax(candidates))
        br_circ = float(br * np.sqrt(q))

    return {
        "bulge_radius_sma_arcsec": br,
        "bulge_radius_circularized_arcsec": br_circ,
        "disk_mu0": mu0,
        "disk_slope": slope,
    }


def kpc_per_arcsec_from_z(z: Optional[float]) -> Optional[float]:
    if z is None:
        return None
    try:
        from astropy.cosmology import Planck18 as cosmo
    except Exception:
        return None
    return float(cosmo.kpc_proper_per_arcmin(float(z)).to_value(u.kpc / u.arcsec))


def finite_or_none(x: Any) -> Optional[float]:
    try:
        xf = float(x)
        return xf if np.isfinite(xf) else None
    except Exception:
        return None


def json_safe_value(x: Any) -> Any:
    """
    Convert common numpy/path/function values into JSON-safe values.

    This is used for reproducibility metadata so the geometry JSON records the
    actual command-line inputs and target dictionary without failing on objects
    such as pathlib.Path, numpy scalars, NaN/Inf, or argparse callback functions.
    """
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        xf = float(x)
        return xf if np.isfinite(xf) else None
    if isinstance(x, float):
        return x if np.isfinite(x) else None
    if isinstance(x, (str, int, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): json_safe_value(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe_value(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_safe_value(x.tolist())
    return str(x)


def collect_input_parameters(
    args: argparse.Namespace,
    target: dict[str, Any],
    data_shape: tuple[int, int],
    hdu_used: str,
    redshift_used: Optional[float],
    kpc_per_arcsec: Optional[float],
) -> dict[str, Any]:
    """
    Build a reproducibility record of the input parameters used for this run.

    This goes into the geometry JSON and is also serialized into the commented
    header of the CSV.
    """
    cli_args = {k: v for k, v in vars(args).items() if k != "func"}

    return json_safe_value(
        {
            "script": (
                Path(sys.argv[0]).name if sys.argv else "legacy_ellipse_profile.py"
            ),
            "command_line": sys.argv,
            "run_utc": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "input_files": {
                "image_fits": args.fits,
                "target_dict": args.target_dict,
                "mask_fits": args.mask_fits,
                "invvar_fits": args.invvar_fits,
                "tractor_fits": args.tractor_fits,
            },
            "output_files": {
                "output_csv": args.output_csv,
                "geometry_out": args.geometry_out
                or str(Path(args.output_csv).with_suffix(".geometry.json")),
                "plot_profile": args.plot_profile,
                "diagnostic_png": args.diagnostic_png,
            },
            "cli_arguments": cli_args,
            "target_dict": target,
            "selected_image_hdu": hdu_used,
            "image_shape_yx": list(data_shape),
            "redshift_used": finite_or_none(redshift_used),
            "kpc_per_arcsec": finite_or_none(kpc_per_arcsec),
        }
    )


def estimate_systematic_uncertainties(
    args: argparse.Namespace,
    data: np.ndarray,
    valid: np.ndarray,
    x0: float,
    y0: float,
    theta: float,
    q: float,
    radius_pix: float,
    pixel_scale: float,
    aperture_flux: float,
    formal_flux_error: float,
    light_fraction: Optional[float],
    transmission: Optional[float],
    extinction_applied: bool,
) -> dict[str, Any]:
    """Finite-difference systematic budget for the measured aperture and total.

    All nuisance scales are 1-sigma assumptions supplied by the caller. Sky is
    a coherent per-pixel offset, and geometry variants use the same image/mask.
    """
    scales = {
        "extinction_mag": args.sys_extinction_mag,
        "sky_nmgy_pix": args.sys_sky_nmgy_pix,
        "center_arcsec": args.sys_center_arcsec,
        "q": args.sys_q,
        "pa_deg": args.sys_pa_deg,
        "aperture_sma_arcsec": args.sys_aperture_sma_arcsec,
        "sersic_fraction": args.sys_sersic_fraction,
    }
    for label, value in scales.items():
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError(f"--sys-{label.replace('_', '-')} must be finite and nonnegative")
    if args.sys_q is not None and args.sys_q > 0 and not (0 < q - args.sys_q and q + args.sys_q <= 1):
        raise ValueError("--sys-q must keep both perturbed axis ratios in (0, 1]")
    if args.sys_sersic_fraction is not None and args.sys_sersic_fraction > 0:
        if light_fraction is None or not (0 < light_fraction - args.sys_sersic_fraction < light_fraction + args.sys_sersic_fraction < 1):
            raise ValueError("--sys-sersic-fraction requires an uncapped Sérsic-fraction aperture and fractions in (0, 1)")
    if args.sys_extinction_mag is not None and not (extinction_applied or transmission is not None):
        raise ValueError("--sys-extinction-mag requires corrected input or an extinction correction")

    yy, xx = np.indices(data.shape, dtype=float)
    nominal_mask = (elliptical_radius_pix(xx, yy, x0, y0, theta, q) <= radius_pix) & valid
    n_pix = int(np.count_nonzero(nominal_mask))
    variants: dict[str, list[float]] = {}
    if args.sys_sky_nmgy_pix is not None:
        delta = n_pix * args.sys_sky_nmgy_pix
        variants["sky"] = [aperture_flux - delta, aperture_flux + delta]

    def aperture_at(cx: float, cy: float, angle: float, ratio: float, radius: float) -> float:
        mask = (elliptical_radius_pix(xx, yy, cx, cy, angle, ratio) <= radius) & valid
        return float(np.sum(data[mask]))

    if args.sys_center_arcsec is not None:
        d = args.sys_center_arcsec / pixel_scale
        variants["center"] = [aperture_at(x0 + dx, y0 + dy, theta, q, radius_pix)
                              for dx, dy in ((d, 0), (-d, 0), (0, d), (0, -d))]
    if args.sys_q is not None:
        d = args.sys_q
        variants["axis_ratio"] = [aperture_at(x0, y0, theta, q + sign*d, radius_pix) for sign in (-1, 1)]
    if args.sys_pa_deg is not None:
        d = math.radians(args.sys_pa_deg)
        variants["position_angle"] = [aperture_at(x0, y0, theta + sign*d, q, radius_pix) for sign in (-1, 1)]
    if args.sys_aperture_sma_arcsec is not None:
        d = args.sys_aperture_sma_arcsec / pixel_scale
        if radius_pix - d <= 0:
            raise ValueError("--sys-aperture-sma-arcsec must be smaller than the maximum aperture radius")
        variants["aperture_radius"] = [aperture_at(x0, y0, theta, q, radius_pix + sign*d) for sign in (-1, 1)]

    factor = 1.0 if extinction_applied else (1.0 / transmission if transmission else 1.0)

    def budget(fraction: float, include_model: bool) -> dict[str, Any]:
        nominal = aperture_flux * factor / fraction
        components: dict[str, dict[str, Optional[float]]] = {}
        for name, values in variants.items():
            flux_changes = [(v - aperture_flux) * factor / fraction for v in values]
            components[name] = {
                "flux_nmgy": max(abs(v) for v in flux_changes),
                "mag": max((abs(mag_from_nmgy(nominal + v) - mag_from_nmgy(nominal)) for v in flux_changes), default=np.nan),
            }
        if args.sys_extinction_mag is not None:
            d = args.sys_extinction_mag
            flux_changes = [nominal * (10 ** (sign*0.4*d) - 1) for sign in (-1, 1)]
            components["galactic_extinction"] = {"flux_nmgy": max(abs(v) for v in flux_changes), "mag": d}
        if include_model and args.sys_sersic_fraction is not None:
            d = args.sys_sersic_fraction
            values = [aperture_flux * factor / (fraction + sign*d) for sign in (-1, 1)]
            components["sersic_extrapolation"] = {
                "flux_nmgy": max(abs(v - nominal) for v in values),
                "mag": max(abs(mag_from_nmgy(v) - mag_from_nmgy(nominal)) for v in values),
            }
        valid_components = [c for c in components.values() if np.isfinite(c["flux_nmgy"]) and np.isfinite(c["mag"])]
        systematic_flux = math.sqrt(sum(c["flux_nmgy"]**2 for c in valid_components)) if valid_components else None
        systematic_mag = math.sqrt(sum(c["mag"]**2 for c in valid_components)) if valid_components else None
        formal_flux = formal_flux_error * factor / fraction
        formal_mag = mag_err_from_flux_err(nominal, formal_flux)
        return {
            "flux_nmgy": finite_or_none(nominal),
            "formal_flux_err_nmgy": finite_or_none(formal_flux),
            "formal_mag_err": finite_or_none(formal_mag),
            "components": json_safe_value(components),
            "systematic_flux_err_nmgy": finite_or_none(systematic_flux),
            "systematic_mag_err": finite_or_none(systematic_mag),
            "combined_flux_err_nmgy": finite_or_none(math.hypot(formal_flux, systematic_flux)) if systematic_flux is not None and np.isfinite(formal_flux) else None,
            "combined_mag_err": finite_or_none(math.hypot(formal_mag, systematic_mag)) if systematic_mag is not None and np.isfinite(formal_mag) else None,
        }

    return {
        "assumed_1sigma": scales,
        "flux_convention": "extinction_corrected" if extinction_applied or transmission else "observed",
        "method": "one-at-a-time symmetric perturbations; largest absolute shift per component; independent components combined in quadrature",
        "caveat": "Conditional on supplied nuisance scales and fixed masking. Shared g/r extinction and sky errors may be correlated; no Reff or bulge-radius uncertainty included.",
        "aperture": budget(1.0, False),
        "sersic_total": budget(light_fraction, True) if light_fraction is not None else None,
    }


def measure_profile(args: argparse.Namespace) -> None:
    target = read_target_dict(args.target_dict)
    if "ra" not in target or "dec" not in target:
        raise ValueError(
            "Target dictionary must contain 'ra' and 'dec' in decimal degrees"
        )

    name = str(target.get("name", Path(args.output_csv).stem))
    ra = float(target["ra"])
    dec = float(target["dec"])
    coord = SkyCoord(ra=ra * u.deg, dec=dec * u.deg)

    with fits.open(args.fits) as hdul:
        data, header, hdu_used = get_hdu_data_header(hdul, args.hdu)
        wcs = make_wcs(header, hdul[0].header)

    if data.ndim != 2:
        raise ValueError(f"Selected HDU has {data.ndim} dimensions; expected 2-D image")

    pixel_area_arcsec2 = pixel_area_arcsec2_from_wcs(wcs, args.pixscale_arcsec)
    pixscale_arcsec = math.sqrt(pixel_area_arcsec2)

    x0_initial, y0_initial = wcs.world_to_pixel(coord)
    x0_initial = float(x0_initial)
    y0_initial = float(y0_initial)
    ny, nx = data.shape
    if not (0 <= x0_initial < nx and 0 <= y0_initial < ny):
        raise ValueError(
            f"Target pixel ({x0_initial:.2f}, {y0_initial:.2f}) is outside image shape {data.shape}"
        )

    external_mask = load_mask(args.mask_fits, args.mask_hdu, data.shape)
    invvar = load_invvar(args.invvar_fits, args.invvar_hdu, data.shape)

    valid = np.isfinite(data)
    if external_mask is not None:
        valid &= ~external_mask

    pa_override = args.pa_deg
    q_override = args.q
    tractor_geometry_description = None
    tractor_geometry_sep_arcsec = None
    tractor_geometry_source_id = None

    if args.geometry_from_tractor:
        if args.tractor_fits is None:
            raise ValueError("--tractor-fits is required with --geometry-from-tractor")
        (
            tractor_q,
            tractor_pa_deg,
            tractor_geometry_description,
            tractor_geometry_sep_arcsec,
            tractor_geometry_source_id,
        ) = estimate_geometry_from_tractor(
            tractor_fits=args.tractor_fits,
            ra_deg=ra,
            dec_deg=dec,
            hdu=args.tractor_hdu,
            max_sep_arcsec=args.tractor_max_sep_arcsec,
        )
        if pa_override is None:
            pa_override = tractor_pa_deg
        if q_override is None:
            q_override = tractor_q

    if pa_override is None or q_override is None or not args.no_recenter:
        xc, yc, theta_auto, q_auto = estimate_moment_geometry(
            data=data,
            valid=valid,
            x0=x0_initial,
            y0=y0_initial,
            pixscale_arcsec=pixscale_arcsec,
            moment_radius_arcsec=args.moment_radius_arcsec,
            threshold_sigma=args.moment_threshold_sigma,
            recenter=not args.no_recenter,
        )
    else:
        xc, yc = x0_initial, y0_initial
        theta_auto, q_auto = 0.0, 1.0

    q = float(q_override) if q_override is not None else q_auto
    q = float(np.clip(q, 0.05, 1.0))
    q_source = (
        "user"
        if args.q is not None
        else (
            "tractor"
            if args.geometry_from_tractor and q_override is not None
            else "moments"
        )
    )

    if pa_override is not None:
        theta = pixel_theta_from_pa(wcs, xc, yc, float(pa_override))
        pa_major = float(pa_override % 180.0)
        pa_source = "user" if args.pa_deg is not None else "tractor"
    else:
        theta = theta_auto
        pa_major = pa_from_pixel_theta(wcs, xc, yc, theta)
        pa_source = "moments"

    pa_minor = float((pa_major + 90.0) % 180.0)

    step_pix = args.step_arcsec / pixscale_arcsec
    if step_pix <= 0:
        raise ValueError("--step-arcsec must be positive")

    tractor_max_sma_description = None
    tractor_max_sma_sep_arcsec = None
    tractor_max_sma_source_id = None
    tractor_max_sma_uncapped = None
    tractor_max_sma_was_capped = False

    if args.max_sma_from_tractor:
        if args.tractor_fits is None:
            raise ValueError("--tractor-fits is required with --max-sma-from-tractor")
        (
            max_sma_arcsec_used,
            tractor_max_sma_description,
            tractor_max_sma_sep_arcsec,
            tractor_max_sma_source_id,
        ) = estimate_max_sma_from_tractor(
            tractor_fits=args.tractor_fits,
            ra_deg=ra,
            dec_deg=dec,
            hdu=args.tractor_hdu,
            max_sep_arcsec=args.tractor_max_sep_arcsec,
            method=args.tractor_max_sma_method,
            factor=args.tractor_max_sma_factor,
            light_frac=args.tractor_max_sma_light_frac,
        )

        tractor_max_sma_uncapped = float(max_sma_arcsec_used)
        tractor_max_sma_was_capped = False

        if args.max_sma_arcsec is not None and max_sma_arcsec_used > float(
            args.max_sma_arcsec
        ):
            max_sma_arcsec_used = float(args.max_sma_arcsec)
            tractor_max_sma_was_capped = True
            tractor_max_sma_description += (
                f"; capped_by_user_max_sma={float(args.max_sma_arcsec):.6g} arcsec"
            )

        max_sma_source = "tractor"
    elif args.auto_max_sma:
        if args.auto_max_search_arcsec is not None:
            search_max_arcsec = float(args.auto_max_search_arcsec)
        else:
            # Conservative search cap that keeps apertures away from the image edge.
            min_edge_pix = min(xc, yc, data.shape[1] - 1 - xc, data.shape[0] - 1 - yc)
            search_max_arcsec = 0.9 * min_edge_pix * pixscale_arcsec
        if args.max_sma_arcsec is not None:
            search_max_arcsec = min(search_max_arcsec, float(args.max_sma_arcsec))
        max_sma_arcsec_used = estimate_auto_max_sma_arcsec(
            data=data,
            valid=valid,
            invvar=invvar,
            x0=xc,
            y0=yc,
            theta=theta,
            q=q,
            pixscale_arcsec=pixscale_arcsec,
            step_arcsec=args.step_arcsec,
            search_max_arcsec=search_max_arcsec,
            min_sma_arcsec=args.auto_max_min_arcsec,
            snr_threshold=args.auto_max_snr_threshold,
            frac_growth_threshold=args.auto_max_frac_growth_threshold,
            consecutive=args.auto_max_consecutive,
            noise_corr_factor=args.noise_corr_factor,
        )
        max_sma_source = "auto"
    else:
        if args.max_sma_arcsec is None:
            raise ValueError(
                "--max-sma-arcsec is required unless --auto-max-sma is used"
            )
        max_sma_arcsec_used = float(args.max_sma_arcsec)
        max_sma_source = "user"

    max_sma_pix = max_sma_arcsec_used / pixscale_arcsec

    systematic_padding_pix = (
        ((args.sys_center_arcsec or 0.0) + (args.sys_aperture_sma_arcsec or 0.0))
        / pixscale_arcsec + step_pix + 2.0
    )
    crop_y, crop_x = crop_slices_around(xc, yc, max_sma_pix + systematic_padding_pix, data.shape)
    sub_data = data[crop_y, crop_x]
    sub_valid = valid[crop_y, crop_x]
    sub_invvar = invvar[crop_y, crop_x] if invvar is not None else None

    y_indices, x_indices = np.indices(sub_data.shape, dtype=float)
    x_full = x_indices + crop_x.start
    y_full = y_indices + crop_y.start
    er_pix = elliptical_radius_pix(x_full, y_full, xc, yc, theta, q)

    # Galactic-extinction handling.
    #
    # The generic internal variables always contain values measured directly
    # from the selected input image. Output naming indicates whether those
    # measurements are extinction corrected.
    input_extinction_corrected = bool(args.input_extinction_corrected)

    band = args.band.lower() if args.band else None
    mw_transmission_band = None
    extinction_mag_band = None
    extinction_source = None
    tractor_extinction_column = None
    tractor_extinction_sep_arcsec = None
    tractor_extinction_source_id = None

    if input_extinction_corrected:
        if args.extinction_mag is not None or args.mw_transmission is not None:
            raise ValueError(
                "--input-extinction-corrected cannot be combined with "
                "--extinction-mag or --mw-transmission"
            )

        # Tractor may still be used for geometry and max-SMA selection, but its
        # transmission must not be applied again.
        extinction_source = "input_already_corrected"

    elif args.extinction_mag is not None:
        extinction_mag_band = float(args.extinction_mag)
        mw_transmission_band = 10.0 ** (-0.4 * extinction_mag_band)
        extinction_source = "user_extinction_mag"

    elif args.mw_transmission is not None:
        mw_transmission_band = float(args.mw_transmission)

        if not np.isfinite(mw_transmission_band) or mw_transmission_band <= 0:
            raise ValueError("--mw-transmission must be positive and finite")

        extinction_mag_band = float(-2.5 * np.log10(mw_transmission_band))
        extinction_source = "user_mw_transmission"

    elif args.tractor_fits is not None:
        if band is None:
            raise ValueError(
                "--band is required when using --tractor-fits for "
                "extinction correction"
            )

        (
            mw_transmission_band,
            tractor_extinction_sep_arcsec,
            tractor_extinction_column,
            tractor_extinction_source_id,
        ) = read_mw_transmission_from_tractor(
            tractor_fits=args.tractor_fits,
            ra_deg=ra,
            dec_deg=dec,
            band=band,
            hdu=args.tractor_hdu,
            max_sep_arcsec=args.tractor_max_sep_arcsec,
        )

        extinction_mag_band = float(-2.5 * np.log10(mw_transmission_band))
        extinction_source = "tractor_mw_transmission"

    if args.sky_sigma_nmgy_pix is not None:
        sky_sigma_nmgy_pix = float(args.sky_sigma_nmgy_pix)
        error_method = "sky_sigma_user"
    elif sub_invvar is None:
        outer = (
            (er_pix >= args.error_sky_inner_frac * max_sma_pix)
            & (er_pix <= args.error_sky_outer_frac * max_sma_pix)
            & sub_valid
            & np.isfinite(sub_data)
        )
        if np.count_nonzero(outer) >= 20:
            _, _, sky_sigma_nmgy_pix = robust_sky_sigma(sub_data[outer])
            error_method = "sky_sigma_outer_annulus"
        else:
            sky_sigma_nmgy_pix = np.nan
            error_method = "none"
    else:
        sky_sigma_nmgy_pix = None
        error_method = "invvar"

    sma_arcsec = np.arange(
        args.step_arcsec, max_sma_arcsec_used + 0.5 * args.step_arcsec, args.step_arcsec
    )
    sma_pix = sma_arcsec / pixscale_arcsec

    rows: list[dict[str, Any]] = []
    previous_mask = np.zeros(sub_data.shape, dtype=bool)
    cumulative_flux_values: list[float] = []
    cumulative_flux_error_values: list[float] = []
    annulus_mu_values: list[float] = []
    for a_arcsec, a_pix in zip(sma_arcsec, sma_pix):
        aperture = (er_pix <= a_pix) & sub_valid
        annulus = aperture & ~previous_mask

        n_pix = int(np.count_nonzero(aperture))
        ann_n_pix = int(np.count_nonzero(annulus))
        area_arcsec2 = n_pix * pixel_area_arcsec2
        ann_area_arcsec2 = ann_n_pix * pixel_area_arcsec2

        flux = float(np.nansum(sub_data[aperture])) if n_pix > 0 else np.nan
        ann_flux = float(np.nansum(sub_data[annulus])) if ann_n_pix > 0 else np.nan

        mean_sb = flux / area_arcsec2 if area_arcsec2 > 0 else np.nan
        ann_sb = ann_flux / ann_area_arcsec2 if ann_area_arcsec2 > 0 else np.nan

        flux_err = aperture_flux_error(
            aperture,
            invvar=sub_invvar,
            sky_sigma_nmgy_pix=sky_sigma_nmgy_pix,
            noise_corr_factor=args.noise_corr_factor,
        )
        ann_flux_err = aperture_flux_error(
            annulus,
            invvar=sub_invvar,
            sky_sigma_nmgy_pix=sky_sigma_nmgy_pix,
            noise_corr_factor=args.noise_corr_factor,
        )

        mean_sb_err = (
            flux_err / area_arcsec2
            if area_arcsec2 > 0 and np.isfinite(flux_err)
            else np.nan
        )
        ann_sb_err = (
            ann_flux_err / ann_area_arcsec2
            if ann_area_arcsec2 > 0 and np.isfinite(ann_flux_err)
            else np.nan
        )

        integrated_mag = mag_from_nmgy(flux)
        integrated_mag_err = mag_err_from_flux_err(flux, flux_err)
        mean_mu = mu_from_nmgy_per_arcsec2(mean_sb)
        annulus_mu = mu_from_nmgy_per_arcsec2(ann_sb)

        cumulative_flux_values.append(flux)
        cumulative_flux_error_values.append(flux_err)
        annulus_mu_values.append(annulus_mu)

        row = {
            "sma_pix": float(a_pix),
            "sma_arcsec": float(a_arcsec),
            "sma_kpc": np.nan,
            "semi_minor_arcsec": float(a_arcsec * q),
            "circularized_radius_arcsec": float(a_arcsec * np.sqrt(q)),
            "area_arcsec2": float(area_arcsec2),
            "n_pix": n_pix,
            "annulus_area_arcsec2": float(ann_area_arcsec2),
            "annulus_n_pix": ann_n_pix,
        }

        if input_extinction_corrected:
            # These quantities were measured directly from an input image that was
            # already extinction corrected. Do not rescale them.
            row.update(
                {
                    "flux_extcorr_nmgy": flux,
                    "flux_extcorr_err_nmgy": flux_err,
                    "integrated_mag_extcorr_ab": integrated_mag,
                    "integrated_mag_extcorr_err": integrated_mag_err,
                    "mean_sb_extcorr_nmgy_arcsec2": mean_sb,
                    "mean_sb_extcorr_err_nmgy_arcsec2": mean_sb_err,
                    "mean_mu_extcorr_ab_arcsec2": mean_mu,
                    "annulus_flux_extcorr_nmgy": ann_flux,
                    "annulus_flux_extcorr_err_nmgy": ann_flux_err,
                    "annulus_sb_extcorr_nmgy_arcsec2": ann_sb,
                    "annulus_sb_extcorr_err_nmgy_arcsec2": ann_sb_err,
                    "annulus_mu_extcorr_ab_arcsec2": annulus_mu,
                }
            )

        else:
            # Direct measurements from an image that has not been declared
            # extinction corrected.
            row.update(
                {
                    "flux_nmgy": flux,
                    "flux_err_nmgy": flux_err,
                    "integrated_mag_ab": integrated_mag,
                    "integrated_mag_err": integrated_mag_err,
                    "mean_sb_nmgy_arcsec2": mean_sb,
                    "mean_sb_err_nmgy_arcsec2": mean_sb_err,
                    "mean_mu_ab_arcsec2": mean_mu,
                    "annulus_flux_nmgy": ann_flux,
                    "annulus_flux_err_nmgy": ann_flux_err,
                    "annulus_sb_nmgy_arcsec2": ann_sb,
                    "annulus_sb_err_nmgy_arcsec2": ann_sb_err,
                    "annulus_mu_ab_arcsec2": annulus_mu,
                }
            )

            if mw_transmission_band is not None:
                row.update(
                    {
                        "flux_extcorr_nmgy": flux_extinction_corrected(
                            flux,
                            mw_transmission_band,
                        ),
                        "flux_extcorr_err_nmgy": flux_extinction_corrected(
                            flux_err,
                            mw_transmission_band,
                        ),
                        "integrated_mag_extcorr_ab": mag_extinction_corrected(
                            integrated_mag,
                            extinction_mag_band,
                        ),
                        "integrated_mag_extcorr_err": integrated_mag_err,
                        "mean_sb_extcorr_nmgy_arcsec2": (
                            flux_extinction_corrected(
                                mean_sb,
                                mw_transmission_band,
                            )
                        ),
                        "mean_sb_extcorr_err_nmgy_arcsec2": (
                            flux_extinction_corrected(
                                mean_sb_err,
                                mw_transmission_band,
                            )
                        ),
                        "mean_mu_extcorr_ab_arcsec2": (
                            mag_extinction_corrected(
                                mean_mu,
                                extinction_mag_band,
                            )
                        ),
                        "annulus_flux_extcorr_nmgy": (
                            flux_extinction_corrected(
                                ann_flux,
                                mw_transmission_band,
                            )
                        ),
                        "annulus_flux_extcorr_err_nmgy": (
                            flux_extinction_corrected(
                                ann_flux_err,
                                mw_transmission_band,
                            )
                        ),
                        "annulus_sb_extcorr_nmgy_arcsec2": (
                            flux_extinction_corrected(
                                ann_sb,
                                mw_transmission_band,
                            )
                        ),
                        "annulus_sb_extcorr_err_nmgy_arcsec2": (
                            flux_extinction_corrected(
                                ann_sb_err,
                                mw_transmission_band,
                            )
                        ),
                        "annulus_mu_extcorr_ab_arcsec2": (
                            mag_extinction_corrected(
                                annulus_mu,
                                extinction_mag_band,
                            )
                        ),
                    }
                )

        rows.append(row)
        previous_mask = aperture

    z = args.redshift if args.redshift is not None else target.get("z", None)
    kpc_per_arcsec = kpc_per_arcsec_from_z(z)
    if kpc_per_arcsec is not None:
        for row in rows:
            row["sma_kpc"] = row["sma_arcsec"] * kpc_per_arcsec

    r = np.array([row["sma_arcsec"] for row in rows], dtype=float)
    f = np.asarray(cumulative_flux_values, dtype=float)
    ferr = np.asarray(cumulative_flux_error_values, dtype=float)

    total_flux = float(f[-1]) if f.size else np.nan
    total_flux_err = float(ferr[-1]) if ferr.size else np.nan
    total_flux_extcorr = flux_extinction_corrected(total_flux, mw_transmission_band)
    total_flux_extcorr_err = flux_extinction_corrected(
        total_flux_err, mw_transmission_band
    )
    total_mag = mag_from_nmgy(total_flux)
    total_mag_err = mag_err_from_flux_err(total_flux, total_flux_err)
    total_mag_extcorr = mag_extinction_corrected(total_mag, extinction_mag_band)
    # Optional model-dependent total correction.
    # This is valid only when the measured outer aperture was explicitly defined
    # as a Sérsic model light-fraction radius and was not reduced by a user cap.
    if (
        args.max_sma_from_tractor
        and args.tractor_max_sma_method == "sersic-fraction"
        and not tractor_max_sma_was_capped
    ):
        sersic_frac = float(args.tractor_max_sma_light_frac)
        if not (0.0 < sersic_frac < 1.0):
            raise ValueError("--tractor-max-sma-light-frac must be between 0 and 1")

        total_flux_sersiccorr = total_flux / sersic_frac
        total_flux_sersiccorr_err = (
            total_flux_err / sersic_frac if np.isfinite(total_flux_err) else np.nan
        )
        total_mag_sersiccorr = mag_from_nmgy(total_flux_sersiccorr)
        total_mag_sersiccorr_err = mag_err_from_flux_err(
            total_flux_sersiccorr,
            total_flux_sersiccorr_err,
        )

        total_flux_sersiccorr_extcorr = flux_extinction_corrected(
            total_flux_sersiccorr,
            mw_transmission_band,
        )
        total_flux_sersiccorr_extcorr_err = flux_extinction_corrected(
            total_flux_sersiccorr_err,
            mw_transmission_band,
        )
        total_mag_sersiccorr_extcorr = mag_extinction_corrected(
            total_mag_sersiccorr,
            extinction_mag_band,
        )

        sersic_total_correction_source = "tractor_sersic_fraction"
        sersic_total_correction_light_frac = sersic_frac
    else:
        total_flux_sersiccorr = np.nan
        total_flux_sersiccorr_err = np.nan
        total_mag_sersiccorr = np.nan
        total_mag_sersiccorr_err = np.nan
        total_flux_sersiccorr_extcorr = np.nan
        total_flux_sersiccorr_extcorr_err = np.nan
        total_mag_sersiccorr_extcorr = np.nan
        sersic_total_correction_source = None
        sersic_total_correction_light_frac = None

    re_sma = interpolate_half_light_radius(r, f, total_flux)
    re_circ = re_sma * math.sqrt(q) if np.isfinite(re_sma) else np.nan
    flux_within_reff = (
        0.5 * total_flux if np.isfinite(re_sma) and total_flux > 0 else np.nan
    )
    flux_err_within_reff = (
        0.5 * total_flux_err if np.isfinite(total_flux_err) else np.nan
    )
    flux_extcorr_within_reff = flux_extinction_corrected(
        flux_within_reff, mw_transmission_band
    )
    flux_extcorr_err_within_reff = flux_extinction_corrected(
        flux_err_within_reff, mw_transmission_band
    )
    mag_within_reff = mag_from_nmgy(flux_within_reff)
    mag_err_within_reff = mag_err_from_flux_err(flux_within_reff, flux_err_within_reff)
    mag_extcorr_within_reff = mag_extinction_corrected(
        mag_within_reff, extinction_mag_band
    )

    # Sérsic-corrected Reff estimate.
    #
    # If the outer aperture was defined as enclosing fraction f of the Tractor
    # Sérsic model light, then the corrected total flux estimate is F_ap / f.
    # The corrected half-light radius is therefore where the measured curve of
    # growth reaches:
    #
    #     0.5 * F_total,corr = F_ap * (0.5 / f)
    #
    # This is not a radius scaling. It is recomputed from the measured curve
    # of growth.
    if (
        args.max_sma_from_tractor
        and args.tractor_max_sma_method == "sersic-fraction"
        and not tractor_max_sma_was_capped
    ):
        sersic_reff_frac = float(args.tractor_max_sma_light_frac)

        if not (0.0 < sersic_reff_frac < 1.0):
            raise ValueError("--tractor-max-sma-light-frac must be between 0 and 1")

        if sersic_reff_frac <= 0.5:
            # The corrected half-light radius would lie at or beyond the
            # measured aperture, so it cannot be estimated robustly from this
            # curve of growth.
            re_sersiccorr_sma = np.nan
            re_sersiccorr_circ = np.nan
            flux_within_reff_sersiccorr = np.nan
            mag_within_reff_sersiccorr = np.nan
            mag_extcorr_within_reff_sersiccorr = np.nan
            reff_sersiccorr_source = "not_available_fraction_le_0.5"
            reff_sersiccorr_target_fraction = np.nan
        else:
            reff_sersiccorr_target_fraction = 0.5 / sersic_reff_frac
            flux_within_reff_sersiccorr = total_flux * reff_sersiccorr_target_fraction

            re_sersiccorr_sma = interpolate_radius_at_flux(
                r,
                f,
                flux_within_reff_sersiccorr,
            )
            re_sersiccorr_circ = (
                re_sersiccorr_sma * math.sqrt(q)
                if np.isfinite(re_sersiccorr_sma)
                else np.nan
            )

            mag_within_reff_sersiccorr = mag_from_nmgy(flux_within_reff_sersiccorr)
            mag_extcorr_within_reff_sersiccorr = mag_extinction_corrected(
                mag_within_reff_sersiccorr,
                extinction_mag_band,
            )
            reff_sersiccorr_source = "tractor_sersic_fraction_curve_of_growth"
    else:
        sersic_reff_frac = np.nan
        reff_sersiccorr_target_fraction = np.nan
        re_sersiccorr_sma = np.nan
        re_sersiccorr_circ = np.nan
        flux_within_reff_sersiccorr = np.nan
        mag_within_reff_sersiccorr = np.nan
        mag_extcorr_within_reff_sersiccorr = np.nan
        reff_sersiccorr_source = None

    if input_extinction_corrected:
        # The directly measured input flux is already extinction corrected.
        flux_extcorr_within_reff_sersiccorr = flux_within_reff_sersiccorr
    else:
        flux_extcorr_within_reff_sersiccorr = flux_extinction_corrected(
            flux_within_reff_sersiccorr,
            mw_transmission_band,
        )
    for row, cumulative_flux in zip(rows, f):
        row["isophotal_fraction_of_total_flux"] = (
            cumulative_flux / total_flux if total_flux > 0 else np.nan
        )

    ann_mu = np.asarray(annulus_mu_values, dtype=float)
    bulge = estimate_bulge_radius_from_disk_excess(
        sma_arcsec=r,
        ann_mu=ann_mu,
        q=q,
        disk_fit_min_frac=args.disk_fit_min_frac,
        excess_threshold_mag=args.bulge_excess_threshold_mag,
        min_fit_points=args.disk_fit_min_points,
    )

    if args.bulge_radius_sma_arcsec is not None:
        br = float(args.bulge_radius_sma_arcsec)
        if not np.isfinite(br) or br <= 0:
            raise ValueError(
                "--bulge-radius-sma-arcsec must be a positive finite radius"
            )
        if br > max_sma_arcsec_used:
            raise ValueError(
                f"--bulge-radius-sma-arcsec={br} is larger than the measured max SMA={max_sma_arcsec_used}. "
                "Increase --max-sma-arcsec or --auto-max-search-arcsec so the requested aperture is measured."
            )
        br_circ = float(br * np.sqrt(q))
        bulge_radius_source = "user_override"
    else:
        br = bulge["bulge_radius_sma_arcsec"]
        br_circ = bulge["bulge_radius_circularized_arcsec"]
        bulge_radius_source = "outer_disk_excess" if br is not None else "not_found"

    if br is not None and np.isfinite(br):
        bulge_flux = float(np.interp(float(br), r, f))
        bulge_flux_err = float(np.interp(float(br), r, ferr))
        bulge_flux_extcorr = flux_extinction_corrected(bulge_flux, mw_transmission_band)
        bulge_flux_extcorr_err = flux_extinction_corrected(
            bulge_flux_err, mw_transmission_band
        )
        bulge_mag = mag_from_nmgy(bulge_flux)
        bulge_mag_err = mag_err_from_flux_err(bulge_flux, bulge_flux_err)
        bulge_mag_extcorr = mag_extinction_corrected(bulge_mag, extinction_mag_band)
    else:
        bulge_flux = np.nan
        bulge_flux_err = np.nan
        bulge_flux_extcorr = np.nan
        bulge_flux_extcorr_err = np.nan
        bulge_mag = np.nan
        bulge_mag_err = np.nan
        bulge_mag_extcorr = np.nan

    input_parameters = collect_input_parameters(
        args=args,
        target=target,
        data_shape=data.shape,
        hdu_used=hdu_used,
        redshift_used=z,
        kpc_per_arcsec=kpc_per_arcsec,
    )

    systematic_uncertainties = None
    if any(value is not None for value in (
        args.sys_extinction_mag, args.sys_sky_nmgy_pix, args.sys_center_arcsec,
        args.sys_q, args.sys_pa_deg, args.sys_aperture_sma_arcsec,
        args.sys_sersic_fraction,
    )):
        systematic_uncertainties = estimate_systematic_uncertainties(
            args=args, data=sub_data, valid=sub_valid,
            x0=xc - crop_x.start, y0=yc - crop_y.start, theta=theta, q=q,
            radius_pix=float(sma_pix[-1]), pixel_scale=pixscale_arcsec,
            aperture_flux=total_flux, formal_flux_error=total_flux_err,
            light_fraction=sersic_total_correction_light_frac,
            transmission=mw_transmission_band,
            extinction_applied=input_extinction_corrected,
        )

    if input_extinction_corrected:
        # The generic internally measured values are already extinction corrected.
        total_flux_extcorr = total_flux
        total_flux_extcorr_err = total_flux_err
        total_mag_extcorr = total_mag

        flux_extcorr_within_reff = flux_within_reff
        flux_extcorr_err_within_reff = flux_err_within_reff
        mag_extcorr_within_reff = mag_within_reff

        mag_extcorr_within_reff_sersiccorr = mag_within_reff_sersiccorr

        total_flux_sersiccorr_extcorr = total_flux_sersiccorr
        total_flux_sersiccorr_extcorr_err = total_flux_sersiccorr_err
        total_mag_sersiccorr_extcorr = total_mag_sersiccorr

        bulge_flux_extcorr = bulge_flux
        bulge_flux_extcorr_err = bulge_flux_err
        bulge_mag_extcorr = bulge_mag

    geom = Geometry(
        target_name=name,
        ra_deg=ra,
        dec_deg=dec,
        x0_pix_initial=x0_initial,
        y0_pix_initial=y0_initial,
        x0_pix_used=xc,
        y0_pix_used=yc,
        hdu_used=hdu_used,
        pixscale_arcsec=pixscale_arcsec,
        pixel_area_arcsec2=pixel_area_arcsec2,
        theta_image_deg=float(np.rad2deg(theta)),
        pa_major_deg_east_of_north=pa_major,
        pa_minor_deg_east_of_north=pa_minor,
        q_b_over_a=q,
        ellipticity=float(1.0 - q),
        moment_radius_arcsec=args.moment_radius_arcsec,
        moment_threshold_sigma=args.moment_threshold_sigma,
        error_method=error_method,
        sky_sigma_nmgy_pix_for_errors=finite_or_none(sky_sigma_nmgy_pix),
        noise_corr_factor=float(args.noise_corr_factor),
        band=band,
        extinction_source=extinction_source,
        extinction_mag_band=finite_or_none(extinction_mag_band),
        mw_transmission_band=finite_or_none(mw_transmission_band),
        tractor_extinction_column=tractor_extinction_column,
        tractor_extinction_sep_arcsec=finite_or_none(tractor_extinction_sep_arcsec),
        tractor_extinction_source_id=tractor_extinction_source_id,
        tractor_max_sma_description=tractor_max_sma_description,
        tractor_max_sma_sep_arcsec=finite_or_none(tractor_max_sma_sep_arcsec),
        tractor_max_sma_source_id=tractor_max_sma_source_id,
        tractor_geometry_description=tractor_geometry_description,
        tractor_geometry_sep_arcsec=finite_or_none(tractor_geometry_sep_arcsec),
        tractor_geometry_source_id=tractor_geometry_source_id,
        pa_source=pa_source,
        q_source=q_source,
        reff_elliptical_sma_arcsec=finite_or_none(re_sma),
        reff_circularized_arcsec=finite_or_none(re_circ),
        reff_elliptical_sma_kpc=(
            finite_or_none(re_sma * kpc_per_arcsec)
            if kpc_per_arcsec and np.isfinite(re_sma)
            else None
        ),
        reff_circularized_kpc=(
            finite_or_none(re_circ * kpc_per_arcsec)
            if kpc_per_arcsec and np.isfinite(re_circ)
            else None
        ),
        reff_sersiccorr_source=reff_sersiccorr_source,
        reff_sersiccorr_light_frac=finite_or_none(sersic_reff_frac),
        reff_sersiccorr_target_fraction_of_aperture_flux=finite_or_none(
            reff_sersiccorr_target_fraction
        ),
        reff_sersiccorr_elliptical_sma_arcsec=finite_or_none(re_sersiccorr_sma),
        reff_sersiccorr_circularized_arcsec=finite_or_none(re_sersiccorr_circ),
        reff_sersiccorr_elliptical_sma_kpc=(
            finite_or_none(re_sersiccorr_sma * kpc_per_arcsec)
            if kpc_per_arcsec and np.isfinite(re_sersiccorr_sma)
            else None
        ),
        reff_sersiccorr_circularized_kpc=(
            finite_or_none(re_sersiccorr_circ * kpc_per_arcsec)
            if kpc_per_arcsec and np.isfinite(re_sersiccorr_circ)
            else None
        ),
        flux_nmgy_within_reff_sersiccorr=(
            None
            if input_extinction_corrected
            else finite_or_none(flux_within_reff_sersiccorr)
        ),
        mag_ab_within_reff_sersiccorr=(
            None
            if input_extinction_corrected
            else finite_or_none(mag_within_reff_sersiccorr)
        ),
        flux_extcorr_nmgy_within_reff_sersiccorr=finite_or_none(
            flux_within_reff_sersiccorr
            if input_extinction_corrected
            else flux_extinction_corrected(
                flux_within_reff_sersiccorr,
                mw_transmission_band,
            )
        ),
        mag_extcorr_ab_within_reff_sersiccorr=finite_or_none(
            mag_extcorr_within_reff_sersiccorr
        ),
        total_flux_nmgy_used_for_reff=(
            None if input_extinction_corrected else finite_or_none(total_flux)
        ),
        total_flux_err_nmgy_used_for_reff=(
            None if input_extinction_corrected else finite_or_none(total_flux_err)
        ),
        total_flux_extcorr_nmgy_used_for_reff=finite_or_none(total_flux_extcorr),
        total_flux_extcorr_err_nmgy_used_for_reff=finite_or_none(
            total_flux_extcorr_err
        ),
        total_mag_used_for_reff=(
            None if input_extinction_corrected else finite_or_none(total_mag)
        ),
        total_mag_err_used_for_reff=(
            None if input_extinction_corrected else finite_or_none(total_mag_err)
        ),
        total_mag_extcorr_used_for_reff=finite_or_none(total_mag_extcorr),
        sersic_total_correction_source=sersic_total_correction_source,
        sersic_total_correction_light_frac=finite_or_none(
            sersic_total_correction_light_frac
        ),
        total_flux_sersiccorr_nmgy=(
            None
            if input_extinction_corrected
            else finite_or_none(total_flux_sersiccorr)
        ),
        total_flux_sersiccorr_err_nmgy=(
            None
            if input_extinction_corrected
            else finite_or_none(total_flux_sersiccorr_err)
        ),
        total_mag_sersiccorr_ab=(
            None if input_extinction_corrected else finite_or_none(total_mag_sersiccorr)
        ),
        total_mag_sersiccorr_err_ab=(
            None
            if input_extinction_corrected
            else finite_or_none(total_mag_sersiccorr_err)
        ),
        total_flux_sersiccorr_extcorr_nmgy=finite_or_none(
            total_flux_sersiccorr_extcorr
        ),
        total_flux_sersiccorr_extcorr_err_nmgy=finite_or_none(
            total_flux_sersiccorr_extcorr_err
        ),
        total_mag_sersiccorr_extcorr_ab=finite_or_none(total_mag_sersiccorr_extcorr),
        flux_nmgy_within_reff=(
            None if input_extinction_corrected else finite_or_none(flux_within_reff)
        ),
        flux_err_nmgy_within_reff=(
            None if input_extinction_corrected else finite_or_none(flux_err_within_reff)
        ),
        flux_extcorr_nmgy_within_reff=finite_or_none(flux_extcorr_within_reff),
        flux_extcorr_err_nmgy_within_reff=finite_or_none(flux_extcorr_err_within_reff),
        mag_ab_within_reff=(
            None if input_extinction_corrected else finite_or_none(mag_within_reff)
        ),
        mag_err_ab_within_reff=(
            None if input_extinction_corrected else finite_or_none(mag_err_within_reff)
        ),
        mag_extcorr_ab_within_reff=finite_or_none(mag_extcorr_within_reff),
        max_sma_arcsec=float(max_sma_arcsec_used),
        max_sma_source=max_sma_source,
        bulge_radius_source=bulge_radius_source,
        bulge_radius_sma_arcsec=finite_or_none(br),
        bulge_radius_circularized_arcsec=finite_or_none(br_circ),
        bulge_radius_sma_kpc=(
            finite_or_none(br * kpc_per_arcsec)
            if br is not None and kpc_per_arcsec
            else None
        ),
        bulge_radius_circularized_kpc=(
            finite_or_none(br_circ * kpc_per_arcsec)
            if br_circ is not None and kpc_per_arcsec
            else None
        ),
        bulge_flux_nmgy=(
            None if input_extinction_corrected else finite_or_none(bulge_flux)
        ),
        bulge_flux_err_nmgy=(
            None if input_extinction_corrected else finite_or_none(bulge_flux_err)
        ),
        bulge_flux_extcorr_nmgy=finite_or_none(bulge_flux_extcorr),
        bulge_flux_extcorr_err_nmgy=finite_or_none(bulge_flux_extcorr_err),
        bulge_mag_ab=finite_or_none(bulge_mag),
        bulge_mag_err_ab=(
            None if input_extinction_corrected else finite_or_none(bulge_mag_err)
        ),
        bulge_mag_extcorr_ab=(
            None if input_extinction_corrected else finite_or_none(bulge_mag_extcorr)
        ),
        bulge_disk_fit_mu0=bulge["disk_mu0"],
        bulge_disk_fit_slope_mag_per_arcsec=bulge["disk_slope"],
        bulge_excess_threshold_mag=args.bulge_excess_threshold_mag,
        input_parameters=input_parameters,
        systematic_uncertainties=systematic_uncertainties,
        notes=(
            "Reff is measured from the curve of growth using the cumulative flux inside max_sma_arcsec. "
            "Existing flux-error fields are formal random errors from invvar or empirical sky RMS. "
            "Optional systematic_uncertainties separately describes conditional systematic and combined errors. "
            "No uncertainty in Reff/bulge radius is included. "
            "The bulge radius is a diagnostic excess-light estimate, not a formal bulge+disk decomposition."
        ),
    )

    write_profile_csv(args.output_csv, rows, geom)

    geometry_out = args.geometry_out or str(
        Path(args.output_csv).with_suffix(".geometry.json")
    )
    Path(geometry_out).write_text(
        json.dumps(geometry_output_dict(geom), indent=2) + "\n"
    )

    if args.magnitudes_out and band not in ("g", "r"):
        raise ValueError("--magnitudes-out requires --band g or --band r")
    if band in ("g", "r") and geom.sersic_total_correction_source is None:
        message = (
            "Sérsic magnitude JSON requires an uncapped Tractor Sérsic-fraction "
            "aperture (--max-sma-from-tractor --tractor-max-sma-method sersic-fraction)"
        )
        if args.magnitudes_out:
            raise ValueError(message)
        print(f"Skipped Sérsic magnitude JSON: {message}")
    elif band in ("g", "r"):
        magnitudes_out = args.magnitudes_out or str(
            Path(args.output_csv).with_suffix(".magnitudes.json")
        )
        write_sersic_magnitudes_json(magnitudes_out, geom)
        print(f"Wrote Sérsic total magnitudes JSON: {magnitudes_out}")

    print(f"Wrote profile CSV: {args.output_csv}")
    print(f"Wrote geometry JSON: {geometry_out}")
    print(f"Target: {name}")
    print(
        f"  PA_major = {geom.pa_major_deg_east_of_north:.3f} deg east of north ({geom.pa_source})"
    )
    print(f"  PA_minor = {geom.pa_minor_deg_east_of_north:.3f} deg east of north")
    print(
        f"  q=b/a    = {geom.q_b_over_a:.4f}  (ellipticity={geom.ellipticity:.4f}; source={geom.q_source})"
    )
    if geom.tractor_geometry_description is not None:
        print(f"  Tractor geometry = {geom.tractor_geometry_description}")
        print(
            f"  Tractor geometry match = {geom.tractor_geometry_sep_arcsec:.3f} arcsec; "
            f"{geom.tractor_geometry_source_id}"
        )
    print(f"  Max SMA = {geom.max_sma_arcsec:.3f} arcsec ({geom.max_sma_source})")
    if geom.tractor_max_sma_description is not None:
        print(f"  Tractor max-SMA rule = {geom.tractor_max_sma_description}")
        print(
            f"  Tractor max-SMA match = {geom.tractor_max_sma_sep_arcsec:.3f} arcsec; "
            f"{geom.tractor_max_sma_source_id}"
        )
    print(f"  Error method = {geom.error_method}")
    if geom.sky_sigma_nmgy_pix_for_errors is not None:
        print(
            f"  Sky sigma for errors = {geom.sky_sigma_nmgy_pix_for_errors:.6g} nMgy/pix"
        )
    if geom.noise_corr_factor != 1.0:
        print(f"  Noise correlation factor = {geom.noise_corr_factor:.3f}")
    if geom.systematic_uncertainties is not None:
        for label in ("aperture", "sersic_total"):
            item = geom.systematic_uncertainties[label]
            if item is not None:
                print(
                    f"  {label} magnitude error: formal={item['formal_mag_err']} "
                    f"systematic={item['systematic_mag_err']} "
                    f"combined={item['combined_mag_err']} mag"
                )
    if geom.extinction_mag_band is not None:
        print(
            f"  Extinction correction = A_{geom.band or 'band'}={geom.extinction_mag_band:.4f} mag; "
            f"MW transmission={geom.mw_transmission_band:.6f}; source={geom.extinction_source}"
        )
        if geom.tractor_extinction_sep_arcsec is not None:
            print(
                f"  Tractor extinction match = {geom.tractor_extinction_sep_arcsec:.3f} arcsec; "
                f"{geom.tractor_extinction_column}; {geom.tractor_extinction_source_id}"
            )

    if geom.input_extinction_corrected:
        if geom.total_flux_extcorr_nmgy_used_for_reff is not None:
            print(
                f"  Total extinction-corrected flux(<max sma) = "
                f"{geom.total_flux_extcorr_nmgy_used_for_reff:.6g}"
                f" +/- {geom.total_flux_extcorr_err_nmgy_used_for_reff:.6g} nMgy "
                f"(mag = {geom.total_mag_extcorr_used_for_reff:.4f} AB)"
            )

    else:
        if geom.total_flux_nmgy_used_for_reff is not None:
            print(
                f"  Total flux(<max sma) = "
                f"{geom.total_flux_nmgy_used_for_reff:.6g}"
                f" +/- {geom.total_flux_err_nmgy_used_for_reff:.6g} nMgy "
                f"(mag = {geom.total_mag_used_for_reff:.4f}"
                f" +/- {geom.total_mag_err_used_for_reff:.4f} AB)"
            )

        if geom.total_flux_extcorr_nmgy_used_for_reff is not None:
            print(
                f"  Total extinction-corrected flux(<max sma) = "
                f"{geom.total_flux_extcorr_nmgy_used_for_reff:.6g}"
                f" +/- {geom.total_flux_extcorr_err_nmgy_used_for_reff:.6g} nMgy"
            )

        if geom.total_mag_extcorr_used_for_reff is not None:
            print(
                f"  Total extinction-corrected mag(<max sma) = "
                f"{geom.total_mag_extcorr_used_for_reff:.4f} AB"
            )

    if geom.input_extinction_corrected:
        if geom.total_flux_sersiccorr_extcorr_nmgy is not None:
            print(
                f"  Sérsic-corrected extinction-corrected total flux = "
                f"{geom.total_flux_sersiccorr_extcorr_nmgy:.6g}"
                f" +/- {geom.total_flux_sersiccorr_extcorr_err_nmgy:.6g} nMgy "
                f"(light-fraction correction = "
                f"1/{geom.sersic_total_correction_light_frac:.4f})"
            )

        if geom.total_mag_sersiccorr_extcorr_ab is not None:
            print(
                f"  Sérsic-corrected extinction-corrected total mag = "
                f"{geom.total_mag_sersiccorr_extcorr_ab:.4f} AB"
            )

    else:
        if geom.total_flux_sersiccorr_nmgy is not None:
            print(
                f"  Sérsic-corrected total flux = "
                f"{geom.total_flux_sersiccorr_nmgy:.6g}"
                f" +/- {geom.total_flux_sersiccorr_err_nmgy:.6g} nMgy "
                f"(light-fraction correction = "
                f"1/{geom.sersic_total_correction_light_frac:.4f})"
            )

            print(
                f"  Sérsic-corrected total mag = "
                f"{geom.total_mag_sersiccorr_ab:.4f}"
                f" +/- {geom.total_mag_sersiccorr_err_ab:.4f} AB"
            )

        if geom.total_flux_sersiccorr_extcorr_nmgy is not None:
            print(
                f"  Sérsic-corrected extinction-corrected total flux = "
                f"{geom.total_flux_sersiccorr_extcorr_nmgy:.6g}"
                f" +/- {geom.total_flux_sersiccorr_extcorr_err_nmgy:.6g} nMgy"
            )

        if geom.total_mag_sersiccorr_extcorr_ab is not None:
            print(
                f"  Sérsic-corrected extinction-corrected total mag = "
                f"{geom.total_mag_sersiccorr_extcorr_ab:.4f} AB"
            )

    if geom.reff_elliptical_sma_arcsec is not None:
        print(
            f"  Reff     = {geom.reff_elliptical_sma_arcsec:.3f} arcsec "
            f"along semi-major axis; "
            f"{geom.reff_circularized_arcsec:.3f} arcsec circularized"
        )

        if geom.input_extinction_corrected:
            if geom.flux_extcorr_nmgy_within_reff is not None:
                print(
                    f"  Extinction-corrected flux(<Reff) = "
                    f"{geom.flux_extcorr_nmgy_within_reff:.6g}"
                    f" +/- {geom.flux_extcorr_err_nmgy_within_reff:.6g} nMgy "
                    f"(mag = {geom.mag_extcorr_ab_within_reff:.4f} AB)"
                )

        else:
            if geom.flux_nmgy_within_reff is not None:
                print(
                    f"  Flux(<Reff) = {geom.flux_nmgy_within_reff:.6g}"
                    f" +/- {geom.flux_err_nmgy_within_reff:.6g} nMgy "
                    f"(mag = {geom.mag_ab_within_reff:.4f}"
                    f" +/- {geom.mag_err_ab_within_reff:.4f} AB)"
                )

            if geom.flux_extcorr_nmgy_within_reff is not None:
                print(
                    f"  Extinction-corrected flux(<Reff) = "
                    f"{geom.flux_extcorr_nmgy_within_reff:.6g}"
                    f" +/- {geom.flux_extcorr_err_nmgy_within_reff:.6g} nMgy"
                )

            if geom.mag_extcorr_ab_within_reff is not None:
                print(
                    f"  Extinction-corrected mag(<Reff) = "
                    f"{geom.mag_extcorr_ab_within_reff:.4f} AB"
                )

    else:
        print("  Reff     = not available; check total flux/background/max radius")
        if geom.reff_elliptical_sma_arcsec is not None:
            print(
                f"  Reff     = {geom.reff_elliptical_sma_arcsec:.3f} arcsec along semi-major axis; "
                f"{geom.reff_circularized_arcsec:.3f} arcsec circularized"
            )
            if geom.flux_nmgy_within_reff is not None:
                print(
                    f"  Flux(<Reff) = {geom.flux_nmgy_within_reff:.6g}"
                    f" +/- {geom.flux_err_nmgy_within_reff:.6g} nMgy "
                    f"(mag = {geom.mag_ab_within_reff:.4f} +/- {geom.mag_err_ab_within_reff:.4f} AB)"
                )
                if geom.flux_extcorr_nmgy_within_reff is not None:
                    print(
                        f"  Extinction-corrected flux(<Reff) = "
                        f"{geom.flux_extcorr_nmgy_within_reff:.6g}"
                        f" +/- {geom.flux_extcorr_err_nmgy_within_reff:.6g} nMgy"
                    )
                if geom.mag_extcorr_ab_within_reff is not None:
                    print(
                        f"  Extinction-corrected mag(<Reff) = {geom.mag_extcorr_ab_within_reff:.4f} AB"
                    )
        else:
            print("  Reff     = not available; check total flux/background/max radius")

    if geom.reff_sersiccorr_elliptical_sma_arcsec is not None:
        print(
            f"  Sérsic-corrected Reff = "
            f"{geom.reff_sersiccorr_elliptical_sma_arcsec:.3f} arcsec "
            f"along semi-major axis; "
            f"{geom.reff_sersiccorr_circularized_arcsec:.3f} arcsec circularized"
        )

        print(
            f"    Correction source = {geom.reff_sersiccorr_source}; "
            f"Sérsic aperture fraction = "
            f"{geom.reff_sersiccorr_light_frac:.4f}; "
            f"target = "
            f"{geom.reff_sersiccorr_target_fraction_of_aperture_flux:.4f} "
            f"of measured aperture flux"
        )

        if geom.input_extinction_corrected:
            if geom.flux_extcorr_nmgy_within_reff_sersiccorr is not None:
                print(
                    f"    Extinction-corrected flux"
                    f"(<Sérsic-corrected Reff) = "
                    f"{geom.flux_extcorr_nmgy_within_reff_sersiccorr:.6g} nMgy"
                )

            if geom.mag_extcorr_ab_within_reff_sersiccorr is not None:
                print(
                    f"    Extinction-corrected mag"
                    f"(<Sérsic-corrected Reff) = "
                    f"{geom.mag_extcorr_ab_within_reff_sersiccorr:.4f} AB"
                )

        else:
            if geom.flux_nmgy_within_reff_sersiccorr is not None:
                print(
                    f"    Flux(<Sérsic-corrected Reff) = "
                    f"{geom.flux_nmgy_within_reff_sersiccorr:.6g} nMgy "
                    f"(mag = "
                    f"{geom.mag_ab_within_reff_sersiccorr:.4f} AB)"
                )

            if geom.flux_extcorr_nmgy_within_reff_sersiccorr is not None:
                print(
                    f"    Extinction-corrected flux"
                    f"(<Sérsic-corrected Reff) = "
                    f"{geom.flux_extcorr_nmgy_within_reff_sersiccorr:.6g} nMgy"
                )

            if geom.mag_extcorr_ab_within_reff_sersiccorr is not None:
                print(
                    f"    Extinction-corrected mag"
                    f"(<Sérsic-corrected Reff) = "
                    f"{geom.mag_extcorr_ab_within_reff_sersiccorr:.4f} AB"
                )

    if geom.bulge_radius_sma_arcsec is not None:
        print(
            f"  Bulge radius ({geom.bulge_radius_source}) = "
            f"{geom.bulge_radius_sma_arcsec:.3f} arcsec "
            f"along semi-major axis; "
            f"{geom.bulge_radius_circularized_arcsec:.3f} arcsec circularized"
        )

        if geom.input_extinction_corrected:
            if geom.bulge_flux_extcorr_nmgy is not None:
                print(
                    f"  Extinction-corrected flux(<bulge radius) = "
                    f"{geom.bulge_flux_extcorr_nmgy:.6g}"
                    f" +/- {geom.bulge_flux_extcorr_err_nmgy:.6g} nMgy "
                    f"(mag = {geom.bulge_mag_extcorr_ab:.4f} AB)"
                )

        else:
            if geom.bulge_flux_nmgy is not None:
                print(
                    f"  Flux(<bulge radius) = {geom.bulge_flux_nmgy:.6g}"
                    f" +/- {geom.bulge_flux_err_nmgy:.6g} nMgy "
                    f"(mag = {geom.bulge_mag_ab:.4f}"
                    f" +/- {geom.bulge_mag_err_ab:.4f} AB)"
                )

            if geom.bulge_flux_extcorr_nmgy is not None:
                print(
                    f"  Extinction-corrected flux(<bulge radius) = "
                    f"{geom.bulge_flux_extcorr_nmgy:.6g}"
                    f" +/- {geom.bulge_flux_extcorr_err_nmgy:.6g} nMgy"
                )

            if geom.bulge_mag_extcorr_ab is not None:
                print(
                    f"  Extinction-corrected mag(<bulge radius) = "
                    f"{geom.bulge_mag_extcorr_ab:.4f} AB"
                )

    else:
        print(
            "  Bulge diagnostic radius = not found/reliable from the "
            "outer-disk excess test"
        )
        if geom.bulge_radius_sma_arcsec is not None:
            print(
                f"  Bulge radius ({geom.bulge_radius_source}) = {geom.bulge_radius_sma_arcsec:.3f} arcsec along semi-major axis; "
                f"{geom.bulge_radius_circularized_arcsec:.3f} arcsec circularized"
            )
            if geom.bulge_flux_nmgy is not None:
                print(
                    f"  Flux(<bulge radius) = {geom.bulge_flux_nmgy:.6g}"
                    f" +/- {geom.bulge_flux_err_nmgy:.6g} nMgy "
                    f"(mag = {geom.bulge_mag_ab:.4f} +/- {geom.bulge_mag_err_ab:.4f} AB)"
                )
                if geom.bulge_flux_extcorr_nmgy is not None:
                    print(
                        f"  Extinction-corrected flux(<bulge radius) = "
                        f"{geom.bulge_flux_extcorr_nmgy:.6g}"
                        f" +/- {geom.bulge_flux_extcorr_err_nmgy:.6g} nMgy"
                    )
                if geom.bulge_mag_extcorr_ab is not None:
                    print(
                        f"  Extinction-corrected mag(<bulge radius) = {geom.bulge_mag_extcorr_ab:.4f} AB"
                    )
        else:
            print(
                "  Bulge diagnostic radius = not found/reliable from the outer-disk excess test"
            )

    if args.plot_profile:
        plot_profile(args.output_csv, args.plot_profile, show=args.show)
        print(f"Wrote profile plot: {args.plot_profile}")

    if args.diagnostic_png:
        make_diagnostic_plot(
            data=data,
            valid=valid,
            x0=xc,
            y0=yc,
            theta=theta,
            q=q,
            max_sma_pix=max_sma_pix,
            output=args.diagnostic_png,
            title=name,
        )
        print(f"Wrote diagnostic plot: {args.diagnostic_png}")


UNCORRECTED_GEOMETRY_PHOTOMETRY_FIELDS = {
    "total_flux_nmgy_used_for_reff",
    "total_flux_err_nmgy_used_for_reff",
    "total_mag_used_for_reff",
    "total_mag_err_used_for_reff",
    "total_flux_sersiccorr_nmgy",
    "total_flux_sersiccorr_err_nmgy",
    "total_mag_sersiccorr_ab",
    "total_mag_sersiccorr_err_ab",
    "flux_nmgy_within_reff_sersiccorr",
    "mag_ab_within_reff_sersiccorr",
    "flux_nmgy_within_reff",
    "flux_err_nmgy_within_reff",
    "mag_ab_within_reff",
    "mag_err_ab_within_reff",
    "bulge_flux_nmgy",
    "bulge_flux_err_nmgy",
    "bulge_mag_ab",
    "bulge_mag_err_ab",
}


def geometry_output_dict(geom: Geometry) -> dict[str, Any]:
    output = asdict(geom)

    if geom.input_extinction_corrected:
        for key in UNCORRECTED_GEOMETRY_PHOTOMETRY_FIELDS:
            output.pop(key, None)

        # No correction factor was applied by this script.
        for key in [
            "extinction_mag_band",
            "mw_transmission_band",
            "tractor_extinction_column",
            "tractor_extinction_sep_arcsec",
            "tractor_extinction_source_id",
        ]:
            output.pop(key, None)

    return output


def write_sersic_magnitudes_json(path: str, geom: Geometry) -> None:
    """Update one band's Sérsic total magnitude and circularized Reff."""
    band = geom.band
    if band not in ("g", "r"):
        raise ValueError("The Sérsic magnitude JSON requires --band g or --band r")
    if geom.sersic_total_correction_source is None:
        raise ValueError(
            "The Sérsic magnitude JSON requires an uncapped Tractor Sérsic-fraction "
            "aperture (--max-sma-from-tractor --tractor-max-sma-method sersic-fraction)"
        )

    # Prefer extinction-corrected magnitudes when a correction is available;
    # otherwise the measured magnitude is already in the appropriate system.
    magnitude = (
        geom.total_mag_sersiccorr_extcorr_ab
        if geom.total_mag_sersiccorr_extcorr_ab is not None
        else geom.total_mag_sersiccorr_ab
    )
    flux = geom.total_flux_sersiccorr_extcorr_nmgy
    flux_error = geom.total_flux_sersiccorr_extcorr_err_nmgy
    error = (
        finite_or_none(mag_err_from_flux_err(flux, flux_error))
        if flux is not None and flux_error is not None
        else None
    )

    output_path = Path(path)
    result = {
        "r": None, "r_err": None, "r_reff_arcsec": None,
        "g": None, "g_err": None, "g_reff_arcsec": None,
    }
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        if not isinstance(existing, dict):
            raise ValueError(f"Existing magnitude JSON must be an object: {path}")
        # Retain the known band measurements from older or extended files.
        # Other metadata/fields are not part of this compact output schema.
        if not any(key in existing for key in result):
            raise ValueError(f"Existing magnitude JSON has no recognized fields: {path}")
        result.update({key: existing[key] for key in result if key in existing})
    result[band] = finite_or_none(magnitude) if magnitude is not None else None
    result[f"{band}_err"] = error
    result[f"{band}_reff_arcsec"] = geom.reff_sersiccorr_circularized_arcsec

    # Keep JSON numbers (rather than strings) while retaining four decimal
    # places, including trailing zeros, in the file itself.
    def json_number(value: Optional[float]) -> str:
        return f"{value:.4f}" if value is not None and math.isfinite(value) else "null"

    serialized = "{\n" + ",\n".join(
        f"  {json.dumps(key)}: {json_number(value)}" for key, value in result.items()
    ) + "\n}\n"

    # Replace atomically so a rerun of the other band never sees partial JSON.
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output_path.parent, delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(serialized)
    try:
        os.replace(tmp_path, output_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def write_profile_csv(
    path: str | Path, rows: list[dict[str, Any]], geom: Geometry
) -> None:
    path = Path(path)
    if not rows:
        raise ValueError("No profile rows to write")

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        f.write("# legacy_ellipse_profile.py\n")
        for key, value in geometry_output_dict(geom).items():
            if key != "notes":
                if isinstance(value, (dict, list, tuple)):
                    value_txt = json.dumps(json_safe_value(value), sort_keys=True)
                else:
                    value_txt = value
                f.write(f"# {key} = {value_txt}\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_profile_csv(path: str | Path) -> dict[str, np.ndarray]:
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(row for row in f if not row.startswith("#"))
        cols: dict[str, list[float]] = {}
        for row in reader:
            for k, v in row.items():
                cols.setdefault(k, [])
                try:
                    cols[k].append(float(v))
                except Exception:
                    cols[k].append(np.nan)
    return {k: np.asarray(v, dtype=float) for k, v in cols.items()}


def plot_profile(
    csv_path: str | Path,
    output: Optional[str] = None,
    show: bool = False,
) -> None:
    import matplotlib.pyplot as plt

    cols = read_profile_csv(csv_path)
    r = cols["sma_arcsec"]

    # Select the photometric convention actually present in the CSV.
    #
    # For --input-extinction-corrected files, only the extcorr columns exist.
    # For ordinary input files, prefer extinction-corrected values when they
    # are available; otherwise use the native values.
    use_extcorr = "flux_extcorr_nmgy" in cols and np.any(
        np.isfinite(cols["flux_extcorr_nmgy"])
    )

    if use_extcorr:
        flux = cols["flux_extcorr_nmgy"]
        flux_err = cols.get(
            "flux_extcorr_err_nmgy",
            np.full_like(flux, np.nan),
        )

        integrated_mag = cols.get(
            "integrated_mag_extcorr_ab",
            np.asarray(mag_from_nmgy(flux), dtype=float),
        )
        integrated_mag_err = cols.get(
            "integrated_mag_extcorr_err",
            np.full_like(flux, np.nan),
        )

        mean_sb = cols["mean_sb_extcorr_nmgy_arcsec2"]
        mean_sb_err = cols.get(
            "mean_sb_extcorr_err_nmgy_arcsec2",
            np.full_like(mean_sb, np.nan),
        )

        mean_mu = cols.get(
            "mean_mu_extcorr_ab_arcsec2",
            np.asarray(mu_from_nmgy_per_arcsec2(mean_sb), dtype=float),
        )

        annulus_sb = cols["annulus_sb_extcorr_nmgy_arcsec2"]
        annulus_sb_err = cols.get(
            "annulus_sb_extcorr_err_nmgy_arcsec2",
            np.full_like(annulus_sb, np.nan),
        )

        annulus_mu = cols.get(
            "annulus_mu_extcorr_ab_arcsec2",
            np.asarray(
                mu_from_nmgy_per_arcsec2(annulus_sb),
                dtype=float,
            ),
        )

        photometry_label = "Extinction corrected"

    else:
        flux = cols["flux_nmgy"]
        flux_err = cols.get(
            "flux_err_nmgy",
            np.full_like(flux, np.nan),
        )

        integrated_mag = cols["integrated_mag_ab"]
        integrated_mag_err = cols.get(
            "integrated_mag_err",
            np.full_like(flux, np.nan),
        )

        mean_sb = cols["mean_sb_nmgy_arcsec2"]
        mean_sb_err = cols.get(
            "mean_sb_err_nmgy_arcsec2",
            np.full_like(mean_sb, np.nan),
        )

        mean_mu = cols.get(
            "mean_mu_ab_arcsec2",
            np.asarray(mu_from_nmgy_per_arcsec2(mean_sb), dtype=float),
        )

        annulus_sb = cols["annulus_sb_nmgy_arcsec2"]
        annulus_sb_err = cols.get(
            "annulus_sb_err_nmgy_arcsec2",
            np.full_like(annulus_sb, np.nan),
        )

        annulus_mu = cols.get(
            "annulus_mu_ab_arcsec2",
            np.asarray(
                mu_from_nmgy_per_arcsec2(annulus_sb),
                dtype=float,
            ),
        )

        photometry_label = "Native"

    # Convert linear surface-brightness errors to magnitude errors:
    #
    # sigma_mu = (2.5 / ln 10) * sigma_I / I
    mean_mu_err = np.full_like(mean_sb, np.nan, dtype=float)
    good = (
        np.isfinite(mean_sb)
        & np.isfinite(mean_sb_err)
        & (mean_sb > 0)
        & (mean_sb_err >= 0)
    )
    mean_mu_err[good] = 2.5 / np.log(10.0) * mean_sb_err[good] / mean_sb[good]

    annulus_mu_err = np.full_like(annulus_sb, np.nan, dtype=float)
    good = (
        np.isfinite(annulus_sb)
        & np.isfinite(annulus_sb_err)
        & (annulus_sb > 0)
        & (annulus_sb_err >= 0)
    )
    annulus_mu_err[good] = 2.5 / np.log(10.0) * annulus_sb_err[good] / annulus_sb[good]

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.5, 10.0),
        sharex=True,
    )

    axes[0].errorbar(
        r,
        flux,
        yerr=flux_err,
        marker="o",
        ms=3,
        lw=1,
        capsize=2,
    )
    axes[0].set_ylabel("Cumulative flux [nMgy]")
    axes[0].grid(alpha=0.25)

    axes[1].errorbar(
        r,
        integrated_mag,
        yerr=integrated_mag_err,
        marker="o",
        ms=3,
        lw=1,
        capsize=2,
    )
    axes[1].invert_yaxis()
    axes[1].set_ylabel("Integrated magnitude [AB]")
    axes[1].grid(alpha=0.25)

    axes[2].errorbar(
        r,
        mean_mu,
        yerr=mean_mu_err,
        marker="o",
        ms=3,
        lw=1,
        capsize=2,
        label="Mean within aperture",
    )

    axes[2].errorbar(
        r,
        annulus_mu,
        yerr=annulus_mu_err,
        marker=".",
        ms=3,
        lw=1,
        capsize=2,
        label="Annular",
    )

    axes[2].invert_yaxis()
    axes[2].set_xlabel("Semi-major-axis radius [arcsec]")
    axes[2].set_ylabel(r"$\mu$ [AB mag arcsec$^{-2}$]")
    axes[2].grid(alpha=0.25)
    axes[2].legend(loc="best", fontsize=9)

    fig.suptitle(f"{Path(csv_path).name}\n{photometry_label} photometry")
    fig.tight_layout()

    if output:
        fig.savefig(output, dpi=160, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def make_diagnostic_plot(
    data: np.ndarray,
    valid: np.ndarray,
    x0: float,
    y0: float,
    theta: float,
    q: float,
    max_sma_pix: float,
    output: str,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    crop_y, crop_x = crop_slices_around(x0, y0, max_sma_pix, data.shape)
    sub = np.array(data[crop_y, crop_x], dtype=float)
    sub[~valid[crop_y, crop_x]] = np.nan

    vals = sub[np.isfinite(sub)]
    if vals.size:
        vmin, vmax = np.nanpercentile(vals, [5, 99.5])
    else:
        vmin, vmax = np.nan, np.nan

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(sub, origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")

    cx = x0 - crop_x.start
    cy = y0 - crop_y.start
    angle_deg = np.rad2deg(theta)

    for frac in [0.25, 0.5, 0.75, 1.0]:
        a = max_sma_pix * frac
        ell = Ellipse(
            (cx, cy), width=2 * a, height=2 * a * q, angle=angle_deg, fill=False, lw=1.2
        )
        ax.add_patch(ell)

    dx = max_sma_pix * np.cos(theta)
    dy = max_sma_pix * np.sin(theta)
    ax.plot([cx - dx, cx + dx], [cy - dy, cy + dy], lw=1.5, label="major axis")
    dxm = max_sma_pix * q * np.cos(theta + np.pi / 2.0)
    dym = max_sma_pix * q * np.sin(theta + np.pi / 2.0)
    ax.plot(
        [cx - dxm, cx + dxm], [cy - dym, cy + dym], lw=1.5, ls="--", label="minor axis"
    )
    ax.plot(cx, cy, marker="+", ms=10, mew=1.5)

    ax.set_title(title)
    ax.set_xlabel("x [pix in crop]")
    ax.set_ylabel("y [pix in crop]")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_command(args: argparse.Namespace) -> None:
    plot_profile(args.profile_csv, args.output, show=args.show)
    if args.output:
        print(f"Wrote profile plot: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Measure elliptical aperture profiles from Legacy Survey nanomaggy images."
    )
    sub = p.add_subparsers(dest="command", required=True)

    m = sub.add_parser("measure", help="measure an elliptical curve of growth")
    m.add_argument(
        "fits", help="Input FITS file, e.g. output from mu_from_legacy_background.py"
    )
    m.add_argument(
        "target_dict", help="Python/JSON dictionary with at least ra and dec in degrees"
    )
    m.add_argument("output_csv", help="Output CSV profile")
    m.add_argument(
        "--hdu",
        default="SKY_SUB_NMGY",
        help="Image HDU name or index [default: SKY_SUB_NMGY]",
    )
    m.add_argument(
        "--band",
        default=None,
        help="Photometric band, e.g. g, r, or z; required for --tractor-fits extinction correction",
    )
    m.add_argument(
        "--input-extinction-corrected",
        action="store_true",
        help=(
            "The selected input image HDU is already corrected for Galactic "
            "extinction. Report measured photometry only in extinction-corrected "
            "output fields and do not apply another correction."
        ),
    )
    m.add_argument(
        "--extinction-mag",
        type=float,
        default=None,
        help="Direct Galactic extinction A_band in magnitudes; corrected mag = observed mag - A_band",
    )
    m.add_argument(
        "--mw-transmission",
        type=float,
        default=None,
        help="Direct Milky Way transmission fraction for this band; dereddened flux = flux / transmission",
    )
    m.add_argument(
        "--tractor-fits",
        default=None,
        help="Optional tractor-<brick>.fits file used to read mw_transmission_<band> near the target",
    )
    m.add_argument(
        "--tractor-hdu",
        default="1",
        help="Tractor table HDU name or index [default: 1]",
    )
    m.add_argument(
        "--tractor-max-sep-arcsec",
        type=float,
        default=5.0,
        help="Maximum match distance for Tractor lookups [default: 5]",
    )
    m.add_argument(
        "--geometry-from-tractor",
        action="store_true",
        help="Use nearest Tractor shape_e1/shape_e2 to set PA and q=b/a unless --pa-deg or --q are explicitly supplied",
    )
    m.add_argument(
        "--mask-fits",
        default=None,
        help="Optional aligned mask FITS; nonzero pixels are excluded",
    )
    m.add_argument(
        "--mask-hdu", default="0", help="Mask HDU name or index [default: 0]"
    )
    m.add_argument(
        "--invvar-fits",
        default=None,
        help="Optional aligned inverse-variance FITS in 1/nMgy^2",
    )
    m.add_argument(
        "--invvar-hdu", default="0", help="Invvar HDU name or index [default: 0]"
    )
    m.add_argument(
        "--sky-sigma-nmgy-pix",
        type=float,
        default=None,
        help="Fallback per-pixel sky RMS in nMgy/pix if no invvar is supplied",
    )
    m.add_argument(
        "--noise-corr-factor",
        type=float,
        default=1.0,
        help="Inflate flux errors for correlated noise [default: 1]",
    )
    m.add_argument("--sys-extinction-mag", type=float, default=None,
                   help="Assumed 1-sigma uncertainty in A_band, mag; also usable with already-dereddened input")
    m.add_argument("--sys-sky-nmgy-pix", type=float, default=None,
                   help="Assumed 1-sigma coherent residual sky offset in input-image nMgy/pixel, from blank regions or alternative background fits")
    m.add_argument("--sys-center-arcsec", type=float, default=None,
                   help="Assumed 1-sigma centroid displacement; perturb x and y independently")
    m.add_argument("--sys-q", type=float, default=None,
                   help="Assumed 1-sigma uncertainty in aperture axis ratio b/a")
    m.add_argument("--sys-pa-deg", type=float, default=None,
                   help="Assumed 1-sigma aperture PA change, degrees")
    m.add_argument("--sys-aperture-sma-arcsec", type=float, default=None,
                   help="Assumed 1-sigma outer-aperture SMA change, arcsec")
    m.add_argument("--sys-sersic-fraction", type=float, default=None,
                   help="Assumed absolute 1-sigma uncertainty in model enclosed-light fraction f; only with uncapped Sérsic-fraction aperture")
    m.add_argument(
        "--error-sky-inner-frac",
        type=float,
        default=0.75,
        help="Inner fraction of max_sma for empirical sky-RMS annulus [default: 0.75]",
    )
    m.add_argument(
        "--error-sky-outer-frac",
        type=float,
        default=1.0,
        help="Outer fraction of max_sma for empirical sky-RMS annulus [default: 1.0]",
    )
    m.add_argument(
        "--pixscale-arcsec",
        type=float,
        default=None,
        help="Fallback pixel scale if WCS pixel area cannot be used",
    )
    m.add_argument(
        "--max-sma-arcsec",
        type=float,
        default=80.0,
        help="Maximum semi-major-axis aperture radius, or cap for --auto-max-sma [default: 80]",
    )
    m.add_argument(
        "--auto-max-sma",
        action="store_true",
        help="Automatically choose max SMA from the curve of growth and outer annular SNR",
    )
    m.add_argument(
        "--max-sma-from-tractor",
        action="store_true",
        help="Set max SMA from nearest Tractor model shape_r",
    )
    m.add_argument(
        "--tractor-max-sma-method",
        choices=["factor", "sersic-fraction"],
        default="factor",
        help="How to derive max SMA from Tractor shape_r [default: factor]",
    )
    m.add_argument(
        "--tractor-max-sma-factor",
        type=float,
        default=5.0,
        help="For --tractor-max-sma-method factor, use this multiplier times shape_r [default: 5]",
    )
    m.add_argument(
        "--tractor-max-sma-light-frac",
        type=float,
        default=0.95,
        help="For --tractor-max-sma-method sersic-fraction, target enclosed model-light fraction [default: 0.95]",
    )
    m.add_argument(
        "--auto-max-search-arcsec",
        type=float,
        default=None,
        help="Maximum search radius for --auto-max-sma; defaults to a conservative image-edge limit, capped by --max-sma-arcsec if set",
    )
    m.add_argument(
        "--auto-max-min-arcsec",
        type=float,
        default=10.0,
        help="Minimum radius allowed for --auto-max-sma [default: 10]",
    )
    m.add_argument(
        "--auto-max-snr-threshold",
        type=float,
        default=1.0,
        help="Stop when outer annular flux S/N stays below this threshold [default: 1]",
    )
    m.add_argument(
        "--auto-max-frac-growth-threshold",
        type=float,
        default=0.005,
        help="Stop when fractional curve-of-growth change per annulus is below this value [default: 0.005]",
    )
    m.add_argument(
        "--auto-max-consecutive",
        type=int,
        default=3,
        help="Number of consecutive quiet annuli required for --auto-max-sma [default: 3]",
    )
    m.add_argument(
        "--step-arcsec",
        type=float,
        default=1.0,
        help="Semi-major-axis radius step [default: 1]",
    )
    m.add_argument(
        "--moment-radius-arcsec",
        type=float,
        default=20.0,
        help="Radius used for moment-based geometry estimate [default: 20]",
    )
    m.add_argument(
        "--moment-threshold-sigma",
        type=float,
        default=1.0,
        help="Signal threshold for moment weights [default: 1]",
    )
    m.add_argument(
        "--no-recenter",
        action="store_true",
        help="Use target RA/Dec pixel as center instead of moment centroid",
    )
    m.add_argument(
        "--pa-deg",
        type=float,
        default=None,
        help="Override major-axis PA in degrees east of north",
    )
    m.add_argument("--q", type=float, default=None, help="Override axis ratio q=b/a")
    m.add_argument(
        "--redshift",
        type=float,
        default=None,
        help="Optional redshift for kpc columns; target dict z is used if present",
    )
    m.add_argument(
        "--geometry-out",
        default=None,
        help="Output JSON geometry file [default: output_csv with .geometry.json]",
    )
    m.add_argument(
        "--magnitudes-out",
        default=None,
        help=(
            "G/r Sérsic-total magnitude and circularized Reff JSON [default: output_csv with "
            ".magnitudes.json]. Supply the same path in g and r runs to combine "
            "both bands; requires --band and an uncapped Tractor Sérsic-fraction aperture."
        ),
    )
    m.add_argument(
        "--plot-profile", default=None, help="Optional output PNG plot of the profile"
    )
    m.add_argument(
        "--diagnostic-png",
        default=None,
        help="Optional output PNG showing apertures and axes",
    )
    m.add_argument("--show", action="store_true", help="Show plots interactively")
    m.add_argument(
        "--disk-fit-min-frac",
        type=float,
        default=0.45,
        help="For bulge diagnostic, fit outer disk beyond this fraction of max_sma [default: 0.45]",
    )
    m.add_argument(
        "--disk-fit-min-points",
        type=int,
        default=8,
        help="Minimum annular-profile points for outer disk fit [default: 8]",
    )
    m.add_argument(
        "--bulge-excess-threshold-mag",
        type=float,
        default=0.20,
        help="Observed annular profile brighter than disk by this many mag [default: 0.20]",
    )
    m.add_argument(
        "--bulge-radius-sma-arcsec",
        type=float,
        default=None,
        help="Override the diagnostic bulge semi-major-axis radius in arcsec; useful for forcing g and r bands to use the same aperture",
    )
    m.set_defaults(func=measure_profile)

    q = sub.add_parser("plot", help="plot a previously written profile CSV")
    q.add_argument("profile_csv", help="Profile CSV from measure command")
    q.add_argument("--output", default=None, help="Output PNG/PDF/etc.")
    q.add_argument("--show", action="store_true", help="Show interactively")
    q.set_defaults(func=plot_command)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
