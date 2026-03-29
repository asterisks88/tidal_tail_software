#!/usr/bin/env python3
"""
Analyze a Legacy Survey r-band host galaxy profile to assess whether it is
consistent with a non-cD BCG or requires a robust outer envelope.

Inputs
------
--image   : science image FITS (e.g. legacysurvey-<brick>-image-r.fits.fz)
--invvar  : inverse-variance FITS (e.g. legacysurvey-<brick>-invvar-r.fits.fz)
--resid   : optional residual FITS, used only as an auxiliary masking aid

Outputs
-------
<outprefix>_profile.csv
<outprefix>_fit_summary.json
<outprefix>_sb_profile.png
<outprefix>_residual_single_sersic.png
<outprefix>_mask.png
<outprefix>_cutout.fits

Notes
-----
- Legacy Survey coadds are assumed to be in nanomaggies/pixel.
- Surface brightness is reported as mag/arcsec^2 using:
      mu = 22.5 - 2.5 log10(I_pix / pixscale^2)
- For paper-quality results, inspect the mask manually and rerun if needed.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.wcs import WCS
from matplotlib.path import Path
from photutils.aperture import EllipticalAperture
from photutils.background import Background2D, MedianBackground
from photutils.isophote import Ellipse, EllipseGeometry
from photutils.segmentation import deblend_sources, detect_sources, detect_threshold
from scipy.optimize import curve_fit

PIX_SCALE_DEFAULT = 0.262  # arcsec / pixel for Legacy Survey coadds
AB_ZP_NM = 22.5  # nanomaggies -> AB mag zeropoint


@dataclass
class FitResult1D:
    model: str
    params: dict
    chi2: float
    aic: float
    bic: float
    n_points: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Surface-photometry workflow for Legacy Survey host galaxy images."
    )
    p.add_argument("--image", required=True, help="Science image FITS")
    p.add_argument("--invvar", required=True, help="Inverse-variance FITS")
    p.add_argument("--model", default=None, help="Optional Legacy Survey model FITS")

    p.add_argument("--outprefix", default="host_profile", help="Output prefix")

    p.add_argument(
        "--x0", type=float, default=None, help="Galaxy x center in full image pixels"
    )
    p.add_argument(
        "--y0", type=float, default=None, help="Galaxy y center in full image pixels"
    )
    p.add_argument("--size", type=int, default=1200, help="Cutout size in pixels")
    p.add_argument(
        "--mask-region",
        action="append",
        default=[],
        help=(
            "Additional DS9 region file to use as a mask. "
            "May be given multiple times. "
            "Supported shapes: circle, ellipse, box, polygon. "
            "Save the region file in IMAGE coordinates."
        ),
    )
    p.add_argument(
        "--pixscale", type=float, default=PIX_SCALE_DEFAULT, help="Arcsec/pixel"
    )
    p.add_argument(
        "--psf-fwhm-pix",
        type=float,
        default=None,
        help=(
            "PSF FWHM in pixels for deciding how far inward the 1D profile fit should go. "
            "If omitted, it will be derived from --psfsize-map if provided."
        ),
    )

    p.add_argument(
        "--psfsize-map",
        default=None,
        help=(
            "Optional Legacy Survey psfsize FITS file (e.g. "
            "legacysurvey-0470m607-psfsize-r.fits.fz). "
            "If provided, a local PSF FWHM will be derived from this map unless "
            "--psf-fwhm-pix is explicitly set."
        ),
    )

    p.add_argument(
        "--psf-box",
        type=int,
        default=25,
        help="Box size in pixels for estimating the local PSF FWHM from --psfsize-map.",
    )
    p.add_argument(
        "--sma0", type=float, default=10.0, help="Initial semi-major axis for isophotes"
    )
    p.add_argument("--eps0", type=float, default=0.3, help="Initial ellipticity guess")
    p.add_argument("--pa0", type=float, default=0.0, help="Initial PA guess in degrees")
    p.add_argument(
        "--maxsma", type=float, default=500.0, help="Maximum semi-major axis in pixels"
    )

    p.add_argument(
        "--seg-nsigma", type=float, default=2.5, help="Detection threshold sigma"
    )
    p.add_argument(
        "--seg-npixels",
        type=int,
        default=20,
        help="Minimum segmentation area in pixels",
    )
    p.add_argument("--box-size", type=int, default=64, help="Background box size")
    p.add_argument("--filter-size", type=int, default=5, help="Background filter size")
    p.add_argument(
        "--mask-model-resid",
        action="store_true",
        help="Compute image-model and segment it as an auxiliary masking image",
    )
    p.add_argument(
        "--model-resid-nsigma",
        type=float,
        default=3.0,
        help="Threshold sigma for segmentation on the internally computed image-model residual",
    )
    p.add_argument(
        "--use-sector-profile",
        action="store_true",
        help="Use a sector-based elliptical-annulus profile for fitting instead of the full isophote intensity profile.",
    )

    p.add_argument(
        "--sector-mode",
        default="minor_both",
        choices=[
            "minor_both",
            "minor_plus",
            "minor_minus",
            "major_both",
            "major_plus",
            "major_minus",
        ],
        help="Which angular sectors to use for the sector-based profile.",
    )

    p.add_argument(
        "--sector-half-width-deg",
        type=float,
        default=25.0,
        help="Half-width of each sector in degrees.",
    )

    p.add_argument(
        "--sector-bin-width-pix",
        type=float,
        default=3.0,
        help="Elliptical-radius bin width in pixels for the sector profile.",
    )

    p.add_argument(
        "--sector-rmin-pix",
        type=float,
        default=None,
        help="Minimum elliptical radius for the sector profile. Defaults to max(2, sma0/2).",
    )

    p.add_argument(
        "--sector-rmax-pix",
        type=float,
        default=None,
        help="Maximum elliptical radius for the sector profile. Defaults to maxsma.",
    )

    p.add_argument(
        "--sector-min-pixels",
        type=int,
        default=30,
        help="Minimum number of unmasked pixels required in a sector annulus bin.",
    )

    p.add_argument(
        "--profile-q",
        type=float,
        default=None,
        help="Override axis ratio q=b/a used for the sector profile.",
    )

    p.add_argument(
        "--profile-pa",
        type=float,
        default=None,
        help="Override major-axis PA in degrees (image-frame convention) used for the sector profile.",
    )
    return p.parse_args()


def find_first_image_hdu(path: str) -> Tuple[np.ndarray, fits.Header, int]:
    with fits.open(path) as hdul:
        for idx, hdu in enumerate(hdul):
            data = getattr(hdu, "data", None)
            if data is not None and np.ndim(data) >= 2:
                return np.asarray(data, dtype=float), hdu.header, idx
    raise ValueError(f"No image HDU found in {path}")


def read_fits_image(path: str) -> Tuple[np.ndarray, fits.Header]:
    data, hdr, _ = find_first_image_hdu(path)
    if data.ndim > 2:
        # keep first plane if unexpectedly 3D
        data = data[0]
    return data.astype(float), hdr


def make_cutout(data, header, x0, y0, size):
    try:
        wcs = WCS(header)
    except Exception:
        wcs = None

    cut = Cutout2D(data, position=(x0, y0), size=(size, size), wcs=wcs, mode="trim")

    # origin of the cutout in the original image, 0-based numpy indices
    x_origin, y_origin = cut.origin_original

    return cut.data.copy(), cut.wcs, x_origin, y_origin


def robust_center_default(data: np.ndarray) -> Tuple[float, float]:
    ny, nx = data.shape
    return 0.5 * (nx - 1), 0.5 * (ny - 1)


def sigma_clip_stats(data: np.ndarray, mask: Optional[np.ndarray] = None):
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, maxiters=10, mask=mask)
    return float(mean), float(median), float(std)


def estimate_background(
    data: np.ndarray,
    mask: Optional[np.ndarray] = None,
    box_size: int = 64,
    filter_size: int = 5,
):
    sigma_clip = SigmaClip(sigma=3.0, maxiters=10)
    bkg = Background2D(
        data,
        box_size=box_size,
        filter_size=(filter_size, filter_size),
        sigma_clip=sigma_clip,
        bkg_estimator=MedianBackground(),
        mask=mask,
    )
    return bkg.background, float(bkg.background_rms_median)


def build_segmentation_mask(
    data_sub: np.ndarray, nsigma: float = 2.5, npixels: int = 20
):
    threshold = detect_threshold(data_sub, nsigma=nsigma)
    segm = detect_sources(data_sub, threshold, npixels=npixels)
    if segm is None:
        return np.zeros_like(data_sub, dtype=bool), None

    try:
        segm = deblend_sources(
            data_sub,
            segm,
            npixels=npixels,
            nlevels=32,
            contrast=0.001,
            progress_bar=False,
        )
    except Exception:
        # deblending can fail on pathological inputs; keep original segmentation
        pass

    return segm.data > 0, segm


def mask_companions_from_segm(segm, x0: float, y0: float):
    if segm is None:
        return np.zeros(segm.shape, dtype=bool) if hasattr(segm, "shape") else None

    labels = segm.data
    x0i = int(np.clip(round(x0), 0, labels.shape[1] - 1))
    y0i = int(np.clip(round(y0), 0, labels.shape[0] - 1))
    host_label = labels[y0i, x0i]

    if host_label > 0:
        return (labels > 0) & (labels != host_label)
    return labels > 0


def mu_from_nanomaggies_per_pix(I_pix: np.ndarray, pixscale: float) -> np.ndarray:
    I_arcsec2 = I_pix / (pixscale**2)
    mu = np.full_like(I_arcsec2, np.nan, dtype=float)
    good = I_arcsec2 > 0
    mu[good] = AB_ZP_NM - 2.5 * np.log10(I_arcsec2[good])
    return mu


def sersic_I(R: np.ndarray, Ie: float, Re: float, n: float) -> np.ndarray:
    n_safe = np.maximum(n, 1e-4)
    bn = 2.0 * n_safe - 1.0 / 3.0 + 0.009876 / n_safe
    return Ie * np.exp(-bn * ((R / np.maximum(Re, 1e-6)) ** (1.0 / n_safe) - 1.0))


def sersic_plus_exp_I(
    R: np.ndarray, Ie: float, Re: float, n: float, I0: float, h: float
) -> np.ndarray:
    return sersic_I(R, Ie, Re, n) + I0 * np.exp(-R / np.maximum(h, 1e-6))


def fit_isophotes(
    data_sub, mask, x0, y0, sma0=20.0, eps0=0.25, pa0_deg=0.0, maxsma=250
):
    data_ma = np.ma.array(data_sub, mask=mask, copy=False)

    geom = EllipseGeometry(
        x0=x0,
        y0=y0,
        sma=sma0,
        eps=eps0,
        pa=np.deg2rad(pa0_deg),
    )

    # Keep centerer off; rely on supplied center.
    ellipse = Ellipse(data_ma, geometry=geom, threshold=1e6)

    isolist = ellipse.fit_image(
        sma0=sma0,
        minsma=max(2.0, sma0 / 2.0),
        maxsma=maxsma,
        step=0.15,
        conver=0.05,
        minit=10,
        maxit=80,
        fflag=0.4,
        maxgerr=1.0,
        sclip=2.5,
        nclip=0,
        fix_center=True,
        fix_pa=False,
        fix_eps=False,
    )
    return isolist


def safe_curve_fit(func, x, y, sigma, p0, bounds, maxfev=50000):
    popt, pcov = curve_fit(
        func,
        x,
        y,
        sigma=sigma,
        p0=p0,
        bounds=bounds,
        absolute_sigma=True,
        maxfev=maxfev,
    )
    return popt, pcov


def fit_1d_models(
    sma: np.ndarray, intens: np.ndarray, intens_err: np.ndarray, psf_fwhm_pix: float
):
    good = np.isfinite(sma) & np.isfinite(intens) & np.isfinite(intens_err)
    good &= (intens > 0) & (intens_err > 0)
    good &= sma > (2.0 * psf_fwhm_pix)

    x = sma[good]
    y = intens[good]
    sig = intens_err[good]

    if len(x) < 8:
        raise RuntimeError("Too few valid profile points to fit.")

    # Single Sérsic
    p0_1 = [np.nanmedian(y), np.nanmedian(x), 4.0]
    bounds_1 = ([1e-8, 1.0, 0.3], [1e8, 1e5, 12.0])
    popt1, pcov1 = safe_curve_fit(sersic_I, x, y, sig, p0_1, bounds_1, maxfev=60000)

    # Sérsic + exponential
    p0_2 = [popt1[0], popt1[1], popt1[2], 0.1 * popt1[0], 2.0 * popt1[1]]
    bounds_2 = ([1e-8, 1.0, 0.3, 1e-8, 1.0], [1e8, 1e5, 12.0, 1e8, 1e5])
    popt2, pcov2 = safe_curve_fit(
        sersic_plus_exp_I, x, y, sig, p0_2, bounds_2, maxfev=100000
    )

    def stats(model_y, k):
        resid = (y - model_y) / sig
        chi2 = float(np.sum(resid**2))
        npts = len(y)
        aic = chi2 + 2 * k
        bic = chi2 + k * np.log(npts)
        return chi2, aic, bic, npts

    chi1, aic1, bic1, npts1 = stats(sersic_I(x, *popt1), 3)
    chi2, aic2, bic2, npts2 = stats(sersic_plus_exp_I(x, *popt2), 5)

    result1 = FitResult1D(
        model="single_sersic",
        params={"Ie": float(popt1[0]), "Re_pix": float(popt1[1]), "n": float(popt1[2])},
        chi2=chi1,
        aic=aic1,
        bic=bic1,
        n_points=npts1,
    )
    result2 = FitResult1D(
        model="sersic_plus_exponential",
        params={
            "Ie": float(popt2[0]),
            "Re_pix": float(popt2[1]),
            "n": float(popt2[2]),
            "I0": float(popt2[3]),
            "h_pix": float(popt2[4]),
        },
        chi2=chi2,
        aic=aic2,
        bic=bic2,
        n_points=npts2,
    )

    return good, result1, result2


def make_2d_sersic_model(shape, x0, y0, q, pa_deg, Ie, Re, n):
    yy, xx = np.indices(shape)
    pa = np.deg2rad(pa_deg)
    dx = xx - x0
    dy = yy - y0
    xp = dx * np.cos(pa) + dy * np.sin(pa)
    yp = -dx * np.sin(pa) + dy * np.cos(pa)
    R = np.sqrt(xp**2 + (yp / max(q, 1e-3)) ** 2)
    return sersic_I(R, Ie, Re, n)


def compute_rff(
    data_sub: np.ndarray,
    model: np.ndarray,
    invvar: np.ndarray,
    x0: float,
    y0: float,
    a: float,
    q: float,
    pa_deg: float,
    extra_mask: Optional[np.ndarray] = None,
) -> float:
    b = a * q
    aper = EllipticalAperture((x0, y0), a=a, b=b, theta=np.deg2rad(pa_deg))
    aper_img = aper.to_mask(method="center").to_image(data_sub.shape)
    aper_mask = aper_img > 0

    sigma = np.zeros_like(data_sub, dtype=float)
    good_iv = invvar > 0
    sigma[good_iv] = 1.0 / np.sqrt(invvar[good_iv])

    use = aper_mask & good_iv & np.isfinite(data_sub) & np.isfinite(model)
    if extra_mask is not None:
        use &= ~extra_mask

    if not np.any(use):
        return float("nan")

    resid = data_sub - model
    num = np.sum(np.abs(resid[use])) - 0.8 * np.sum(sigma[use])
    den = np.sum(data_sub[use])

    return float(num / den) if den != 0 else float("nan")


def fit_single_sersic_under_sky_shift(sma, intens, intens_err, psf_fwhm_pix, delta_sky):
    shifted = intens - delta_sky
    good = np.isfinite(sma) & np.isfinite(shifted) & np.isfinite(intens_err)
    good &= (shifted > 0) & (intens_err > 0)
    good &= sma > (2.0 * psf_fwhm_pix)

    x = sma[good]
    y = shifted[good]
    sig = intens_err[good]

    if len(x) < 8:
        return None

    p0 = [np.nanmedian(y), np.nanmedian(x), 4.0]
    bounds = ([1e-8, 1.0, 0.3], [1e8, 1e5, 12.0])

    try:
        popt, _ = safe_curve_fit(sersic_I, x, y, sig, p0, bounds, maxfev=60000)
        model = sersic_I(x, *popt)
        resid = (y - model) / sig
        chi2 = float(np.sum(resid**2))
        bic = chi2 + 3 * np.log(len(x))
        return {
            "Ie": float(popt[0]),
            "Re_pix": float(popt[1]),
            "n": float(popt[2]),
            "chi2": chi2,
            "bic": bic,
            "n_points": int(len(x)),
            "delta_sky": float(delta_sky),
        }
    except Exception:
        return None


def fit_sersic_plus_exp_under_sky_shift(
    sma, intens, intens_err, psf_fwhm_pix, delta_sky
):
    shifted = intens - delta_sky
    good = np.isfinite(sma) & np.isfinite(shifted) & np.isfinite(intens_err)
    good &= (shifted > 0) & (intens_err > 0)
    good &= sma > (2.0 * psf_fwhm_pix)

    x = sma[good]
    y = shifted[good]
    sig = intens_err[good]

    if len(x) < 8:
        return None

    # First get a single-Sersic guess to seed the 2-component fit
    p0_1 = [np.nanmedian(y), np.nanmedian(x), 4.0]
    bounds_1 = ([1e-8, 1.0, 0.3], [1e8, 1e5, 12.0])

    try:
        popt1, _ = safe_curve_fit(sersic_I, x, y, sig, p0_1, bounds_1, maxfev=60000)

        p0_2 = [popt1[0], popt1[1], popt1[2], 0.1 * popt1[0], 2.0 * popt1[1]]
        bounds_2 = ([1e-8, 1.0, 0.3, 1e-8, 1.0], [1e8, 1e5, 12.0, 1e8, 1e5])

        popt2, _ = safe_curve_fit(
            sersic_plus_exp_I, x, y, sig, p0_2, bounds_2, maxfev=100000
        )

        model = sersic_plus_exp_I(x, *popt2)
        resid = (y - model) / sig
        chi2 = float(np.sum(resid**2))
        bic = chi2 + 5 * np.log(len(x))

        return {
            "Ie": float(popt2[0]),
            "Re_pix": float(popt2[1]),
            "n": float(popt2[2]),
            "I0": float(popt2[3]),
            "h_pix": float(popt2[4]),
            "chi2": chi2,
            "bic": bic,
            "n_points": int(len(x)),
            "delta_sky": float(delta_sky),
        }

    except Exception:
        return None


def save_profile_csv(path: str, sma, intens, intens_err, mu, eps, pa_deg):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "sma_pix",
                "intens_nMgy_per_pix",
                "int_err_nMgy_per_pix",
                "mu_mag_arcsec2",
                "ellipticity",
                "pa_deg",
            ]
        )
        for row in zip(sma, intens, intens_err, mu, eps, pa_deg):
            w.writerow(row)


def save_json(path: str, payload: dict):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def plot_mask(path: str, image: np.ndarray, mask: np.ndarray):
    v1, v2 = np.nanpercentile(image[np.isfinite(image)], [5, 99])
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(image, origin="lower", cmap="gray", vmin=v1, vmax=v2)
    overlay = np.zeros((*mask.shape, 4), dtype=float)
    overlay[..., 0] = 1.0
    overlay[..., 3] = mask.astype(float) * 0.35
    ax.imshow(overlay, origin="lower")
    ax.set_title("Mask overlay")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_sb_profile(
    path: str,
    sma,
    mu,
    fitmask,
    result1: FitResult1D,
    result2: FitResult1D,
    pixscale: float,
):
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(sma, mu, "ko", ms=3, label="Isophotes")

    rr = np.linspace(np.nanmin(sma[fitmask]), np.nanmax(sma[fitmask]), 600)
    mu1 = mu_from_nanomaggies_per_pix(
        sersic_I(
            rr, result1.params["Ie"], result1.params["Re_pix"], result1.params["n"]
        ),
        pixscale,
    )
    mu2 = mu_from_nanomaggies_per_pix(
        sersic_plus_exp_I(
            rr,
            result2.params["Ie"],
            result2.params["Re_pix"],
            result2.params["n"],
            result2.params["I0"],
            result2.params["h_pix"],
        ),
        pixscale,
    )

    ax.plot(rr, mu1, label=f"Single Sersic (BIC={result1.bic:.1f})")
    ax.plot(rr, mu2, label=f"Sersic+exp (BIC={result2.bic:.1f})")

    ax.invert_yaxis()
    ax.set_xlabel("Semimajor axis [pix]")
    ax.set_ylabel(r"$\mu_r$ [mag arcsec$^{-2}$]")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_residual(path: str, resid: np.ndarray):
    p = np.nanpercentile(resid[np.isfinite(resid)], [2, 98])
    vmax = max(abs(p[0]), abs(p[1]))
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(resid, origin="lower", cmap="gray", vmin=-vmax, vmax=vmax)
    ax.set_title("Residual: data - single Sersic")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_cutout_fits(path: str, data: np.ndarray, wcs: Optional[WCS] = None):
    hdr = wcs.to_header() if wcs is not None else fits.Header()
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def save_fits(path: str, data: np.ndarray, wcs: Optional[WCS] = None):
    hdr = wcs.to_header() if wcs is not None else fits.Header()
    fits.PrimaryHDU(data=data, header=hdr).writeto(path, overwrite=True)


def _split_ds9_fields(s: str):
    return [x for x in re.split(r"[,\s]+", s.strip()) if x]


def _parse_ds9_region_line(line: str):
    """
    Parse a single DS9 region specification line.
    Supports: circle, ellipse, box, polygon
    Assumes IMAGE coordinates.

    Returns
    -------
    tuple or None
        (shape_name, params_list) or None
    """
    # Remove comments after '#'
    line = line.split("#", 1)[0].strip()
    if not line:
        return None

    # Ignore common metadata lines
    low = line.lower()
    if low.startswith("global"):
        return None
    if low in {"image", "physical"}:
        return ("coord", [low])
    if low in {"fk5", "icrs", "wcs", "galactic", "ecliptic"}:
        return ("coord", [low])

    # Some DS9 files may have multiple region specs separated by ';'
    # Those are handled outside this function; this expects one spec only.
    m = re.match(
        r"^\s*([+-]?)(circle|ellipse|box|polygon)\s*\((.*)\)\s*$", line, re.IGNORECASE
    )
    if not m:
        return None

    sign = m.group(1)
    shape = m.group(2).lower()
    params = _split_ds9_fields(m.group(3))

    # We treat all provided regions as mask-inclusion regions.
    # A leading '-' is ignored here intentionally.
    return (shape, params)


def parse_ds9_region_file(path: str):
    """
    Parse a DS9 region file.

    Only IMAGE/PHYSICAL coordinate regions are supported.
    For fk5/wcs/etc, save the region file from DS9 in IMAGE coordinates first.
    """
    regions = []
    coord_sys = "image"

    with open(path, "r") as f:
        for raw in f:
            # Split lines that may contain multiple region statements
            parts = [p.strip() for p in raw.strip().split(";") if p.strip()]
            for part in parts:
                parsed = _parse_ds9_region_line(part)
                if parsed is None:
                    continue

                kind, vals = parsed
                if kind == "coord":
                    coord_sys = vals[0].lower()
                    continue

                if coord_sys not in {"image", "physical"}:
                    raise ValueError(
                        f"Region file {path!r} uses coordinate system {coord_sys!r}. "
                        "Please save/export the DS9 region file in IMAGE coordinates."
                    )

                regions.append((kind, vals))

    return regions


def _mask_circle(shape, x, y, r):
    yy, xx = np.indices(shape)
    # DS9 image coords are 1-based pixel-center coordinates
    xx = xx + 1.0
    yy = yy + 1.0
    return (xx - x) ** 2 + (yy - y) ** 2 <= r**2


def _mask_ellipse(shape, x, y, r1, r2, angle_deg):
    yy, xx = np.indices(shape)
    xx = xx + 1.0
    yy = yy + 1.0

    theta = np.deg2rad(angle_deg)
    dx = xx - x
    dy = yy - y

    xp = dx * np.cos(theta) + dy * np.sin(theta)
    yp = -dx * np.sin(theta) + dy * np.cos(theta)

    return (xp / max(r1, 1e-8)) ** 2 + (yp / max(r2, 1e-8)) ** 2 <= 1.0


def _mask_box(shape, x, y, width, height, angle_deg):
    yy, xx = np.indices(shape)
    xx = xx + 1.0
    yy = yy + 1.0

    theta = np.deg2rad(angle_deg)
    dx = xx - x
    dy = yy - y

    xp = dx * np.cos(theta) + dy * np.sin(theta)
    yp = -dx * np.sin(theta) + dy * np.cos(theta)

    return (np.abs(xp) <= width / 2.0) & (np.abs(yp) <= height / 2.0)


def _mask_polygon(shape, xy):
    yy, xx = np.indices(shape)
    pts = np.column_stack([(xx + 1.0).ravel(), (yy + 1.0).ravel()])
    path = Path(xy)
    return path.contains_points(pts).reshape(shape)


def ds9_regions_to_mask(region_file: str, image_shape, x_offset=0.0, y_offset=0.0):
    """
    Build a boolean mask from a DS9 region file.

    x_offset, y_offset are the cutout origin in the ORIGINAL image, in 0-based
    numpy pixel coordinates. For DS9 IMAGE coordinates drawn on the original
    image, local cutout coordinates are:
        x_local = x_ds9 - x_offset
        y_local = y_ds9 - y_offset
    """
    regions = parse_ds9_region_file(region_file)
    mask = np.zeros(image_shape, dtype=bool)

    for shape_name, params in regions:
        try:
            if shape_name == "circle":
                if len(params) < 3:
                    continue
                x, y, r = map(float, params[:3])
                x -= x_offset
                y -= y_offset
                mask |= _mask_circle(image_shape, x, y, r)

            elif shape_name == "ellipse":
                if len(params) < 5:
                    continue
                x, y, r1, r2, ang = map(float, params[:5])
                x -= x_offset
                y -= y_offset
                mask |= _mask_ellipse(image_shape, x, y, r1, r2, ang)

            elif shape_name == "box":
                if len(params) < 5:
                    continue
                x, y, w, h, ang = map(float, params[:5])
                x -= x_offset
                y -= y_offset
                mask |= _mask_box(image_shape, x, y, w, h, ang)

            elif shape_name == "polygon":
                if len(params) < 6 or (len(params) % 2 != 0):
                    continue
                vals = list(map(float, params))
                xy = np.array(vals, dtype=float).reshape(-1, 2)
                xy[:, 0] -= x_offset
                xy[:, 1] -= y_offset
                mask |= _mask_polygon(image_shape, xy)

        except Exception as e:
            print(
                f"Warning: failed to parse region in {region_file}: {shape_name}({params}) -> {e}"
            )

    return mask


def plot_boolean_mask(path: str, mask: np.ndarray, title="Boolean mask"):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(mask.astype(float), origin="lower", cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def estimate_psf_fwhm_pix_from_psfmap(
    psfsize_fits: str,
    x0: float,
    y0: float,
    box: int = 25,
    pixscale: float = PIX_SCALE_DEFAULT,
):
    """
    Estimate the local PSF FWHM in pixels from a Legacy Survey psfsize map.

    Parameters
    ----------
    psfsize_fits : str
        Path to legacysurvey-<brick>-psfsize-r.fits.fz
    x0, y0 : float
        Galaxy center in full-image pixel coordinates
    box : int
        Size of square cutout used to compute a local median
    pixscale : float
        Arcsec/pixel

    Returns
    -------
    psf_fwhm_pix : float
    psf_fwhm_arcsec : float
    """
    psf_arcsec_full, psf_hdr = read_fits_image(psfsize_fits)

    # Local cutout around the host center in the FULL image frame
    psf_cut, _wcs, _xo, _yo = make_cutout(psf_arcsec_full, psf_hdr, x0, y0, box)

    good = np.isfinite(psf_cut) & (psf_cut > 0)
    if not np.any(good):
        raise ValueError(
            f"No valid PSF values found near ({x0}, {y0}) in {psfsize_fits}"
        )

    psf_fwhm_arcsec = float(np.nanmedian(psf_cut[good]))
    psf_fwhm_pix = psf_fwhm_arcsec / pixscale
    return psf_fwhm_pix, psf_fwhm_arcsec


def _angle_diff_deg(a, b):
    """
    Smallest absolute difference between two angles in degrees,
    accounting for wraparound.
    """
    d = (a - b + 180.0) % 360.0 - 180.0
    return np.abs(d)


def _sector_selector(theta_deg, mode, half_width_deg):
    """
    theta_deg is measured in the rotated major-axis frame:
      0 deg   = +major axis
      90 deg  = +minor axis
      180/-180 = -major axis
      -90 deg = -minor axis
    """
    if mode == "major_plus":
        return _angle_diff_deg(theta_deg, 0.0) <= half_width_deg
    elif mode == "major_minus":
        return _angle_diff_deg(theta_deg, 180.0) <= half_width_deg
    elif mode == "major_both":
        return (_angle_diff_deg(theta_deg, 0.0) <= half_width_deg) | (
            _angle_diff_deg(theta_deg, 180.0) <= half_width_deg
        )
    elif mode == "minor_plus":
        return _angle_diff_deg(theta_deg, 90.0) <= half_width_deg
    elif mode == "minor_minus":
        return _angle_diff_deg(theta_deg, -90.0) <= half_width_deg
    elif mode == "minor_both":
        return (_angle_diff_deg(theta_deg, 90.0) <= half_width_deg) | (
            _angle_diff_deg(theta_deg, -90.0) <= half_width_deg
        )
    else:
        raise ValueError(f"Unknown sector mode: {mode}")


def build_sector_profile(
    data_sub,
    invvar,
    mask,
    x0,
    y0,
    q,
    pa_deg,
    r_edges,
    sector_mode="minor_both",
    half_width_deg=25.0,
    min_pixels=30,
):
    """
    Build a 1D profile from elliptical annular sectors.

    Parameters
    ----------
    data_sub : 2D array
        Sky-subtracted image.
    invvar : 2D array
        Inverse-variance map.
    mask : 2D bool array
        Mask of bad pixels / companions / tails.
    x0, y0 : float
        Host center in cutout coordinates.
    q : float
        Axis ratio b/a.
    pa_deg : float
        Major-axis PA in image coordinates (same convention as photutils fit).
    r_edges : 1D array
        Bin edges in elliptical semimajor-axis radius.
    sector_mode : str
        Which sectors to use.
    half_width_deg : float
        Half-opening angle of each sector.
    min_pixels : int
        Minimum number of valid pixels needed in a bin.

    Returns
    -------
    sma_mid, intens, intens_err, npix : arrays
    """
    yy, xx = np.indices(data_sub.shape)

    dx = xx - x0
    dy = yy - y0

    pa = np.deg2rad(pa_deg)

    # Rotate into the host major-axis frame
    xp = dx * np.cos(pa) + dy * np.sin(pa)
    yp = -dx * np.sin(pa) + dy * np.cos(pa)

    # Elliptical radius in semimajor-axis units
    q_safe = max(float(q), 1e-3)
    r_ell = np.sqrt(xp**2 + (yp / q_safe) ** 2)

    # Polar angle in the rotated frame
    theta_deg = np.rad2deg(np.arctan2(yp, xp))

    sector_mask = _sector_selector(theta_deg, sector_mode, half_width_deg)

    valid = (
        np.isfinite(data_sub)
        & np.isfinite(invvar)
        & (invvar > 0)
        & (~mask)
        & sector_mask
    )

    sma_mid = []
    intens = []
    intens_err = []
    npix_used = []

    for r0, r1 in zip(r_edges[:-1], r_edges[1:]):
        ann = (r_ell >= r0) & (r_ell < r1) & valid
        n = int(np.count_nonzero(ann))

        if n < min_pixels:
            continue

        vals = data_sub[ann]

        # robust clip inside the bin
        med = np.nanmedian(vals)
        mad = np.nanmedian(np.abs(vals - med))
        sigma_rob = 1.4826 * mad if np.isfinite(mad) else np.nan

        if np.isfinite(sigma_rob) and sigma_rob > 0:
            keep = np.abs(vals - med) <= 3.0 * sigma_rob
            vals = vals[keep]
            ann_idx = np.where(ann)
            # clip weights the same way
            w_all = invvar[ann]
            w = w_all[keep]
        else:
            w = invvar[ann]

        if len(vals) < min_pixels:
            continue

        # weighted mean intensity
        wsum = np.sum(w)
        if wsum <= 0:
            continue

        I = np.sum(w * vals) / wsum

        # use inverse-variance error, but protect against unrealistically tiny values
        Ierr_iv = np.sqrt(1.0 / wsum)

        # also estimate empirical scatter / sqrt(N)
        if len(vals) > 1:
            s = np.nanstd(vals, ddof=1)
            Ierr_emp = s / np.sqrt(len(vals))
            Ierr = max(Ierr_iv, Ierr_emp)
        else:
            Ierr = Ierr_iv

        sma_mid.append(0.5 * (r0 + r1))
        intens.append(I)
        intens_err.append(Ierr)
        npix_used.append(len(vals))

    return (
        np.asarray(sma_mid, dtype=float),
        np.asarray(intens, dtype=float),
        np.asarray(intens_err, dtype=float),
        np.asarray(npix_used, dtype=int),
    )


def derive_profile_geometry_from_isophotes(
    sma,
    eps,
    pa_deg,
    sma_min=None,
):
    """
    Derive a stable default geometry (q, PA) from the raw isophote solution,
    without relying on the later 1D profile fit mask.

    Parameters
    ----------
    sma : array
        Isophote semimajor axis values in pixels.
    eps : array
        Isophote ellipticities.
    pa_deg : array
        Isophote position angles in degrees (image-frame convention).
    sma_min : float or None
        Use only isophotes with sma >= sma_min. If None, use all finite values.

    Returns
    -------
    q_med : float
        Median axis ratio b/a.
    pa_med : float
        Median PA in degrees, using circular statistics modulo 180 deg.
    """
    good = np.isfinite(sma) & np.isfinite(eps) & np.isfinite(pa_deg)
    good &= (eps >= 0.0) & (eps < 1.0)

    if sma_min is not None:
        good &= sma >= sma_min

    if np.count_nonzero(good) < 3:
        # fallback to something reasonable
        q_med = 0.7
        pa_med = 0.0
        return q_med, pa_med

    q_vals = 1.0 - eps[good]
    q_med = float(np.nanmedian(q_vals))

    # PA is 180-degree periodic, so use doubled-angle circular mean
    pa_rad = np.deg2rad(pa_deg[good])
    c = np.nanmean(np.cos(2.0 * pa_rad))
    s = np.nanmean(np.sin(2.0 * pa_rad))
    pa_med = 0.5 * np.rad2deg(np.arctan2(s, c))
    pa_med = pa_med % 180.0

    return q_med, pa_med


def main():
    args = parse_args()

    img_full, img_hdr = read_fits_image(args.image)
    iv_full, iv_hdr = read_fits_image(args.invvar)

    if img_full.shape != iv_full.shape:
        raise ValueError("Image and invvar shapes do not match.")

    model_full = None
    model_hdr = None
    if args.model is not None:
        model_full, model_hdr = read_fits_image(args.model)
        if model_full.shape != img_full.shape:
            print(
                "Warning: model image shape differs from science image; ignoring model input."
            )
            model_full = None
            model_hdr = None

    if args.x0 is None or args.y0 is None:
        x0_full, y0_full = robust_center_default(img_full)
    else:
        x0_full, y0_full = args.x0, args.y0

    # Decide which PSF FWHM to use for the 1D profile fit cutoff.
    psf_fwhm_pix_used = None
    psf_fwhm_arcsec_used = None
    psf_source = None

    if args.psf_fwhm_pix is not None:
        psf_fwhm_pix_used = float(args.psf_fwhm_pix)
        psf_fwhm_arcsec_used = psf_fwhm_pix_used * args.pixscale
        psf_source = "command_line"

    elif args.psfsize_map is not None:
        try:
            psf_fwhm_pix_used, psf_fwhm_arcsec_used = estimate_psf_fwhm_pix_from_psfmap(
                args.psfsize_map,
                x0_full,
                y0_full,
                box=args.psf_box,
                pixscale=args.pixscale,
            )
            psf_source = "psfsize_map"
        except Exception as e:
            print(f"Warning: could not derive PSF from {args.psfsize_map!r}: {e}")
            psf_fwhm_pix_used = 5.0
            psf_fwhm_arcsec_used = psf_fwhm_pix_used * args.pixscale
            psf_source = "fallback_default_after_psfsize_failure"

    else:
        psf_fwhm_pix_used = 5.0
        psf_fwhm_arcsec_used = psf_fwhm_pix_used * args.pixscale
        psf_source = "fallback_default"

    print(
        f"Using PSF FWHM = {psf_fwhm_pix_used:.3f} pix "
        f"({psf_fwhm_arcsec_used:.3f} arcsec), source = {psf_source}"
    )

    img_cut, wcs_cut, x_origin, y_origin = make_cutout(
        img_full, img_hdr, x0_full, y0_full, args.size
    )
    iv_cut, _, _, _ = make_cutout(iv_full, iv_hdr, x0_full, y0_full, args.size)
    model_cut = None
    legacy_resid_cut = None
    if model_full is not None:
        model_cut, _, _, _ = make_cutout(
            model_full, model_hdr, x0_full, y0_full, args.size
        )
        legacy_resid_cut = img_cut - model_cut

    if legacy_resid_cut is not None:
        save_fits(
            f"{args.outprefix}_legacy_model_residual.fits", legacy_resid_cut, wcs_cut
        )
        plot_residual(f"{args.outprefix}_legacy_model_residual.png", legacy_resid_cut)

    # center in cutout coordinates
    x0, y0 = robust_center_default(img_cut)

    # First-pass mask: invalid invvar
    mask = ~np.isfinite(img_cut) | ~np.isfinite(iv_cut) | (iv_cut <= 0)

    # Additional DS9 region masks
    if args.mask_region:
        for regfile in args.mask_region:
            try:
                reg_mask = ds9_regions_to_mask(
                    regfile,
                    img_cut.shape,
                    x_offset=x_origin,
                    y_offset=y_origin,
                )
                mask |= reg_mask
                print(f"Applied DS9 mask region file: {regfile}")
                print(
                    f"  Masked pixels from this region file: {int(np.count_nonzero(reg_mask))}"
                )
            except Exception as e:
                print(f"Warning: could not apply DS9 mask region file {regfile!r}: {e}")

    if args.mask_region:
        region_only = np.zeros_like(mask, dtype=bool)
        for regfile in args.mask_region:
            reg_mask = ds9_regions_to_mask(
                regfile,
                img_cut.shape,
                x_offset=x_origin,
                y_offset=y_origin,
            )
            region_only |= reg_mask

        plot_boolean_mask(
            f"{args.outprefix}_ds9_region_mask.png",
            region_only,
            title="DS9 region mask only",
        )

    # First-pass sky estimate
    _, sky0, _ = sigma_clip_stats(img_cut, mask=mask)
    img_sub0 = img_cut - sky0

    # Segmentation on first-pass sky-subtracted image
    _, segm = build_segmentation_mask(
        img_sub0, nsigma=args.seg_nsigma, npixels=args.seg_npixels
    )
    comp_mask = (
        mask_companions_from_segm(segm, x0, y0)
        if segm is not None
        else np.zeros_like(mask)
    )

    mask |= comp_mask

    # Optional segmentation on supplied residual image
    if args.mask_model_resid and legacy_resid_cut is not None:
        _, segm_resid = build_segmentation_mask(
            np.abs(legacy_resid_cut),
            nsigma=args.model_resid_nsigma,
            npixels=max(10, args.seg_npixels // 2),
        )
        if segm_resid is not None:
            mask_resid = mask_companions_from_segm(segm_resid, x0, y0)
            mask |= mask_resid

    # Refined background estimate
    sky2d, sky_rms = estimate_background(
        img_cut,
        mask=mask,
        box_size=args.box_size,
        filter_size=args.filter_size,
    )
    img_sub = img_cut - sky2d

    # Isophotes
    isolist = fit_isophotes(
        img_sub,
        mask,
        x0=x0,
        y0=y0,
        sma0=args.sma0,
        eps0=args.eps0,
        pa0_deg=args.pa0,
        maxsma=args.maxsma,
    )
    sma = np.asarray(isolist.sma, dtype=float)
    intens = np.asarray(isolist.intens, dtype=float)
    intens_err = np.asarray(isolist.int_err, dtype=float)
    eps = np.asarray(isolist.eps, dtype=float)
    pa_deg = np.rad2deg(np.asarray(isolist.pa, dtype=float))

    # Derive a stable default geometry directly from the raw isophote solution,
    # without depending on the later 1D fit mask.
    geom_sma_min = max(args.sma0, 2.0 * psf_fwhm_pix_used)
    q_default, pa_default = derive_profile_geometry_from_isophotes(
        sma, eps, pa_deg, sma_min=geom_sma_min
    )

    # Allow manual override if desired
    profile_q = float(args.profile_q) if args.profile_q is not None else q_default
    profile_pa = float(args.profile_pa) if args.profile_pa is not None else pa_default

    # Decide which 1D profile to fit
    if args.use_sector_profile:
        rmin = (
            args.sector_rmin_pix
            if args.sector_rmin_pix is not None
            else max(2.0, args.sma0 / 2.0)
        )
        rmax = args.sector_rmax_pix if args.sector_rmax_pix is not None else args.maxsma

        r_edges = np.arange(
            rmin,
            rmax + args.sector_bin_width_pix,
            args.sector_bin_width_pix,
        )

        sma_used, intens_used, intens_err_used, sector_npix = build_sector_profile(
            img_sub,
            iv_cut,
            mask,
            x0=x0,
            y0=y0,
            q=profile_q,
            pa_deg=profile_pa,
            r_edges=r_edges,
            sector_mode=args.sector_mode,
            half_width_deg=args.sector_half_width_deg,
            min_pixels=args.sector_min_pixels,
        )

        if len(sma_used) < 8:
            raise RuntimeError(
                "Too few valid sector-profile bins to fit. "
                "Try increasing --sector-half-width-deg, decreasing "
                "--sector-min-pixels, or using a larger cutout."
            )
    else:
        sma_used = sma
        intens_used = intens
        intens_err_used = intens_err
        sector_npix = None

    mu = mu_from_nanomaggies_per_pix(intens_used, args.pixscale)

    fitmask, result1, result2 = fit_1d_models(
        sma_used, intens_used, intens_err_used, psf_fwhm_pix=psf_fwhm_pix_used
    )

    # Geometry for the 2D residual model / RFF
    q_med = profile_q
    pa_med = profile_pa

    model1 = make_2d_sersic_model(
        img_sub.shape,
        x0,
        y0,
        q=q_med,
        pa_deg=pa_med,
        Ie=result1.params["Ie"],
        Re=result1.params["Re_pix"],
        n=result1.params["n"],
    )
    resid1 = img_sub - model1

    a_rff = 2.5 * result1.params["Re_pix"]
    rff1 = compute_rff(
        img_sub,
        model1,
        iv_cut,
        x0=x0,
        y0=y0,
        a=a_rff,
        q=q_med,
        pa_deg=pa_med,
        extra_mask=mask,
    )

    # Sky perturbation tests
    sky_nom = fit_single_sersic_under_sky_shift(
        sma_used, intens_used, intens_err_used, psf_fwhm_pix_used, delta_sky=0.0
    )
    sky_plus = fit_single_sersic_under_sky_shift(
        sma_used, intens_used, intens_err_used, psf_fwhm_pix_used, delta_sky=+sky_rms
    )
    sky_minus = fit_single_sersic_under_sky_shift(
        sma_used, intens_used, intens_err_used, psf_fwhm_pix_used, delta_sky=-sky_rms
    )

    sky2_nom = fit_sersic_plus_exp_under_sky_shift(
        sma_used, intens_used, intens_err_used, psf_fwhm_pix_used, delta_sky=0.0
    )
    sky2_plus = fit_sersic_plus_exp_under_sky_shift(
        sma_used, intens_used, intens_err_used, psf_fwhm_pix_used, delta_sky=+sky_rms
    )
    sky2_minus = fit_sersic_plus_exp_under_sky_shift(
        sma_used, intens_used, intens_err_used, psf_fwhm_pix_used, delta_sky=-sky_rms
    )

    # Save outputs
    if args.use_sector_profile:
        # For sector profiles, eps/pa columns are not meaningful per radial bin in the same way,
        # so write NaNs there.
        eps_used = np.full_like(sma_used, np.nan, dtype=float)
        pa_used = np.full_like(sma_used, np.nan, dtype=float)
        save_profile_csv(
            f"{args.outprefix}_profile.csv",
            sma_used,
            intens_used,
            intens_err_used,
            mu,
            eps_used,
            pa_used,
        )
    else:
        save_profile_csv(
            f"{args.outprefix}_profile.csv",
            sma_used,
            intens_used,
            intens_err_used,
            mu,
            eps,
            pa_deg,
        )

    summary = {
        "inputs": {
            "image": args.image,
            "invvar": args.invvar,
            "model": args.model,
            "used_model_residual_for_masking": bool(
                args.mask_model_resid and legacy_resid_cut is not None
            ),
            "psfsize_map": args.psfsize_map,
            "cutout_center_full_image_pix": {"x0": x0_full, "y0": y0_full},
            "cutout_size_pix": args.size,
            "pixscale_arcsec_per_pix": args.pixscale,
            "mask_region_files": args.mask_region,
        },
        "psf": {
            "psf_fwhm_pix_used": psf_fwhm_pix_used,
            "psf_fwhm_arcsec_used": psf_fwhm_arcsec_used,
            "psf_source": psf_source,
            "psfsize_map": args.psfsize_map,
            "psf_box_pix": args.psf_box,
        },
        "profile_extraction": {
            "use_sector_profile": bool(args.use_sector_profile),
            "sector_mode": args.sector_mode if args.use_sector_profile else None,
            "sector_half_width_deg": (
                args.sector_half_width_deg if args.use_sector_profile else None
            ),
            "sector_bin_width_pix": (
                args.sector_bin_width_pix if args.use_sector_profile else None
            ),
            "sector_rmin_pix": (
                args.sector_rmin_pix if args.use_sector_profile else None
            ),
            "sector_rmax_pix": (
                args.sector_rmax_pix if args.use_sector_profile else None
            ),
            "sector_min_pixels": (
                args.sector_min_pixels if args.use_sector_profile else None
            ),
            "profile_q_used": profile_q,
            "profile_pa_used_deg": profile_pa,
            "n_profile_points_used": int(len(sma_used)),
        },
        "background": {
            "background_rms_nMgy_per_pix": sky_rms,
        },
        "single_sersic": asdict(result1),
        "sersic_plus_exponential": asdict(result2),
        "single_sersic_rff": rff1,
        "median_geometry_for_2d_model": {
            "q": q_med,
            "pa_deg": pa_med,
            "q_default_from_isophotes": q_default,
            "pa_default_from_isophotes_deg": pa_default,
            "rff_aperture_a_pix": a_rff,
            "rff_aperture_b_pix": a_rff * q_med,
        },
        "sky_shift_tests_single_sersic": {
            "nominal": sky_nom,
            "plus_1sigma_sky": sky_plus,
            "minus_1sigma_sky": sky_minus,
        },
        "sky_shift_tests_sersic_plus_exponential": {
            "nominal": sky2_nom,
            "plus_1sigma_sky": sky2_plus,
            "minus_1sigma_sky": sky2_minus,
        },
        "interpretation_hints": {
            "lower_bic_is_better": True,
            "robust_non_cd_case": [
                "single-Sersic remains acceptable under ±1 sigma sky shifts",
                "two-component model does not produce a robust outer excess",
                "single-Sersic residual image shows no smooth diffuse large-radius envelope",
                "RFF is modest rather than strongly elevated",
            ],
        },
    }
    save_json(f"{args.outprefix}_fit_summary.json", summary)
    plot_sb_profile(
        f"{args.outprefix}_sb_profile.png",
        sma_used,
        mu,
        fitmask,
        result1,
        result2,
        args.pixscale,
    )
    plot_residual(f"{args.outprefix}_residual_single_sersic.png", resid1)
    plot_mask(f"{args.outprefix}_mask.png", img_sub, mask)
    save_cutout_fits(f"{args.outprefix}_cutout.fits", img_sub, wcs_cut)

    print(f"Wrote: {args.outprefix}_profile.csv")
    print(f"Wrote: {args.outprefix}_fit_summary.json")
    print(f"Wrote: {args.outprefix}_sb_profile.png")
    print(f"Wrote: {args.outprefix}_residual_single_sersic.png")
    print(f"Wrote: {args.outprefix}_mask.png")
    print(f"Wrote: {args.outprefix}_cutout.fits")
    print()
    print("Quick readout:")
    print(f"  Single Sersic BIC        : {result1.bic:.2f}")
    print(f"  Sersic+exponential BIC   : {result2.bic:.2f}")
    print(f"  Single Sersic RFF        : {rff1:.4f}")
    print(f"  Sky RMS                  : {sky_rms:.6g} nMgy/pix")
    print("  Check the mask and residual image before using this for the paper.")


if __name__ == "__main__":
    main()
