#!/usr/bin/env python3
"""
mu_from_legacy.py  —  Sky-subtract Legacy r (nMgy/pix) then convert to mag/arcsec^2,
with optional smoothing in flux units BEFORE the SB conversion.

Outputs a multi-product MEF:
  - DATA_NMGY, MASK, BKG_NMGY, SKY_SUB_NMGY
  - SKY_SUB_SMOOTH_NMGY (if smoothing enabled)
  - MU_OBS, MU_REST (if --z provided)
  - Optional standalone background FITS via --out-background-fits

Requires: astropy, photutils; (optional) scipy for mask dilation fallback.

New:
  - Can use an externally supplied FITS mask via --mask-fits instead of
    computing a source mask internally.
  - For Legacy images, --bin-factor now block-sums the image/mask before
    background estimation, matching a rebin_fits.py --sum preprocessing step.
  - Synthetic images retain native-resolution extinction/noise processing before
    optional block-sum binning.
"""

import argparse
import math
import os
import warnings

import astropy.units as u
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.table import Table
from astropy.time import Time
from astropy.units import UnitsWarning
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.background import Background2D, MedianBackground

warnings.filterwarnings(
    "ignore",
    message=r".*did not parse as fits unit.*",
    category=UnitsWarning,
)

try:
    from regions import Regions
except Exception:
    Regions = None

# ---- photutils compatibility for source masking
try:
    from photutils.segmentation import make_source_mask  # old API
except Exception:
    make_source_mask = None
    try:
        from photutils.segmentation import detect_sources, detect_threshold
    except Exception:
        detect_threshold = None
        detect_sources = None
    try:
        from scipy.ndimage import binary_dilation
    except Exception:
        binary_dilation = None
    from astropy.convolution import Box2DKernel
    from astropy.convolution import convolve as _conv

AB_ZP_NANOMAGGY = 22.5  # AB mag of 1 nanomaggy


# ----------------- helpers -----------------


def pixel_scale_arcsec(header):
    """Return (sx, sy) arcsec/pixel using WCS (handles CD/PC)."""
    w = WCS(header)
    scales_deg = proj_plane_pixel_scales(w) * u.deg
    sx = scales_deg[0].to(u.arcsec).value
    sy = scales_deg[1].to(u.arcsec).value
    if not np.isfinite(sx * sy) or sx <= 0 or sy <= 0:
        sx = abs(header["CDELT1"]) * 3600.0
        sy = abs(header["CDELT2"]) * 3600.0
    if sx <= 0 or sy <= 0:
        raise RuntimeError("Could not determine a positive pixel scale.")
    return float(sx), float(sy)


def pixel_area_arcsec2(header):
    sx, sy = pixel_scale_arcsec(header)
    return sx * sy


def mu_from_nmgy_arcsec2(nmgy_arcsec2):
    mu = np.full_like(nmgy_arcsec2, np.nan, dtype=float)
    m = nmgy_arcsec2 > 0
    mu[m] = AB_ZP_NANOMAGGY - 2.5 * np.log10(nmgy_arcsec2[m])
    return mu


def k_from_alpha(z, alpha=0.0):
    return -2.5 * (1.0 + alpha) * math.log10(1.0 + z)


def restframe_mu(mu_obs, z, kcorr=None, alpha=None):
    tolman = 10.0 * math.log10(1.0 + z)
    if kcorr is not None:
        K = float(kcorr)
    elif alpha is not None:
        K = k_from_alpha(z, alpha)
    else:
        K = 0.0
    return mu_obs - tolman - K, tolman, K


def _kernel_stats(ker: np.ndarray, pix_area_arcsec2: float):
    """
    Return (neff_px2, area_arcsec2, dmu_mag) for a 2D kernel.
    neff = 1 / sum(w^2) where w is the kernel normalized to unit sum.
    """
    w = ker.astype(float)
    w /= w.sum()
    neff = 1.0 / float((w * w).sum())
    area_eff = neff * pix_area_arcsec2
    dmu = 1.25 * math.log10(neff)
    return neff, area_eff, dmu


def _fallback_source_mask(data, nsigma=2.0, npixels=20, dilate_size=5, maxiters=3):
    _, med, std = sigma_clipped_stats(data, sigma=3.0, maxiters=maxiters)
    if (
        "detect_threshold" in globals()
        and detect_threshold is not None
        and detect_sources is not None
    ):
        thr = detect_threshold(
            data, nsigma=nsigma, sigma_clip=SigmaClip(sigma=3.0, maxiters=maxiters)
        )
        segm = detect_sources(data, thr, npixels=npixels)
        base = (segm.data > 0) if segm is not None else (data > med + nsigma * std)
    else:
        base = data > med + nsigma * std

    if "binary_dilation" in globals() and binary_dilation is not None:
        return binary_dilation(base, iterations=max(1, int(dilate_size // 2))).astype(
            np.uint8
        )

    ker = Box2DKernel(max(1, int(dilate_size)))
    return (_conv(base.astype(float), ker, normalize_kernel=False) > 0).astype(np.uint8)


def build_source_mask(data, nsigma=2.0, npixels=20, dilate_size=5, maxiters=3):
    if make_source_mask is not None:
        mask = make_source_mask(
            data,
            nsigma=nsigma,
            npixels=npixels,
            dilate_size=dilate_size,
            sigclip=SigmaClip(sigma=3.0, maxiters=maxiters),
        )
        return mask.astype(np.uint8)

    return _fallback_source_mask(
        data, nsigma=nsigma, npixels=npixels, dilate_size=dilate_size, maxiters=maxiters
    )


def _read_first_image_hdu(path):
    with fits.open(path, memmap=True) as hdul:
        hdu = next((h for h in hdul if getattr(h, "data", None) is not None), None)
        if hdu is None or hdu.data is None:
            raise RuntimeError(f"No image data found in {path}")
        return np.array(hdu.data), hdu.header.copy()


def _read_hdu_by_spec(path, hdu_spec):
    with fits.open(path, memmap=True) as hdul:
        if hdu_spec is None:
            hdu = next((h for h in hdul if getattr(h, "data", None) is not None), None)
            if hdu is None or hdu.data is None:
                raise RuntimeError(f"No image data found in {path}")
            return np.array(hdu.data), hdu.header.copy()

        try:
            idx = int(hdu_spec)
            hdu = hdul[idx]
            if hdu.data is None:
                raise RuntimeError(f"HDU {idx} in {path} has no data")
            return np.array(hdu.data), hdu.header.copy()
        except ValueError:
            pass

        target = str(hdu_spec).strip().upper()
        for hdu in hdul:
            name = str(getattr(hdu, "name", "")).strip().upper()
            if name == target and getattr(hdu, "data", None) is not None:
                return np.array(hdu.data), hdu.header.copy()

        raise RuntimeError(f"Could not find HDU '{hdu_spec}' with image data in {path}")


def read_external_mask(mask_fits, mask_hdu=None, expected_shape=None):
    """
    Read an external mask FITS.

    Interpretation:
      - 0       => unmasked
      - nonzero => masked
      - NaN/inf => masked

    Returns uint8 mask with 1=masked, 0=good.
    """
    mask_data, _ = _read_hdu_by_spec(mask_fits, mask_hdu)
    mask_data = np.squeeze(mask_data)

    if mask_data.ndim != 2:
        raise RuntimeError(
            f"External mask must be 2D after squeeze(); got shape {mask_data.shape}"
        )

    if expected_shape is not None and mask_data.shape != expected_shape:
        raise RuntimeError(
            f"External mask shape {mask_data.shape} does not match input image shape "
            f"{expected_shape}"
        )

    mask = ((~np.isfinite(mask_data)) | (mask_data != 0)).astype(np.uint8)
    return mask


def _global_background(data, mask, sigma_clip=3.0, maxiters=5, mode_label="GLOBAL"):
    good = (~mask.astype(bool)) & np.isfinite(data)
    if np.count_nonzero(good) < 10:
        raise RuntimeError(
            "Too few unmasked finite pixels to estimate a global background."
        )
    mean, med, std = sigma_clipped_stats(
        data[good], sigma=sigma_clip, maxiters=maxiters
    )
    bkg = np.full_like(data, med, dtype=float)
    return bkg, {
        "BKGMODE": mode_label,
        "BKGMEAN": float(mean),
        "BKGMED": float(med),
        "BKGSTD": float(std),
    }


def estimate_background(
    data,
    mask,
    mode,
    box_size,
    filter_size,
    sigma_clip=3.0,
    maxiters=5,
    exclude_percentile=10.0,
    fallback_to_global=False,
):
    if mode == "none":
        return np.zeros_like(data, dtype=float), {"BKGMODE": "NONE"}

    sc = SigmaClip(sigma=sigma_clip, maxiters=maxiters)

    if mode == "global":
        return _global_background(
            data, mask, sigma_clip=sigma_clip, maxiters=maxiters, mode_label="GLOBAL"
        )

    try:
        bkg2d = Background2D(
            data,
            box_size=box_size,
            filter_size=filter_size,
            mask=mask.astype(bool),
            sigma_clip=sc,
            bkg_estimator=MedianBackground(),
            exclude_percentile=float(exclude_percentile),
        )
    except ValueError as e:
        msg = (
            "Background2D failed. This usually means every background box is too "
            "masked for the current exclude_percentile setting. For binned images, "
            "try a smaller --box-size in binned pixels and/or a larger "
            "--bkg-exclude-percentile, e.g. --box-size 32 --filter-size 3 "
            "--bkg-exclude-percentile 50. Original Photutils error: "
            f"{e}"
        )
        if fallback_to_global:
            print("WARNING:", msg)
            bkg, meta = _global_background(
                data,
                mask,
                sigma_clip=sigma_clip,
                maxiters=maxiters,
                mode_label="GLOBAL_FALLBACK",
            )
            meta["BKGFAIL"] = "BKG2D"
            meta["BKGEXCL"] = float(exclude_percentile)
            return bkg, meta
        raise RuntimeError(msg) from e

    bkg = np.array(bkg2d.background, dtype=float)
    return bkg, {
        "BKGMODE": "BKG2D",
        "BOXSIZE": int(box_size if np.isscalar(box_size) else box_size[0]),
        "FILTSIZE": int(filter_size if np.isscalar(filter_size) else filter_size[0]),
        "BKGEXCL": float(exclude_percentile),
        "BKGMEAN": float(np.nanmean(bkg)),
        "BKGSTD": float(np.nanstd(bkg)),
    }


# ----- extinction / region / noise helpers -----


def find_col_case_insensitive(tab, name):
    """Return actual table column name matching name case-insensitively."""
    target = name.lower()
    for col in tab.colnames:
        if col.lower() == target:
            return col
    raise KeyError(
        f"Could not find column {name!r}. Available columns include: "
        f"{tab.colnames[:30]}"
    )


def table_column_as_deg_quantity(tab, colname):
    """
    Return a table column as an astropy Quantity in degrees.

    Handles both unitless numeric FITS columns and Astropy Table columns
    that already carry unit='deg'. This avoids accidentally creating deg^2.
    """
    col = tab[colname]
    if getattr(col, "unit", None) is not None:
        return col.quantity.to(u.deg)
    return np.asarray(col, dtype=float) * u.deg


def build_region_weights(region_file, shape, header, subpixels=8):
    """
    Rasterize one or more DS9 regions onto an image grid.

    Returns fractional pixel weights in [0, 1]. If multiple regions are present,
    the union is represented by taking the maximum weight at each pixel.
    """
    if Regions is None:
        raise RuntimeError(
            "The 'regions' package is required for --dered-region. "
            "Install with: pip install regions"
        )

    regs = Regions.read(region_file, format="ds9")
    wcs = WCS(header)
    weights = np.zeros(shape, dtype=float)

    for reg in regs:
        # PixelRegion objects already have to_mask(); SkyRegion objects need WCS.
        preg = reg if hasattr(reg, "to_mask") else reg.to_pixel(wcs)
        rmask = preg.to_mask(mode="subpixels", subpixels=int(subpixels))
        if rmask is None:
            continue

        img = rmask.to_image(shape)
        if img is None:
            continue

        weights = np.maximum(weights, np.asarray(img, dtype=float))

    return weights


def tractor_transmission_from_region(
    tractor_file,
    region_file,
    header,
    shape,
    band="r",
    min_sources=3,
    min_region_weight=0.0,
    subpixels=8,
):
    """
    Estimate a single foreground MW transmission for a DS9 region using the
    median MW_TRANSMISSION_<band> of Tractor sources inside the same region.

    This is intended for long tails where a single object-center RA/Dec is
    less representative than the measurement aperture itself.
    """
    band = band.lower().strip()
    trans_col_requested = f"mw_transmission_{band}"

    region_weights = build_region_weights(
        region_file=region_file,
        shape=shape,
        header=header,
        subpixels=subpixels,
    )
    if np.sum(region_weights > 0) == 0:
        raise RuntimeError(
            f"Region {region_file!r} does not overlap the image or could not be rasterized."
        )

    tab = Table.read(tractor_file, hdu=1)

    ra_col = find_col_case_insensitive(tab, "ra")
    dec_col = find_col_case_insensitive(tab, "dec")
    trans_col = find_col_case_insensitive(tab, trans_col_requested)

    ra_q = table_column_as_deg_quantity(tab, ra_col)
    dec_q = table_column_as_deg_quantity(tab, dec_col)
    trans = np.asarray(tab[trans_col], dtype=float)

    valid = (
        np.isfinite(ra_q.to_value(u.deg))
        & np.isfinite(dec_q.to_value(u.deg))
        & np.isfinite(trans)
        & (trans > 0)
    )

    if not np.any(valid):
        raise RuntimeError(
            f"No valid rows found in {tractor_file} with finite RA/Dec and "
            f"positive {trans_col}."
        )

    wcs = WCS(header)
    coords = SkyCoord(ra=ra_q[valid], dec=dec_q[valid], frame="icrs")
    x, y = wcs.world_to_pixel(coords)

    ny, nx = shape
    xi = np.rint(x).astype(int)
    yi = np.rint(y).astype(int)

    in_bounds = (
        np.isfinite(x) & np.isfinite(y) & (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
    )

    valid_indices = np.where(valid)[0]
    valid_indices = valid_indices[in_bounds]
    xi = xi[in_bounds]
    yi = yi[in_bounds]

    if valid_indices.size == 0:
        raise RuntimeError("No valid Tractor rows project onto the image footprint.")

    inside_region = region_weights[yi, xi] > float(min_region_weight)
    region_indices = valid_indices[inside_region]

    vals = trans[region_indices]
    vals = vals[np.isfinite(vals) & (vals > 0)]

    if vals.size < int(min_sources):
        raise RuntimeError(
            f"Only {vals.size} valid Tractor source(s) found inside the region, "
            f"fewer than --min-tractor-sources={min_sources}. "
            f"Use a larger region, lower --min-tractor-sources, or use a centroid-based "
            f"correction in a separate preprocessing step."
        )

    T_med = float(np.median(vals))
    A_band = float(-2.5 * np.log10(T_med))
    p16, p84 = np.percentile(vals, [16, 84])

    return {
        "method": "median_tractor_sources_inside_ds9_region",
        "tractor_file": tractor_file,
        "region_file": region_file,
        "band": band,
        "transmission_column": trans_col,
        "n_sources": int(vals.size),
        "mw_transmission_median": T_med,
        "mw_transmission_mean": float(np.mean(vals)),
        "mw_transmission_std": float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0,
        "mw_transmission_p16": float(p16),
        "mw_transmission_p84": float(p84),
        "mw_transmission_min": float(np.min(vals)),
        "mw_transmission_max": float(np.max(vals)),
        "A_band_mag": A_band,
        # Larger T means smaller A, so the percentile order reverses in magnitude.
        "A_band_p16_mag": float(-2.5 * np.log10(p84)),
        "A_band_p84_mag": float(-2.5 * np.log10(p16)),
    }


def read_invvar_sigma(invvar_fits, invvar_hdu, expected_shape):
    """
    Read a Legacy inverse-variance image and return observed-frame sigma [nMgy/pix].

    For valid pixels:
        sigma_obs = 1 / sqrt(invvar_obs)

    Invalid, nonfinite, or non-positive invvar pixels are assigned sigma=0 so
    they do not inject NaNs into synthetic realizations. They should normally
    be excluded by the same science mask used in downstream measurements.
    """
    invvar, inv_hdr = _read_hdu_by_spec(invvar_fits, invvar_hdu)
    invvar = np.squeeze(np.asarray(invvar, dtype=float))

    if invvar.ndim != 2:
        raise RuntimeError(
            f"Inverse-variance image must be 2D after squeeze(); got shape {invvar.shape}"
        )
    if invvar.shape != expected_shape:
        raise RuntimeError(
            f"Inverse-variance shape {invvar.shape} does not match input image shape "
            f"{expected_shape}"
        )

    good = np.isfinite(invvar) & (invvar > 0)
    sigma = np.zeros_like(invvar, dtype=float)
    sigma[good] = 1.0 / np.sqrt(invvar[good])

    return sigma, inv_hdr, int(np.sum(good)), int(invvar.size - np.sum(good))


def add_science_header_keywords(
    header,
    image_kind,
    extinction_mode,
    science_frame,
    ext_info=None,
    noise_info=None,
):
    """Add extinction/noise provenance to a flux or mu HDU header."""
    h = header.copy()
    h["IMGKIND"] = (str(image_kind).upper(), "Input image kind")
    h["EXTMODE"] = (str(extinction_mode).upper(), "Flux extinction operation")
    h["SCIFRAME"] = (str(science_frame).upper(), "Flux convention of this HDU")

    if ext_info is not None:
        h["DEREDMTH"] = ("REGION", "MW transmission from DS9 region")
        h["DEREDBND"] = (ext_info["band"], "Band used for MW transmission")
        h["MWTRANS"] = (
            ext_info["mw_transmission_median"],
            "Median MW transmission in region",
        )
        h["A_BAND"] = (ext_info["A_band_mag"], "Extinction correction [mag]")
        h["NTRSRC"] = (
            ext_info["n_sources"],
            "Tractor sources used for transmission median",
        )
        h["TRP16"] = (ext_info["mw_transmission_p16"], "MW transmission p16")
        h["TRP84"] = (ext_info["mw_transmission_p84"], "MW transmission p84")

        if extinction_mode == "deredden":
            h["DEREDDEN"] = (True, "Flux divided by MW transmission")
        elif extinction_mode == "redden":
            h["REDDENED"] = (True, "Flux multiplied by MW transmission")

    if noise_info is not None:
        h["NOISADD"] = (True, "Gaussian noise realization added")
        h["NOISFRAM"] = (noise_info["noise_frame"].upper(), "Noise flux convention")
        h["NOISSCL"] = (float(noise_info["noise_scale"]), "Multiplicative noise scale")
        h["NOISSEED"] = (int(noise_info["seed"]), "Noise RNG seed")
        h["INVGOOD"] = (int(noise_info["invvar_good"]), "Positive finite invvar pixels")
        h["INVBAD"] = (int(noise_info["invvar_bad"]), "Bad invvar pixels")
        h["SIGMED"] = (
            float(noise_info["sigma_median"]),
            "Median injected sigma [nMgy/pix]",
        )
        if "sigma_median_binned" in noise_info:
            h["SIGMBIN"] = (
                float(noise_info["sigma_median_binned"]),
                "Median absolute binned noise [nMgy/binned pix]",
            )

    return h


def surface_brightness_image_from_flux(flux_nmgy_pix, area_arcsec2, mu_floor):
    """Convert a nMgy/pix image into mag/arcsec^2 with a faint-end floor."""
    nmgy_arcsec2 = flux_nmgy_pix / area_arcsec2
    fmin = 10 ** ((AB_ZP_NANOMAGGY - float(mu_floor)) / 2.5)
    nmgy_arcsec2 = np.clip(nmgy_arcsec2, fmin, None)
    return mu_from_nmgy_arcsec2(nmgy_arcsec2)


# ----- binning helpers -----


def _trimmed_shape_for_bin(shape, factor, trim_mode="crop"):
    """Return the shape trimmed to an integer number of factor x factor blocks."""
    if factor <= 1:
        return tuple(shape)

    ny, nx = shape
    ny_trim = (ny // factor) * factor
    nx_trim = (nx // factor) * factor

    if ny_trim <= 0 or nx_trim <= 0:
        raise RuntimeError(
            f"Image shape {shape} is too small for --bin-factor={factor}."
        )

    if trim_mode == "error" and (ny_trim != ny or nx_trim != nx):
        raise RuntimeError(
            f"Image shape {shape} is not divisible by --bin-factor={factor}. "
            f"Use --bin-trim crop or pad/crop the input image first."
        )

    return ny_trim, nx_trim


def bin2d_sum(data, factor, trim_mode="crop", name="image"):
    """
    Bin a 2D nMgy/pix image by summing factor x factor blocks.

    Summing, not averaging, is the correct operation for flux images in nMgy/pix:
    total flux is conserved, while the binned-pixel area grows by factor^2.
    """
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise RuntimeError(f"{name} must be 2D for binning; got shape {arr.shape}")
    if factor <= 1:
        return np.array(arr, copy=True)

    ny_trim, nx_trim = _trimmed_shape_for_bin(arr.shape, factor, trim_mode)
    arr = arr[:ny_trim, :nx_trim]
    return arr.reshape(ny_trim // factor, factor, nx_trim // factor, factor).sum(
        axis=(1, 3)
    )


def bin2d_mask_any(mask, factor, trim_mode="crop", name="mask"):
    """
    Bin a 2D mask by logical OR over factor x factor blocks.

    Any masked native pixel makes the corresponding binned pixel masked.
    """
    arr = np.asarray(mask).astype(bool)
    if arr.ndim != 2:
        raise RuntimeError(f"{name} must be 2D for binning; got shape {arr.shape}")
    if factor <= 1:
        return arr.astype(np.uint8)

    ny_trim, nx_trim = _trimmed_shape_for_bin(arr.shape, factor, trim_mode)
    arr = arr[:ny_trim, :nx_trim]
    out = arr.reshape(ny_trim // factor, factor, nx_trim // factor, factor).any(
        axis=(1, 3)
    )
    return out.astype(np.uint8)


def update_header_for_binning(
    header, factor, original_shape, new_shape, stage="EXTNOISE"
):
    """
    Update a FITS WCS header after block-summing factor x factor pixels.

    The new pixel value is the sum over each block. The WCS is adjusted so the
    center of a binned pixel maps to the center of the corresponding native-pixel
    block. This assumes binning starts at the lower-left array corner and crops
    only the high-x/high-y edges if the shape is not divisible by the factor.

    stage records when binning occurred:
      PREBKG   = Legacy image binned before background estimation
      EXTNOISE = synthetic image binned after extinction/noise processing
    """
    if factor <= 1:
        return header.copy()

    h = header.copy()
    ny_old, nx_old = original_shape
    ny_new, nx_new = new_shape

    # Update CRPIX for block centers. FITS CRPIX is 1-based.
    for key in ("CRPIX1", "CRPIX2"):
        if key in h:
            h[key] = (float(h[key]) + 0.5 * (factor - 1)) / float(factor)

    # Pixel scale grows by the bin factor. Handle either CD matrix or CDELT.
    cd_keys = ["CD1_1", "CD1_2", "CD2_1", "CD2_2"]
    if any(k in h for k in cd_keys):
        for key in cd_keys:
            if key in h:
                h[key] = float(h[key]) * float(factor)
    else:
        if "CDELT1" in h:
            h["CDELT1"] = float(h["CDELT1"]) * float(factor)
        if "CDELT2" in h:
            h["CDELT2"] = float(h["CDELT2"]) * float(factor)

    h["NAXIS1"] = int(nx_new)
    h["NAXIS2"] = int(ny_new)
    h["BINFAC"] = (int(factor), "Block-sum binning factor")
    h["BINMODE"] = ("SUM", "Flux-conserving block binning")
    h["BINORIG1"] = (int(nx_old), "Original unbinned NAXIS1")
    h["BINORIG2"] = (int(ny_old), "Original unbinned NAXIS2")
    h["BINNEW1"] = (int(nx_new), "Binned NAXIS1")
    h["BINNEW2"] = (int(ny_new), "Binned NAXIS2")
    h["BINSTAGE"] = (str(stage).upper(), "Processing stage of block binning")

    return h


def add_binning_header_keywords(header, bin_info):
    """Add common binning provenance to an HDU header."""
    h = header.copy()
    if bin_info is not None:
        stage = str(bin_info.get("stage", "EXTNOISE")).upper()
        h["BINFAC"] = (int(bin_info["factor"]), "Block-sum binning factor")
        h["BINMODE"] = ("SUM", "Flux-conserving block binning")
        h["BINTRIM"] = (bin_info["trim_mode"].upper(), "Binning edge handling")
        h["BINORIG1"] = (int(bin_info["original_shape"][1]), "Original unbinned NAXIS1")
        h["BINORIG2"] = (int(bin_info["original_shape"][0]), "Original unbinned NAXIS2")
        h["BINNEW1"] = (int(bin_info["new_shape"][1]), "Binned NAXIS1")
        h["BINNEW2"] = (int(bin_info["new_shape"][0]), "Binned NAXIS2")
        h["BINSTAGE"] = (stage, "Processing stage of block binning")
        if stage == "PREBKG":
            h["BINBEFOR"] = ("BACKGROUND", "Binned before background estimation")
        elif stage == "EXTNOISE":
            h["BINAFTER"] = ("EXTNOISE", "Binned after extinction/noise operations")
        h["BINFOOT"] = (
            "high-x/high-y" if bin_info["cropped"] else "none",
            "Cropped edge",
        )
    return h


# ----- smoothing helpers -----


def _box_kernel(box_px, normalize=True):
    k = np.ones((box_px, box_px), dtype=float)
    if normalize:
        k /= k.sum()
    return k


def _gauss_kernel(sigma_px):
    return Gaussian2DKernel(x_stddev=float(sigma_px), y_stddev=float(sigma_px)).array


def _size_to_pixels(size_value, size_in_arcsec, header, is_sigma=False, is_box=False):
    """Convert a user size (arcsec or px) to pixels."""
    if size_in_arcsec:
        sx, sy = pixel_scale_arcsec(header)
        s_px = float(size_value) / sx
    else:
        s_px = float(size_value)

    if is_sigma:
        return max(s_px, 0.1)
    if is_box:
        return max(int(round(s_px)), 1)
    return s_px


def smooth_flux(data_nmgy_pix, hdr, args):
    """Return (smoothed_flux[nMgy/pix], meta_dict) or (None, {}) if disabled."""
    if args.smooth == "none":
        return None, {}

    img = np.array(data_nmgy_pix, dtype=float)

    if args.smooth == "box":
        box_px = _size_to_pixels(args.box_size, args.size_arcsec, hdr, is_box=True)
        ker = _box_kernel(box_px, normalize=(args.kernel_mode == "mean"))
        sigma_px = None
        fwhm_px = None

    elif args.smooth == "gauss":
        if args.gauss_sigma is not None:
            sigma_px = _size_to_pixels(
                args.gauss_sigma, args.size_arcsec, hdr, is_sigma=True
            )
            fwhm_px = 2.354820045 * float(sigma_px)
        elif args.gauss_fwhm is not None:
            fwhm_px = _size_to_pixels(
                args.gauss_fwhm, args.size_arcsec, hdr, is_sigma=False
            )
            sigma_px = float(fwhm_px) / 2.354820045
        else:
            raise ValueError(
                "For --smooth gauss, provide --gauss-sigma or --gauss-fwhm"
            )
        ker = _gauss_kernel(sigma_px)

    else:
        raise ValueError("Unknown smoothing mode.")

    sm = convolve(
        img, ker, boundary="extend", nan_treatment="interpolate", normalize_kernel=False
    )

    meta = {"SMOOTH": args.smooth}
    if args.smooth == "box":
        meta["BOXPX"] = int(ker.shape[0])
        meta["KMODE"] = args.kernel_mode
    if args.smooth == "gauss":
        meta["GSIGPX"] = float(sigma_px)
        meta["GFWHMPX"] = float(fwhm_px)

    neff, area_eff_as2, dmu = _kernel_stats(ker, pixel_area_arcsec2(hdr))
    meta["NEFFPX"] = float(neff)
    meta["NEFFAS"] = float(area_eff_as2)
    meta["DMUGAIN"] = float(dmu)

    return sm, meta


# ----------------- main -----------------


def main():
    ap = argparse.ArgumentParser(
        description="Sky-subtract Legacy r image (nMgy/pix) and convert to mag/arcsec^2 with optional smoothing."
    )
    ap.add_argument("input_fits")
    ap.add_argument("output_fits")

    # Optional external mask
    ap.add_argument(
        "--mask-fits",
        default=None,
        help=(
            "Optional FITS file containing a precomputed mask. "
            "Nonzero pixels are treated as masked and used instead of computing "
            "a source mask from the image."
        ),
    )
    ap.add_argument(
        "--mask-hdu",
        default=None,
        help=(
            "HDU index or EXTNAME within --mask-fits. "
            "Default: first HDU with image data."
        ),
    )

    # Background
    ap.add_argument("--bkg-mode", choices=["2d", "global", "none"], default="2d")
    ap.add_argument(
        "--box-size", type=int, default=128, help="Background2D box size (px)."
    )
    ap.add_argument(
        "--filter-size", type=int, default=5, help="Background2D filter size (odd)."
    )
    ap.add_argument(
        "--bkg-exclude-percentile",
        type=float,
        default=10.0,
        help=(
            "Background2D exclude_percentile. A box is excluded if more than this "
            "percent of its pixels are masked. For heavily masked or binned images, "
            "values like 50--90 may be needed."
        ),
    )
    ap.add_argument(
        "--bkg-fallback-global",
        action="store_true",
        help=(
            "If Background2D fails because too few boxes are usable, fall back to "
            "a single global sigma-clipped median background instead of aborting."
        ),
    )
    ap.add_argument(
        "--nsigma", type=float, default=2.0, help="Source-mask threshold in sigma."
    )
    ap.add_argument(
        "--npixels", type=int, default=20, help="Min connected pixels in auto mask."
    )
    ap.add_argument("--dilate", type=int, default=7, help="Auto-mask dilation (px).")

    # Smoothing
    ap.add_argument(
        "--smooth",
        choices=["none", "box", "gauss"],
        default="none",
        help="Apply smoothing in flux (nMgy/pix) BEFORE SB conversion.",
    )
    ap.add_argument(
        "--size-arcsec",
        action="store_true",
        help="Interpret smoothing sizes as arcsec instead of pixels.",
    )
    ap.add_argument(
        "--box-size-smooth",
        type=float,
        default=13,
        help="Box kernel side (px or arcsec) for --smooth box.",
    )
    ap.add_argument(
        "--gauss-sigma",
        type=float,
        default=None,
        help="Gaussian sigma (px or arcsec) for --smooth gauss.",
    )
    ap.add_argument(
        "--gauss-fwhm",
        type=float,
        default=None,
        help="Gaussian FWHM (px or arcsec) for --smooth gauss.",
    )
    ap.add_argument(
        "--kernel-mode",
        choices=["mean", "sum"],
        default="mean",
        help="For box smoothing: 'mean' (normalized) or 'sum'. Gaussian is always unit-integral.",
    )

    # mu floor & cosmology
    ap.add_argument(
        "--mu-floor",
        type=float,
        default=32.0,
        help="Faintest mu allowed (mag/arcsec^2).",
    )
    ap.add_argument("--z", type=float, default=None, help="Redshift for rest-frame mu.")
    ap.add_argument("--kcorr", type=float, default=None, help="K-correction [mag].")
    ap.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Spectral index alpha if --kcorr omitted (f_nu ∝ nu^alpha).",
    )
    ap.add_argument(
        "--report-kernel-area",
        action="store_true",
        help="Print effective area and expected SB gain for the smoothing kernel.",
    )
    ap.add_argument(
        "--fixed-background-fits",
        default=None,
        help=(
            "Optional FITS image containing a fixed background model in nMgy/pix "
            "to subtract instead of estimating a new background."
        ),
    )

    ap.add_argument(
        "--fixed-background-hdu",
        default=None,
        help=(
            "HDU index or EXTNAME in --fixed-background-fits. "
            "Default: first image HDU."
        ),
    )
    # Image convention, foreground extinction, and synthetic noise controls
    ap.add_argument(
        "--image-kind",
        choices=["legacy", "synthetic"],
        default="legacy",
        help=(
            "Input image type. 'legacy' means an observed Legacy Survey image. "
            "'synthetic' means a noise-free model image in intrinsic/dereddened flux units."
        ),
    )
    ap.add_argument(
        "--extinction-mode",
        choices=["none", "deredden", "redden"],
        default="none",
        help=(
            "Apply a scalar Galactic foreground correction from --tractor-fits and "
            "--dered-region to the sky-subtracted flux. Use 'deredden' for Legacy "
            "surface-brightness/luminosity comparisons; use 'redden' for synthetic "
            "observed-frame tail-length comparisons."
        ),
    )
    ap.add_argument(
        "--tractor-fits",
        default=None,
        help=(
            "Legacy Tractor FITS catalog used to obtain MW_TRANSMISSION_<band> "
            "inside --dered-region."
        ),
    )
    ap.add_argument(
        "--dered-region",
        default=None,
        help=(
            "DS9 region defining the aperture over which the median Tractor "
            "MW_TRANSMISSION_<band> is computed."
        ),
    )
    ap.add_argument(
        "--dered-band",
        choices=["g", "r", "z"],
        default="r",
        help="Band used for MW_TRANSMISSION_<band>.",
    )
    ap.add_argument(
        "--min-tractor-sources",
        type=int,
        default=3,
        help="Minimum Tractor sources required inside --dered-region.",
    )
    ap.add_argument(
        "--dered-region-min-weight",
        type=float,
        default=0.0,
        help=(
            "Minimum rasterized DS9-region weight for a Tractor source to count "
            "as inside the region."
        ),
    )
    ap.add_argument(
        "--dered-subpixels",
        type=int,
        default=8,
        help="Subpixel sampling used to rasterize --dered-region.",
    )
    ap.add_argument(
        "--invvar-fits",
        default=None,
        help=(
            "Optional Legacy inverse-variance FITS image. Required with --add-noise."
        ),
    )
    ap.add_argument(
        "--invvar-hdu",
        default=None,
        help="HDU index or EXTNAME in --invvar-fits. Default: first image HDU.",
    )
    ap.add_argument(
        "--add-noise",
        action="store_true",
        help=(
            "Add a Gaussian noise realization to the final science flux image. "
            "For synthetic images, this is how to create Legacy-like observed or "
            "dereddened noisy realizations."
        ),
    )
    ap.add_argument(
        "--noise-frame",
        choices=["auto", "observed", "dereddened"],
        default="auto",
        help=(
            "Flux convention for injected noise. 'observed' uses sigma=1/sqrt(invvar). "
            "'dereddened' uses sigma=(1/sqrt(invvar))/MW_TRANSMISSION. "
            "'auto' matches the science frame implied by --image-kind and --extinction-mode."
        ),
    )
    ap.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Multiplicative factor applied to the invvar-derived sigma image.",
    )
    ap.add_argument(
        "--noise-seed",
        type=int,
        default=12345,
        help="Random seed for synthetic noise realization.",
    )

    # Optional output binning
    ap.add_argument(
        "--bin-factor",
        type=int,
        default=1,
        help=(
            "Block-sum bin the image by this factor. For --image-kind legacy, "
            "binning is performed before background estimation so it matches an "
            "externally pre-binned Legacy image. For --image-kind synthetic, "
            "binning remains after extinction/noise and before smoothing/SB conversion. "
            "Use --bin-factor 4 for 4x binning. Flux is summed, not averaged."
        ),
    )
    ap.add_argument(
        "--bin-trim",
        choices=["crop", "error"],
        default="crop",
        help=(
            "How to handle image dimensions not divisible by --bin-factor. "
            "'crop' removes high-x/high-y edge pixels; 'error' aborts."
        ),
    )

    ap.add_argument(
        "--out-background-fits",
        default=None,
        help=(
            "Optional standalone FITS file containing the fitted background model "
            "in nMgy/pix. This is equivalent to the BKG_NMGY extension and can "
            "be passed to gadget_movie.py with --background-file."
        ),
    )

    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.bin_factor < 1:
        raise RuntimeError("--bin-factor must be >= 1")

    data_nmgy_pix, hdr = _read_first_image_hdu(args.input_fits)
    data_nmgy_pix = np.squeeze(data_nmgy_pix)
    if data_nmgy_pix.ndim != 2:
        raise RuntimeError(
            f"Input image must be 2D after squeeze(); got shape {data_nmgy_pix.shape}"
        )

    native_shape = tuple(data_nmgy_pix.shape)
    area_arcsec2 = pixel_area_arcsec2(hdr)

    # Legacy images must be binned BEFORE background estimation so --bin-factor
    # reproduces the workflow of first block-summing the Legacy image externally
    # and then running this script on that binned image. Synthetic images retain
    # the original ordering: extinction/noise at native resolution, then binning.
    legacy_prebin = args.image_kind == "legacy" and args.bin_factor > 1
    bin_info = None

    # Load an external mask on the native grid first. For Legacy pre-binning it
    # is then OR-binned to the same grid as the block-summed science image.
    # If no external mask is supplied, auto-masking is performed after Legacy
    # pre-binning so it matches the externally pre-binned workflow.
    mask = None
    if args.mask_fits is not None:
        mask = read_external_mask(
            args.mask_fits, mask_hdu=args.mask_hdu, expected_shape=native_shape
        )
        mask_meta = {
            "MASKMODE": "EXTERN",
            "MASKPATH": os.path.basename(args.mask_fits),
        }
        if args.mask_hdu is not None:
            mask_meta["MASKHDU"] = str(args.mask_hdu)
        print(f"Using external mask: {args.mask_fits}")
        if args.mask_hdu is not None:
            print(f"  mask HDU: {args.mask_hdu}")
    else:
        mask_meta = {
            "MASKMODE": "AUTO",
            "MASKNSIG": float(args.nsigma),
            "MASKNPIX": int(args.npixels),
            "MASKDIL": int(args.dilate),
        }

    if legacy_prebin:
        bin_factor = int(args.bin_factor)
        original_shape = native_shape
        ny_trim, nx_trim = _trimmed_shape_for_bin(
            original_shape, bin_factor, args.bin_trim
        )
        new_shape = (ny_trim // bin_factor, nx_trim // bin_factor)
        cropped = (ny_trim, nx_trim) != original_shape

        data_nmgy_pix = bin2d_sum(
            data_nmgy_pix, bin_factor, args.bin_trim, name="DATA_NMGY"
        )
        if mask is not None:
            mask = bin2d_mask_any(mask, bin_factor, args.bin_trim, name="MASK")
            mask_meta["BINFAC"] = int(bin_factor)
            mask_meta["BINMODE"] = "ANY"
            mask_meta["MASKGRID"] = "BINNED"

        hdr = update_header_for_binning(
            hdr,
            bin_factor,
            original_shape=original_shape,
            new_shape=new_shape,
            stage="PREBKG",
        )
        area_arcsec2 = pixel_area_arcsec2(hdr)

        bin_info = {
            "factor": bin_factor,
            "trim_mode": args.bin_trim,
            "original_shape": original_shape,
            "trimmed_shape": (ny_trim, nx_trim),
            "new_shape": tuple(data_nmgy_pix.shape),
            "cropped": cropped,
            "stage": "PREBKG",
        }

        print()
        print("=== Legacy pre-background binning ===")
        print(f"Block-sum bin factor:     {bin_factor}x{bin_factor}")
        print(f"Original shape:           {original_shape}")
        if cropped:
            print(f"Trimmed shape:            {(ny_trim, nx_trim)}")
        print(f"Binned shape:             {data_nmgy_pix.shape}")
        print(f"Binned pixel area:        {area_arcsec2:.6f} arcsec^2/pixel")
        print("Background estimation will use the binned image and binned mask.")

    # Auto-mask on the image grid that will be used for background estimation.
    # For Legacy + --bin-factor this is deliberately the binned grid.
    if mask is None:
        mask = build_source_mask(
            data_nmgy_pix,
            nsigma=args.nsigma,
            npixels=args.npixels,
            dilate_size=args.dilate,
        )
        if legacy_prebin:
            mask_meta["MASKGRID"] = "BINNED"
            mask_meta["BINFAC"] = int(args.bin_factor)
            mask_meta["BINMODE"] = "AUTO_BINNED"
        print(
            f"Using auto mask: nsigma={args.nsigma}, npixels={args.npixels}, dilate={args.dilate}"
        )

    # Background. For Legacy pre-binning, Background2D/global estimation now sees
    # the binned data and binned mask, matching the rebin_fits.py --sum workflow.
    if args.fixed_background_fits is not None:
        fixed_hdu = args.fixed_background_hdu

        bkg_map_nmgy, bkg_hdr = _read_hdu_by_spec(
            args.fixed_background_fits,
            fixed_hdu,
        )
        bkg_map_nmgy = np.squeeze(np.asarray(bkg_map_nmgy, dtype=float))

        if bkg_map_nmgy.ndim != 2:
            raise RuntimeError(
                f"Fixed background must be 2D after squeeze(); got shape {bkg_map_nmgy.shape}"
            )

        if legacy_prebin and bkg_map_nmgy.shape == native_shape:
            # A native-resolution fixed background must be block-summed just like
            # the Legacy image before subtraction.
            bkg_map_nmgy = bin2d_sum(
                bkg_map_nmgy,
                int(args.bin_factor),
                args.bin_trim,
                name="fixed BKG_NMGY",
            )
        elif bkg_map_nmgy.shape != data_nmgy_pix.shape:
            raise RuntimeError(
                f"Fixed background shape {bkg_map_nmgy.shape} does not match "
                f"the science image shape {data_nmgy_pix.shape}. "
                + (
                    f"For Legacy pre-binning, a native background of shape {native_shape} "
                    "or an already-binned background matching the science image is accepted."
                    if legacy_prebin
                    else ""
                )
            )

        bkg_meta = {
            "BKGMODE": "FIXED",
            "BKGFILE": os.path.basename(args.fixed_background_fits),
            "BKGHDU": str(fixed_hdu) if fixed_hdu is not None else "FIRST",
            "BKGMEAN": float(np.nanmean(bkg_map_nmgy)),
            "BKGSTD": float(np.nanstd(bkg_map_nmgy)),
        }
        if legacy_prebin:
            bkg_meta["BINFAC"] = int(args.bin_factor)
            bkg_meta["BINSTAGE"] = "PREBKG"

        print(f"Using fixed background: {args.fixed_background_fits}")
        if fixed_hdu is not None:
            print(f"  background HDU: {fixed_hdu}")

    else:
        bkg_map_nmgy, bkg_meta = estimate_background(
            data_nmgy_pix,
            mask,
            mode=args.bkg_mode,
            box_size=args.box_size,
            filter_size=args.filter_size,
            exclude_percentile=args.bkg_exclude_percentile,
            fallback_to_global=args.bkg_fallback_global,
        )
        if legacy_prebin:
            bkg_meta["BINFAC"] = int(args.bin_factor)
            bkg_meta["BINSTAGE"] = "PREBKG"

    # Sky subtraction in flux
    sky_sub_nmgy_pix = data_nmgy_pix - bkg_map_nmgy

    # Foreground-extinction convention and optional noise realization.
    #
    # The raw sky_sub_nmgy_pix image is always preserved as SKY_SUB_NMGY.
    # The final product for matched comparisons is SCIENCE_NMGY:
    #   - legacy + --extinction-mode deredden:
    #       SCIENCE_NMGY = SKY_SUB_NMGY / MW_TRANSMISSION
    #   - synthetic + --extinction-mode redden:
    #       SCIENCE_NMGY = synthetic_intrinsic * MW_TRANSMISSION
    #   - optional --add-noise:
    #       noise is injected in the requested/auto flux convention.
    ext_info = None
    T_region = None

    if (
        args.image_kind == "synthetic"
        and args.fixed_background_fits is None
        and args.bkg_mode != "none"
    ):
        print(
            "WARNING: --image-kind synthetic is usually used with --bkg-mode none "
            "unless you intentionally want to subtract a fitted background from the model."
        )

    # Determine the nominal flux convention of the final science image.
    if args.extinction_mode == "redden":
        science_frame = "observed"
    elif args.extinction_mode == "deredden":
        science_frame = "dereddened"
    else:
        science_frame = "dereddened" if args.image_kind == "synthetic" else "observed"

    # Decide whether MW_TRANSMISSION is needed.
    auto_noise_frame = science_frame
    need_transmission = args.extinction_mode != "none"
    if args.add_noise:
        if args.noise_frame == "dereddened":
            need_transmission = True
        elif args.noise_frame == "auto" and auto_noise_frame == "dereddened":
            need_transmission = True

    if need_transmission:
        if args.tractor_fits is None or args.dered_region is None:
            raise RuntimeError(
                "--tractor-fits and --dered-region are required when applying "
                "foreground extinction/reddening, or when adding dereddened-frame noise."
            )

        ext_info = tractor_transmission_from_region(
            tractor_file=args.tractor_fits,
            region_file=args.dered_region,
            header=hdr,
            shape=data_nmgy_pix.shape,
            band=args.dered_band,
            min_sources=args.min_tractor_sources,
            min_region_weight=args.dered_region_min_weight,
            subpixels=args.dered_subpixels,
        )
        T_region = ext_info["mw_transmission_median"]

        print()
        print("=== Region-based Galactic foreground correction ===")
        print(f"Tractor file:             {args.tractor_fits}")
        print(f"Region file:              {args.dered_region}")
        print(f"Band:                     {args.dered_band}")
        print(f"Method:                   {ext_info['method']}")
        print(f"Tractor sources used:     {ext_info['n_sources']}")
        print(f"MW transmission median:   {T_region:.8f}")
        print(
            f"A_{args.dered_band}:                    {ext_info['A_band_mag']:.5f} mag"
        )
        print(
            f"Transmission p16-p84:     "
            f"{ext_info['mw_transmission_p16']:.8f} – "
            f"{ext_info['mw_transmission_p84']:.8f}"
        )

    ext_flux_nmgy_pix = np.array(sky_sub_nmgy_pix, dtype=float, copy=True)

    if args.extinction_mode == "deredden":
        ext_flux_nmgy_pix = ext_flux_nmgy_pix / T_region
        print("Applied extinction operation: deredden, flux /= MW_TRANSMISSION")
    elif args.extinction_mode == "redden":
        ext_flux_nmgy_pix = ext_flux_nmgy_pix * T_region
        print("Applied extinction operation: redden, flux *= MW_TRANSMISSION")
    else:
        print(f"Applied extinction operation: none; science frame = {science_frame}")

    noise_nmgy_pix = None
    noise_info = None
    science_nmgy_pix = np.array(ext_flux_nmgy_pix, dtype=float, copy=True)

    if args.add_noise:
        if args.invvar_fits is None:
            raise RuntimeError("--add-noise requires --invvar-fits.")

        # For Legacy pre-binning the supplied inverse-variance map is still on
        # the native Legacy grid. Combine native variances when converting it to
        # the binned grid: sigma_bin^2 = sum(sigma_native^2). This is equivalent
        # to drawing independent native-pixel Gaussian noise and block-summing it.
        invvar_expected_shape = (
            native_shape if legacy_prebin else tuple(data_nmgy_pix.shape)
        )
        sigma_obs, inv_hdr, inv_good, inv_bad = read_invvar_sigma(
            args.invvar_fits,
            args.invvar_hdu,
            expected_shape=invvar_expected_shape,
        )

        noise_frame = (
            auto_noise_frame if args.noise_frame == "auto" else args.noise_frame
        )

        if noise_frame == "dereddened":
            if T_region is None:
                raise RuntimeError(
                    "Dereddened-frame noise requires MW_TRANSMISSION. "
                    "Provide --tractor-fits and --dered-region."
                )
            sigma_use = sigma_obs / T_region
        else:
            sigma_use = sigma_obs

        sigma_use = float(args.noise_scale) * sigma_use

        if noise_frame != science_frame:
            print(
                f"WARNING: injecting {noise_frame} noise into a {science_frame} science image. "
                "This is allowed, but is usually not the matched convention."
            )

        if legacy_prebin:
            sigma_use = np.sqrt(
                bin2d_sum(
                    sigma_use * sigma_use,
                    int(args.bin_factor),
                    args.bin_trim,
                    name="noise variance",
                )
            )

        rng = np.random.default_rng(args.noise_seed)
        noise_nmgy_pix = rng.normal(loc=0.0, scale=sigma_use).astype(float)
        science_nmgy_pix = science_nmgy_pix + noise_nmgy_pix

        good_sigma = np.isfinite(sigma_use) & (sigma_use > 0)
        sigma_median = (
            float(np.nanmedian(sigma_use[good_sigma])) if np.any(good_sigma) else 0.0
        )

        noise_info = {
            "noise_frame": noise_frame,
            "noise_scale": float(args.noise_scale),
            "seed": int(args.noise_seed),
            "invvar_good": int(inv_good),
            "invvar_bad": int(inv_bad),
            "sigma_median": sigma_median,
        }
        if legacy_prebin:
            noise_info["sigma_median_binned"] = sigma_median

        print()
        print("=== Noise realization ===")
        print(f"Invvar file:              {args.invvar_fits}")
        print(f"Noise frame:              {noise_frame}")
        print(f"Noise scale:              {args.noise_scale}")
        print(f"Seed:                     {args.noise_seed}")
        print(f"Valid invvar pixels:      {inv_good}")
        print(f"Bad invvar pixels:        {inv_bad}")
        if legacy_prebin:
            print(f"Median binned sigma:      {sigma_median:.6g} nMgy/binned pix")
        else:
            print(f"Median injected sigma:    {sigma_median:.6g} nMgy/pix")

    # Synthetic images retain the original binning sequence: extinction/reddening
    # and optional native-resolution noise first, then block-sum before smoothing
    # and surface-brightness conversion. Legacy images were already binned above,
    # before background estimation, and must not be binned a second time here.
    if args.bin_factor > 1 and not legacy_prebin:
        bin_factor = int(args.bin_factor)
        original_shape = tuple(data_nmgy_pix.shape)
        ny_trim, nx_trim = _trimmed_shape_for_bin(
            original_shape, bin_factor, args.bin_trim
        )
        new_shape = (ny_trim // bin_factor, nx_trim // bin_factor)
        cropped = (ny_trim, nx_trim) != original_shape

        data_nmgy_pix = bin2d_sum(
            data_nmgy_pix, bin_factor, args.bin_trim, name="DATA_NMGY"
        )
        bkg_map_nmgy = bin2d_sum(
            bkg_map_nmgy, bin_factor, args.bin_trim, name="BKG_NMGY"
        )
        sky_sub_nmgy_pix = bin2d_sum(
            sky_sub_nmgy_pix, bin_factor, args.bin_trim, name="SKY_SUB_NMGY"
        )
        ext_flux_nmgy_pix = bin2d_sum(
            ext_flux_nmgy_pix,
            bin_factor,
            args.bin_trim,
            name="extinction-corrected flux",
        )
        science_nmgy_pix = bin2d_sum(
            science_nmgy_pix, bin_factor, args.bin_trim, name="SCIENCE_NMGY"
        )
        mask = bin2d_mask_any(mask, bin_factor, args.bin_trim, name="MASK")

        if noise_nmgy_pix is not None:
            noise_nmgy_pix = bin2d_sum(
                noise_nmgy_pix, bin_factor, args.bin_trim, name="NOISE_NMGY"
            )
            if noise_info is not None:
                sigma_binned = np.sqrt(
                    bin2d_sum(
                        sigma_use * sigma_use,
                        bin_factor,
                        args.bin_trim,
                        name="noise variance",
                    )
                )
                good_sigma_binned = np.isfinite(sigma_binned) & (sigma_binned > 0)
                if np.any(good_sigma_binned):
                    noise_info["sigma_median_binned"] = float(
                        np.nanmedian(sigma_binned[good_sigma_binned])
                    )

        hdr = update_header_for_binning(
            hdr,
            bin_factor,
            original_shape=original_shape,
            new_shape=new_shape,
            stage="EXTNOISE",
        )
        area_arcsec2 = pixel_area_arcsec2(hdr)

        bin_info = {
            "factor": bin_factor,
            "trim_mode": args.bin_trim,
            "original_shape": original_shape,
            "trimmed_shape": (ny_trim, nx_trim),
            "new_shape": tuple(science_nmgy_pix.shape),
            "cropped": cropped,
            "stage": "EXTNOISE",
        }

        bkg_meta["BINFAC"] = int(bin_factor)
        bkg_meta["BINMODE"] = "SUM"
        bkg_meta["BINSTAGE"] = "EXTNOISE"
        bkg_meta["BKGMEAN"] = float(np.nanmean(bkg_map_nmgy))
        bkg_meta["BKGSTD"] = float(np.nanstd(bkg_map_nmgy))
        mask_meta["BINFAC"] = int(bin_factor)
        mask_meta["BINMODE"] = "ANY"

        print()
        print("=== Synthetic post-noise binning ===")
        print(f"Block-sum bin factor:     {bin_factor}x{bin_factor}")
        print(f"Original shape:           {original_shape}")
        if cropped:
            print(f"Trimmed shape:            {(ny_trim, nx_trim)}")
        print(f"Binned shape:             {science_nmgy_pix.shape}")
        print(f"Binned pixel area:        {area_arcsec2:.6f} arcsec^2/pixel")

    # Optional smoothing in flux before SB conversion.
    # Smoothing is applied to the final science image, so the mu image and the
    # SCIENCE_SMOOTH_NMGY HDU use the same extinction/noise convention.
    smooth_nmgy_pix = None
    smooth_meta = {}
    if args.smooth != "none":

        class _Shim:
            pass

        shim = _Shim()
        shim.smooth = args.smooth
        shim.size_arcsec = args.size_arcsec
        shim.box_size = args.box_size_smooth
        shim.kernel_mode = args.kernel_mode
        shim.gauss_sigma = args.gauss_sigma
        shim.gauss_fwhm = args.gauss_fwhm
        smooth_nmgy_pix, smooth_meta = smooth_flux(science_nmgy_pix, hdr, shim)

    if (
        args.smooth != "none"
        and args.report_kernel_area
        and smooth_nmgy_pix is not None
    ):
        ktype = smooth_meta.get("SMOOTH", "?")
        if ktype == "box":
            desc = f'box {int(smooth_meta["BOXPX"])} px ({smooth_meta.get("KMODE", "mean")})'
        elif ktype == "gauss":
            desc = (
                f'gauss FWHM={smooth_meta["GFWHMPX"]:.2f}px '
                f'(sigma={smooth_meta["GSIGPX"]:.2f}px)'
            )
        else:
            desc = ktype

        print(
            f"[kernel] {desc}  neff={smooth_meta['NEFFPX']:.2f} px^2, "
            f"A_eff={smooth_meta['NEFFAS']:.1f} arcsec^2, "
            f"Delta_mu={smooth_meta['DMUGAIN']:.2f} mag"
        )

    flux_for_mu = smooth_nmgy_pix if smooth_nmgy_pix is not None else science_nmgy_pix
    mu_obs = surface_brightness_image_from_flux(
        flux_for_mu,
        area_arcsec2=area_arcsec2,
        mu_floor=args.mu_floor,
    )

    # Assemble outputs
    now = Time.now().isot.split(".")[0] + "Z"
    ph = fits.Header()
    ph["DATE"] = now
    ph["ORIGIN"] = "mu_from_nmgy"
    ph["BUNIT"] = "mag/arcsec^2"
    ph["PIXA2"] = (area_arcsec2, "Pixel area [arcsec^2]")
    ph["NMZP"] = (AB_ZP_NANOMAGGY, "AB mag of 1 nanomaggy")
    ph["MUFLOOR"] = (args.mu_floor, "mu floor [mag/arcsec^2]")
    ph["IMGKIND"] = (args.image_kind.upper(), "Input image kind")
    ph["EXTMODE"] = (args.extinction_mode.upper(), "Flux extinction operation")
    ph["SCIFRAME"] = (science_frame.upper(), "Final science flux convention")
    if bin_info is not None:
        ph = add_binning_header_keywords(ph, bin_info)
    if ext_info is not None:
        ph["DEREDBND"] = (ext_info["band"], "Band used for MW transmission")
        ph["MWTRANS"] = (ext_info["mw_transmission_median"], "Median MW transmission")
        ph["A_BAND"] = (ext_info["A_band_mag"], "Extinction correction [mag]")
        ph["NTRSRC"] = (ext_info["n_sources"], "Tractor sources used")
    if noise_info is not None:
        ph["NOISADD"] = (True, "Gaussian noise realization added")
        ph["NOISFRAM"] = (noise_info["noise_frame"].upper(), "Noise convention")
        ph["NOISSCL"] = (noise_info["noise_scale"], "Multiplicative noise scale")
        ph["NOISSEED"] = (noise_info["seed"], "Noise RNG seed")

    for d in (mask_meta, bkg_meta):
        for k, v in d.items():
            try:
                ph[k] = v
            except Exception:
                pass

    if smooth_nmgy_pix is not None:
        if "SMOOTH" in smooth_meta:
            ph["SMOOTH"] = smooth_meta["SMOOTH"]
        if "BOXPX" in smooth_meta:
            ph["BOXPX"] = int(smooth_meta["BOXPX"])
        if "KMODE" in smooth_meta:
            ph["KMODE"] = smooth_meta["KMODE"]
        if "GSIGPX" in smooth_meta:
            ph["GSIGPX"] = float(smooth_meta["GSIGPX"])
        if "GFWHMPX" in smooth_meta:
            ph["GFWHMP"] = float(smooth_meta["GFWHMPX"])

    prim = fits.PrimaryHDU(header=ph)

    def add_common(h, bunit):
        hh = h.copy()
        hh = add_binning_header_keywords(hh, bin_info)
        hh["A_SEC2"] = (area_arcsec2, "Pixel area [arcsec^2]")
        hh["MUFLOOR"] = (args.mu_floor, "mu floor [mag/arcsec^2]")
        hh["BUNIT"] = bunit
        return hh

    mask_hdr = fits.Header()
    for k, v in mask_meta.items():
        try:
            mask_hdr[k] = v
        except Exception:
            pass

    hdus = [
        prim,
        fits.ImageHDU(
            data=data_nmgy_pix.astype(np.float32),
            header=add_common(hdr, "nMgy/pix"),
            name="DATA_NMGY",
        ),
        fits.ImageHDU(
            data=mask.astype(np.uint8),
            header=mask_hdr,
            name="MASK",
        ),
        fits.ImageHDU(
            data=bkg_map_nmgy.astype(np.float32),
            header=add_common(hdr, "nMgy/pix"),
            name="BKG_NMGY",
        ),
        fits.ImageHDU(
            data=sky_sub_nmgy_pix.astype(np.float32),
            header=add_common(hdr, "nMgy/pix"),
            name="SKY_SUB_NMGY",
        ),
    ]

    # Extinction-only intermediate product, before optional noise.
    if args.extinction_mode == "deredden":
        h_ext = add_science_header_keywords(
            add_common(hdr, "nMgy/pix"),
            args.image_kind,
            args.extinction_mode,
            "dereddened",
            ext_info=ext_info,
            noise_info=None,
        )
        hdus.append(
            fits.ImageHDU(
                data=ext_flux_nmgy_pix.astype(np.float32),
                header=h_ext,
                name="SKY_SUB_DERED_NMGY",
            )
        )
    elif args.extinction_mode == "redden":
        h_ext = add_science_header_keywords(
            add_common(hdr, "nMgy/pix"),
            args.image_kind,
            args.extinction_mode,
            "observed",
            ext_info=ext_info,
            noise_info=None,
        )
        hdus.append(
            fits.ImageHDU(
                data=ext_flux_nmgy_pix.astype(np.float32),
                header=h_ext,
                name="SKY_SUB_REDDENED_NMGY",
            )
        )

    if noise_nmgy_pix is not None:
        h_noise = add_science_header_keywords(
            add_common(hdr, "nMgy/pix"),
            args.image_kind,
            args.extinction_mode,
            science_frame,
            ext_info=ext_info,
            noise_info=noise_info,
        )
        hdus.append(
            fits.ImageHDU(
                data=noise_nmgy_pix.astype(np.float32),
                header=h_noise,
                name="NOISE_NMGY",
            )
        )

    # Final comparison image after requested extinction operation and optional noise.
    h_science = add_science_header_keywords(
        add_common(hdr, "nMgy/pix"),
        args.image_kind,
        args.extinction_mode,
        science_frame,
        ext_info=ext_info,
        noise_info=noise_info,
    )
    hdus.append(
        fits.ImageHDU(
            data=science_nmgy_pix.astype(np.float32),
            header=h_science,
            name="SCIENCE_NMGY",
        )
    )

    # Convenience aliases for the two main science use-cases.
    if args.image_kind == "synthetic" and science_frame == "observed":
        hdus.append(
            fits.ImageHDU(
                data=science_nmgy_pix.astype(np.float32),
                header=h_science.copy(),
                name="SYNTH_OBS_NMGY",
            )
        )
    if args.image_kind == "synthetic" and science_frame == "dereddened":
        hdus.append(
            fits.ImageHDU(
                data=science_nmgy_pix.astype(np.float32),
                header=h_science.copy(),
                name="SYNTH_DERED_NMGY",
            )
        )

    if smooth_nmgy_pix is not None:
        h_smooth = add_science_header_keywords(
            add_common(hdr, "nMgy/pix"),
            args.image_kind,
            args.extinction_mode,
            science_frame,
            ext_info=ext_info,
            noise_info=noise_info,
        )
        hdus.append(
            fits.ImageHDU(
                data=smooth_nmgy_pix.astype(np.float32),
                header=h_smooth,
                name="SCIENCE_SMOOTH_NMGY",
            )
        )

    h_mu = add_science_header_keywords(
        add_common(hdr, "mag/arcsec^2"),
        args.image_kind,
        args.extinction_mode,
        science_frame,
        ext_info=ext_info,
        noise_info=noise_info,
    )
    h_mu["MU_KIND"] = ("SCIENCE", "SB from SCIENCE_NMGY or SCIENCE_SMOOTH_NMGY")
    hdus.append(
        fits.ImageHDU(
            data=mu_obs.astype(np.float32),
            header=h_mu,
            name="MU_SCIENCE",
        )
    )

    # Backward-compatible and convenience SB aliases.
    hdus.append(
        fits.ImageHDU(
            data=mu_obs.astype(np.float32),
            header=h_mu.copy(),
            name="MU_OBS",
        )
    )
    if args.extinction_mode == "deredden":
        hdus.append(
            fits.ImageHDU(
                data=mu_obs.astype(np.float32),
                header=h_mu.copy(),
                name="MU_OBS_DERED",
            )
        )
    elif args.extinction_mode == "redden":
        hdus.append(
            fits.ImageHDU(
                data=mu_obs.astype(np.float32),
                header=h_mu.copy(),
                name="MU_OBS_REDDENED",
            )
        )

    if args.z is not None:
        mu_rest, tolman, K = restframe_mu(mu_obs, args.z, args.kcorr, args.alpha)
        h_mur = add_common(hdr, "mag/arcsec^2")
        h_mur["MU_KIND"] = "REST"
        h_mur["Z"] = (args.z, "Redshift")
        h_mur["TOLMAN"] = (tolman, "mag subtracted")
        h_mur["KCORR"] = (K, "mag subtracted")
        if args.kcorr is None and args.alpha is not None:
            h_mur["KALPHA"] = (args.alpha, "f_nu ∝ nu^alpha")
        hdus.append(
            fits.ImageHDU(
                data=mu_rest.astype(np.float32),
                header=h_mur,
                name="MU_REST",
            )
        )

    fits.HDUList(hdus).writeto(args.output_fits, overwrite=args.overwrite)

    if args.out_background_fits is not None:
        bkg_hdr = add_common(hdr, "nMgy/pix")
        bkg_hdr["IMTYPE"] = ("BKG_NMGY", "Fitted background model")
        bkg_hdr["SRCFILE"] = (os.path.basename(args.input_fits), "Input Legacy image")
        bkg_hdr["SRCEXT"] = ("BKG_NMGY", "Equivalent extension in output MEF")
        for k, v in bkg_meta.items():
            try:
                bkg_hdr[k] = v
            except Exception:
                pass
        fits.PrimaryHDU(
            data=bkg_map_nmgy.astype(np.float32),
            header=bkg_hdr,
        ).writeto(args.out_background_fits, overwrite=args.overwrite)
        print("Wrote background model:", args.out_background_fits)

    print("Wrote:", args.output_fits)
    print("HDU names:", [h.name for h in hdus])


if __name__ == "__main__":
    main()
