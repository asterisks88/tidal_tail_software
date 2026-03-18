#!/usr/bin/env python3
"""
mu_from_nmgy.py  —  Sky-subtract Legacy r (nMgy/pix) then convert to mag/arcsec^2,
with optional smoothing in flux units BEFORE the SB conversion.

Outputs a multi-product MEF:
  - DATA_NMGY, MASK, BKG_NMGY, SKY_SUB_NMGY
  - SKY_SUB_SMOOTH_NMGY (if smoothing enabled)
  - MU_OBS, MU_REST (if --z provided)

Requires: astropy, photutils; (optional) scipy for mask dilation fallback.
"""

import argparse
import math

import astropy.units as u
import numpy as np
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.io import fits
from astropy.stats import SigmaClip, sigma_clipped_stats
from astropy.time import Time
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from photutils.background import Background2D, MedianBackground

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
    neff = 1.0 / float((w * w).sum())  # effective number of pixels
    area_eff = neff * pix_area_arcsec2  # effective area in arcsec^2
    dmu = 1.25 * math.log10(neff)  # expected SB gain in mag
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


def estimate_background(
    data, mask, mode, box_size, filter_size, sigma_clip=3.0, maxiters=5
):
    if mode == "none":
        return np.zeros_like(data, dtype=float), {"BKGMODE": "NONE"}
    sc = SigmaClip(sigma=sigma_clip, maxiters=maxiters)
    if mode == "global":
        mean, med, std = sigma_clipped_stats(
            data[~mask.astype(bool)], sigma=sigma_clip, maxiters=maxiters
        )
        bkg = np.full_like(data, med, dtype=float)
        return bkg, {
            "BKGMODE": "GLOBAL",
            "BKGMEAN": float(mean),
            "BKGMED": float(med),
            "BKGSTD": float(std),
        }
    bkg2d = Background2D(
        data,
        box_size=box_size,
        filter_size=filter_size,
        mask=mask.astype(bool),
        sigma_clip=sc,
        bkg_estimator=MedianBackground(),
    )
    bkg = np.array(bkg2d.background, dtype=float)
    return bkg, {
        "BKGMODE": "BKG2D",
        "BOXSIZE": int(box_size if np.isscalar(box_size) else box_size[0]),
        "FILTSIZE": int(filter_size if np.isscalar(filter_size) else filter_size[0]),
        "BKGMEAN": float(np.nanmean(bkg)),
        "BKGSTD": float(np.nanstd(bkg)),
    }


# ----- smoothing helpers -----


def _box_kernel(box_px, normalize=True):
    k = np.ones((box_px, box_px), dtype=float)
    if normalize:
        k /= k.sum()
    return k


def _gauss_kernel(sigma_px):
    # Gaussian2DKernel is already unit-integral (sum≈1)
    return Gaussian2DKernel(x_stddev=float(sigma_px), y_stddev=float(sigma_px)).array


def _size_to_pixels(size_value, size_in_arcsec, header, is_sigma=False, is_box=False):
    """Convert a user size (arcsec or px) to pixels (float for sigma, int for box side)."""
    if size_in_arcsec:
        sx, sy = pixel_scale_arcsec(header)
        # Use x-scale for conversion (Legacy pixels are effectively square)
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
        # size in px or arcsec
        box_px = _size_to_pixels(args.box_size, args.size_arcsec, hdr, is_box=True)
        ker = _box_kernel(box_px, normalize=(args.kernel_mode == "mean"))
        sigma_px = None
        fwhm_px = None

    elif args.smooth == "gauss":
        # prefer sigma; otherwise convert FWHM to sigma
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
        ker = _gauss_kernel(sigma_px)  # unit-integral

    else:
        raise ValueError("Unknown smoothing mode.")

    # Convolve with NaN-aware handling; we control normalization via ker itself
    sm = convolve(
        img, ker, boundary="extend", nan_treatment="interpolate", normalize_kernel=False
    )

    # Metadata (SHORT, FITS-classic keys elsewhere)
    meta = {"SMOOTH": args.smooth}
    if args.smooth == "box":
        meta["BOXPX"] = int(ker.shape[0])
        meta["KMODE"] = args.kernel_mode
    if args.smooth == "gauss":
        meta["GSIGPX"] = float(sigma_px)
        meta["GFWHMPX"] = float(fwhm_px)

    # Effective area / SB gain
    neff, area_eff_as2, dmu = _kernel_stats(ker, pixel_area_arcsec2(hdr))
    meta["NEFFPX"] = float(neff)  # px^2
    meta["NEFFAS"] = float(area_eff_as2)  # arcsec^2
    meta["DMUGAIN"] = float(dmu)  # mag

    return sm, meta


# ----------------- main -----------------


def main():
    ap = argparse.ArgumentParser(
        description="Sky-subtract Legacy r image (nMgy/pix) and convert to mag/arcsec^2 with optional smoothing."
    )
    ap.add_argument("input_fits")
    ap.add_argument("output_fits")
    # Background
    ap.add_argument("--bkg-mode", choices=["2d", "global", "none"], default="2d")
    ap.add_argument(
        "--box-size", type=int, default=128, help="Background2D box size (px)."
    )
    ap.add_argument(
        "--filter-size", type=int, default=5, help="Background2D filter size (odd)."
    )
    ap.add_argument(
        "--nsigma", type=float, default=2.0, help="Source-mask threshold in σ."
    )
    ap.add_argument(
        "--npixels", type=int, default=20, help="Min connected pixels in mask."
    )
    ap.add_argument("--dilate", type=int, default=7, help="Mask dilation (px).")
    # Smoothing (new)
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
        help="Box kernel side (px or arcsec) for --smooth box. Default 13.",
    )
    ap.add_argument(
        "--gauss-sigma",
        type=float,
        default=None,
        help="Gaussian σ (px or arcsec) for --smooth gauss.",
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
    # μ floor & cosmology
    ap.add_argument(
        "--mu-floor",
        type=float,
        default=32.0,
        help="Faintest μ allowed (mag/arcsec^2).",
    )
    ap.add_argument("--z", type=float, default=None, help="Redshift for rest-frame μ.")
    ap.add_argument("--kcorr", type=float, default=None, help="K-correction [mag].")
    ap.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Spectral index α if --kcorr omitted (f_nu ∝ nu^α).",
    )
    ap.add_argument(
        "--report-kernel-area",
        action="store_true",
        help="Print effective area (px^2 / arcsec^2) and expected SB gain Δμ for the smoothing kernel.",
    )

    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    with fits.open(args.input_fits, memmap=True) as hdul:
        hdu = next((h for h in hdul if h.data is not None), None)
        if hdu is None:
            raise RuntimeError("No image data found.")
        data_nmgy_pix = np.array(hdu.data, dtype=float)
        hdr = hdu.header.copy()

    area_arcsec2 = pixel_area_arcsec2(hdr)

    # Build mask & background (on original flux)
    mask = build_source_mask(
        data_nmgy_pix, nsigma=args.nsigma, npixels=args.npixels, dilate_size=args.dilate
    )
    bkg_map_nmgy, bkg_meta = estimate_background(
        data_nmgy_pix,
        mask,
        mode=args.bkg_mode,
        box_size=args.box_size,
        filter_size=args.filter_size,
    )

    # Sky subtraction in flux
    sky_sub_nmgy_pix = data_nmgy_pix - bkg_map_nmgy

    # ----- optional smoothing (in flux) BEFORE SB conversion -----
    # NOTE: we smooth the sky-subtracted flux image
    smooth_nmgy_pix = None
    smooth_meta = {}
    if args.smooth != "none":
        # Use --box-size-smooth (not the background box)
        class _Shim:
            pass

        shim = _Shim()
        shim.smooth = args.smooth
        shim.size_arcsec = args.size_arcsec
        shim.box_size = args.box_size_smooth
        shim.kernel_mode = args.kernel_mode
        shim.gauss_sigma = args.gauss_sigma
        shim.gauss_fwhm = args.gauss_fwhm
        smooth_nmgy_pix, smooth_meta = smooth_flux(sky_sub_nmgy_pix, hdr, shim)
    if (
        args.smooth != "none"
        and args.report_kernel_area
        and smooth_nmgy_pix is not None
    ):
        ktype = smooth_meta.get("SMOOTH", "?")
        if ktype == "box":
            desc = f'box {int(smooth_meta["BOXPX"])} px ({smooth_meta.get("KMODE","mean")})'
        elif ktype == "gauss":
            desc = f'gauss FWHM={smooth_meta["GFWHMPX"]:.2f}px (σ={smooth_meta["GSIGPX"]:.2f}px)'
        else:
            desc = ktype
        print(
            f"[kernel] {desc}  neff={smooth_meta['NEFFPX']:.2f} px^2, "
            f"A_eff={smooth_meta['NEFFAS']:.1f} arcsec^2,  Δμ={smooth_meta['DMUGAIN']:.2f} mag"
        )

    # Choose which flux to convert to SB: smoothed if present, else sky-subtracted
    flux_for_mu = smooth_nmgy_pix if smooth_nmgy_pix is not None else sky_sub_nmgy_pix

    # SB conversion (nMgy/arcsec^2) and μ with floor
    nmgy_arcsec2 = flux_for_mu / area_arcsec2
    fmin = 10 ** ((AB_ZP_NANOMAGGY - args.mu_floor) / 2.5)
    nmgy_arcsec2 = np.clip(nmgy_arcsec2, fmin, None)
    mu_obs = mu_from_nmgy_arcsec2(nmgy_arcsec2)

    # ----- assemble outputs -----
    now = Time.now().isot.split(".")[0] + "Z"
    ph = fits.Header()
    ph["DATE"] = now
    ph["ORIGIN"] = "mu_from_nmgy (with smoothing)"
    ph["BUNIT"] = "mag/arcsec^2"
    ph["PIXA2"] = (area_arcsec2, "Pixel area [arcsec^2]")
    ph["NMZP"] = (AB_ZP_NANOMAGGY, "AB mag of 1 nanomaggy")
    ph["MUFLOOR"] = (args.mu_floor, "mu floor [mag/arcsec^2]")
    for k, v in bkg_meta.items():
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
        hh["A_SEC2"] = (area_arcsec2, "Pixel area [arcsec^2]")
        hh["MUFLOOR"] = (args.mu_floor, "mu floor [mag/arcsec^2]")
        hh["BUNIT"] = bunit
        return hh

    hdus = [
        prim,
        fits.ImageHDU(
            data=data_nmgy_pix.astype(np.float32),
            header=add_common(hdr, "nMgy/pix"),
            name="DATA_NMGY",
        ),
        fits.ImageHDU(data=mask.astype(np.uint8), header=fits.Header(), name="MASK"),
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
    if smooth_nmgy_pix is not None:
        hdus.append(
            fits.ImageHDU(
                data=smooth_nmgy_pix.astype(np.float32),
                header=add_common(hdr, "nMgy/pix"),
                name="SKY_SUB_SMOOTH_NMGY",
            )
        )
    hdus.append(
        fits.ImageHDU(
            data=mu_obs.astype(np.float32),
            header=add_common(hdr, "mag/arcsec^2"),
            name="MU_OBS",
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
            fits.ImageHDU(data=mu_rest.astype(np.float32), header=h_mur, name="MU_REST")
        )

    fits.HDUList(hdus).writeto(args.output_fits, overwrite=args.overwrite)
    print("Wrote:", args.output_fits)
    print("HDU names:", [h.name for h in hdus])


if __name__ == "__main__":
    main()
