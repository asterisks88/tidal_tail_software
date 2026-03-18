#!/usr/bin/env python3
"""
Make residual images (flux residual + residual SNR) for (a) Legacy brick r-band image
and (b) a synthetic r-band image, using the same background subtraction + smooth modeling.

Default recommended usage for fair comparison:
  - Use --residual-mode auto for BOTH real and synthetic
  - Use the SAME invvar + maskbits + bbox for BOTH (if the synthetic is on the same pixel grid)
  - Compare *_resid_snr.fits with your matched-filter script

Outputs (prefix = --output-prefix):
  - <prefix>_resid_flux.fits
  - <prefix>_resid_snr.fits
  - <prefix>_model_used.fits        (if --write-model)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter


def read_fits_data(path: Path, hdu: int) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path, memmap=True) as hdul:
        data = np.asarray(hdul[hdu].data, dtype=np.float64)
        hdr = hdul[hdu].header.copy()
    return data, hdr


def write_fits(path: Path, data: np.ndarray, hdr: Optional[fits.Header]) -> None:
    hdu = fits.PrimaryHDU(data=data, header=hdr)
    fits.HDUList([hdu]).writeto(path, overwrite=True)


def parse_bbox(bbox: Optional[str]) -> Optional[tuple[int, int, int, int]]:
    if bbox is None:
        return None
    parts = [p.strip() for p in bbox.split(",")]
    if len(parts) != 4:
        raise ValueError("bbox must be 'x0,x1,y0,y1'")
    x0, x1, y0, y1 = map(int, parts)
    if x1 <= x0 or y1 <= y0:
        raise ValueError("bbox must satisfy x1>x0 and y1>y0")
    return x0, x1, y0, y1


def apply_bbox(
    arr: np.ndarray, bbox: Optional[tuple[int, int, int, int]]
) -> np.ndarray:
    if bbox is None:
        return arr
    x0, x1, y0, y1 = bbox
    return arr[y0:y1, x0:x1]


def adjust_header_for_bbox(
    hdr: fits.Header, bbox: tuple[int, int, int, int]
) -> fits.Header:
    x0, x1, y0, y1 = bbox
    h = hdr.copy()
    if "CRPIX1" in h:
        h["CRPIX1"] = h["CRPIX1"] - x0
    if "CRPIX2" in h:
        h["CRPIX2"] = h["CRPIX2"] - y0
    h["NAXIS1"] = x1 - x0
    h["NAXIS2"] = y1 - y0
    return h


def sigma_from_invvar(invvar: np.ndarray, sigma_floor: float = 0.0) -> np.ndarray:
    sig = np.full_like(invvar, np.inf, dtype=np.float64)
    good = np.isfinite(invvar) & (invvar > 0)
    sig[good] = 1.0 / np.sqrt(invvar[good])
    if sigma_floor > 0:
        sig = np.maximum(sig, sigma_floor)
    return sig


def robust_median_bg(
    img: np.ndarray, good: np.ndarray, clip_sigma: float = 3.0, iters: int = 3
) -> float:
    vals = img[good & np.isfinite(img)]
    if vals.size == 0:
        return 0.0
    v = vals
    for _ in range(iters):
        med = np.median(v)
        mad = np.median(np.abs(v - med))
        rsig = 1.4826 * mad if mad > 0 else np.std(v)
        if rsig <= 0:
            break
        keep = np.abs(v - med) < clip_sigma * rsig
        if np.count_nonzero(keep) < 100:
            break
        v = v[keep]
    return float(np.median(v))


def masked_gaussian_lowpass(
    img: np.ndarray, good: np.ndarray, sigma_pix: float
) -> np.ndarray:
    """
    Normalized convolution low-pass:
      smooth = G*(img*w) / G*(w), w in {0,1}
    """
    w = good.astype(np.float64)
    num = gaussian_filter(img * w, sigma=sigma_pix, mode="nearest")
    den = gaussian_filter(w, sigma=sigma_pix, mode="nearest")
    out = np.zeros_like(img, dtype=np.float64)
    m = den > 1e-12
    out[m] = num[m] / den[m]
    out[~m] = 0.0
    return out


def masked_gaussian_smooth_postmask(
    img_out: np.ndarray, ok: np.ndarray, sigma_pix: float
) -> np.ndarray:
    """
    Smooth an already-masked output image (bad pixels are 0 or NaN) using normalized
    convolution with ok as weights, so masked pixels do not bleed into good pixels.
    """
    img0 = np.array(img_out, dtype=np.float64, copy=True)
    img0[~ok] = 0.0
    img0[~np.isfinite(img0)] = 0.0
    return masked_gaussian_lowpass(img0, ok, sigma_pix=float(sigma_pix))


def auto_smooth_model_iterative(
    img: np.ndarray,
    sigma_map: np.ndarray,
    good: np.ndarray,
    smooth_sigma_pix: float,
    clip_snr: float,
    n_iter: int,
    clip_sign: str,
) -> np.ndarray:
    """
    Iteratively build a smooth model while excluding significant residual features from the mask.

    clip_sign:
      - 'positive': excludes positive residuals (tails in image-model if tails are positive features)
      - 'negative': excludes negative residuals
    """
    mask_i = good.copy()

    for _ in range(max(1, int(n_iter))):
        model_i = masked_gaussian_lowpass(
            img, mask_i, sigma_pix=float(smooth_sigma_pix)
        )
        resid_i = img - model_i

        snr = np.zeros_like(img, dtype=np.float64)
        ok = (
            mask_i
            & np.isfinite(resid_i)
            & np.isfinite(sigma_map)
            & (sigma_map > 0)
            & (sigma_map < np.inf)
        )
        snr[ok] = resid_i[ok] / sigma_map[ok]

        if clip_sign == "positive":
            mask_i = mask_i & (snr < clip_snr)
        elif clip_sign == "negative":
            mask_i = mask_i & (snr > -clip_snr)
        else:
            raise ValueError("clip_sign must be 'positive' or 'negative'")

    return masked_gaussian_lowpass(img, mask_i, sigma_pix=float(smooth_sigma_pix))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Create residual flux + residual SNR images with consistent modeling."
    )

    ap.add_argument("--image", required=True, type=Path, help="Input r-band image FITS")
    ap.add_argument("--image-hdu", type=int, default=0)

    ap.add_argument(
        "--invvar",
        type=Path,
        default=None,
        help="invvar FITS (optional but recommended)",
    )
    ap.add_argument("--invvar-hdu", type=int, default=0)

    ap.add_argument(
        "--maskbits", type=Path, default=None, help="maskbits FITS (optional)"
    )
    ap.add_argument("--maskbits-hdu", type=int, default=1)

    ap.add_argument(
        "--bbox",
        type=str,
        default=None,
        help="Crop bbox 'x0,x1,y0,y1' in full-image coords",
    )

    ap.add_argument(
        "--maskbits-mode",
        choices=["any", "mask", "none"],
        default="any",
        help="'any' masks all nonzero maskbits; 'mask' uses --maskbits-mask; 'none' ignores maskbits.",
    )
    ap.add_argument("--maskbits-mask", type=int, default=0)

    ap.add_argument(
        "--residual-mode",
        choices=["auto", "provided", "none"],
        default="auto",
        help="auto: build smooth model by iterative masked low-pass; provided: subtract --model; none: residual=image",
    )
    ap.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Provided model FITS (only for --residual-mode provided)",
    )
    ap.add_argument("--model-hdu", type=int, default=0)

    ap.add_argument(
        "--bg",
        choices=["none", "median"],
        default="median",
        help="Background subtraction using robust median on good pixels",
    )
    ap.add_argument("--bg-clip-sigma", type=float, default=3.0)
    ap.add_argument("--bg-iters", type=int, default=3)

    # Auto-model controls
    ap.add_argument(
        "--auto-model-sigma-pix",
        type=float,
        default=50.0,
        help="Low-pass sigma (pix) for auto model. Larger preserves tails more (default: 50).",
    )
    ap.add_argument("--auto-model-iters", type=int, default=3)
    ap.add_argument(
        "--auto-model-clip-snr",
        type=float,
        default=2.0,
        help="Exclude residual pixels above this SNR from model-building mask (default: 2).",
    )
    ap.add_argument(
        "--auto-model-clip-sign", choices=["positive", "negative"], default="positive"
    )

    ap.add_argument(
        "--smooth-sigma-pix",
        type=float,
        default=0.0,
        help=(
            "Gaussian sigma (pix) to smooth the FINAL masked outputs (after bad pixels are set to 0/NaN). "
            "If >0, writes *_resid_flux_smooth.fits and *_resid_snr_smooth.fits."
        ),
    )
    ap.add_argument(
        "--smooth-mode",
        choices=["masked", "plain"],
        default="masked",
        help=(
            "Smoothing mode after masking: 'masked' uses normalized convolution with ok weights (recommended); "
            "'plain' uses gaussian_filter on NaN->0 (can smear zeros inward)."
        ),
    )
    ap.add_argument(
        "--smooth-snr-from-flux",
        action="store_true",
        help=(
            "If set (and smoothing enabled), compute smoothed SNR as (smoothed residual flux)/sigma "
            "instead of smoothing the SNR image directly."
        ),
    )

    ap.add_argument(
        "--sigma-floor", type=float, default=0.0, help="Optional floor on sigma map"
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument(
        "--bad-to-nan",
        dest="bad_to_nan",
        action="store_true",
        help="Write bad pixels as NaN (recommended; default).",
    )
    g.add_argument(
        "--bad-to-zero",
        dest="bad_to_nan",
        action="store_false",
        help="Write bad pixels as 0.0 (legacy behavior).",
    )
    ap.set_defaults(bad_to_nan=True)
    ap.add_argument("--write-model", action="store_true", help="Write the model used")

    ap.add_argument("--output-prefix", required=True, type=Path)

    args = ap.parse_args()

    bbox = parse_bbox(args.bbox)

    img_full, hdr_full = read_fits_data(args.image, args.image_hdu)
    img = apply_bbox(img_full, bbox)

    # invvar (optional)
    if args.invvar is not None:
        inv_full, _ = read_fits_data(args.invvar, args.invvar_hdu)
        inv = apply_bbox(inv_full, bbox)
        if inv.shape != img.shape:
            raise RuntimeError(f"invvar shape {inv.shape} != image shape {img.shape}")
    else:
        inv = np.ones_like(img, dtype=np.float64)

    # maskbits (optional)
    if args.maskbits is not None:
        mb_full, _ = read_fits_data(args.maskbits, args.maskbits_hdu)
        mb = apply_bbox(mb_full, bbox)
        if mb.shape != img.shape:
            raise RuntimeError(f"maskbits shape {mb.shape} != image shape {img.shape}")
    else:
        mb = np.zeros_like(img, dtype=np.int64)

    # good pixels
    good = np.isfinite(img) & np.isfinite(inv) & (inv > 0) & np.isfinite(mb)

    if args.maskbits_mode == "any":
        good &= mb == 0
    elif args.maskbits_mode == "mask":
        good &= (mb.astype(np.int64) & np.int64(args.maskbits_mask)) == 0
    elif args.maskbits_mode == "none":
        pass

    sigma = sigma_from_invvar(inv, sigma_floor=float(args.sigma_floor))

    # Background subtract on the "work image" for consistency
    work = img.copy()
    if args.bg == "median":
        bg = robust_median_bg(
            work, good, clip_sigma=float(args.bg_clip_sigma), iters=int(args.bg_iters)
        )
        work = work - bg

    # Model and residual
    model_used = np.zeros_like(work)

    if args.residual_mode == "none":
        resid = work.copy()

    elif args.residual_mode == "provided":
        if args.model is None:
            raise RuntimeError("--residual-mode provided requires --model")
        mod_full, _ = read_fits_data(args.model, args.model_hdu)
        mod = apply_bbox(mod_full, bbox)
        if mod.shape != work.shape:
            raise RuntimeError(f"model shape {mod.shape} != image shape {work.shape}")
        model_used = mod.copy()
        resid = work - model_used

    elif args.residual_mode == "auto":
        model_used = auto_smooth_model_iterative(
            img=work,
            sigma_map=sigma,
            good=good,
            smooth_sigma_pix=float(args.auto_model_sigma_pix),
            clip_snr=float(args.auto_model_clip_snr),
            n_iter=int(args.auto_model_iters),
            clip_sign=args.auto_model_clip_sign,
        )
        resid = work - model_used

    else:
        raise RuntimeError(f"Unknown residual mode: {args.residual_mode}")

    # Residual SNR
    resid_snr = np.zeros_like(resid)
    ok = good & np.isfinite(resid) & np.isfinite(sigma) & (sigma > 0) & (sigma < np.inf)
    resid_snr[ok] = resid[ok] / sigma[ok]

    if args.bad_to_nan:
        resid_out = resid.copy()
        snr_out = resid_snr.copy()
        resid_out[~ok] = np.nan
        snr_out[~ok] = np.nan
    else:
        resid_out = resid.copy()
        snr_out = resid_snr.copy()
        resid_out[~ok] = 0.0
        snr_out[~ok] = 0.0

    # Optional smoothing AFTER the output masking/zeroing step.
    smooth_sig = float(args.smooth_sigma_pix)
    resid_out_sm = None
    snr_out_sm = None

    if smooth_sig > 0:
        if args.smooth_mode == "masked":
            # Smooth the masked flux residual without bleeding masked pixels into the tail.
            resid_out_sm = masked_gaussian_smooth_postmask(resid_out, ok, smooth_sig)

            if args.smooth_snr_from_flux:
                # Derive SNR from smoothed flux (preferred)
                snr_out_sm = np.zeros_like(resid_out_sm, dtype=np.float64)
                snr_out_sm[ok] = resid_out_sm[ok] / sigma[ok]
            else:
                # Smooth the SNR map directly (less preferred)
                snr_out_sm = masked_gaussian_smooth_postmask(snr_out, ok, smooth_sig)

        else:
            # Plain smoothing: treat NaN as 0; can smear zeros inward
            resid_out_sm = gaussian_filter(
                np.nan_to_num(resid_out, nan=0.0), sigma=smooth_sig, mode="nearest"
            )

            if args.smooth_snr_from_flux:
                snr_out_sm = np.zeros_like(resid_out_sm, dtype=np.float64)
                snr_out_sm[ok] = resid_out_sm[ok] / sigma[ok]
            else:
                snr_out_sm = gaussian_filter(
                    np.nan_to_num(snr_out, nan=0.0), sigma=smooth_sig, mode="nearest"
                )

        # Re-impose your bad pixel policy on the smoothed outputs
        if args.bad_to_nan:
            resid_out_sm[~ok] = np.nan
            snr_out_sm[~ok] = np.nan
        else:
            resid_out_sm[~ok] = 0.0
            snr_out_sm[~ok] = 0.0

    hdr_out = hdr_full
    if bbox is not None:
        hdr_out = adjust_header_for_bbox(hdr_full, bbox)

    outp = args.output_prefix
    outp.parent.mkdir(parents=True, exist_ok=True)

    write_fits(
        Path(str(outp) + "_resid_flux.fits"), resid_out.astype(np.float32), hdr_out
    )
    write_fits(Path(str(outp) + "_resid_snr.fits"), snr_out.astype(np.float32), hdr_out)
    if resid_out_sm is not None and snr_out_sm is not None:
        write_fits(
            Path(str(outp) + "_resid_flux_smooth.fits"),
            resid_out_sm.astype(np.float32),
            hdr_out,
        )
        write_fits(
            Path(str(outp) + "_resid_snr_smooth.fits"),
            snr_out_sm.astype(np.float32),
            hdr_out,
        )

    if args.write_model:
        write_fits(
            Path(str(outp) + "_model_used.fits"), model_used.astype(np.float32), hdr_out
        )

    print("Wrote:")
    print(f"  {Path(str(outp) + '_resid_flux.fits')}")
    print(f"  {Path(str(outp) + '_resid_snr.fits')}")
    if resid_out_sm is not None:
        print(f"  {Path(str(outp) + '_resid_flux_smooth.fits')}")
        print(f"  {Path(str(outp) + '_resid_snr_smooth.fits')}")

    if args.write_model:
        print(f"  {Path(str(outp) + '_model_used.fits')}")


if __name__ == "__main__":
    main()
