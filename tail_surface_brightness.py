#!/usr/bin/env python3
"""
measure_region_sb.py

Measure the mean surface brightness of a DS9-defined region from the output
MEF produced by mu_from_nmgy.py, excluding masked pixels from a supplied mask FITS.

Primary output quantities:
  - mean SB in nMgy/arcsec^2
  - mean SB in mag/arcsec^2
  - estimated uncertainty from random translated apertures of the same shape
  - naive background-noise uncertainty from sigma-clipped sky pixels

Dependencies:
  numpy, astropy, regions

Example:
  ./measure_region_sb.py \
      mu_output.fits \
      tail.reg \
      mask.fits \
      --flux-hdu SKY_SUB_NMGY \
      --mask-hdu 0 \
      --nrand 1000 \
      --subpixels 8
"""

import argparse
import math
import sys

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

try:
    from regions import Regions
except Exception:
    print("ERROR: this script requires the 'regions' package.", file=sys.stderr)
    print("Install with: pip install regions", file=sys.stderr)
    sys.exit(1)

AB_ZP_NANOMAGGY = 22.5


def parse_hdu_spec(spec):
    if spec is None:
        return None
    try:
        return int(spec)
    except ValueError:
        return str(spec)


def read_hdu_data(path, hdu_spec=None):
    with fits.open(path, memmap=True) as hdul:
        if hdu_spec is None:
            hdu = next((h for h in hdul if getattr(h, "data", None) is not None), None)
            if hdu is None or hdu.data is None:
                raise RuntimeError(f"No image HDU with data found in {path}")
            return np.squeeze(np.array(hdu.data)), hdu.header.copy()

        if isinstance(hdu_spec, int):
            hdu = hdul[hdu_spec]
            if hdu.data is None:
                raise RuntimeError(f"HDU {hdu_spec} in {path} has no data")
            return np.squeeze(np.array(hdu.data)), hdu.header.copy()

        target = str(hdu_spec).strip().upper()
        for hdu in hdul:
            name = str(getattr(hdu, "name", "")).strip().upper()
            if name == target and getattr(hdu, "data", None) is not None:
                return np.squeeze(np.array(hdu.data)), hdu.header.copy()

    raise RuntimeError(f"Could not find data HDU '{hdu_spec}' in {path}")


def pixel_scale_arcsec(header):
    w = WCS(header)
    scales_deg = proj_plane_pixel_scales(w) * u.deg
    sx = scales_deg[0].to(u.arcsec).value
    sy = scales_deg[1].to(u.arcsec).value
    if not np.isfinite(sx * sy) or sx <= 0 or sy <= 0:
        if "CDELT1" in header and "CDELT2" in header:
            sx = abs(header["CDELT1"]) * 3600.0
            sy = abs(header["CDELT2"]) * 3600.0
        else:
            raise RuntimeError("Could not determine pixel scale from header.")
    return float(sx), float(sy)


def pixel_area_arcsec2(header):
    if "A_SEC2" in header:
        return float(header["A_SEC2"])
    if "PIXA2" in header:
        return float(header["PIXA2"])
    sx, sy = pixel_scale_arcsec(header)
    return sx * sy


def nmgy_as2_to_mu(sb_nmgy_arcsec2):
    if not np.isfinite(sb_nmgy_arcsec2) or sb_nmgy_arcsec2 <= 0:
        return np.nan
    return AB_ZP_NANOMAGGY - 2.5 * math.log10(sb_nmgy_arcsec2)


def sb_err_to_mag_err(sb, sb_err):
    if not np.isfinite(sb) or not np.isfinite(sb_err) or sb <= 0 or sb_err <= 0:
        return np.nan
    return (2.5 / math.log(10.0)) * (sb_err / sb)


def read_mask(mask_fits, mask_hdu=None, expected_shape=None):
    data, _ = read_hdu_data(mask_fits, mask_hdu)
    if data.ndim != 2:
        raise RuntimeError(
            f"Mask FITS must be 2D after squeeze(); got shape {data.shape}"
        )
    if expected_shape is not None and data.shape != expected_shape:
        raise RuntimeError(
            f"Mask shape {data.shape} does not match flux image shape {expected_shape}"
        )
    return ((~np.isfinite(data)) | (data != 0)).astype(bool)


def build_region_weights(region_file, shape, header, subpixels=8):
    regs = Regions.read(region_file, format="ds9")
    wcs = WCS(header)
    weights = np.zeros(shape, dtype=float)

    for reg in regs:
        if hasattr(reg, "to_mask"):
            preg = reg
        else:
            preg = reg.to_pixel(wcs)

        rmask = preg.to_mask(mode="subpixels", subpixels=subpixels)
        if rmask is None:
            continue

        img = rmask.to_image(shape)
        if img is None:
            continue

        weights = np.maximum(weights, np.asarray(img, dtype=float))

    return weights


def sigma_clipped_sky_sigma(data, good_sky):
    vals = data[good_sky]
    if vals.size < 10:
        return np.nan, np.nan, np.nan
    mean, med, std = sigma_clipped_stats(vals, sigma=3.0, maxiters=5)
    return float(mean), float(med), float(std)


def random_aperture_uncertainty(
    data,
    region_weights,
    ext_mask,
    nrand=1000,
    seed=12345,
    max_trials=None,
):
    """
    Empirical uncertainty from random translated apertures of the same effective shape.

    Uses the *unmasked* weighted region footprint as the aperture stamp.
    Candidate background placements must:
      - stay fully inside the image
      - avoid external mask
      - avoid the original target region
      - use finite data only

    Returns:
      sb_samples_nmgy_as2, area_arcsec2, sumw
    """
    finite = np.isfinite(data)
    valid_weights = np.where((~ext_mask) & finite, region_weights, 0.0)

    yy, xx = np.where(valid_weights > 0)
    if yy.size == 0:
        return np.array([]), np.nan, np.nan

    y0, y1 = yy.min(), yy.max() + 1
    x0, x1 = xx.min(), xx.max() + 1

    stamp = valid_weights[y0:y1, x0:x1]
    stamp_support = stamp > 0
    sh, sw = stamp.shape
    ny, nx = data.shape

    sumw = stamp.sum()
    if sumw <= 0:
        return np.array([]), np.nan, np.nan

    region_support = region_weights > 0
    rng = np.random.default_rng(seed)

    if max_trials is None:
        max_trials = max(10000, 50 * nrand)

    samples = []
    trials = 0

    while len(samples) < nrand and trials < max_trials:
        trials += 1
        ys = rng.integers(0, ny - sh + 1)
        xs = rng.integers(0, nx - sw + 1)

        data_sub = data[ys : ys + sh, xs : xs + sw]
        mask_sub = ext_mask[ys : ys + sh, xs : xs + sw]
        reg_sub = region_support[ys : ys + sh, xs : xs + sw]
        fin_sub = np.isfinite(data_sub)

        bad = stamp_support & (mask_sub | reg_sub | (~fin_sub))
        if np.any(bad):
            continue

        flux_sum = np.sum(data_sub * stamp)
        samples.append(flux_sum)

    return np.asarray(samples, dtype=float), float(sumw)


def write_check_fits(
    outpath,
    data,
    hdr,
    region_weights,
    valid_weights,
    ext_mask,
    pix_area,
    flux_sum_nmgy,
    area_arcsec2,
    sb_nmgy_arcsec2,
    mu_mag_arcsec2,
    fill_mode="nan",
):
    """
    Write a FITS file containing products useful for checking the measurement in DS9.

    HDUs:
      PRIMARY       = MASKED_NMGY (original data in valid region; fill elsewhere)
      WEIGHTED_NMGY = data * valid_weights; sum matches flux_sum_nmgy exactly
      AREA_AS2      = valid_weights * pix_area; sum matches area_arcsec2 exactly
      VALID_WT      = effective fractional weights after external masking
      REGION_WT     = pure region fractional weights before external masking
      EXT_MASK      = external mask (1=masked, 0=good)
    """
    fill_value = np.nan if fill_mode == "nan" else 0.0
    support = valid_weights > 0

    masked_nmgy = np.full(data.shape, fill_value, dtype=np.float32)
    weighted_nmgy = np.full(data.shape, fill_value, dtype=np.float32)
    area_img = np.full(data.shape, fill_value, dtype=np.float32)

    masked_nmgy[support] = data[support].astype(np.float32)
    weighted_nmgy[support] = (data[support] * valid_weights[support]).astype(np.float32)
    area_img[support] = (valid_weights[support] * pix_area).astype(np.float32)

    ph = hdr.copy()
    ph["BUNIT"] = "nMgy/pix"
    ph["PIXAS2"] = (float(pix_area), "Pixel area [arcsec^2]")
    ph["SUMFLUX"] = (float(flux_sum_nmgy), "Weighted flux sum in region [nMgy]")
    ph["AREAAS2"] = (float(area_arcsec2), "Weighted unmasked area [arcsec^2]")
    ph["MEANSB"] = (float(sb_nmgy_arcsec2), "Mean SB [nMgy/arcsec^2]")
    if np.isfinite(mu_mag_arcsec2):
        ph["MU_SB"] = (float(mu_mag_arcsec2), "Mean SB [mag/arcsec^2]")

    hdus = [
        fits.PrimaryHDU(data=masked_nmgy, header=ph),
        fits.ImageHDU(
            data=weighted_nmgy,
            header=fits.Header({"BUNIT": "nMgy/pix"}),
            name="WEIGHTED_NMGY",
        ),
        fits.ImageHDU(
            data=area_img,
            header=fits.Header({"BUNIT": "arcsec^2"}),
            name="AREA_AS2",
        ),
        fits.ImageHDU(
            data=valid_weights.astype(np.float32),
            header=fits.Header({"BUNIT": "fraction"}),
            name="VALID_WT",
        ),
        fits.ImageHDU(
            data=region_weights.astype(np.float32),
            header=fits.Header({"BUNIT": "fraction"}),
            name="REGION_WT",
        ),
        fits.ImageHDU(
            data=ext_mask.astype(np.uint8),
            header=fits.Header({"BUNIT": "mask"}),
            name="EXT_MASK",
        ),
    ]

    fits.HDUList(hdus).writeto(outpath, overwrite=True)


def main():
    ap = argparse.ArgumentParser(
        description="Measure mean surface brightness in a DS9 region from mu_from_nmgy.py output, excluding a supplied mask."
    )
    ap.add_argument("mu_fits", help="MEF output from mu_from_nmgy.py")
    ap.add_argument("region_file", help="DS9 region file defining the tail")
    ap.add_argument("mask_fits", help="Mask FITS; nonzero pixels are masked")
    ap.add_argument(
        "--flux-hdu",
        default="SKY_SUB_NMGY",
        help="Flux HDU to measure from (default: SKY_SUB_NMGY). "
        "Use SKY_SUB_SMOOTH_NMGY if you want the smoothed flux image.",
    )
    ap.add_argument(
        "--mask-hdu",
        default=None,
        help="Mask HDU index or EXTNAME (default: first image HDU in mask FITS).",
    )
    ap.add_argument(
        "--write-check-fits",
        default=None,
        help=("Optional FITS file to write masked/weighted images for DS9 checking."),
    )
    ap.add_argument(
        "--check-fill",
        choices=["nan", "zero"],
        default="nan",
        help=(
            "Fill value outside the valid region in the check FITS " "(default: nan)."
        ),
    )
    ap.add_argument(
        "--subpixels",
        type=int,
        default=8,
        help="Subpixel sampling for region rasterization (default: 8).",
    )
    ap.add_argument(
        "--nrand",
        type=int,
        default=1000,
        help="Number of random background apertures for empirical error (default: 1000).",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Random seed for aperture placement (default: 12345).",
    )
    args = ap.parse_args()

    flux_hdu = parse_hdu_spec(args.flux_hdu)
    mask_hdu = parse_hdu_spec(args.mask_hdu)

    data, hdr = read_hdu_data(args.mu_fits, flux_hdu)
    if data.ndim != 2:
        raise RuntimeError(f"Flux HDU must be 2D after squeeze(); got {data.shape}")

    pix_area = pixel_area_arcsec2(hdr)
    ext_mask = read_mask(args.mask_fits, mask_hdu=mask_hdu, expected_shape=data.shape)
    region_weights = build_region_weights(
        args.region_file, data.shape, hdr, subpixels=args.subpixels
    )

    if np.sum(region_weights > 0) == 0:
        raise RuntimeError(
            "Region does not overlap the image, or could not be rasterized."
        )

    finite = np.isfinite(data)
    valid_weights = np.where((~ext_mask) & finite, region_weights, 0.0)

    sumw = float(valid_weights.sum())
    if sumw <= 0:
        raise RuntimeError("No unmasked pixels remain inside the region.")

    flux_sum_nmgy = float(np.sum(data * valid_weights))
    area_arcsec2 = sumw * pix_area
    sb_nmgy_arcsec2 = flux_sum_nmgy / area_arcsec2
    mu_mag_arcsec2 = nmgy_as2_to_mu(sb_nmgy_arcsec2)

    if args.write_check_fits is not None:
        write_check_fits(
            outpath=args.write_check_fits,
            data=data,
            hdr=hdr,
            region_weights=region_weights,
            valid_weights=valid_weights,
            ext_mask=ext_mask,
            pix_area=pix_area,
            flux_sum_nmgy=flux_sum_nmgy,
            area_arcsec2=area_arcsec2,
            sb_nmgy_arcsec2=sb_nmgy_arcsec2,
            mu_mag_arcsec2=mu_mag_arcsec2,
            fill_mode=args.check_fill,
        )
        print()
        print(f"Wrote check FITS:         {args.write_check_fits}")
        print("  PRIMARY       = MASKED_NMGY")
        print("  WEIGHTED_NMGY = exact weighted flux image")
        print("  AREA_AS2      = exact weighted area image")

    # Background pool for naive sigma estimate
    good_sky = (~ext_mask) & finite & (region_weights == 0)
    sky_mean, sky_med, sky_sigma_pix = sigma_clipped_sky_sigma(data, good_sky)

    # Naive weighted error from clipped pixel sigma
    w2sum = float(np.sum(valid_weights * valid_weights))
    if np.isfinite(sky_sigma_pix) and w2sum > 0 and sumw > 0:
        sb_err_naive = sky_sigma_pix * math.sqrt(w2sum) / (pix_area * sumw)
    else:
        sb_err_naive = np.nan
    mu_err_naive = sb_err_to_mag_err(sb_nmgy_arcsec2, sb_err_naive)

    # Empirical random-aperture error
    rand_flux_sums, stamp_sumw = random_aperture_uncertainty(
        data=data,
        region_weights=region_weights,
        ext_mask=ext_mask,
        nrand=args.nrand,
        seed=args.seed,
    )

    if rand_flux_sums.size >= 10 and np.isfinite(stamp_sumw) and stamp_sumw > 0:
        rand_sb = rand_flux_sums / (stamp_sumw * pix_area)
        sb_err_emp = float(np.std(rand_sb, ddof=1))
        mu_err_emp = sb_err_to_mag_err(sb_nmgy_arcsec2, sb_err_emp)
        rand_sb_mean = float(np.mean(rand_sb))
        nrand_ok = int(rand_sb.size)
    else:
        sb_err_emp = np.nan
        mu_err_emp = np.nan
        rand_sb_mean = np.nan
        nrand_ok = int(rand_flux_sums.size)

    print()
    print("=== Region surface brightness measurement ===")
    print(f"Flux HDU:                 {args.flux_hdu}")
    print(f"Mask FITS:                {args.mask_fits}")
    print(
        f"Mask HDU:                 {args.mask_hdu if args.mask_hdu is not None else 'first data HDU'}"
    )
    print(f"Region file:              {args.region_file}")
    print(f"Pixel area:               {pix_area:.6f} arcsec^2/pixel")
    print(f"Subpixels:                {args.subpixels}")
    print()

    print(f"Weighted unmasked area:   {area_arcsec2:.3f} arcsec^2")
    print(f"Equivalent pixels:        {sumw:.3f}")
    print(f"Flux sum in region:       {flux_sum_nmgy:.6g} nMgy")
    print(f"Mean SB:                  {sb_nmgy_arcsec2:.6g} nMgy/arcsec^2")

    if np.isfinite(mu_mag_arcsec2):
        print(f"Mean SB:                  {mu_mag_arcsec2:.5f} mag/arcsec^2")
    else:
        print("Mean SB:                  undefined in mag/arcsec^2 (non-positive flux)")

    print()
    print("=== Error estimates ===")
    print(f"Sky sigma per pixel:      {sky_sigma_pix:.6g} nMgy/pixel")
    print(f"Sky mean (clipped):       {sky_mean:.6g} nMgy/pixel")
    print(f"Sky median (clipped):     {sky_med:.6g} nMgy/pixel")
    print()

    if np.isfinite(sb_err_naive):
        print(f"Naive SB error:           {sb_err_naive:.6g} nMgy/arcsec^2")
        if np.isfinite(mu_err_naive):
            print(f"Naive mag error:          {mu_err_naive:.5f} mag")
        else:
            print("Naive mag error:          undefined")
    else:
        print("Naive SB error:           unavailable")
        print("Naive mag error:          unavailable")

    print()

    print(f"Random apertures used:    {nrand_ok}")
    if np.isfinite(rand_sb_mean):
        print(f"Random-aperture mean SB:  {rand_sb_mean:.6g} nMgy/arcsec^2")
    else:
        print("Random-aperture mean SB:  unavailable")

    if np.isfinite(sb_err_emp):
        print(f"Empirical SB error:       {sb_err_emp:.6g} nMgy/arcsec^2")
        if np.isfinite(mu_err_emp):
            print(f"Empirical mag error:      {mu_err_emp:.5f} mag")
        else:
            print("Empirical mag error:      undefined")
    else:
        print("Empirical SB error:       unavailable")
        print("Empirical mag error:      unavailable")

    print()
    print("Recommended error to quote:")
    if np.isfinite(sb_err_emp):
        if np.isfinite(mu_err_emp):
            print(f"  {sb_nmgy_arcsec2:.6g} ± {sb_err_emp:.6g} nMgy/arcsec^2")
            if np.isfinite(mu_mag_arcsec2):
                print(f"  {mu_mag_arcsec2:.5f} ± {mu_err_emp:.5f} mag/arcsec^2")
        else:
            print(f"  {sb_nmgy_arcsec2:.6g} ± {sb_err_emp:.6g} nMgy/arcsec^2")
    elif np.isfinite(sb_err_naive):
        if np.isfinite(mu_err_naive):
            print(f"  {sb_nmgy_arcsec2:.6g} ± {sb_err_naive:.6g} nMgy/arcsec^2")
            if np.isfinite(mu_mag_arcsec2):
                print(f"  {mu_mag_arcsec2:.5f} ± {mu_err_naive:.5f} mag/arcsec^2")
        else:
            print(f"  {sb_nmgy_arcsec2:.6g} ± {sb_err_naive:.6g} nMgy/arcsec^2")
    else:
        print("  Could not compute a reliable uncertainty.")


if __name__ == "__main__":
    main()
