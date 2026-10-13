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
import csv
import math
import sys
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.table import Table
from astropy.units import UnitsWarning
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

warnings.filterwarnings(
    "ignore",
    message=r".*did not parse as fits unit.*",
    category=UnitsWarning,
)

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
    that already carry unit='deg'.
    """
    col = tab[colname]

    if getattr(col, "unit", None) is not None:
        return col.quantity.to(u.deg)

    return np.asarray(col, dtype=float) * u.deg


def tractor_transmission_from_region(
    tractor_file,
    region_weights,
    header,
    band="r",
    min_sources=3,
    min_region_weight=0.0,
):
    """
    Estimate the Galactic transmission for a DS9 region using Tractor rows
    whose RA/Dec fall inside the rasterized region.

    Returns the median MW_TRANSMISSION_<band> and diagnostic statistics.

    This uses nearest-pixel membership in the already rasterized region_weights.
    For Galactic extinction this is sufficient because MW transmission varies
    smoothly compared with the pixel scale.
    """
    band = band.lower().strip()
    trans_col_requested = f"mw_transmission_{band}"

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

    ny, nx = region_weights.shape
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

    if region_indices.size < int(min_sources):
        raise RuntimeError(
            f"Only {region_indices.size} Tractor source(s) found inside the "
            f"tail region, fewer than --min-tractor-sources={min_sources}. "
            f"Use a larger region, lower --min-tractor-sources, or fall back "
            f"to a tail-centroid RA/Dec correction."
        )

    vals = trans[region_indices]
    vals = vals[np.isfinite(vals) & (vals > 0)]

    if vals.size < int(min_sources):
        raise RuntimeError(
            f"Only {vals.size} valid positive {trans_col} values found inside "
            f"the tail region."
        )

    T_med = float(np.median(vals))
    A_band = float(-2.5 * np.log10(T_med))

    p16, p84 = np.percentile(vals, [16, 84])

    return {
        "method": "median_tractor_sources_inside_region",
        "tractor_file": tractor_file,
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
        "A_band_p16_mag": float(-2.5 * np.log10(p84)),
        "A_band_p84_mag": float(-2.5 * np.log10(p16)),
    }


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


def get_dereddening_metadata(header):
    """
    Extract dereddening metadata from the flux HDU header, if present.

    Expected keywords written by mu_from_legacy.py:
      DEREDDEN, DEREDBND, MWTRANS, A_BAND, DEREDRA, DEREDDEC
    """
    dered = bool(header.get("DEREDDEN", False))

    return {
        "dereddened": dered,
        "dered_band": header.get("DEREDBND", ""),
        "mw_transmission": header.get("MWTRANS", np.nan),
        "A_band_mag": header.get("A_BAND", np.nan),
        "dered_ra": header.get("DEREDRA", np.nan),
        "dered_dec": header.get("DEREDDEC", np.nan),
        "tractor_match_sep_arcsec": header.get("TRSEP", np.nan),
    }


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
    dered_info=None,
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
    if dered_info is not None:
        ph["DEREDDEN"] = (True, "Fluxes corrected for Galactic extinction")
        ph["DEREDMTH"] = ("REGION", "Dereddening method")
        ph["DEREDBND"] = (dered_info["band"], "Band used for MW transmission")
        ph["MWTRANS"] = (
            dered_info["mw_transmission_median"],
            "Median MW transmission in region",
        )
        ph["A_BAND"] = (dered_info["A_band_mag"], "Extinction correction [mag]")
        ph["NTRSRC"] = (
            dered_info["n_sources"],
            "Tractor sources used for transmission median",
        )
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


def write_summary_csv(outpath, row):
    """
    Write a one-row CSV summary for this run.
    If the file does not exist yet, write a header first.
    """
    fieldnames = list(row.keys())

    try:
        with open(outpath, "r", newline="") as f:
            has_header = True
    except FileNotFoundError:
        has_header = False

    with open(outpath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not has_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(
        description="Measure mean surface brightness in a DS9 region from mu_from_nmgy.py output, excluding a supplied mask."
    )
    ap.add_argument("mu_fits", help="MEF output from mu_from_nmgy.py")
    ap.add_argument("region_file", help="DS9 region file defining the tail")
    ap.add_argument("mask_fits", help="Mask FITS; nonzero pixels are masked")
    ap.add_argument(
        "--flux-hdu",
        default="DATA_NMGY",
        help=(
            "Flux HDU to measure from (default: DATA_NMGY). "
            "For dereddened Legacy comparisons use SKY_SUB_DERED_NMGY. "
            "Use SKY_SUB_SMOOTH_NMGY or SKY_SUB_SMOOTH_DERED_NMGY only if "
            "you intentionally want the smoothed flux image."
        ),
    )
    ap.add_argument(
        "--mask-hdu",
        default=None,
        help="Mask HDU index or EXTNAME (default: first image HDU in mask FITS).",
    )
    ap.add_argument(
        "--summary-csv",
        default=None,
        help="Optional CSV file to append a one-line summary of the measurement.",
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
    ap.add_argument(
        "--tractor-fits",
        default=None,
        help=(
            "Optional Legacy Tractor FITS catalog. Used with "
            "--deredden-from-tractor-region to estimate MW_TRANSMISSION_<band> "
            "from Tractor sources inside the DS9 region."
        ),
    )
    ap.add_argument(
        "--deredden-from-tractor-region",
        action="store_true",
        help=(
            "Deredden the flux image by dividing by the median "
            "MW_TRANSMISSION_<band> of Tractor sources inside the DS9 region."
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
        help=(
            "Minimum number of Tractor sources required inside the region "
            "for region-median dereddening."
        ),
    )
    ap.add_argument(
        "--dered-region-min-weight",
        type=float,
        default=0.0,
        help=(
            "Minimum rasterized region weight for a Tractor source to count "
            "as inside the region. Default 0 means any overlap."
        ),
    )
    ap.add_argument(
        "--allow-double-deredden",
        action="store_true",
        help=(
            "Allow dereddening even if the selected flux HDU is already marked "
            "DEREDDEN=True. Normally this is blocked to avoid double correction."
        ),
    )
    args = ap.parse_args()

    flux_hdu = parse_hdu_spec(args.flux_hdu)
    mask_hdu = parse_hdu_spec(args.mask_hdu)

    data, hdr = read_hdu_data(args.mu_fits, flux_hdu)
    if data.ndim != 2:
        raise RuntimeError(f"Flux HDU must be 2D after squeeze(); got {data.shape}")
    dered_meta = get_dereddening_metadata(hdr)
    pix_area = pixel_area_arcsec2(hdr)

    ext_mask = read_mask(args.mask_fits, mask_hdu=mask_hdu, expected_shape=data.shape)
    region_weights = build_region_weights(
        args.region_file, data.shape, hdr, subpixels=args.subpixels
    )

    if np.sum(region_weights > 0) == 0:
        raise RuntimeError(
            "Region does not overlap the image, or could not be rasterized."
        )

    dered_info = None

    if args.deredden_from_tractor_region:
        if args.tractor_fits is None:
            raise RuntimeError(
                "--deredden-from-tractor-region requires --tractor-fits."
            )

        already_dered = bool(hdr.get("DEREDDEN", False))
        if already_dered and not args.allow_double_deredden:
            raise RuntimeError(
                f"The selected flux HDU {args.flux_hdu!r} is already marked "
                "DEREDDEN=True. Use SKY_SUB_NMGY instead, or pass "
                "--allow-double-deredden if you really intend this."
            )

        dered_info = tractor_transmission_from_region(
            tractor_file=args.tractor_fits,
            region_weights=region_weights,
            header=hdr,
            band=args.dered_band,
            min_sources=args.min_tractor_sources,
            min_region_weight=args.dered_region_min_weight,
        )

        T_region = dered_info["mw_transmission_median"]

        # Apply Galactic-extinction correction in flux space.
        data = data / T_region

        print()
        print("=== Region-based Galactic dereddening ===")
        print(f"Tractor file:             {args.tractor_fits}")
        print(f"Band:                     {args.dered_band}")
        print(f"Method:                   {dered_info['method']}")
        print(f"Tractor sources used:     {dered_info['n_sources']}")
        print(f"MW transmission median:   {T_region:.8f}")
        print(
            f"A_{args.dered_band}:                    {dered_info['A_band_mag']:.5f} mag"
        )
        print(
            f"Transmission p16-p84:     "
            f"{dered_info['mw_transmission_p16']:.8f} – "
            f"{dered_info['mw_transmission_p84']:.8f}"
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
            dered_info=dered_info,
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

    # Preferred quoted uncertainties
    quoted_sb_err = sb_err_emp if np.isfinite(sb_err_emp) else sb_err_naive
    quoted_mu_err = mu_err_emp if np.isfinite(mu_err_emp) else mu_err_naive
    quoted_err_source = (
        "empirical_random_apertures"
        if np.isfinite(sb_err_emp)
        else "naive_sigma_clipped_sky" if np.isfinite(sb_err_naive) else "unavailable"
    )

    summary_row = {
        "mu_fits": args.mu_fits,
        "region_file": args.region_file,
        "mask_fits": args.mask_fits,
        "flux_hdu": args.flux_hdu,
        "mask_hdu": args.mask_hdu if args.mask_hdu is not None else "first data HDU",
        "dereddened": dered_info is not None,
        "dered_method": "" if dered_info is None else dered_info["method"],
        "dered_band": "" if dered_info is None else dered_info["band"],
        "tractor_sources_for_dered": (
            "" if dered_info is None else dered_info["n_sources"]
        ),
        "mw_transmission_region_median": (
            "" if dered_info is None else dered_info["mw_transmission_median"]
        ),
        "mw_transmission_region_p16": (
            "" if dered_info is None else dered_info["mw_transmission_p16"]
        ),
        "mw_transmission_region_p84": (
            "" if dered_info is None else dered_info["mw_transmission_p84"]
        ),
        "A_band_mag": "" if dered_info is None else dered_info["A_band_mag"],
        "subpixels": args.subpixels,
        "nrand_requested": args.nrand,
        "nrand_used": nrand_ok,
        "seed": args.seed,
        "pixel_area_arcsec2": pix_area,
        "weighted_unmasked_area_arcsec2": area_arcsec2,
        "equivalent_pixels": sumw,
        "flux_sum_nmgy": flux_sum_nmgy,
        "mean_sb_nmgy_arcsec2": sb_nmgy_arcsec2,
        "mean_sb_mag_arcsec2": mu_mag_arcsec2,
        "sky_sigma_pix_nmgy": sky_sigma_pix,
        "sky_mean_nmgy": sky_mean,
        "sky_median_nmgy": sky_med,
        "naive_sb_err_nmgy_arcsec2": sb_err_naive,
        "naive_mag_err": mu_err_naive,
        "rand_ap_mean_sb_nmgy_arcsec2": rand_sb_mean,
        "empirical_sb_err_nmgy_arcsec2": sb_err_emp,
        "empirical_mag_err": mu_err_emp,
        "quoted_sb_err_nmgy_arcsec2": quoted_sb_err,
        "quoted_mag_err": quoted_mu_err,
        "quoted_error_source": quoted_err_source,
    }

    if args.summary_csv is not None:
        write_summary_csv(args.summary_csv, summary_row)
        print()
        print(f"Wrote summary CSV:        {args.summary_csv}")

    print()
    print("=== Region surface brightness measurement ===")
    print(f"Flux HDU:                 {args.flux_hdu}")
    if dered_info is not None:
        print(
            f"Dereddened:               yes, region median "
            f"MW_TRANSMISSION_{args.dered_band.upper()}="
            f"{dered_info['mw_transmission_median']:.8f}, "
            f"A_{args.dered_band}={dered_info['A_band_mag']:.5f} mag, "
            f"N={dered_info['n_sources']}"
        )
    else:
        print("Dereddened:               no")
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
