#!/usr/bin/env python3
"""
Compare two FITS images (actual vs synthetic) by sliding the synthetic image
up to +/- max_shift pixels in x and y.

Key features:
  - HDU selection by index or name (e.g. MU_OBS).
  - Compute:
      * CC(dy,dx): unnormalized cross-correlation (matched-filter response)
      * SIM(dy,dx): normalized mean-subtracted correlation coefficient in [-1, 1]
  - Optional smoothing of CC(dy,dx) *in shift space* before peak selection.
  - Plot similarity vs shift (SIM map).
  - Optionally save a circularly shifted synthetic FITS using the selected (dy, dx).

Conventions:
  dy > 0 => synthetic must be shifted DOWN relative to actual
  dx > 0 => synthetic must be shifted RIGHT relative to actual
"""

import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

# ---------------------------------------------------------------------
# FITS helpers
# ---------------------------------------------------------------------


def parse_hdu_spec(spec):
    """Parse HDU spec: int index or EXTNAME string. None -> 0."""
    if spec is None:
        return 0
    try:
        return int(spec)
    except (TypeError, ValueError):
        return spec


def load_fits_2d(path, hdu_spec):
    """Load a 2D image from FITS using HDU index or name."""
    with fits.open(path, memmap=True) as hdul:
        try:
            hdu = hdul[hdu_spec]
        except (KeyError, IndexError, TypeError) as exc:
            names = [f"{i}: {h.name!r}" for i, h in enumerate(hdul)]
            msg = (
                f"Could not find HDU {hdu_spec!r} in {path!r}.\n"
                f"Available HDUs (index: EXTNAME):\n  " + "\n  ".join(names)
            )
            raise ValueError(msg) from exc
        data = hdu.data

    if data is None:
        raise ValueError(
            f"No data found in HDU {hdu_spec!r} of {path!r} (data is None)."
        )

    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError(
            f"Expected a 2D image in {path!r} (HDU={hdu_spec!r}), got shape {data.shape}"
        )
    return data


# ---------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------


def preprocess_for_correlation(
    img,
    *,
    treat_zero_as_nan: bool = False,
    img_smooth_sigma: float = 0.0,
    clip_sigma: float = 0.0,
    standardize: bool = True,
):
    """
    Prepare an image for correlation:
      - Optionally treat exact zeros as invalid (useful if masked pixels were written as 0)
      - Optional Gaussian smoothing in IMAGE space (NOT shift space)
      - Replace non-finite pixels by the median of finite pixels
      - Robustly clip extreme values to reduce core dominance
      - Subtract median (remove DC/background)
      - Optionally divide by robust sigma (MAD-based) so both images are comparable

    Returns a float64 array.
    """
    arr = np.asarray(img, dtype=np.float64).copy()

    if treat_zero_as_nan:
        arr[arr == 0.0] = np.nan

    if img_smooth_sigma and img_smooth_sigma > 0:
        # Image-space smoothing for both actual and synthetic prior to correlation
        arr = gaussian_filter(arr, sigma=float(img_smooth_sigma), mode="nearest")

    finite = np.isfinite(arr)
    if not np.any(finite):
        raise ValueError("Image has no finite pixels after preprocessing.")

    med = np.median(arr[finite])

    # Robust sigma from MAD (fallback to std if needed)
    mad = np.median(np.abs(arr[finite] - med))
    rsig = 1.4826 * mad
    if not np.isfinite(rsig) or rsig <= 0:
        rsig = np.std(arr[finite])
    if not np.isfinite(rsig) or rsig <= 0:
        rsig = 1.0

    # Clip extremes (helps tails by suppressing bright residual knots/cores)
    if clip_sigma and clip_sigma > 0:
        lo = med - float(clip_sigma) * rsig
        hi = med + float(clip_sigma) * rsig
        arr[finite] = np.clip(arr[finite], lo, hi)

    # Fill invalid with the (possibly clipped) median
    arr[~finite] = med

    # Remove DC
    arr -= med

    # Normalize scale (optional but recommended)
    if standardize:
        arr /= rsig

    return arr


def overlap_slices(shape, dy, dx):
    """
    Overlap slices for two same-shaped images when synthetic is shifted by (dy, dx)
    relative to actual (no wrap; only overlapping region).
    """
    H, W = shape

    # Rows
    if dy >= 0:
        y0_a, y1_a = dy, H
        y0_s, y1_s = 0, H - dy
    else:
        y0_a, y1_a = 0, H + dy
        y0_s, y1_s = -dy, H

    # Cols
    if dx >= 0:
        x0_a, x1_a = dx, W
        x0_s, x1_s = 0, W - dx
    else:
        x0_a, x1_a = 0, W + dx
        x0_s, x1_s = -dx, W

    return (slice(y0_a, y1_a), slice(x0_a, x1_a), slice(y0_s, y1_s), slice(x0_s, x1_s))


def integral_image(img):
    """Summed-area table with shape (H+1, W+1)."""
    H, W = img.shape
    ii = np.zeros((H + 1, W + 1), dtype=np.float64)
    ii[1:, 1:] = np.cumsum(np.cumsum(img, axis=0), axis=1)
    return ii


def rect_sum(ii, y0, y1, x0, x1):
    """Sum over rectangle [y0:y1, x0:x1] using an integral image."""
    return ii[y1, x1] - ii[y0, x1] - ii[y1, x0] + ii[y0, x0]


def shift_image_nonwrap(
    img: np.ndarray, dy: int, dx: int, fill_value: float
) -> np.ndarray:
    """
    Shift an image by (dy, dx) WITHOUT wrapping.
    Pixels that roll off the edge are discarded; new pixels are filled with fill_value.

    Conventions match the rest of this script:
      dy > 0 => move content DOWN
      dx > 0 => move content RIGHT
    """
    H, W = img.shape
    out = np.full((H, W), fill_value, dtype=np.float64)

    # Destination bounds
    y0d = max(dy, 0)
    y1d = min(H + dy, H)
    x0d = max(dx, 0)
    x1d = min(W + dx, W)

    # Source bounds
    y0s = max(-dy, 0)
    y1s = y0s + (y1d - y0d)
    x0s = max(-dx, 0)
    x1s = x0s + (x1d - x0d)

    if (y1d > y0d) and (x1d > x0d):
        out[y0d:y1d, x0d:x1d] = img[y0s:y1s, x0s:x1s]

    return out


def apply_zero_mask_policy(img: np.ndarray, treat_zero_as_nan: bool) -> np.ndarray:
    x = np.asarray(img, dtype=np.float64, copy=True)
    if treat_zero_as_nan:
        x[x == 0.0] = np.nan
    return x


def compute_maps(
    actual,
    synthetic,
    max_shift,
    *,
    treat_zero_as_nan=False,
    img_smooth_sigma=0.0,
    clip_sigma=0.0,
    standardize=True,
):
    """
    Compute within the +/- max_shift window:
      - cc_map(dy,dx): unnormalized cross-correlation (matched-filter response)
      - sim_map(dy,dx): normalized mean-subtracted correlation coefficient in [-1,1]
      - n_pix_map(dy,dx): overlap pixel count

    Returns:
      shift_ys, shift_xs, sim_map, cc_map, n_pix_map
    """
    if actual.shape != synthetic.shape:
        raise ValueError(
            f"Image shapes differ: actual={actual.shape}, synthetic={synthetic.shape}"
        )

    H, W = actual.shape
    max_shift_y = min(max_shift, H - 1)
    max_shift_x = min(max_shift, W - 1)

    a = preprocess_for_correlation(
        actual,
        treat_zero_as_nan=treat_zero_as_nan,
        img_smooth_sigma=img_smooth_sigma,
        clip_sigma=clip_sigma,
        standardize=standardize,
    )
    s = preprocess_for_correlation(
        synthetic,
        treat_zero_as_nan=treat_zero_as_nan,
        img_smooth_sigma=img_smooth_sigma,
        clip_sigma=clip_sigma,
        standardize=standardize,
    )

    # Full cross-correlation via FFT: correlate2d(a, s) == fftconvolve(a, s[::-1,::-1])
    cc_full = fftconvolve(a, s[::-1, ::-1], mode="full").astype(np.float64)

    # For normalized SIM map we need local sums/variances over overlap
    ii_a = integral_image(a)
    ii_s = integral_image(s)
    ii_a2 = integral_image(a**2)
    ii_s2 = integral_image(s**2)

    cy, cx = H - 1, W - 1  # index of zero shift in cc_full

    shift_ys = np.arange(-max_shift_y, max_shift_y + 1, dtype=int)
    shift_xs = np.arange(-max_shift_x, max_shift_x + 1, dtype=int)

    sim_map = np.full((shift_ys.size, shift_xs.size), np.nan, dtype=np.float64)
    cc_map = np.full_like(sim_map, np.nan, dtype=np.float64)
    n_pix_map = np.zeros_like(sim_map, dtype=int)

    for iy, dy in enumerate(shift_ys):
        for ix, dx in enumerate(shift_xs):
            sa_y, sa_x, ss_y, ss_x = overlap_slices(actual.shape, dy, dx)
            y0a, y1a = sa_y.start, sa_y.stop
            x0a, x1a = sa_x.start, sa_x.stop
            y0s, y1s = ss_y.start, ss_y.stop
            x0s, x1s = ss_x.start, ss_x.stop

            h = y1a - y0a
            w = x1a - x0a
            n = h * w
            if n <= 0:
                continue

            # Unnormalized cross-correlation (matched-filter response) at this shift
            iy_cc = cy + dy
            ix_cc = cx + dx
            sum_as = cc_full[iy_cc, ix_cc]
            cc_map[iy, ix] = sum_as

            # Normalized similarity (mean-subtracted corrcoef over overlap)
            sum_a = rect_sum(ii_a, y0a, y1a, x0a, x1a)
            sum_s = rect_sum(ii_s, y0s, y1s, x0s, x1s)
            sum_a2 = rect_sum(ii_a2, y0a, y1a, x0a, x1a)
            sum_s2 = rect_sum(ii_s2, y0s, y1s, x0s, x1s)

            mean_a = sum_a / n
            mean_s = sum_s / n

            cov = sum_as - n * mean_a * mean_s
            var_a = sum_a2 - n * mean_a**2
            var_s = sum_s2 - n * mean_s**2

            denom = math.sqrt(var_a * var_s) if (var_a > 0 and var_s > 0) else 0.0
            if denom > 0:
                sim_map[iy, ix] = cov / denom
                n_pix_map[iy, ix] = n

    if np.all(~np.isfinite(cc_map)):
        raise ValueError(
            "Could not compute any finite cross-correlation values in the shift window."
        )

    return shift_ys, shift_xs, sim_map, cc_map, n_pix_map


def compute_maps_masked_fft(
    actual,
    synthetic,
    max_shift,
    *,
    treat_zero_as_nan: bool = True,
    treat_nans_as_zeros: bool = False,
    clip_sigma: float = 0.0,
    standardize: bool = True,
    min_pairs: int = 10,
):
    """
    Mask-aware version of compute_maps:
      - Invalid pixels are those that are non-finite (and optionally exact zeros).
      - If treat_nans_as_zeros=True: NaNs are treated as valid zeros (included in weights).
      - Uses FFT-based weighted sums to compute covariance/variance per shift.
      - cc_map is the mean-subtracted matched-filter response (covariance term).
    """
    if actual.shape != synthetic.shape:
        raise ValueError(
            f"Image shapes differ: actual={actual.shape}, synthetic={synthetic.shape}"
        )

    H, W = actual.shape
    max_shift_y = min(max_shift, H - 1)
    max_shift_x = min(max_shift, W - 1)

    A = np.asarray(actual, dtype=np.float64).copy()
    S = np.asarray(synthetic, dtype=np.float64).copy()

    if treat_zero_as_nan:
        A[A == 0.0] = np.nan
        S[S == 0.0] = np.nan

    # Weights: valid pixels
    WA = np.isfinite(A).astype(np.float64)
    WS = np.isfinite(S).astype(np.float64)

    # If NaNs should count as *valid* zeros, force weights to 1 there and fill NaNs with 0
    if treat_nans_as_zeros:
        WA[:] = 1.0
        WS[:] = 1.0
        A = np.nan_to_num(A, nan=0.0)
        S = np.nan_to_num(S, nan=0.0)

    # Robust centering / optional clipping / optional standardization (finite pixels only)
    def _robust_center_scale(X, Wt):
        fin = Wt > 0
        if not np.any(fin):
            return X, 0.0, 1.0
        med = np.nanmedian(X[fin])
        mad = np.nanmedian(np.abs(X[fin] - med))
        rsig = 1.4826 * mad
        if not np.isfinite(rsig) or rsig <= 0:
            rsig = np.nanstd(X[fin])
        if not np.isfinite(rsig) or rsig <= 0:
            rsig = 1.0

        if clip_sigma and clip_sigma > 0:
            lo = med - clip_sigma * rsig
            hi = med + clip_sigma * rsig
            X[fin] = np.clip(X[fin], lo, hi)

        X[fin] = X[fin] - med
        if standardize:
            X[fin] = X[fin] / rsig

        return X, med, rsig

    A, _, _ = _robust_center_scale(A, WA)
    S, _, _ = _robust_center_scale(S, WS)

    # Replace invalids by 0 for convolution; weights handle validity
    A0 = np.nan_to_num(A, nan=0.0)
    S0 = np.nan_to_num(S, nan=0.0)

    # Flip for correlation
    S0f = S0[::-1, ::-1]
    WSf = WS[::-1, ::-1]

    sum_as_full = fftconvolve(A0, S0f, mode="full").astype(np.float64)
    n_full = fftconvolve(WA, WSf, mode="full").astype(np.float64)

    sum_a_full = fftconvolve(A0, WSf, mode="full").astype(np.float64)
    sum_s_full = fftconvolve(WA, S0f, mode="full").astype(np.float64)
    sum_a2_full = fftconvolve(A0 * A0, WSf, mode="full").astype(np.float64)
    sum_s2_full = fftconvolve(WA, (S0 * S0)[::-1, ::-1], mode="full").astype(np.float64)

    cy, cx = H - 1, W - 1
    shift_ys = np.arange(-max_shift_y, max_shift_y + 1, dtype=int)
    shift_xs = np.arange(-max_shift_x, max_shift_x + 1, dtype=int)

    sim_map = np.full((shift_ys.size, shift_xs.size), np.nan, dtype=np.float64)
    cc_map = np.full_like(sim_map, np.nan, dtype=np.float64)
    n_pix_map = np.zeros_like(sim_map, dtype=int)

    for iy, dy in enumerate(shift_ys):
        for ix, dx in enumerate(shift_xs):
            yy = cy + dy
            xx = cx + dx
            n = n_full[yy, xx]
            if not np.isfinite(n) or n < min_pairs:
                continue

            sum_as = sum_as_full[yy, xx]
            sum_a = sum_a_full[yy, xx]
            sum_s = sum_s_full[yy, xx]
            sum_a2 = sum_a2_full[yy, xx]
            sum_s2 = sum_s2_full[yy, xx]

            cov = sum_as - (sum_a * sum_s) / n
            var_a = sum_a2 - (sum_a * sum_a) / n
            var_s = sum_s2 - (sum_s * sum_s) / n

            denom = math.sqrt(var_a * var_s) if (var_a > 0 and var_s > 0) else 0.0
            if denom > 0:
                sim_map[iy, ix] = cov / denom
                n_pix_map[iy, ix] = int(round(n))

            cc_map[iy, ix] = cov  # use covariance for peak selection

    if np.all(~np.isfinite(cc_map)):
        raise ValueError("No finite values computed in masked FFT maps.")

    return shift_ys, shift_xs, sim_map, cc_map, n_pix_map


def pick_best_shift_from_cc(shift_ys, shift_xs, cc_map, cc_smooth_sigma=0.0):
    """
    Pick (dy, dx) from the cross-correlation surface, optionally smoothed
    in shift space with a Gaussian kernel.

    Returns:
      dy_best, dx_best, cc_best_raw, cc_best_pick, cc_pick_map
    """
    cc_pick_map = cc_map
    if cc_smooth_sigma and cc_smooth_sigma > 0:
        # cc_map is finite in practice; smooth in shift-space
        cc_pick_map = gaussian_filter(cc_map, sigma=cc_smooth_sigma, mode="nearest")

    best_flat = np.nanargmax(cc_pick_map)
    best_iy, best_ix = np.unravel_index(best_flat, cc_pick_map.shape)

    dy_best = int(shift_ys[best_iy])
    dx_best = int(shift_xs[best_ix])

    cc_best_raw = float(cc_map[best_iy, best_ix])
    cc_best_pick = float(cc_pick_map[best_iy, best_ix])

    return dy_best, dx_best, cc_best_raw, cc_best_pick, cc_pick_map


def residual_metrics_at_shift(actual, synthetic, dy, dx):
    """
    Residual metrics between actual and synthetic at a given shift:
      - overlap only (no wrap)
      - mask non-finite
      - subtract local medians
    """
    sa_y, sa_x, ss_y, ss_x = overlap_slices(actual.shape, dy, dx)
    a = actual[sa_y, sa_x]
    s = synthetic[ss_y, ss_x]

    mask = np.isfinite(a) & np.isfinite(s)
    if not np.any(mask):
        return math.nan, math.nan, math.nan, 0

    a = a[mask]
    s = s[mask]

    a = a - np.median(a)
    s = s - np.median(s)

    diff = a - s
    mse = float(np.mean(diff**2))
    rms = float(np.sqrt(mse))
    std_a = float(np.std(a))
    nrmse = rms / std_a if std_a > 0 else math.nan

    return mse, rms, nrmse, int(a.size)


def residual_metrics_at_shift_masked_pairs(
    actual, synthetic, dy, dx, *, fit_affine=True
):
    """
    Residual metrics at a shift using the same valid-pair logic as masked FFT:
      - valid pairs: finite(actual) & finite(shifted_synthetic) in the overlap region
      - subtract local medians on the valid pairs
      - optionally fit avc ≈ alpha*svc + beta (least squares) before computing residuals

    Returns:
      mse, rms, nrmse, n,
      mse_fit, rms_fit, nrmse_fit, alpha, beta
    """
    sa_y, sa_x, ss_y, ss_x = overlap_slices(actual.shape, dy, dx)
    a = actual[sa_y, sa_x]
    s = synthetic[ss_y, ss_x]

    ok = np.isfinite(a) & np.isfinite(s)
    if not np.any(ok):
        return (
            math.nan,
            math.nan,
            math.nan,
            0,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
            math.nan,
        )

    av = np.asarray(a[ok], dtype=np.float64)
    sv = np.asarray(s[ok], dtype=np.float64)

    # Local median subtraction (your current policy)
    avc = av - np.median(av)
    svc = sv - np.median(sv)

    # Unscaled residuals
    diff = avc - svc
    mse = float(np.mean(diff * diff))
    rms = float(np.sqrt(mse))

    std_a = float(np.std(avc))
    nrmse = rms / std_a if std_a > 0 else math.nan

    # Optional affine fit: avc ≈ alpha*svc + beta
    if fit_affine:
        X = np.vstack([svc, np.ones_like(svc)]).T  # [svc, 1]
        alpha, beta = np.linalg.lstsq(X, avc, rcond=None)[0]

        diff_fit = avc - (alpha * svc + beta)
        mse_fit = float(np.mean(diff_fit * diff_fit))
        rms_fit = float(np.sqrt(mse_fit))
        nrmse_fit = rms_fit / std_a if std_a > 0 else math.nan
    else:
        alpha, beta = 1.0, 0.0
        mse_fit, rms_fit, nrmse_fit = math.nan, math.nan, math.nan
    corr_med = float(np.corrcoef(avc, svc)[0, 1])

    return (
        mse,
        rms,
        nrmse,
        int(avc.size),
        mse_fit,
        rms_fit,
        nrmse_fit,
        float(alpha),
        float(beta),
        float(corr_med),
    )


def smooth_map_nanaware(x: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smooth in shift-space, ignoring NaNs (normalized by weights)."""
    if not sigma or sigma <= 0:
        return x
    x = np.asarray(x, dtype=np.float64)
    w = np.isfinite(x).astype(np.float64)
    x0 = np.nan_to_num(x, nan=0.0)

    num = gaussian_filter(x0 * w, sigma=float(sigma), mode="nearest")
    den = gaussian_filter(w, sigma=float(sigma), mode="nearest")
    out = np.full_like(x0, np.nan, dtype=np.float64)

    # IMPORTANT: do NOT "inpaint" NaN pixels. Only define output where the *raw* value was finite.
    m = (den > 1e-12) & (w > 0)
    out[m] = num[m] / den[m]
    return out


def pick_best_shift_from_metric(
    shift_ys: np.ndarray,
    shift_xs: np.ndarray,
    metric_map: np.ndarray,
    smooth_sigma: float = 0.0,
):
    """
    Pick (dy, dx) by maximizing metric_map (optionally NaN-aware smoothed).
    Returns: dy_best, dx_best, metric_best_raw, metric_best_pick, metric_pick_map
    """
    metric_pick_map = smooth_map_nanaware(metric_map, smooth_sigma)

    # argmax while ignoring NaNs
    if not np.any(np.isfinite(metric_pick_map)):
        raise ValueError("Peak metric map has no finite values.")
    best_flat = np.nanargmax(metric_pick_map)
    best_iy, best_ix = np.unravel_index(best_flat, metric_pick_map.shape)

    dy_best = int(shift_ys[best_iy])
    dx_best = int(shift_xs[best_ix])

    metric_best_raw = float(metric_map[best_iy, best_ix])
    metric_best_pick = float(metric_pick_map[best_iy, best_ix])

    return dy_best, dx_best, metric_best_raw, metric_best_pick, metric_pick_map


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def plot_similarity_map(
    shift_ys,
    shift_xs,
    sim_map,
    dy_best,
    dx_best,
    output=None,
    show=True,
    title="Similarity vs shift",
):
    """2D heatmap of normalized similarity vs shift; marks best and zero shifts."""
    fig, ax = plt.subplots()

    extent = [
        shift_xs[0] - 0.5,
        shift_xs[-1] + 0.5,
        shift_ys[0] - 0.5,
        shift_ys[-1] + 0.5,
    ]

    im = ax.imshow(sim_map, origin="lower", extent=extent, aspect="auto")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized similarity")

    ax.set_xlabel("dx (pixels, synthetic RIGHT relative to actual)")
    ax.set_ylabel("dy (pixels, synthetic DOWN relative to actual)")
    ax.set_title(title)

    ax.plot(
        dx_best,
        dy_best,
        marker="o",
        markersize=8,
        markerfacecolor="none",
        markeredgecolor="white",
        markeredgewidth=2,
        label="Selected shift",
    )
    ax.plot(
        0,
        0,
        marker="+",
        markersize=10,
        markeredgecolor="red",
        label="Zero shift",
    )

    ax.legend(loc="best")
    fig.tight_layout()

    if output is not None:
        fig.savefig(output, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two FITS images (actual vs synthetic) by scanning shifts.\n"
            "Peak selection is performed on the cross-correlation (matched-filter) surface,\n"
            "optionally smoothed in shift space before choosing the peak."
        )
    )
    parser.add_argument("actual", help="FITS file with the actual (observed) image.")
    parser.add_argument("synthetic", help="FITS file with the synthetic/model image.")

    parser.add_argument(
        "--actual-hdu",
        default=None,
        help="HDU for actual image: integer index or extension name (e.g. MU_OBS). Default: 0.",
    )
    parser.add_argument(
        "--synthetic-hdu",
        default=None,
        help="HDU for synthetic image: integer index or extension name. Default: 0.",
    )

    parser.add_argument(
        "--max-shift",
        type=int,
        default=100,
        help="Maximum absolute shift (pixels) to search in y and x (default: 100).",
    )
    parser.add_argument(
        "--cc-smooth-sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma (pixels in shift space) to smooth cross-correlation before peak selection (default: 0=off).",
    )
    parser.add_argument(
        "--peak-metric",
        choices=["sim", "sim_weighted", "cc", "cc_per_pair"],
        default="sim",
        help=(
            "Which surface to maximize to choose the shift. "
            "sim = normalized correlation coefficient (recommended if you want the SIM peak). "
            "cc = covariance/matched-filter response (biases toward shifts with more valid pixels). "
            "cc_per_pair = cc / n_valid (reduces overlap bias)."
        ),
    )
    parser.add_argument(
        "--plot-out",
        type=str,
        default=None,
        help="Optional path to save the similarity-vs-shift plot (e.g. similarity.png).",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the plot interactively (useful on headless systems).",
    )
    parser.add_argument(
        "--keep-zeros",
        action="store_true",
        help="Do NOT treat 0.0 as masked. Default behavior treats 0.0 as invalid.",
    )
    parser.add_argument(
        "--img-smooth-sigma",
        type=float,
        default=0.0,
        help="Gaussian sigma (pixels) for IMAGE-space smoothing applied to both images before computing CC/SIM (default: 0=off).",
    )
    parser.add_argument(
        "--clip-sigma",
        type=float,
        default=0.0,
        help="Robust clip threshold in units of MAD-sigma before correlation (e.g., 5 or 8). Helps prevent cores from dominating.",
    )
    parser.add_argument(
        "--no-standardize",
        action="store_true",
        help="Disable robust standardization (division by MAD-sigma) in preprocessing.",
    )
    parser.add_argument(
        "--shifted-image",
        type=str,
        default=None,
        help=(
            "If provided, write a new FITS file where the chosen synthetic HDU "
            "has been circularly shifted by the selected (dy, dx) using np.roll (wrap)."
        ),
    )
    parser.add_argument(
        "--shifted-image-mode",
        choices=["wrap", "constant", "nan"],
        default="nan",
        help="How to shift when writing --shifted-image: wrap=np.roll, constant=fill with --shifted-fill, nan=fill with NaN (float output).",
    )
    parser.add_argument(
        "--overlap-gamma",
        type=float,
        default=1.0,
        help="For sim_weighted: metric = sim * (n_valid/n_max)**gamma (default: 1.0).",
    )
    parser.add_argument(
        "--shifted-fill",
        type=float,
        default=0.0,
        help="Fill value used when --shifted-image-mode=constant (default: 0.0).",
    )
    parser.add_argument(
        "--min-overlap-frac",
        type=float,
        default=0.80,
        help="Require n_valid >= this fraction of the maximum n_valid before considering a shift (default: 0.80).",
    )
    parser.add_argument(
        "--min-pairs",
        type=int,
        default=10000,
        help="Absolute minimum number of valid pixel pairs required for a shift (default: 50000).",
    )

    args = parser.parse_args()

    actual_hdu = parse_hdu_spec(args.actual_hdu)
    synthetic_hdu = parse_hdu_spec(args.synthetic_hdu)

    actual = load_fits_2d(args.actual, actual_hdu)
    synthetic = load_fits_2d(args.synthetic, synthetic_hdu)

    treat_zero_as_nan = not args.keep_zeros

    actual = apply_zero_mask_policy(actual, treat_zero_as_nan)
    synthetic = apply_zero_mask_policy(synthetic, treat_zero_as_nan)
    if treat_zero_as_nan:
        frac_zero_a = float(np.mean(actual == 0.0))
        frac_zero_s = float(np.mean(synthetic == 0.0))
        frac_nan_a = float(np.mean(~np.isfinite(actual)))
        frac_nan_s = float(np.mean(~np.isfinite(synthetic)))

        # If you're expecting NaN-encoded masks but still see lots of zeros, warn.
        if frac_zero_a > 0.01 and frac_nan_a < 1e-4:
            print(
                f"[Warn] actual has {frac_zero_a:.3%} zeros but {frac_nan_a:.3%} NaNs; upstream mask may be 0-encoded."
            )
        if frac_zero_s > 0.01 and frac_nan_s < 1e-4:
            print(
                f"[Warn] synthetic has {frac_zero_s:.3%} zeros but {frac_nan_s:.3%} NaNs; upstream mask may be 0-encoded."
            )

    if actual.shape != synthetic.shape:
        raise SystemExit(
            f"ERROR: Images must have the same shape. Got actual={actual.shape}, synthetic={synthetic.shape}"
        )

    print(f"Image shape: {actual.shape}")
    print(f"Using actual HDU: {actual_hdu!r}, synthetic HDU: {synthetic_hdu!r}")
    print(f"Searching shifts up to +/- {args.max_shift} px in y and x...")
    shift_ys, shift_xs, sim_map, cc_map, n_pix_map = compute_maps_masked_fft(
        actual,
        synthetic,
        max_shift=args.max_shift,
        treat_zero_as_nan=False,  # already applied
        treat_nans_as_zeros=False,
        clip_sigma=float(args.clip_sigma),
        standardize=(not args.no_standardize),
        min_pairs=args.min_pairs,
    )
    nmax = float(np.nanmax(n_pix_map.astype(np.float64)))

    if args.peak_metric == "sim":
        peak_map = sim_map
        metric_label = "SIM (corrcoef)"
    elif args.peak_metric == "sim_weighted":
        frac = n_pix_map.astype(np.float64) / max(nmax, 1.0)
        peak_map = sim_map * np.power(frac, float(args.overlap_gamma))
        metric_label = f"SIM * (n_valid/n_max)^{args.overlap_gamma:g}"
    elif args.peak_metric == "cc":
        peak_map = cc_map
        metric_label = "CC (cov)"
    elif args.peak_metric == "cc_per_pair":
        peak_map = cc_map / np.maximum(n_pix_map.astype(np.float64), 1.0)
        metric_label = "CC / n_valid"
    else:
        raise RuntimeError("Unknown --peak-metric")

    # -------------------------
    # enforce overlap/valid-pair gating so peak selection matches what you see in SIM
    # -------------------------
    nmax = int(np.nanmax(n_pix_map)) if np.any(n_pix_map > 0) else 0
    n_required = max(
        int(args.min_pairs), int(round(float(args.min_overlap_frac) * nmax))
    )

    if nmax <= 0:
        raise RuntimeError("n_pix_map has no valid entries; check masks / inputs.")

    # Mask out shifts with insufficient valid pixel pairs
    peak_map_gated = np.array(peak_map, dtype=np.float64, copy=True)
    bad = n_pix_map < n_required
    peak_map_gated[bad] = np.nan

    kept = np.count_nonzero(~bad)
    total = peak_map_gated.size
    print(
        f"Peak gating: require n_valid >= {n_required} (max={nmax}, frac={args.min_overlap_frac:.3f}). "
        f"Kept {kept}/{total} shifts."
    )
    dy_best, dx_best, peak_best_raw, peak_best_pick, _ = pick_best_shift_from_metric(
        shift_ys, shift_xs, peak_map_gated, smooth_sigma=args.cc_smooth_sigma
    )

    if args.cc_smooth_sigma and args.cc_smooth_sigma > 0:
        print(
            f"Peak selection maximized {metric_label} with Gaussian smoothing sigma={args.cc_smooth_sigma} px (shift space)."
        )
        print(f"Peak metric at selected shift (smoothed surface): {peak_best_pick:.6e}")
        print(f"Peak metric at selected shift (raw surface):      {peak_best_raw:.6e}")
    else:
        print(f"Peak selection maximized {metric_label} (no smoothing).")
        print(f"Peak metric at selected shift: {peak_best_raw:.6e}")

    # Extract similarity at selected and at zero shift
    iy0 = int(np.where(shift_ys == 0)[0][0])
    ix0 = int(np.where(shift_xs == 0)[0][0])
    sim_zero = float(sim_map[iy0, ix0])

    iyb = int(np.where(shift_ys == dy_best)[0][0])
    ixb = int(np.where(shift_xs == dx_best)[0][0])
    sim_best = float(sim_map[iyb, ixb])
    n_best = int(n_pix_map[iyb, ixb])
    # ---- Debug: FFT n_valid vs direct masked-pair count at the selected shift ----
    sa_y, sa_x, ss_y, ss_x = overlap_slices(actual.shape, dy_best, dx_best)
    n_direct = int(
        np.sum(np.isfinite(actual[sa_y, sa_x]) & np.isfinite(synthetic[ss_y, ss_x]))
    )
    print(
        f"[Debug] n_valid at selected shift: FFT={n_best} direct={n_direct} diff={n_best - n_direct}"
    )

    print(f"\n=== Selected shift (from {metric_label}) ===")
    print(f"  dy = {dy_best:+d} px (down is +)")
    print(f"  dx = {dx_best:+d} px (right is +)")
    print(f"  Normalized similarity at selected shift: {sim_best:.6f}")
    print(f"  Overlapping pixels at selected shift: {n_best}")

    print("\n=== Zero-shift similarity (dy=0, dx=0) ===")
    print(f"  similarity: {sim_zero:.6f}")

    # Residual metrics
    mse0, rms0, nrmse0, n0, mse0f, rms0f, nrmse0f, a0, b0, corr_med0 = (
        residual_metrics_at_shift_masked_pairs(actual, synthetic, 0, 0, fit_affine=True)
    )

    mseb, rmsb, nrmseb, nb, mseb_f, rmsb_f, nrmseb_f, ab, bb, corr_med = (
        residual_metrics_at_shift_masked_pairs(
            actual, synthetic, dy_best, dx_best, fit_affine=True
        )
    )

    print("\n=== Residual metrics (median-subtracted overlap) ===")
    print("Zero-shift metrics (median-subtracted overlap):")
    print(f"  Local corration coeff median: {corr_med0:.6e}")
    print(f"  RMS:   {rms0:.6e}   NRMSE: {nrmse0:.6f}   Npix: {n0}")
    print(f"  RMSfit:{rms0f:.6e}  NRMSEfit:{nrmse0f:.6f}  alpha={a0:.6g} beta={b0:.6g}")

    print(f"\nSelected-shift metrics (dy={dy_best:+d}, dx={dx_best:+d}):")
    print(f"  Local corration coeff median: {corr_med:.6e}")
    print(f"  RMS:   {rmsb:.6e}   NRMSE: {nrmseb:.6f}   Npix: {nb}")
    print(
        f"  RMSfit:{rmsb_f:.6e}  NRMSEfit:{nrmseb_f:.6f}  alpha={ab:.6g} beta={bb:.6g}"
    )

    if args.shifted_image is not None:
        print(f"\nWriting shifted synthetic image to {args.shifted_image!r} ...")

        if args.shifted_image_mode == "wrap":
            shifted_data = np.roll(
                synthetic, shift=(dy_best, dx_best), axis=(0, 1)
            ).astype(np.float64)
        elif args.shifted_image_mode == "constant":
            shifted_data = shift_image_nonwrap(
                synthetic, dy_best, dx_best, float(args.shifted_fill)
            )
        elif args.shifted_image_mode == "nan":
            shifted_data = shift_image_nonwrap(synthetic, dy_best, dx_best, np.nan)
        else:
            raise RuntimeError("Unknown --shifted-image-mode")

        with fits.open(args.synthetic, memmap=True) as hdul:
            hdu = hdul[synthetic_hdu]
            hdu.data = shifted_data.astype(np.float32, copy=False)
            hdu.header.add_history(
                f"Shifted by dy={dy_best}, dx={dx_best} using mode={args.shifted_image_mode}"
            )
            hdul.writeto(args.shifted_image, overwrite=True)

        print("Done writing shifted synthetic image.")

    # Plot similarity vs shift (normalized SIM surface), marking the shift picked from CC peak
    plot_similarity_map(
        shift_ys,
        shift_xs,
        sim_map,
        dy_best,
        dx_best,
        output=args.plot_out,
        show=not args.no_show,
        title="Normalized similarity vs shift (peak selected from cross-correlation)",
    )


if __name__ == "__main__":
    main()
