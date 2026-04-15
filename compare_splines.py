#!/usr/bin/env python3

import argparse
import csv

import numpy as np
from astropy import units as u
from astropy.cosmology import Planck18

LEGACY_PIXSCALE_ARCSEC = 0.262


def load_xy_csv(path, x_col=0, y_col=1):
    """
    Load x,y points from CSV.
    Supports an optional header row and extra columns.
    """
    pts = []
    skipped_header = False

    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, start=1):
            if not row or all(cell.strip() == "" for cell in row):
                continue

            if max(x_col, y_col) >= len(row):
                raise ValueError(
                    f"{path}: line {line_num} does not have columns {x_col} and {y_col}"
                )

            try:
                x = float(row[x_col].strip())
                y = float(row[y_col].strip())
                pts.append((x, y))
            except ValueError:
                if not pts and not skipped_header:
                    skipped_header = True
                    continue
                raise ValueError(
                    f"{path}: non-numeric x/y value on line {line_num}: {row}"
                )

    arr = np.asarray(pts, dtype=float)
    if arr.shape[0] < 3:
        raise ValueError(f"{path}: need at least 3 valid points")
    return arr


def curve_length(points):
    diffs = np.diff(points, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def cumulative_arclength(points):
    diffs = np.diff(points, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seglen)))
    return s


def resample_curve(points, n_samples):
    """
    Resample curve to n_samples equally spaced in arc length.
    """
    if n_samples < 5:
        raise ValueError("n_samples must be >= 5")

    s = cumulative_arclength(points)
    total = s[-1]
    if total <= 0:
        raise ValueError("Curve has zero total length")

    s_new = np.linspace(0.0, total, n_samples)
    x_new = np.interp(s_new, s, points[:, 0])
    y_new = np.interp(s_new, s, points[:, 1])
    return np.column_stack((x_new, y_new)), s_new


def gaussian_kernel1d(sigma, radius=None):
    if sigma <= 0:
        return np.array([1.0], dtype=float)

    if radius is None:
        radius = max(1, int(np.ceil(4.0 * sigma)))

    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    k /= np.sum(k)
    return k


def smooth_1d_reflect(arr, sigma):
    if sigma <= 0:
        return arr.copy()

    k = gaussian_kernel1d(sigma)
    pad = len(k) // 2
    padded = np.pad(arr, pad_width=pad, mode="reflect")
    smoothed = np.convolve(padded, k, mode="valid")
    return smoothed


def smooth_curve(points, sigma_samples):
    if sigma_samples <= 0:
        return points.copy()

    x = smooth_1d_reflect(points[:, 0], sigma_samples)
    y = smooth_1d_reflect(points[:, 1], sigma_samples)
    return np.column_stack((x, y))


def compute_curvature(points, s):
    """
    Compute signed curvature k(s) for a 2D parametric curve.

    k = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)
    """
    x = points[:, 0]
    y = points[:, 1]

    dx = np.gradient(x, s)
    dy = np.gradient(y, s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)

    denom = (dx * dx + dy * dy) ** 1.5
    eps = 1e-14
    kappa = (dx * ddy - dy * ddx) / np.maximum(denom, eps)

    return kappa


def correlation(a, b):
    a0 = a - np.mean(a)
    b0 = b - np.mean(b)
    sa = np.sqrt(np.mean(a0**2))
    sb = np.sqrt(np.mean(b0**2))
    if sa == 0 or sb == 0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.mean((a0 / sa) * (b0 / sb)))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def cumulative_turning_from_normalized_curvature(kappa_norm):
    """
    If u = s/L in [0,1], then kappa_norm = kappa * L = dtheta/du.
    Integrate over u to get turning angle relative to the start.
    """
    n = len(kappa_norm)
    u = np.linspace(0.0, 1.0, n)
    du = u[1] - u[0]

    theta = np.zeros(n, dtype=float)
    theta[1:] = np.cumsum(0.5 * (kappa_norm[:-1] + kappa_norm[1:]) * du)
    return theta


def build_profile(points):
    L = curve_length(points)
    s = cumulative_arclength(points)
    kappa = compute_curvature(points, s)
    kappa_norm = kappa * L
    theta = cumulative_turning_from_normalized_curvature(kappa_norm)

    return {
        "points": points,
        "s": s,
        "length": L,
        "kappa": kappa,
        "kappa_abs": np.abs(kappa),
        "kappa_norm": kappa_norm,
        "kappa_norm_abs": np.abs(kappa_norm),
        "theta": theta,
    }


def unit_tangent_and_normal(points, s):
    x = points[:, 0]
    y = points[:, 1]

    dx = np.gradient(x, s)
    dy = np.gradient(y, s)

    speed = np.hypot(dx, dy)
    eps = 1e-14
    tx = dx / np.maximum(speed, eps)
    ty = dy / np.maximum(speed, eps)

    tangent = np.column_stack((tx, ty))
    normal = np.column_stack((-ty, tx))
    return tangent, normal


def compute_residual_profile(points_ref, points_cmp):
    """
    Residuals of comparison curve relative to the reference curve,
    evaluated point-by-point at matched fractional arc length.
    """
    s_ref = cumulative_arclength(points_ref)
    tangent, normal = unit_tangent_and_normal(points_ref, s_ref)

    delta = points_cmp - points_ref
    along_track = np.sum(delta * tangent, axis=1)
    cross_track = np.sum(delta * normal, axis=1)
    separation = np.linalg.norm(delta, axis=1)

    return {
        "s_pixels": s_ref,
        "along_track_pixels": along_track,
        "cross_track_pixels": cross_track,
        "separation_pixels": separation,
        "along_track_rmse_pixels": rmse(along_track, np.zeros_like(along_track)),
        "cross_track_rmse_pixels": rmse(cross_track, np.zeros_like(cross_track)),
        "separation_rmse_pixels": rmse(separation, np.zeros_like(separation)),
        "separation_mean_pixels": float(np.mean(separation)),
        "separation_median_pixels": float(np.median(separation)),
        "separation_max_pixels": float(np.max(separation)),
    }


def compare_curvature_profiles(points_a, points_b):
    """
    Compare two curves by curvature profile after both have been:
      - resampled uniformly in arc length
      - optionally smoothed

    Returns metrics for best of forward/reversed ordering of curve B.
    """
    prof_a = build_profile(points_a)

    candidates = []

    # Forward
    points_b_fwd = points_b.copy()
    prof_b_fwd = build_profile(points_b_fwd)
    candidates.append(("forward", points_b_fwd, prof_b_fwd))

    # Reversed
    points_b_rev = points_b[::-1].copy()
    prof_b_rev = build_profile(points_b_rev)

    # For a reversed traversal of the same geometric curve,
    # signed curvature changes sign. Flip it back so that
    # "same shape, opposite tracing direction" compares fairly.
    prof_b_rev["kappa"] *= -1.0
    prof_b_rev["kappa_norm"] *= -1.0
    prof_b_rev["theta"] *= -1.0

    candidates.append(("reversed", points_b_rev, prof_b_rev))

    best = None

    for direction, points_b_used, prof_b in candidates:
        residuals = compute_residual_profile(prof_a["points"], points_b_used)

        result = {
            "direction_used": direction,
            "length_a": prof_a["length"],
            "length_b": prof_b["length"],
            "length_ratio_b_over_a": prof_b["length"] / prof_a["length"],
            "signed_curvature_rmse": rmse(prof_a["kappa_norm"], prof_b["kappa_norm"]),
            "abs_curvature_rmse": rmse(
                prof_a["kappa_norm_abs"], prof_b["kappa_norm_abs"]
            ),
            "signed_curvature_corr": correlation(
                prof_a["kappa_norm"], prof_b["kappa_norm"]
            ),
            "abs_curvature_corr": correlation(
                prof_a["kappa_norm_abs"], prof_b["kappa_norm_abs"]
            ),
            "turning_angle_rmse": rmse(prof_a["theta"], prof_b["theta"]),
            "total_turning_a_rad": float(prof_a["theta"][-1]),
            "total_turning_b_rad": float(prof_b["theta"][-1]),
            "profiles_a": prof_a,
            "profiles_b": prof_b,
            "residuals": residuals,
        }

        # Heuristic combined score in [0,1]-ish, higher is better.
        # Based mostly on abs curvature profile, because that is more
        # robust to sign/orientation conventions.
        result["shape_similarity_score"] = float(
            np.exp(-result["abs_curvature_rmse"])
            * max(0.0, 0.5 * (result["abs_curvature_corr"] + 1.0))
        )

        if best is None:
            best = result
        else:
            # Pick the better match primarily by abs curvature RMSE,
            # then by turning angle RMSE.
            if result["abs_curvature_rmse"] < best["abs_curvature_rmse"] or (
                np.isclose(result["abs_curvature_rmse"], best["abs_curvature_rmse"])
                and result["turning_angle_rmse"] < best["turning_angle_rmse"]
            ):
                best = result

    return best


def make_scale_info(z=None, pixscale_arcsec=LEGACY_PIXSCALE_ARCSEC):
    arcsec_per_pixel = pixscale_arcsec * u.arcsec
    arcmin_per_pixel = arcsec_per_pixel.to(u.arcmin)

    info = {
        "pixscale_arcsec": pixscale_arcsec,
        "arcsec_per_pixel": arcsec_per_pixel,
        "arcmin_per_pixel": arcmin_per_pixel,
        "z": z,
        "has_cosmology": z is not None,
    }

    if z is not None:
        if z < 0:
            raise ValueError("z must be >= 0")
        da = Planck18.angular_diameter_distance(z)
        kpc_per_arcmin = Planck18.kpc_proper_per_arcmin(z)
        kpc_per_pixel = (kpc_per_arcmin * arcmin_per_pixel).to(u.kpc)

        info.update(
            {
                "angular_diameter_distance": da,
                "kpc_per_arcmin": kpc_per_arcmin,
                "kpc_per_pixel": kpc_per_pixel,
            }
        )

    return info


def length_summary(length_pixels, scale_info):
    out = {
        "pixels": float(length_pixels),
        "arcsec": (length_pixels * scale_info["arcsec_per_pixel"]).to_value(u.arcsec),
        "arcmin": (length_pixels * scale_info["arcmin_per_pixel"]).to_value(u.arcmin),
    }
    if scale_info["has_cosmology"]:
        out["kpc"] = (length_pixels * scale_info["kpc_per_pixel"]).to_value(u.kpc)
    return out


def convert_profile_pixels_to_display(values_pixels, scale_info):
    values_pixels = np.asarray(values_pixels, dtype=float)

    if scale_info["has_cosmology"]:
        return (
            values_pixels * scale_info["kpc_per_pixel"].to_value(u.kpc),
            "kpc",
        )

    return (
        values_pixels * scale_info["arcsec_per_pixel"].to_value(u.arcsec),
        "arcsec",
    )


def print_length_block(label, summary):
    print(f"{label}")
    print(f"  pixels                  : {summary['pixels']:.6g}")
    print(f"  arcsec                  : {summary['arcsec']:.6g}")
    print(f"  arcmin                  : {summary['arcmin']:.6g}")
    if "kpc" in summary:
        print(f"  proper length (kpc)     : {summary['kpc']:.6g}")


def print_report(
    result,
    file_a,
    file_b,
    n_samples,
    smooth_sigma,
    raw_length_a,
    raw_length_b,
    scale_info,
):
    length_a = length_summary(raw_length_a, scale_info)
    length_b = length_summary(raw_length_b, scale_info)

    print("Curvature-based spline comparison")
    print("--------------------------------")
    print(f"file A                    : {file_a}")
    print(f"file B                    : {file_b}")
    print(f"resampled points          : {n_samples}")
    print(f"smoothing sigma (samples) : {smooth_sigma}")
    print(f"direction_used            : {result['direction_used']}")
    print(
        f"pixel scale               : {scale_info['pixscale_arcsec']:.6g} arcsec/pixel"
    )
    if scale_info["has_cosmology"]:
        print(f"redshift z                : {scale_info['z']:.6g}")
        print(
            "angular diameter distance : "
            f"{scale_info['angular_diameter_distance'].to_value(u.Mpc):.6g} Mpc"
        )
        print(
            "proper scale              : "
            f"{scale_info['kpc_per_arcmin'].to_value(u.kpc / u.arcmin):.6g} kpc/arcmin"
        )
    print()

    print_length_block("Input spline length A", length_a)
    print_length_block("Input spline length B", length_b)
    print(f"  raw length ratio B/A    : {raw_length_b / raw_length_a:.6g}")
    print()

    residuals = result["residuals"]
    cross_vals, cross_unit = convert_profile_pixels_to_display(
        residuals["cross_track_pixels"], scale_info
    )
    sep_vals, sep_unit = convert_profile_pixels_to_display(
        residuals["separation_pixels"], scale_info
    )

    print(f"signed curvature RMSE     : {result['signed_curvature_rmse']:.6g}")
    print(f"abs curvature RMSE        : {result['abs_curvature_rmse']:.6g}")
    print(f"signed curvature corr     : {result['signed_curvature_corr']:.6f}")
    print(f"abs curvature corr        : {result['abs_curvature_corr']:.6f}")
    print(f"turning angle RMSE (rad)  : {result['turning_angle_rmse']:.6g}")
    print(f"total turning A (rad)     : {result['total_turning_a_rad']:.6g}")
    print(f"total turning B (rad)     : {result['total_turning_b_rad']:.6g}")
    print(f"shape similarity score    : {result['shape_similarity_score']:.6f}")
    print()
    print(
        f"cross-track RMSE ({cross_unit}) : {rmse(cross_vals, np.zeros_like(cross_vals)):.6g}"
    )
    print(f"mean residual ({sep_unit})      : {np.mean(sep_vals):.6g}")
    print(f"median residual ({sep_unit})    : {np.median(sep_vals):.6g}")
    print(f"max residual ({sep_unit})       : {np.max(sep_vals):.6g}")


def maybe_plot(result, scale_info, outpath=None):
    import matplotlib.pyplot as plt

    residuals = result["residuals"]

    s_disp, s_unit = convert_profile_pixels_to_display(
        residuals["s_pixels"], scale_info
    )
    cross_disp, y_unit = convert_profile_pixels_to_display(
        residuals["cross_track_pixels"], scale_info
    )
    sep_disp, _ = convert_profile_pixels_to_display(
        residuals["separation_pixels"], scale_info
    )

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax1.plot(s_disp, cross_disp)
    ax1.axhline(0.0, linewidth=1.0, linestyle="--")
    ax1.set_ylabel(f"Cross-track offset [{y_unit}]")
    ax1.set_title("Spline residuals at matched fractional arc length")

    ax2.plot(s_disp, sep_disp)
    ax2.set_xlabel(f"Arc length along spline A [{s_unit}]")
    ax2.set_ylabel(f"Residual offset [{y_unit}]")

    text = "\n".join(
        [
            f"direction_used = {result['direction_used']}",
            f"shape similarity score = {result['shape_similarity_score']:.6f}",
            (
                f"cross-track RMSE [{y_unit}] = "
                f"{rmse(cross_disp, np.zeros_like(cross_disp)):.6g}"
            ),
            f"mean residual [{y_unit}] = {np.mean(sep_disp):.6g}",
            f"max residual [{y_unit}] = {np.max(sep_disp):.6g}",
        ]
    )
    ax1.text(
        0.99,
        0.97,
        text,
        transform=ax1.transAxes,
        ha="right",
        va="top",
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=200, bbox_inches="tight")
        print(f"\nSaved plot to: {outpath}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare two spline CSV files by curvature profile."
    )
    parser.add_argument("csv_a", help="First spline CSV")
    parser.add_argument("csv_b", help="Second spline CSV")
    parser.add_argument("--x-col", type=int, default=0, help="Zero-based x column")
    parser.add_argument("--y-col", type=int, default=1, help="Zero-based y column")
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Number of arc-length resampled points",
    )
    parser.add_argument(
        "--smooth-sigma",
        type=float,
        default=2.0,
        help="Gaussian smoothing sigma in resampled-point units",
    )
    parser.add_argument(
        "--z",
        type=float,
        default=None,
        help="Redshift for converting spline lengths and residuals to proper kpc",
    )
    parser.add_argument(
        "--pixscale-arcsec",
        type=float,
        default=LEGACY_PIXSCALE_ARCSEC,
        help="Pixel scale in arcsec/pixel (default: Legacy Survey 0.262)",
    )
    parser.add_argument(
        "--plot",
        nargs="?",
        const="",
        default=None,
        help="Show residual plot, or save to a file if a path is provided",
    )
    args = parser.parse_args()

    raw_a = load_xy_csv(args.csv_a, x_col=args.x_col, y_col=args.y_col)
    raw_b = load_xy_csv(args.csv_b, x_col=args.x_col, y_col=args.y_col)

    raw_length_a = curve_length(raw_a)
    raw_length_b = curve_length(raw_b)
    scale_info = make_scale_info(z=args.z, pixscale_arcsec=args.pixscale_arcsec)

    a_resamp, _ = resample_curve(raw_a, args.n_samples)
    b_resamp, _ = resample_curve(raw_b, args.n_samples)

    a_proc = smooth_curve(a_resamp, args.smooth_sigma)
    b_proc = smooth_curve(b_resamp, args.smooth_sigma)

    result = compare_curvature_profiles(a_proc, b_proc)

    print_report(
        result=result,
        file_a=args.csv_a,
        file_b=args.csv_b,
        n_samples=args.n_samples,
        smooth_sigma=args.smooth_sigma,
        raw_length_a=raw_length_a,
        raw_length_b=raw_length_b,
        scale_info=scale_info,
    )

    if args.plot is not None:
        outpath = args.plot if args.plot != "" else None
        maybe_plot(result, scale_info=scale_info, outpath=outpath)


if __name__ == "__main__":
    main()
