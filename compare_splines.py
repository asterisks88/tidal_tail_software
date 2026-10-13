#!/usr/bin/env python3
"""Compare spline geometry and export publication-sized residual figures.

Example for CSV coordinates from 4x-binned Legacy images:
    python compare_splines.py observed.csv model.csv \
        --pixscale-arcsec 1.048 --plot tail_residuals.pdf

A saved plot also produces matching PDF and PNG files. PDF curves and text
remain vector graphics; PNG output defaults to 600 dpi. Residuals retain the
original matched-fractional-arc-length definition, with spline A as reference.
The single-column figure has a 5:8 aspect ratio, matching a 1000 x 1600 image.
"""

import argparse
import csv
from pathlib import Path

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


def publication_plot_style():
    """Local Matplotlib settings for readable journal-sized line figures."""
    return {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 10.0,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "axes.unicode_minus": True,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.2,
        "ytick.minor.size": 2.2,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "xtick.major.pad": 3.0,
        "ytick.major.pad": 3.0,
        "lines.linewidth": 1.3,
        "figure.facecolor": "white",
        "figure.autolayout": False,
        "figure.constrained_layout.use": False,
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.transparent": False,
        "savefig.bbox": None,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "text.usetex": False,
    }


def save_publication_plot(fig, outpath, dpi=600):
    """Save the requested format plus a vector PDF and high-resolution PNG."""
    path = Path(outpath).expanduser()
    suffix = path.suffix.lower()
    supported = fig.canvas.get_supported_filetypes()
    if suffix and suffix.lstrip(".") not in supported:
        raise ValueError(
            f"Unsupported plot extension {path.suffix!r}. "
            "Use .pdf, .png, .svg, .eps, or another Matplotlib image format."
        )

    if suffix:
        targets = [(path, suffix.lstrip("."))]
        stem = path.with_suffix("")
    else:
        targets = []
        stem = path

    for fmt in ("pdf", "png"):
        if not any(existing_fmt == fmt for _, existing_fmt in targets):
            targets.append((Path(str(stem) + "." + fmt), fmt))

    for target, fmt in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Keep the chosen physical figure size; explicit margins below ensure
        # labels fit without bbox_inches='tight' changing the export dimensions.
        fig.savefig(target, format=fmt, dpi=dpi, bbox_inches=None)
        print(f"Saved plot to: {target}")
    return [target for target, _ in targets]


def maybe_plot(
    result, scale_info, outpath=None, *, plot_size="single", dpi=600,
    show_stats=False, title=None,
):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import AutoMinorLocator, MaxNLocator

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

    sizes = {"single": (3.5, 5.6), "double": (7.1, 4.8)}
    if plot_size not in sizes:
        raise ValueError("plot_size must be 'single' or 'double'")
    if dpi < 300:
        raise ValueError("Plot dpi must be at least 300")

    with plt.rc_context(publication_plot_style()):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=sizes[plot_size], sharex=True,
        )
        fig.subplots_adjust(
            left=0.21 if plot_size == "single" else 0.12,
            right=0.97, bottom=0.12, top=0.92 if title else 0.97,
            hspace=0.08,
        )

        # Colors are accompanied by different line styles and explicit labels.
        ax1.plot(s_disp, cross_disp, color="#0072B2", linestyle="-", zorder=3)
        ax1.axhline(0.0, color="0.4", linewidth=0.8, linestyle="--", zorder=1)
        ax1.set_ylabel(r"$d_\perp$" + f" [{y_unit}]")

        ax2.plot(s_disp, sep_disp, color="#D55E00", linestyle="--", zorder=3)
        ax2.set_xlabel(f"Arc length along spline A [{s_unit}]")
        ax2.set_ylabel(r"$d_{\mathrm{2D}}$" + f" [{y_unit}]")
        ax2.set_ylim(bottom=0.0)
        ax2.set_xlim(float(s_disp[0]), float(s_disp[-1]))

        for ax, label, description in (
            (ax1, "(a)", "Cross-track offset"),
            (ax2, "(b)", "Point separation"),
        ):
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))
            ax.tick_params(which="both", direction="in", top=True, right=True)
            ax.margins(y=0.12)
            ax.text(
                0.035, 0.95, label + "  " + description,
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9,
                      "pad": 1.5},
                zorder=4,
            )

        if show_stats:
            cross_rms = rmse(cross_disp, np.zeros_like(cross_disp))
            ax1.text(
                0.965, 0.08, f"RMS = {cross_rms:.3g} {y_unit}",
                transform=ax1.transAxes, ha="right", va="bottom", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9,
                      "pad": 1.5},
            )
            ax2.text(
                0.965, 0.08, f"Median = {np.median(sep_disp):.3g} {y_unit}",
                transform=ax2.transAxes, ha="right", va="bottom", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9,
                      "pad": 1.5},
            )
        if title:
            fig.suptitle(title, fontsize=10)

        try:
            if outpath:
                return save_publication_plot(fig, outpath, dpi=dpi)
            plt.show()
            return []
        finally:
            plt.close(fig)


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
        help=(
            "Show the publication-style residual plot, or save to a path/prefix. "
            "Saved plots include matching vector PDF and high-resolution PNG."
        ),
    )
    parser.add_argument(
        "--plot-size", choices=("single", "double"), default="single",
        help=(
            "Journal figure size: single column (3.5 x 5.6 in; 5:8 aspect "
            "ratio) or double column (7.1 x 4.8 in)."
        ),
    )
    parser.add_argument(
        "--plot-dpi", type=int, default=600,
        help="Resolution of raster plot exports (default: 600; minimum: 300).",
    )
    parser.add_argument(
        "--plot-stats", action="store_true",
        help="Add compact RMS and median annotations (default: omit them).",
    )
    parser.add_argument(
        "--plot-title", default=None,
        help="Optional figure title (default: no title; use the paper caption).",
    )
    args = parser.parse_args()
    if args.plot_dpi < 300:
        parser.error("--plot-dpi must be >= 300")

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
        maybe_plot(
            result, scale_info=scale_info, outpath=outpath,
            plot_size=args.plot_size, dpi=args.plot_dpi,
            show_stats=args.plot_stats, title=args.plot_title,
        )


if __name__ == "__main__":
    main()
