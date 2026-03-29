#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import numpy as np


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


def compare_curvature_profiles(points_a, points_b):
    """
    Compare two curves by curvature profile after both have been:
      - resampled uniformly in arc length
      - optionally smoothed

    Returns metrics for best of forward/reversed ordering of curve B.
    """

    def make_profiles(points):
        L = curve_length(points)
        s = cumulative_arclength(points)
        kappa = compute_curvature(points, s)

        # Dimensionless curvature profile: shape-focused, scale-reduced
        kappa_norm = kappa * L

        theta = cumulative_turning_from_normalized_curvature(kappa_norm)

        return {
            "length": L,
            "kappa": kappa,
            "kappa_abs": np.abs(kappa),
            "kappa_norm": kappa_norm,
            "kappa_norm_abs": np.abs(kappa_norm),
            "theta": theta,
        }

    prof_a = make_profiles(points_a)

    candidates = []

    # Forward
    prof_b_fwd = make_profiles(points_b)
    candidates.append(("forward", prof_b_fwd))

    # Reversed
    prof_b_rev = make_profiles(points_b[::-1].copy())

    # For a reversed traversal of the same geometric curve,
    # signed curvature changes sign. Flip it back so that
    # "same shape, opposite tracing direction" compares fairly.
    prof_b_rev["kappa"] *= -1.0
    prof_b_rev["kappa_norm"] *= -1.0
    prof_b_rev["theta"] *= -1.0

    candidates.append(("reversed", prof_b_rev))

    best = None

    for direction, prof_b in candidates:
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


def print_report(result, file_a, file_b, n_samples, smooth_sigma):
    print("Curvature-based spline comparison")
    print("--------------------------------")
    print(f"file A                    : {file_a}")
    print(f"file B                    : {file_b}")
    print(f"resampled points          : {n_samples}")
    print(f"smoothing sigma (samples) : {smooth_sigma}")
    print(f"direction_used            : {result['direction_used']}")
    print()
    print(f"length A                  : {result['length_a']:.6g}")
    print(f"length B                  : {result['length_b']:.6g}")
    print(f"length ratio B/A          : {result['length_ratio_b_over_a']:.6g}")
    print()
    print(f"signed curvature RMSE     : {result['signed_curvature_rmse']:.6g}")
    print(f"abs curvature RMSE        : {result['abs_curvature_rmse']:.6g}")
    print(f"signed curvature corr     : {result['signed_curvature_corr']:.6f}")
    print(f"abs curvature corr        : {result['abs_curvature_corr']:.6f}")
    print(f"turning angle RMSE (rad)  : {result['turning_angle_rmse']:.6g}")
    print(f"total turning A (rad)     : {result['total_turning_a_rad']:.6g}")
    print(f"total turning B (rad)     : {result['total_turning_b_rad']:.6g}")
    print(f"shape similarity score    : {result['shape_similarity_score']:.6f}")


def maybe_plot(result, outpath=None):
    import matplotlib.pyplot as plt

    prof_a = result["profiles_a"]
    prof_b = result["profiles_b"]

    n = len(prof_a["kappa_norm"])
    u = np.linspace(0.0, 1.0, n)

    fig = plt.figure(figsize=(12, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.plot(u, prof_a["kappa_norm"], label="A")
    ax1.plot(u, prof_b["kappa_norm"], label="B")
    ax1.set_title("Signed normalized curvature")
    ax1.set_xlabel("Normalized arc length")
    ax1.set_ylabel("kappa * L")
    ax1.legend()

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.plot(u, prof_a["kappa_norm_abs"], label="A")
    ax2.plot(u, prof_b["kappa_norm_abs"], label="B")
    ax2.set_title("Absolute normalized curvature")
    ax2.set_xlabel("Normalized arc length")
    ax2.set_ylabel("|kappa| * L")
    ax2.legend()

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.plot(u, prof_a["theta"], label="A")
    ax3.plot(u, prof_b["theta"], label="B")
    ax3.set_title("Cumulative turning angle")
    ax3.set_xlabel("Normalized arc length")
    ax3.set_ylabel("Theta(u) [rad]")
    ax3.legend()

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    txt = "\n".join(
        [
            f"direction_used = {result['direction_used']}",
            f"length_ratio B/A = {result['length_ratio_b_over_a']:.6g}",
            f"signed curvature RMSE = {result['signed_curvature_rmse']:.6g}",
            f"abs curvature RMSE = {result['abs_curvature_rmse']:.6g}",
            f"signed curvature corr = {result['signed_curvature_corr']:.6f}",
            f"abs curvature corr = {result['abs_curvature_corr']:.6f}",
            f"turning angle RMSE = {result['turning_angle_rmse']:.6g}",
            f"shape similarity score = {result['shape_similarity_score']:.6f}",
        ]
    )
    ax4.text(0.01, 0.99, txt, va="top", ha="left", family="monospace")

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
        "--plot",
        nargs="?",
        const="",
        default=None,
        help="Show plot, or save to a file if a path is provided",
    )
    args = parser.parse_args()

    raw_a = load_xy_csv(args.csv_a, x_col=args.x_col, y_col=args.y_col)
    raw_b = load_xy_csv(args.csv_b, x_col=args.x_col, y_col=args.y_col)

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
    )

    if args.plot is not None:
        outpath = args.plot if args.plot != "" else None
        maybe_plot(result, outpath=outpath)


if __name__ == "__main__":
    main()
