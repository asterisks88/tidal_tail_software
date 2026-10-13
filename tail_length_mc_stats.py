#!/usr/bin/env python3
"""Trace noise realizations and summarize length and signed total turning.

Turning uses the compare_splines.py estimator (default: 500 arc-length
samples, Gaussian coordinate smoothing with sigma=2 samples). All realizations
must be traced in the same direction. The Monte Carlo standard deviation is
the noise uncertainty of one measurement; the SEM is not that uncertainty.
"""
import argparse
import csv
import glob
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def natural_key(s):
    s = str(s)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def read_summary_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def to_float_list(rows, key):
    vals = []
    for row in rows:
        try:
            v = float(row[key])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    return np.asarray(vals, dtype=float)


def total_turning_angle(path, n_samples=500, smooth_sigma=2.0):
    """Return signed total turning [rad], matching compare_splines.py.

    This is its integral of length-normalized signed curvature on the uniform
    fractional-arc-length grid, after resampling and reflect-padded smoothing.
    Pixel scale is unnecessary: total turning is invariant under uniform scaling.
    """
    if n_samples < 5:
        raise ValueError("turning n_samples must be >= 5")
    if not np.isfinite(smooth_sigma) or smooth_sigma < 0:
        raise ValueError("turning smooth_sigma must be finite and >= 0")

    points = []
    skipped_header = False
    with open(path, newline="") as f:
        for line_num, row in enumerate(csv.reader(f), start=1):
            if not row or all(not cell.strip() for cell in row):
                continue
            try:
                point = (float(row[0]), float(row[1]))
            except (ValueError, IndexError):
                if not points and not skipped_header:
                    skipped_header = True
                    continue
                raise ValueError(f"{path}: invalid x/y on line {line_num}")
            if not np.all(np.isfinite(point)):
                raise ValueError(f"{path}: non-finite x/y on line {line_num}")
            points.append(point)

    points = np.asarray(points, dtype=float)
    if len(points) < 3:
        raise ValueError(f"{path}: need at least 3 points")
    # Remove consecutive duplicate points before interpolation/derivatives.
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 0]
    points = points[keep]
    if len(points) < 3:
        raise ValueError(f"{path}: need at least 3 distinct consecutive points")
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    s_new = np.linspace(0.0, s[-1], n_samples)
    points = np.column_stack([np.interp(s_new, s, points[:, j]) for j in (0, 1)])

    if smooth_sigma > 0:
        radius = max(1, int(np.ceil(4.0 * smooth_sigma)))
        offsets = np.arange(-radius, radius + 1, dtype=float)
        kernel = np.exp(-0.5 * (offsets / smooth_sigma) ** 2)
        kernel /= kernel.sum()
        points = np.column_stack([
            np.convolve(np.pad(points[:, j], radius, mode="reflect"),
                        kernel, mode="valid")
            for j in (0, 1)
        ])

    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    if np.any(np.diff(s) <= 0):
        raise ValueError(f"{path}: degenerate curve after smoothing")
    dx = np.gradient(points[:, 0], s)
    dy = np.gradient(points[:, 1], s)
    ddx = np.gradient(dx, s)
    ddy = np.gradient(dy, s)
    kappa = (dx * ddy - dy * ddx) / np.maximum((dx**2 + dy**2)**1.5, 1e-14)
    kappa_norm = kappa * s[-1]
    increments = 0.5 * (kappa_norm[:-1] + kappa_norm[1:]) / (n_samples - 1)
    angle = float(np.cumsum(increments)[-1])
    if not np.isfinite(angle):
        raise ValueError(f"{path}: non-finite total turning angle")
    return angle


def add_turning_to_summary(path, rows, n_samples, smooth_sigma):
    """Add one total-angle measurement per successful tracer summary row."""
    new_fields = ["total_turning_rad", "total_turning_deg", "turning_status",
                  "turning_error"]
    with open(path, newline="") as f:
        fieldnames = list(csv.DictReader(f).fieldnames or [])
    fieldnames.extend(key for key in new_fields if key not in fieldnames)
    n_failed = 0
    for row in rows:
        try:
            spline_path = row.get("spline_pix_csv", "")
            if not spline_path:
                raise ValueError("tracer summary has no spline_pix_csv")
            angle = total_turning_angle(spline_path, n_samples, smooth_sigma)
            row.update(total_turning_rad=angle, total_turning_deg=float(np.degrees(angle)),
                       turning_status="ok", turning_error="")
        except (OSError, ValueError) as exc:
            n_failed += 1
            row.update(total_turning_rad="", total_turning_deg="",
                       turning_status="failed", turning_error=str(exc))
            print(f"  Turning measurement failed for {row.get('input_fits', '?')}: {exc}")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return n_failed


def summarize(arr):
    if arr.size == 0:
        return None

    out = {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else np.nan,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p16": float(np.percentile(arr, 16.0)),
        "p84": float(np.percentile(arr, 84.0)),
    }
    out["sem"] = out["std"] / np.sqrt(out["n"]) if out["n"] > 1 else np.nan
    out["half_68_range"] = 0.5 * (out["p84"] - out["p16"]) if out["n"] > 1 else np.nan
    out["err_minus"] = out["median"] - out["p16"] if out["n"] > 1 else np.nan
    out["err_plus"] = out["p84"] - out["median"] if out["n"] > 1 else np.nan
    return out


def compute_bias(arr, true_value):
    bias = np.median(np.array(arr) - true_value)
    return bias


def write_stats_csv(path, pix_stats, arcmin_stats, n_total, n_success, n_failed,
                    turning_stats=None, turning_deg_stats=None, n_turning_failed=0,
                    turning_n_samples=500, turning_smooth_sigma=2.0):
    fieldnames = [
        "n_total",
        "n_success",
        "n_failed",
        "length_pix_n",
        "length_pix_mean",
        "length_pix_median",
        "length_pix_bias",
        "length_pix_adjusted",
        "length_pix_std",
        "length_pix_sem",
        "length_pix_p16",
        "length_pix_p84",
        "length_pix_half_68_range",
        "length_pix_min",
        "length_pix_max",
        "length_arcmin_n",
        "length_arcmin_mean",
        "length_arcmin_median",
        "length_arcmin_bias",
        "length_arcmin_adjusted",
        "length_arcmin_std",
        "length_arcmin_sem",
        "length_arcmin_p16",
        "length_arcmin_p84",
        "length_arcmin_half_68_range",
        "length_arcmin_min",
        "length_arcmin_max",
    ]
    turning_keys = ["n", "mean", "median", "std", "sem", "p16", "p84",
                    "half_68_range", "err_minus", "err_plus", "min", "max"]
    fieldnames += ["n_turning_failed", "turning_n_samples", "turning_smooth_sigma"]
    fieldnames += [f"turning_{unit}_{key}" for unit in ("rad", "deg")
                   for key in turning_keys]

    row = {
        "n_total": n_total,
        "n_success": n_success,
        "n_failed": n_failed,
        "n_turning_failed": n_turning_failed,
        "turning_n_samples": turning_n_samples,
        "turning_smooth_sigma": turning_smooth_sigma,
    }
    for unit, stats in [("rad", turning_stats), ("deg", turning_deg_stats)]:
        if stats is not None:
            row.update({f"turning_{unit}_{key}": stats[key] for key in turning_keys})
        else:
            row[f"turning_{unit}_n"] = 0

    if pix_stats is not None:
        row.update(
            {
                "length_pix_n": pix_stats["n"],
                "length_pix_mean": pix_stats["mean"],
                "length_pix_median": pix_stats["median"],
                "length_pix_bias": pix_stats["bias"],
                "length_pix_adjusted": pix_stats["adjusted"],
                "length_pix_std": pix_stats["std"],
                "length_pix_sem": pix_stats["sem"],
                "length_pix_p16": pix_stats["p16"],
                "length_pix_p84": pix_stats["p84"],
                "length_pix_half_68_range": pix_stats["half_68_range"],
                "length_pix_min": pix_stats["min"],
                "length_pix_max": pix_stats["max"],
            }
        )

    if arcmin_stats is not None:
        row.update(
            {
                "length_arcmin_n": arcmin_stats["n"],
                "length_arcmin_mean": arcmin_stats["mean"],
                "length_arcmin_median": arcmin_stats["median"],
                "length_arcmin_bias": arcmin_stats["bias"],
                "length_arcmin_adjusted": arcmin_stats["adjusted"],
                "length_arcmin_std": arcmin_stats["std"],
                "length_arcmin_sem": arcmin_stats["sem"],
                "length_arcmin_p16": arcmin_stats["p16"],
                "length_arcmin_p84": arcmin_stats["p84"],
                "length_arcmin_half_68_range": arcmin_stats["half_68_range"],
                "length_arcmin_min": arcmin_stats["min"],
                "length_arcmin_max": arcmin_stats["max"],
            }
        )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run trace_sat_tail_spline.py on many noise realizations and report "
            "length and signed total-turning-angle statistics."
        )
    )
    ap.add_argument(
        "pattern",
        help="Glob pattern for realization FITS files, e.g. 's*_bin4_mu.fits'",
    )
    ap.add_argument(
        "--tracer",
        default=str(Path.home() / "tidal-tail-software" / "trace_sat_tail_spline.py"),
        help="Path to your single-image tail tracing script",
    )
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for the tracer",
    )
    ap.add_argument(
        "--summary-csv",
        default="tail_mc_lengths.csv",
        help="Per-realization summary CSV written by the tracer",
    )
    ap.add_argument(
        "--stats-csv",
        default="tail_mc_length_stats.csv",
        help="One-row CSV with aggregate length and turning-angle statistics",
    )
    ap.add_argument(
        "--products-dir",
        default="tail_mc_products",
        help="Directory for per-realization tracer outputs",
    )
    ap.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any realization fails",
    )
    ap.add_argument(
        "--true-length",
        type=float,
        default=5.545,
        help="True length in arcmin derived from noise free synthetic image from model",
    )
    ap.add_argument(
        "--pixel-scale",
        type=float,
        default=0.262 * 4,
        help="Arcsec per pixel.  Default is bin4 Legacy Survey scale.",
    )
    ap.add_argument(
        "--turning-n-samples", type=int, default=500,
        help="Arc-length samples for turning calculation (compare_splines.py default: 500)",
    )
    ap.add_argument(
        "--turning-smooth-sigma", type=float, default=2.0,
        help="Gaussian coordinate smoothing in resampled points for turning (default: 2)",
    )

    args, tracer_args = ap.parse_known_args()
    if args.turning_n_samples < 5:
        ap.error("--turning-n-samples must be >= 5")
    if not np.isfinite(args.turning_smooth_sigma) or args.turning_smooth_sigma < 0:
        ap.error("--turning-smooth-sigma must be finite and >= 0")
    if not np.isfinite(args.pixel_scale) or args.pixel_scale <= 0:
        ap.error("--pixel-scale must be finite and > 0")

    forbidden = {"--summary-csv", "--out-prefix"}
    for tok in tracer_args:
        if tok in forbidden:
            raise SystemExit(
                f"Do not pass {tok} to the wrapper; it manages that itself."
            )

    files = sorted(glob.glob(args.pattern), key=natural_key)
    if not files:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    tracer = Path(args.tracer)
    if not tracer.exists():
        raise SystemExit(f"Tracer script not found: {tracer}")

    products_dir = Path(args.products_dir)
    products_dir.mkdir(parents=True, exist_ok=True)

    summary_csv = Path(args.summary_csv)
    if summary_csv.exists():
        summary_csv.unlink()

    failures = []

    print(f"Found {len(files)} realization files.")
    print(f"Per-realization summary CSV: {summary_csv}")
    print(f"Aggregate stats CSV:         {args.stats_csv}")
    print(f"Products directory:          {products_dir}")
    print()

    for i, fits_path in enumerate(files, start=1):
        fits_path = Path(fits_path)
        stem = fits_path.stem
        out_prefix = products_dir / stem

        cmd = [
            args.python,
            str(tracer),
            str(fits_path),
            "--out-prefix",
            str(out_prefix),
            "--summary-csv",
            str(summary_csv),
        ] + tracer_args

        print(f"[{i:>3d}/{len(files)}] {fits_path.name}")
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            failures.append(str(fits_path))
            print(f"  FAILED with exit code {e.returncode}")
            if args.stop_on_error:
                raise
        except Exception as e:
            failures.append(str(fits_path))
            print(f"  FAILED: {e}")
            if args.stop_on_error:
                raise

    if not summary_csv.exists():
        raise SystemExit("No successful runs produced a summary CSV.")

    rows = read_summary_rows(summary_csv)
    n_turning_failed = add_turning_to_summary(
        summary_csv, rows, args.turning_n_samples, args.turning_smooth_sigma
    )
    turning = to_float_list(rows, "total_turning_rad")
    turning_stats = summarize(turning)
    turning_deg_stats = summarize(np.degrees(turning))
    length_pix = to_float_list(rows, "length_pix")
    length_arcmin = to_float_list(rows, "length_arcmin")

    pix_stats = summarize(length_pix)
    if pix_stats is not None:
        pix_stats["bias"] = compute_bias(
            length_pix, 60 * args.true_length / args.pixel_scale
        )
        pix_stats["adjusted"] = pix_stats["median"] - pix_stats["bias"]

    arcmin_stats = summarize(length_arcmin)
    if arcmin_stats is not None:
        arcmin_stats["bias"] = compute_bias(length_arcmin, args.true_length)
        arcmin_stats["adjusted"] = arcmin_stats["median"] - arcmin_stats["bias"]

    n_total = len(files)
    n_success = len(rows)
    n_failed = len(failures)

    write_stats_csv(
        args.stats_csv, pix_stats, arcmin_stats, n_total, n_success, n_failed,
        turning_stats, turning_deg_stats, n_turning_failed,
        args.turning_n_samples, args.turning_smooth_sigma,
    )

    print()
    print("=== Monte Carlo tail-length and turning-angle statistics ===")
    print(f"Total realizations:        {n_total}")
    print(f"Successful traces:         {n_success}")
    print(f"Failed traces:             {n_failed}")
    print(f"Valid turning angles:      {len(turning)}")
    print(f"Failed turning measures:   {n_turning_failed}")

    if pix_stats is not None:
        print()
        print("Length [pix]")
        print(f"  mean     = {pix_stats['mean']:.6f}")
        print(f"  median   = {pix_stats['median']:.6f}")
        print(f"  bias     = {pix_stats['bias']:.6f}")
        print(f"  adjusted = {pix_stats['adjusted']:.6f}")
        print(f"  std      = {pix_stats['std']:.6f}")
        print(f"  sem      = {pix_stats['sem']:.6f}")
        print(f"  p16      = {pix_stats['p16']:.6f}")
        print(f"  p84      = {pix_stats['p84']:.6f}")
        print(f"  half68   = {pix_stats['half_68_range']:.6f}")
        print(f"  min/max  = {pix_stats['min']:.6f} / {pix_stats['max']:.6f}")

    if arcmin_stats is not None:
        print()
        print("Length [arcmin]")
        print(f"  mean     = {arcmin_stats['mean']:.6f}")
        print(f"  median   = {arcmin_stats['median']:.6f}")
        print(f"  bias     = {arcmin_stats['bias']:.6f}")
        print(f"  adjusted = {arcmin_stats['adjusted']:.6f}")
        print(f"  std      = {arcmin_stats['std']:.6f}")
        print(f"  sem      = {arcmin_stats['sem']:.6f}")
        print(f"  p16      = {arcmin_stats['p16']:.6f}")
        print(f"  p84      = {arcmin_stats['p84']:.6f}")
        print(f"  half68   = {arcmin_stats['half_68_range']:.6f}")
        print(f"  min/max  = {arcmin_stats['min']:.6f} / {arcmin_stats['max']:.6f}")

    print()
    print("Signed total turning angle (same tracing direction for all realizations)")
    print(f"  estimator: compare_splines.py, {args.turning_n_samples} samples, "
          f"smoothing sigma={args.turning_smooth_sigma:g} samples")
    if turning_stats is None:
        print("  No valid turning-angle measurements; see turning_error in the summary CSV.")
    else:
        for unit, stats in [("rad", turning_stats), ("deg", turning_deg_stats)]:
            print(f"  [{unit}] mean             = {stats['mean']:.6f}")
            print(f"  [{unit}] median           = {stats['median']:.6f}")
            print(f"  [{unit}] MC std (1-sigma)  = {stats['std']:.6f}")
            print(f"  [{unit}] p16 / p84        = {stats['p16']:.6f} / {stats['p84']:.6f}")
            print(f"  [{unit}] median -err/+err = {stats['median']:.6f} "
                  f"-{stats['err_minus']:.6f}/+{stats['err_plus']:.6f}")
        if turning_stats['n'] > 1:
            print(f"  Report symmetric noise uncertainty: "
                  f"Delta_theta = {turning_stats['median']:.6f} "
                  f"+/- {turning_stats['std']:.6f} rad (MC standard deviation)")
        else:
            print("  At least two valid realizations are required to estimate noise uncertainty.")
        print("  This measures noise scatter in total turning; it excludes tracing-choice systematics.")

    print()
    print(f"Wrote per-realization CSV: {summary_csv}")
    print(f"Wrote stats CSV:           {args.stats_csv}")

    if failures:
        fail_txt = Path(args.stats_csv).with_suffix(".failures.txt")
        with open(fail_txt, "w") as f:
            for item in failures:
                f.write(item + "\n")
        print(f"Wrote failure list:        {fail_txt}")


if __name__ == "__main__":
    main()
