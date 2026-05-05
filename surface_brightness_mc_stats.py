#!/usr/bin/env python3
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


def summarize(arr):
    if arr.size == 0:
        return None

    out = {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "p16": float(np.percentile(arr, 16.0)),
        "p84": float(np.percentile(arr, 84.0)),
    }
    out["sem"] = out["std"] / np.sqrt(out["n"]) if out["n"] > 1 else 0.0
    out["half_68_range"] = 0.5 * (out["p84"] - out["p16"])
    return out


def write_stats_csv(
    path,
    sb_nmgy_stats,
    sb_mag_stats,
    quoted_sb_err_stats,
    quoted_mag_err_stats,
    n_total,
    n_success,
    n_failed,
):
    fieldnames = [
        "n_total",
        "n_success",
        "n_failed",
        "sb_nmgy_n",
        "sb_nmgy_mean",
        "sb_nmgy_median",
        "sb_nmgy_std",
        "sb_nmgy_sem",
        "sb_nmgy_p16",
        "sb_nmgy_p84",
        "sb_nmgy_half_68_range",
        "sb_nmgy_min",
        "sb_nmgy_max",
        "sb_mag_n",
        "sb_mag_mean",
        "sb_mag_median",
        "sb_mag_std",
        "sb_mag_sem",
        "sb_mag_p16",
        "sb_mag_p84",
        "sb_mag_half_68_range",
        "sb_mag_min",
        "sb_mag_max",
        "quoted_sb_err_n",
        "quoted_sb_err_mean",
        "quoted_sb_err_median",
        "quoted_sb_err_std",
        "quoted_sb_err_sem",
        "quoted_sb_err_p16",
        "quoted_sb_err_p84",
        "quoted_sb_err_half_68_range",
        "quoted_sb_err_min",
        "quoted_sb_err_max",
        "quoted_mag_err_n",
        "quoted_mag_err_mean",
        "quoted_mag_err_median",
        "quoted_mag_err_std",
        "quoted_mag_err_sem",
        "quoted_mag_err_p16",
        "quoted_mag_err_p84",
        "quoted_mag_err_half_68_range",
        "quoted_mag_err_min",
        "quoted_mag_err_max",
    ]

    row = {
        "n_total": n_total,
        "n_success": n_success,
        "n_failed": n_failed,
    }

    if sb_nmgy_stats is not None:
        row.update(
            {
                "sb_nmgy_n": sb_nmgy_stats["n"],
                "sb_nmgy_mean": sb_nmgy_stats["mean"],
                "sb_nmgy_median": sb_nmgy_stats["median"],
                "sb_nmgy_std": sb_nmgy_stats["std"],
                "sb_nmgy_sem": sb_nmgy_stats["sem"],
                "sb_nmgy_p16": sb_nmgy_stats["p16"],
                "sb_nmgy_p84": sb_nmgy_stats["p84"],
                "sb_nmgy_half_68_range": sb_nmgy_stats["half_68_range"],
                "sb_nmgy_min": sb_nmgy_stats["min"],
                "sb_nmgy_max": sb_nmgy_stats["max"],
            }
        )

    if sb_mag_stats is not None:
        row.update(
            {
                "sb_mag_n": sb_mag_stats["n"],
                "sb_mag_mean": sb_mag_stats["mean"],
                "sb_mag_median": sb_mag_stats["median"],
                "sb_mag_std": sb_mag_stats["std"],
                "sb_mag_sem": sb_mag_stats["sem"],
                "sb_mag_p16": sb_mag_stats["p16"],
                "sb_mag_p84": sb_mag_stats["p84"],
                "sb_mag_half_68_range": sb_mag_stats["half_68_range"],
                "sb_mag_min": sb_mag_stats["min"],
                "sb_mag_max": sb_mag_stats["max"],
            }
        )

    if quoted_sb_err_stats is not None:
        row.update(
            {
                "quoted_sb_err_n": quoted_sb_err_stats["n"],
                "quoted_sb_err_mean": quoted_sb_err_stats["mean"],
                "quoted_sb_err_median": quoted_sb_err_stats["median"],
                "quoted_sb_err_std": quoted_sb_err_stats["std"],
                "quoted_sb_err_sem": quoted_sb_err_stats["sem"],
                "quoted_sb_err_p16": quoted_sb_err_stats["p16"],
                "quoted_sb_err_p84": quoted_sb_err_stats["p84"],
                "quoted_sb_err_half_68_range": quoted_sb_err_stats["half_68_range"],
                "quoted_sb_err_min": quoted_sb_err_stats["min"],
                "quoted_sb_err_max": quoted_sb_err_stats["max"],
            }
        )

    if quoted_mag_err_stats is not None:
        row.update(
            {
                "quoted_mag_err_n": quoted_mag_err_stats["n"],
                "quoted_mag_err_mean": quoted_mag_err_stats["mean"],
                "quoted_mag_err_median": quoted_mag_err_stats["median"],
                "quoted_mag_err_std": quoted_mag_err_stats["std"],
                "quoted_mag_err_sem": quoted_mag_err_stats["sem"],
                "quoted_mag_err_p16": quoted_mag_err_stats["p16"],
                "quoted_mag_err_p84": quoted_mag_err_stats["p84"],
                "quoted_mag_err_half_68_range": quoted_mag_err_stats["half_68_range"],
                "quoted_mag_err_min": quoted_mag_err_stats["min"],
                "quoted_mag_err_max": quoted_mag_err_stats["max"],
            }
        )

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Run measure_region_sb.py on many noise realizations and report "
            "surface-brightness statistics."
        )
    )
    ap.add_argument(
        "pattern",
        help="Glob pattern for realization FITS files, e.g. 's*_bin4_mu.fits'",
    )
    ap.add_argument(
        "region_file",
        help="DS9 region file used for all realizations",
    )
    ap.add_argument(
        "mask_fits",
        help="Mask FITS used for all realizations",
    )
    ap.add_argument(
        "--measurer",
        default=str(Path.home() / "tidal-tail-software" / "tail_surface_brightness.py"),
        help="Path to your single-image surface-brightness script",
    )
    ap.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter to use for the measurer",
    )
    ap.add_argument(
        "--summary-csv",
        default="sb_mc_measurements.csv",
        help="Per-realization summary CSV written by the measurer",
    )
    ap.add_argument(
        "--stats-csv",
        default="sb_mc_stats.csv",
        help="One-row CSV with aggregate surface-brightness statistics",
    )
    ap.add_argument(
        "--check-fits-dir",
        default=None,
        help="Optional directory for per-realization check FITS products",
    )
    ap.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any realization fails",
    )

    args, measurer_args = ap.parse_known_args()

    forbidden = {"--summary-csv", "--write-check-fits"}
    for tok in measurer_args:
        if tok in forbidden:
            raise SystemExit(
                f"Do not pass {tok} to the wrapper; it manages that itself."
            )

    files = sorted(glob.glob(args.pattern), key=natural_key)
    if not files:
        raise SystemExit(f"No files matched pattern: {args.pattern}")

    measurer = Path(args.measurer)
    if not measurer.exists():
        raise SystemExit(f"Measurement script not found: {measurer}")

    summary_csv = Path(args.summary_csv)
    if summary_csv.exists():
        summary_csv.unlink()

    check_fits_dir = None
    if args.check_fits_dir is not None:
        check_fits_dir = Path(args.check_fits_dir)
        check_fits_dir.mkdir(parents=True, exist_ok=True)

    failures = []

    print(f"Found {len(files)} realization files.")
    print(f"Per-realization summary CSV: {summary_csv}")
    print(f"Aggregate stats CSV:         {args.stats_csv}")
    if check_fits_dir is not None:
        print(f"Check FITS directory:        {check_fits_dir}")
    print()

    for i, fits_path in enumerate(files, start=1):
        fits_path = Path(fits_path)

        cmd = [
            args.python,
            str(measurer),
            str(fits_path),
            str(args.region_file),
            str(args.mask_fits),
            "--summary-csv",
            str(summary_csv),
        ] + measurer_args

        if check_fits_dir is not None:
            check_path = check_fits_dir / f"{fits_path.stem}.check.fits"
            cmd += ["--write-check-fits", str(check_path)]

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

    sb_nmgy = to_float_list(rows, "mean_sb_nmgy_arcsec2")
    sb_mag = to_float_list(rows, "mean_sb_mag_arcsec2")
    quoted_sb_err = to_float_list(rows, "quoted_sb_err_nmgy_arcsec2")
    quoted_mag_err = to_float_list(rows, "quoted_mag_err")

    sb_nmgy_stats = summarize(sb_nmgy)
    sb_mag_stats = summarize(sb_mag)
    quoted_sb_err_stats = summarize(quoted_sb_err)
    quoted_mag_err_stats = summarize(quoted_mag_err)

    n_total = len(files)
    n_success = len(rows)
    n_failed = len(failures)

    write_stats_csv(
        args.stats_csv,
        sb_nmgy_stats,
        sb_mag_stats,
        quoted_sb_err_stats,
        quoted_mag_err_stats,
        n_total,
        n_success,
        n_failed,
    )

    print()
    print("=== Monte Carlo surface-brightness statistics ===")
    print(f"Total realizations:        {n_total}")
    print(f"Successful measurements:   {n_success}")
    print(f"Failed measurements:       {n_failed}")

    if sb_nmgy_stats is not None:
        print()
        print("Mean SB [nMgy/arcsec^2]")
        print(f"  mean     = {sb_nmgy_stats['mean']:.6g}")
        print(f"  median   = {sb_nmgy_stats['median']:.6g}")
        print(f"  std      = {sb_nmgy_stats['std']:.6g}")
        print(f"  sem      = {sb_nmgy_stats['sem']:.6g}")
        print(f"  p16      = {sb_nmgy_stats['p16']:.6g}")
        print(f"  p84      = {sb_nmgy_stats['p84']:.6g}")
        print(f"  half68   = {sb_nmgy_stats['half_68_range']:.6g}")
        print(f"  min/max  = {sb_nmgy_stats['min']:.6g} / {sb_nmgy_stats['max']:.6g}")

    if sb_mag_stats is not None:
        print()
        print("Mean SB [mag/arcsec^2]")
        print(f"  mean     = {sb_mag_stats['mean']:.6f}")
        print(f"  median   = {sb_mag_stats['median']:.6f}")
        print(f"  std      = {sb_mag_stats['std']:.6f}")
        print(f"  sem      = {sb_mag_stats['sem']:.6f}")
        print(f"  p16      = {sb_mag_stats['p16']:.6f}")
        print(f"  p84      = {sb_mag_stats['p84']:.6f}")
        print(f"  half68   = {sb_mag_stats['half_68_range']:.6f}")
        print(f"  min/max  = {sb_mag_stats['min']:.6f} / {sb_mag_stats['max']:.6f}")

    if quoted_sb_err_stats is not None:
        print()
        print("Per-run quoted SB error [nMgy/arcsec^2]")
        print(f"  mean     = {quoted_sb_err_stats['mean']:.6g}")
        print(f"  median   = {quoted_sb_err_stats['median']:.6g}")
        print(f"  std      = {quoted_sb_err_stats['std']:.6g}")

    if quoted_mag_err_stats is not None:
        print()
        print("Per-run quoted mag error [mag]")
        print(f"  mean     = {quoted_mag_err_stats['mean']:.6f}")
        print(f"  median   = {quoted_mag_err_stats['median']:.6f}")
        print(f"  std      = {quoted_mag_err_stats['std']:.6f}")

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
