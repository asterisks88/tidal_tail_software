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


def compute_bias(arr, true_value):
    bias = np.median(np.array(arr) - true_value)
    return bias


def write_stats_csv(path, pix_stats, arcmin_stats, n_total, n_success, n_failed):
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

    row = {
        "n_total": n_total,
        "n_success": n_success,
        "n_failed": n_failed,
    }

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
                "length_arcmin_bias": pix_stats["bias"],
                "length_arcmin_adjusted": pix_stats["adjusted"],
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
            "length statistics."
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
        help="One-row CSV with aggregate length statistics",
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
        default=5.574,
        help="True length in arcmin derived from noise free synthetic image from model",
    )
    ap.add_argument(
        "--pixel-scale",
        default=0.262 * 4,
        help="Arcsec per pixel.  Default is bin4 Legacy Survey scale.",
    )

    args, tracer_args = ap.parse_known_args()

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
    length_pix = to_float_list(rows, "length_pix")
    length_arcmin = to_float_list(rows, "length_arcmin")

    pix_stats = summarize(length_pix)
    pix_stats["bias"] = compute_bias(
        length_pix, 60 * args.true_length / args.pixel_scale
    )
    pix_stats["adjusted"] = pix_stats["median"] - pix_stats["bias"]

    arcmin_stats = summarize(length_arcmin)
    arcmin_stats["bias"] = compute_bias(length_arcmin, args.true_length)
    arcmin_stats["adjusted"] = arcmin_stats["median"] - arcmin_stats["bias"]

    n_total = len(files)
    n_success = len(rows)
    n_failed = len(failures)

    write_stats_csv(
        args.stats_csv, pix_stats, arcmin_stats, n_total, n_success, n_failed
    )

    print()
    print("=== Monte Carlo tail-length statistics ===")
    print(f"Total realizations:        {n_total}")
    print(f"Successful traces:         {n_success}")
    print(f"Failed traces:             {n_failed}")

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
