#!/usr/bin/env python3
"""
make_synthetic_noise_realizations.py

Wrapper around mu_from_legacy_extinction_noise.py to generate many noisy
realizations from one noise-free intrinsic synthetic image.

Two intended modes:

  observed
      For tail-length comparisons.
      The synthetic image is reddened by MW_TRANSMISSION_<band> and then
      observed-frame Legacy noise from invvar is added.

  dereddened
      For intrinsic/dereddened surface-brightness comparisons with optional
      noise in the dereddened flux convention.
      The synthetic image is left intrinsic and the invvar-derived noise is
      scaled by 1/MW_TRANSMISSION_<band>.

The wrapper does not duplicate the extinction/noise logic. It builds and runs
commands for mu_from_legacy_extinction_noise.py, then writes a manifest CSV.
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Sequence


def shell_join(cmd: Sequence[str]) -> str:
    """Return a shell-readable command string for logging."""
    return " ".join(shlex.quote(str(x)) for x in cmd)


def build_driver_command(
    *,
    driver_script: Path,
    input_fits: Path,
    output_fits: Path,
    frame: str,
    invvar_fits: Path,
    invvar_hdu: str | None,
    tractor_fits: Path | None,
    dered_region: Path | None,
    dered_band: str,
    min_tractor_sources: int,
    dered_region_min_weight: float,
    dered_subpixels: int,
    seed: int,
    noise_scale: float,
    smooth: str,
    size_arcsec: bool,
    box_size_smooth: float,
    gauss_sigma: float | None,
    gauss_fwhm: float | None,
    kernel_mode: str,
    mu_floor: float,
    overwrite: bool,
    extra_args: Sequence[str],
) -> List[str]:
    """Construct one mu_from_legacy_extinction_noise.py command."""
    if frame not in {"observed", "dereddened"}:
        raise ValueError(f"Unknown frame: {frame}")

    if frame == "observed":
        # Intrinsic synthetic -> reddened observed-frame synthetic.
        extinction_mode = "redden"
        noise_frame = "observed"
        if tractor_fits is None or dered_region is None:
            raise ValueError(
                "--frame observed requires --tractor-fits and --dered-region "
                "because the synthetic image must be reddened."
            )
    else:
        # Intrinsic synthetic remains intrinsic/dereddened, but the observed
        # invvar noise must be scaled to the dereddened flux convention.
        extinction_mode = "none"
        noise_frame = "dereddened"
        if tractor_fits is None or dered_region is None:
            raise ValueError(
                "--frame dereddened requires --tractor-fits and --dered-region "
                "because dereddened-frame noise requires MW_TRANSMISSION."
            )

    cmd: List[str] = [
        sys.executable,
        str(driver_script),
        str(input_fits),
        str(output_fits),
        "--image-kind",
        "synthetic",
        "--bkg-mode",
        "none",
        "--extinction-mode",
        extinction_mode,
        "--tractor-fits",
        str(tractor_fits),
        "--dered-region",
        str(dered_region),
        "--dered-band",
        dered_band,
        "--min-tractor-sources",
        str(min_tractor_sources),
        "--dered-region-min-weight",
        str(dered_region_min_weight),
        "--dered-subpixels",
        str(dered_subpixels),
        "--invvar-fits",
        str(invvar_fits),
        "--add-noise",
        "--noise-frame",
        noise_frame,
        "--noise-scale",
        str(noise_scale),
        "--noise-seed",
        str(seed),
        "--smooth",
        smooth,
        "--mu-floor",
        str(mu_floor),
    ]

    if invvar_hdu is not None:
        cmd.extend(["--invvar-hdu", str(invvar_hdu)])

    if size_arcsec:
        cmd.append("--size-arcsec")

    if smooth == "box":
        cmd.extend(
            ["--box-size-smooth", str(box_size_smooth), "--kernel-mode", kernel_mode]
        )
    elif smooth == "gauss":
        if gauss_sigma is not None:
            cmd.extend(["--gauss-sigma", str(gauss_sigma)])
        if gauss_fwhm is not None:
            cmd.extend(["--gauss-fwhm", str(gauss_fwhm)])

    if overwrite:
        cmd.append("--overwrite")

    cmd.extend(list(extra_args))
    return cmd


def run_one(
    index: int, cmd: Sequence[str], log_file: Path | None, dry_run: bool
) -> dict:
    """Run one realization and return manifest information."""
    if dry_run:
        return {
            "index": index,
            "returncode": 0,
            "status": "dry_run",
            "log_file": "",
            "command": shell_join(cmd),
        }

    stdout = subprocess.PIPE if log_file is not None else None
    stderr = subprocess.STDOUT if log_file is not None else None

    proc = subprocess.run(
        list(cmd),
        text=True,
        stdout=stdout,
        stderr=stderr,
        check=False,
    )

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(proc.stdout or "")

    return {
        "index": index,
        "returncode": proc.returncode,
        "status": "ok" if proc.returncode == 0 else "failed",
        "log_file": "" if log_file is None else str(log_file),
        "command": shell_join(cmd),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Generate many noisy synthetic realizations by repeatedly calling "
            "mu_from_legacy_extinction_noise.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    ap.add_argument(
        "synthetic_fits", help="Noise-free intrinsic synthetic image [nMgy/pix]."
    )
    ap.add_argument(
        "outdir", help="Directory where output realization FITS files are written."
    )

    ap.add_argument(
        "--driver-script",
        default="mu_from_legacy_extinction_noise.py",
        help="Path to mu_from_legacy_extinction_noise.py or the bkgfix version.",
    )
    ap.add_argument(
        "--n-realizations",
        type=int,
        required=True,
        help="Number of noisy realizations to produce.",
    )
    ap.add_argument(
        "--frame",
        choices=["observed", "dereddened"],
        default="observed",
        help=(
            "observed: redden synthetic image and add observed-frame invvar noise. "
            "dereddened: keep intrinsic image and add dereddened-frame invvar noise."
        ),
    )
    ap.add_argument(
        "--prefix",
        default="synthetic_realization",
        help="Output filename prefix.",
    )
    ap.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="First realization index used in filenames.",
    )

    # Noise inputs
    ap.add_argument(
        "--invvar-fits", required=True, help="Legacy inverse-variance FITS image."
    )
    ap.add_argument(
        "--invvar-hdu", default=None, help="HDU index or EXTNAME for invvar image."
    )
    ap.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Multiplier for invvar-derived sigma.",
    )
    ap.add_argument(
        "--base-seed", type=int, default=12345, help="Seed for first realization."
    )
    ap.add_argument(
        "--seed-stride", type=int, default=1, help="Seed increment per realization."
    )

    # Extinction-region inputs
    ap.add_argument(
        "--tractor-fits", required=True, help="Legacy Tractor FITS catalog."
    )
    ap.add_argument(
        "--dered-region",
        required=True,
        help="DS9 region used to estimate MW_TRANSMISSION.",
    )
    ap.add_argument("--dered-band", choices=["g", "r", "z"], default="r")
    ap.add_argument("--min-tractor-sources", type=int, default=3)
    ap.add_argument("--dered-region-min-weight", type=float, default=0.0)
    ap.add_argument("--dered-subpixels", type=int, default=8)

    # Optional smoothing passed through to the driver
    ap.add_argument("--smooth", choices=["none", "box", "gauss"], default="none")
    ap.add_argument("--size-arcsec", action="store_true")
    ap.add_argument("--box-size-smooth", type=float, default=13)
    ap.add_argument("--gauss-sigma", type=float, default=None)
    ap.add_argument("--gauss-fwhm", type=float, default=None)
    ap.add_argument("--kernel-mode", choices=["mean", "sum"], default="mean")
    ap.add_argument("--mu-floor", type=float, default=32.0)

    # Execution controls
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of subprocesses to run concurrently.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass --overwrite to the driver script.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print/write commands without executing them.",
    )
    ap.add_argument(
        "--no-logs",
        action="store_true",
        help="Do not write per-realization stdout logs.",
    )
    ap.add_argument(
        "--manifest",
        default=None,
        help="Manifest CSV path. Default: <outdir>/<prefix>_<frame>_manifest.csv",
    )
    ap.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments passed verbatim to the driver. Put this last.",
    )

    args = ap.parse_args()

    if args.n_realizations <= 0:
        raise SystemExit("--n-realizations must be positive.")
    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive.")
    if args.smooth == "gauss" and args.gauss_sigma is None and args.gauss_fwhm is None:
        raise SystemExit("--smooth gauss requires --gauss-sigma or --gauss-fwhm.")

    synthetic_fits = Path(args.synthetic_fits)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    driver_script = Path(args.driver_script)
    invvar_fits = Path(args.invvar_fits)
    tractor_fits = Path(args.tractor_fits)
    dered_region = Path(args.dered_region)

    manifest_path = (
        Path(args.manifest)
        if args.manifest
        else outdir / f"{args.prefix}_{args.frame}_manifest.csv"
    )
    logs_dir = outdir / "logs"

    rows = []
    jobs = []

    for k in range(args.n_realizations):
        idx = args.start_index + k
        seed = args.base_seed + k * args.seed_stride
        output_fits = outdir / f"{args.prefix}_{args.frame}_{idx:04d}.fits"
        log_file = (
            None
            if args.no_logs
            else logs_dir / f"{args.prefix}_{args.frame}_{idx:04d}.log"
        )

        cmd = build_driver_command(
            driver_script=driver_script,
            input_fits=synthetic_fits,
            output_fits=output_fits,
            frame=args.frame,
            invvar_fits=invvar_fits,
            invvar_hdu=args.invvar_hdu,
            tractor_fits=tractor_fits,
            dered_region=dered_region,
            dered_band=args.dered_band,
            min_tractor_sources=args.min_tractor_sources,
            dered_region_min_weight=args.dered_region_min_weight,
            dered_subpixels=args.dered_subpixels,
            seed=seed,
            noise_scale=args.noise_scale,
            smooth=args.smooth,
            size_arcsec=args.size_arcsec,
            box_size_smooth=args.box_size_smooth,
            gauss_sigma=args.gauss_sigma,
            gauss_fwhm=args.gauss_fwhm,
            kernel_mode=args.kernel_mode,
            mu_floor=args.mu_floor,
            overwrite=args.overwrite,
            extra_args=args.extra_args,
        )

        rows.append(
            {
                "index": idx,
                "seed": seed,
                "frame": args.frame,
                "output_fits": str(output_fits),
                "log_file": "" if log_file is None else str(log_file),
                "command": shell_join(cmd),
                "returncode": "",
                "status": "pending",
            }
        )
        jobs.append((idx, cmd, log_file))

    print(f"Prepared {len(jobs)} realization(s).")
    print(f"Frame:      {args.frame}")
    print(f"Outdir:     {outdir}")
    print(f"Manifest:   {manifest_path}")
    if args.dry_run:
        print("Dry run: commands will not be executed.")

    results_by_index = {}
    if args.jobs == 1:
        for idx, cmd, log_file in jobs:
            print(
                f"Running realization {idx}: seed={args.base_seed + (idx - args.start_index) * args.seed_stride}"
            )
            print(shell_join(cmd))
            results_by_index[idx] = run_one(idx, cmd, log_file, args.dry_run)
            if results_by_index[idx]["returncode"] != 0:
                print(
                    f"ERROR: realization {idx} failed. See {log_file}", file=sys.stderr
                )
                break
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as exe:
            futs = {
                exe.submit(run_one, idx, cmd, log_file, args.dry_run): idx
                for idx, cmd, log_file in jobs
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    res = fut.result()
                except Exception as exc:
                    res = {
                        "index": idx,
                        "returncode": -999,
                        "status": f"exception: {exc}",
                        "log_file": "",
                        "command": "",
                    }
                results_by_index[idx] = res
                print(f"Finished realization {idx}: {res['status']}")

    n_failed = 0
    for row in rows:
        idx = int(row["index"])
        res = results_by_index.get(idx)
        if res is not None:
            row["returncode"] = res["returncode"]
            row["status"] = res["status"]
            if res.get("log_file"):
                row["log_file"] = res["log_file"]
        if row["status"] not in {"ok", "dry_run"}:
            n_failed += 1

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote manifest: {manifest_path}")
    if n_failed:
        raise SystemExit(f"{n_failed} realization(s) failed.")


if __name__ == "__main__":
    main()
