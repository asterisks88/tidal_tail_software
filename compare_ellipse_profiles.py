#!/usr/bin/env python3
"""
compare_ellipse_profiles.py

Compare two profile CSV files produced by legacy_ellipse_profile.py.

Layouts
-------
full:
  1. Cumulative aperture flux
  2. Integrated magnitude
  3. Mean surface brightness within the aperture
  4. Annular surface brightness
  5. Annular surface-brightness residual

paper:
  1. Annular surface brightness
  2. Annular surface-brightness residual

The residual convention is:

    Delta mu_ann = mu_ann,profile2 - mu_ann,profile1

Thus, when profile 1 is observed and profile 2 is synthetic:
  - Delta mu > 0: synthetic profile is fainter.
  - Delta mu < 0: synthetic profile is brighter.

Error bars
----------
Flux and linear surface-brightness errors are read from the profile CSV.
Magnitude-space errors are either read directly or calculated as

    sigma_mu = (2.5 / ln(10)) * sigma_I / I.

Residual uncertainties are propagated as

    sigma_Delta = sqrt(sigma_1^2 + sigma_2^2),

which assumes independent errors between the two profiles.

Residual radii
--------------
Profile 1 defines the residual radius grid. If the two profiles do not have
the same radii, profile 2 and its variance are linearly interpolated onto the
finite profile-1 radii within the overlapping radial range.

Examples
--------
Paper-focused observed/synthetic comparison:

  python compare_ellipse_profiles.py observed.csv synthetic.csv \
      --label1 "Observed" --label2 "Synthetic" \
      --layout paper \
      --output observed_synthetic_profiles.pdf

Full diagnostic comparison:

  python compare_ellipse_profiles.py observed.csv synthetic.csv \
      --label1 "Observed" --label2 "Synthetic" \
      --layout full \
      --output observed_synthetic_diagnostics.png

Use physical radius:

  python compare_ellipse_profiles.py observed.csv synthetic.csv \
      --layout paper \
      --x-axis sma_kpc \
      --output observed_synthetic_kpc.pdf
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


AB_ZP_NMGY = 22.5
MAG_ERROR_FACTOR = 2.5 / np.log(10.0)


def read_profile_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Read a profile CSV while ignoring commented metadata lines."""
    path = Path(path)

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(
            line for line in handle if not line.lstrip().startswith("#")
        )

        if reader.fieldnames is None:
            raise ValueError(f"{path} does not contain a CSV header")

        columns: dict[str, list[float]] = {
            name: [] for name in reader.fieldnames
        }

        for row in reader:
            for name in reader.fieldnames:
                value = row.get(name, "")
                try:
                    columns[name].append(float(value))
                except (TypeError, ValueError):
                    columns[name].append(np.nan)

    if not columns or not any(len(values) for values in columns.values()):
        raise ValueError(f"{path} contains no profile rows")

    return {
        name: np.asarray(values, dtype=float)
        for name, values in columns.items()
    }


def require_column(
    profile: dict[str, np.ndarray],
    name: str,
    profile_name: str,
) -> np.ndarray:
    if name not in profile:
        raise ValueError(
            f"{profile_name} does not contain required column {name!r}"
        )
    return profile[name]


def has_finite_column(
    profile: dict[str, np.ndarray],
    name: str,
) -> bool:
    return (
        name in profile
        and np.any(np.isfinite(np.asarray(profile[name], dtype=float)))
    )


def magnitude_from_flux(flux: np.ndarray) -> np.ndarray:
    flux = np.asarray(flux, dtype=float)
    result = np.full(flux.shape, np.nan, dtype=float)
    good = np.isfinite(flux) & (flux > 0)
    result[good] = AB_ZP_NMGY - 2.5 * np.log10(flux[good])
    return result


def magnitude_error_from_linear(
    value: np.ndarray,
    error: np.ndarray,
) -> np.ndarray:
    """Convert a linear uncertainty to a magnitude uncertainty."""
    value = np.asarray(value, dtype=float)
    error = np.asarray(error, dtype=float)
    value_b, error_b = np.broadcast_arrays(value, error)

    result = np.full(value_b.shape, np.nan, dtype=float)
    good = (
        np.isfinite(value_b)
        & np.isfinite(error_b)
        & (value_b > 0)
        & (error_b >= 0)
    )
    result[good] = (
        MAG_ERROR_FACTOR * error_b[good] / value_b[good]
    )
    return result


def choose_mode(
    requested_mode: str,
    profile1: dict[str, np.ndarray],
    profile2: dict[str, np.ndarray],
) -> str:
    """
    Select a common photometric basis.

    Auto mode prefers extinction-corrected columns when both files contain
    them. It uses native columns only when both files contain native values.
    It never silently compares corrected values in one file with uncorrected
    values in the other.
    """
    extcorr_required = [
        "flux_extcorr_nmgy",
        "flux_extcorr_err_nmgy",
        "annulus_sb_extcorr_nmgy_arcsec2",
        "annulus_sb_extcorr_err_nmgy_arcsec2",
    ]
    native_required = [
        "flux_nmgy",
        "flux_err_nmgy",
        "annulus_sb_nmgy_arcsec2",
        "annulus_sb_err_nmgy_arcsec2",
    ]

    both_extcorr = all(
        has_finite_column(profile1, name)
        and has_finite_column(profile2, name)
        for name in extcorr_required
    )
    both_native = all(
        has_finite_column(profile1, name)
        and has_finite_column(profile2, name)
        for name in native_required
    )

    if requested_mode == "extcorr":
        if not both_extcorr:
            raise ValueError(
                "--mode extcorr requires valid extinction-corrected "
                "photometry columns in both files"
            )
        return "extcorr"

    if requested_mode == "native":
        if not both_native:
            raise ValueError(
                "--mode native requires valid native photometry columns "
                "in both files"
            )
        return "native"

    if both_extcorr:
        return "extcorr"

    if both_native:
        return "native"

    raise ValueError(
        "The profiles do not share a common photometric basis. One file may "
        "contain corrected-only values while the other lacks extinction-"
        "corrected columns. Generate corrected columns for both profiles or "
        "regenerate both on the same basis."
    )


def get_profile_quantities(
    profile: dict[str, np.ndarray],
    profile_name: str,
    mode: str,
    x_column: str,
) -> dict[str, np.ndarray]:
    x = require_column(profile, x_column, profile_name)

    if not np.any(np.isfinite(x)):
        raise ValueError(
            f"{profile_name}: column {x_column!r} contains no finite values"
        )

    if mode == "extcorr":
        flux = require_column(profile, "flux_extcorr_nmgy", profile_name)
        flux_err = require_column(
            profile, "flux_extcorr_err_nmgy", profile_name
        )

        integrated_mag = profile.get(
            "integrated_mag_extcorr_ab",
            magnitude_from_flux(flux),
        )
        integrated_mag_err = profile.get(
            "integrated_mag_extcorr_err",
            magnitude_error_from_linear(flux, flux_err),
        )

        mean_sb = require_column(
            profile, "mean_sb_extcorr_nmgy_arcsec2", profile_name
        )
        mean_sb_err = require_column(
            profile, "mean_sb_extcorr_err_nmgy_arcsec2", profile_name
        )

        annular_sb = require_column(
            profile, "annulus_sb_extcorr_nmgy_arcsec2", profile_name
        )
        annular_sb_err = require_column(
            profile,
            "annulus_sb_extcorr_err_nmgy_arcsec2",
            profile_name,
        )

    else:
        flux = require_column(profile, "flux_nmgy", profile_name)
        flux_err = require_column(profile, "flux_err_nmgy", profile_name)

        integrated_mag = profile.get(
            "integrated_mag_ab",
            magnitude_from_flux(flux),
        )
        integrated_mag_err = profile.get(
            "integrated_mag_err",
            magnitude_error_from_linear(flux, flux_err),
        )

        mean_sb = require_column(
            profile, "mean_sb_nmgy_arcsec2", profile_name
        )
        mean_sb_err = require_column(
            profile, "mean_sb_err_nmgy_arcsec2", profile_name
        )

        annular_sb = require_column(
            profile, "annulus_sb_nmgy_arcsec2", profile_name
        )
        annular_sb_err = require_column(
            profile, "annulus_sb_err_nmgy_arcsec2", profile_name
        )

    return {
        "x": np.asarray(x, dtype=float),
        "flux": np.asarray(flux, dtype=float),
        "flux_err": np.asarray(flux_err, dtype=float),
        "integrated_mag": np.asarray(integrated_mag, dtype=float),
        "integrated_mag_err": np.asarray(
            integrated_mag_err, dtype=float
        ),
        "mean_mu": magnitude_from_flux(mean_sb),
        "mean_mu_err": magnitude_error_from_linear(
            mean_sb, mean_sb_err
        ),
        "annular_mu": magnitude_from_flux(annular_sb),
        "annular_mu_err": magnitude_error_from_linear(
            annular_sb, annular_sb_err
        ),
    }


def finite_for_errorbar(
    x: np.ndarray,
    y: np.ndarray,
    yerr: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    good = np.isfinite(x) & np.isfinite(y)

    if yerr is not None:
        good &= np.isfinite(yerr) & (yerr >= 0)
        return x[good], y[good], yerr[good]

    return x[good], y[good], None


def plot_series(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
    label: str,
    marker: str,
) -> None:
    x_plot, y_plot, yerr_plot = finite_for_errorbar(x, y, yerr)

    if x_plot.size == 0:
        return

    ax.errorbar(
        x_plot,
        y_plot,
        yerr=yerr_plot,
        marker=marker,
        markersize=3.5,
        linewidth=1.0,
        capsize=2.0,
        elinewidth=0.8,
        label=label,
    )


def x_axis_label(column: str) -> str:
    labels = {
        "sma_arcsec": "Semi-major-axis radius [arcsec]",
        "sma_kpc": "Semi-major-axis radius [kpc]",
        "circularized_radius_arcsec": "Circularized radius [arcsec]",
        "semi_minor_arcsec": "Semi-minor-axis radius [arcsec]",
        "sma_pix": "Semi-major-axis radius [pixel]",
    }
    return labels.get(column, column)


def sorted_finite_profile(
    x: np.ndarray,
    y: np.ndarray,
    yerr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    good = (
        np.isfinite(x)
        & np.isfinite(y)
        & np.isfinite(yerr)
        & (yerr >= 0)
    )
    x_out = np.asarray(x[good], dtype=float)
    y_out = np.asarray(y[good], dtype=float)
    e_out = np.asarray(yerr[good], dtype=float)

    if x_out.size == 0:
        return x_out, y_out, e_out

    order = np.argsort(x_out)
    x_out = x_out[order]
    y_out = y_out[order]
    e_out = e_out[order]

    # np.interp requires a strictly increasing grid. If duplicate radii exist,
    # retain the first occurrence.
    unique_x, unique_index = np.unique(x_out, return_index=True)
    return (
        unique_x,
        y_out[unique_index],
        e_out[unique_index],
    )


def annular_residuals(
    quantities1: dict[str, np.ndarray],
    quantities2: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate profile2 - profile1 on profile 1's finite radius grid.

    Profile 2's magnitude profile is linearly interpolated. Its variance is
    interpolated before taking the square root.
    """
    x1, mu1, err1 = sorted_finite_profile(
        quantities1["x"],
        quantities1["annular_mu"],
        quantities1["annular_mu_err"],
    )
    x2, mu2, err2 = sorted_finite_profile(
        quantities2["x"],
        quantities2["annular_mu"],
        quantities2["annular_mu_err"],
    )

    if x1.size == 0 or x2.size == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    overlap = (x1 >= x2[0]) & (x1 <= x2[-1])
    x_res = x1[overlap]
    mu1_res = mu1[overlap]
    err1_res = err1[overlap]

    if x_res.size == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    mu2_interp = np.interp(x_res, x2, mu2)
    var2_interp = np.interp(x_res, x2, err2**2)
    err2_interp = np.sqrt(np.maximum(var2_interp, 0.0))

    delta_mu = mu2_interp - mu1_res
    delta_mu_err = np.sqrt(err1_res**2 + err2_interp**2)

    return x_res, delta_mu, delta_mu_err


def configure_annular_panel(
    ax: plt.Axes,
    quantities1: dict[str, np.ndarray],
    quantities2: dict[str, np.ndarray],
    label1: str,
    label2: str,
) -> None:
    plot_series(
        ax,
        quantities1["x"],
        quantities1["annular_mu"],
        quantities1["annular_mu_err"],
        label1,
        "o",
    )
    plot_series(
        ax,
        quantities2["x"],
        quantities2["annular_mu"],
        quantities2["annular_mu_err"],
        label2,
        "s",
    )
    ax.invert_yaxis()
    ax.set_ylabel(
        r"Annular $\mu$ [AB mag arcsec$^{-2}$]"
    )
    ax.grid(alpha=0.25)
    ax.legend(loc="best")


def configure_residual_panel(
    ax: plt.Axes,
    x_res: np.ndarray,
    delta_mu: np.ndarray,
    delta_mu_err: np.ndarray,
    label1: str,
    label2: str,
    x_column: str,
    residual_limit: Optional[float],
) -> None:
    if x_res.size:
        ax.errorbar(
            x_res,
            delta_mu,
            yerr=delta_mu_err,
            marker="o",
            markersize=3.5,
            linewidth=1.0,
            capsize=2.0,
            elinewidth=0.8,
        )

    ax.axhline(0.0, linewidth=1.0, linestyle="--")
    ax.set_ylabel(
        r"$\Delta\mu_{\rm ann}$"
        "\n"
        f"({label2} - {label1})"
    )
    ax.set_xlabel(x_axis_label(x_column))
    ax.grid(alpha=0.25)

    if residual_limit is not None:
        if not np.isfinite(residual_limit) or residual_limit <= 0:
            raise ValueError("--residual-limit must be positive and finite")
        ax.set_ylim(-residual_limit, residual_limit)


def make_full_figure(
    quantities1: dict[str, np.ndarray],
    quantities2: dict[str, np.ndarray],
    label1: str,
    label2: str,
    x_res: np.ndarray,
    delta_mu: np.ndarray,
    delta_mu_err: np.ndarray,
    args: argparse.Namespace,
) -> plt.Figure:
    fig, axes = plt.subplots(
        5,
        1,
        figsize=(8.5, 15.5),
        sharex=True,
        constrained_layout=True,
    )

    plot_series(
        axes[0],
        quantities1["x"],
        quantities1["flux"],
        quantities1["flux_err"],
        label1,
        "o",
    )
    plot_series(
        axes[0],
        quantities2["x"],
        quantities2["flux"],
        quantities2["flux_err"],
        label2,
        "s",
    )
    axes[0].set_ylabel("Cumulative flux [nMgy]")
    axes[0].grid(alpha=0.25)
    axes[0].legend(loc="best")

    plot_series(
        axes[1],
        quantities1["x"],
        quantities1["integrated_mag"],
        quantities1["integrated_mag_err"],
        label1,
        "o",
    )
    plot_series(
        axes[1],
        quantities2["x"],
        quantities2["integrated_mag"],
        quantities2["integrated_mag_err"],
        label2,
        "s",
    )
    axes[1].invert_yaxis()
    axes[1].set_ylabel("Integrated magnitude [AB]")
    axes[1].grid(alpha=0.25)

    plot_series(
        axes[2],
        quantities1["x"],
        quantities1["mean_mu"],
        quantities1["mean_mu_err"],
        label1,
        "o",
    )
    plot_series(
        axes[2],
        quantities2["x"],
        quantities2["mean_mu"],
        quantities2["mean_mu_err"],
        label2,
        "s",
    )
    axes[2].invert_yaxis()
    axes[2].set_ylabel(
        r"Mean $\mu$ within aperture [AB mag arcsec$^{-2}$]"
    )
    axes[2].grid(alpha=0.25)

    configure_annular_panel(
        axes[3],
        quantities1,
        quantities2,
        label1,
        label2,
    )
    configure_residual_panel(
        axes[4],
        x_res,
        delta_mu,
        delta_mu_err,
        label1,
        label2,
        args.x_axis,
        args.residual_limit,
    )

    return fig


def make_paper_figure(
    quantities1: dict[str, np.ndarray],
    quantities2: dict[str, np.ndarray],
    label1: str,
    label2: str,
    x_res: np.ndarray,
    delta_mu: np.ndarray,
    delta_mu_err: np.ndarray,
    args: argparse.Namespace,
) -> plt.Figure:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(7.5, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3.0, 1.35]},
        constrained_layout=True,
    )

    configure_annular_panel(
        axes[0],
        quantities1,
        quantities2,
        label1,
        label2,
    )
    configure_residual_panel(
        axes[1],
        x_res,
        delta_mu,
        delta_mu_err,
        label1,
        label2,
        args.x_axis,
        args.residual_limit,
    )

    return fig


def compare_profiles(args: argparse.Namespace) -> None:
    profile1 = read_profile_csv(args.profile1)
    profile2 = read_profile_csv(args.profile2)

    mode = choose_mode(args.mode, profile1, profile2)

    quantities1 = get_profile_quantities(
        profile1,
        str(args.profile1),
        mode,
        args.x_axis,
    )
    quantities2 = get_profile_quantities(
        profile2,
        str(args.profile2),
        mode,
        args.x_axis,
    )

    label1 = args.label1 or Path(args.profile1).stem
    label2 = args.label2 or Path(args.profile2).stem

    x_res, delta_mu, delta_mu_err = annular_residuals(
        quantities1,
        quantities2,
    )

    if x_res.size == 0:
        raise ValueError(
            "No finite overlapping annular surface-brightness points were "
            "available for the residual panel"
        )

    if args.layout == "paper":
        fig = make_paper_figure(
            quantities1,
            quantities2,
            label1,
            label2,
            x_res,
            delta_mu,
            delta_mu_err,
            args,
        )
    else:
        fig = make_full_figure(
            quantities1,
            quantities2,
            label1,
            label2,
            x_res,
            delta_mu,
            delta_mu_err,
            args,
        )

    if args.title:
        fig.suptitle(args.title)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi, bbox_inches="tight")

    if args.show:
        plt.show()

    plt.close(fig)

    rms = float(np.sqrt(np.mean(delta_mu**2)))
    median = float(np.median(delta_mu))
    reduced_chi2 = float(
        np.mean((delta_mu / delta_mu_err) ** 2)
    )

    print(f"Wrote comparison plot: {output}")
    print(f"Layout: {args.layout}")
    print(f"Photometry mode: {mode}")
    print(f"Residual convention: {label2} - {label1}")
    print(f"Residual points: {x_res.size}")
    print(f"Median annular residual: {median:.4f} mag arcsec^-2")
    print(f"RMS annular residual: {rms:.4f} mag arcsec^-2")
    print(
        "Mean squared normalized residual: "
        f"{reduced_chi2:.4f}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two legacy_ellipse_profile.py CSV files with error "
            "bars and an annular surface-brightness residual panel."
        )
    )

    parser.add_argument("profile1", help="First profile CSV")
    parser.add_argument("profile2", help="Second profile CSV")
    parser.add_argument(
        "--label1",
        default=None,
        help="Legend label for the first profile",
    )
    parser.add_argument(
        "--label2",
        default=None,
        help="Legend label for the second profile",
    )
    parser.add_argument(
        "--output",
        default="profile_comparison.png",
        help="Output image or PDF [default: profile_comparison.png]",
    )
    parser.add_argument(
        "--title",
        default=None,
        help="Optional figure title; omitted by default",
    )
    parser.add_argument(
        "--layout",
        choices=["full", "paper"],
        default="full",
        help=(
            "Figure layout: full shows all diagnostic panels; paper shows "
            "only annular surface brightness and residuals [default: full]"
        ),
    )
    parser.add_argument(
        "--x-axis",
        choices=[
            "sma_arcsec",
            "sma_kpc",
            "circularized_radius_arcsec",
            "semi_minor_arcsec",
            "sma_pix",
        ],
        default="sma_arcsec",
        help="Radius column used for the x-axis [default: sma_arcsec]",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "native", "extcorr"],
        default="auto",
        help=(
            "Common photometric basis. Auto prefers extinction-corrected "
            "columns when both files contain them and otherwise uses native "
            "columns only when both files contain them [default: auto]"
        ),
    )
    parser.add_argument(
        "--residual-limit",
        type=float,
        default=None,
        help=(
            "Optional symmetric residual y-axis limit in mag arcsec^-2, "
            "for example --residual-limit 0.5"
        ),
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Raster output resolution [default: 180]",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot interactively",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        compare_profiles(args)
    except Exception as exc:
        parser.error(str(exc))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
