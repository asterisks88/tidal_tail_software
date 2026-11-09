#!/usr/bin/env python3
"""Plot measured elliptical-annulus profiles without modifying the photometry.

Dependencies: Python 3.10+, numpy, matplotlib.
See README.txt for an example using the supplied observed and synthetic CSVs.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MaxNLocator, MultipleLocator

MODEL_COLOR = "#D55E00"


@dataclass
class Profile:
    path: Path
    radius: np.ndarray
    mu: np.ndarray
    mu_error: np.ndarray
    minor_axis: np.ndarray
    metadata: dict[str, str]


@dataclass
class Comparison:
    name: str
    observed: Profile
    synthetic: Profile

    @property
    def residual(self) -> np.ndarray:
        return self.synthetic.mu - self.observed.mu

    @property
    def residual_error(self) -> np.ndarray:
        # Formal errors only; covariance and additional systematics are unknown.
        return np.hypot(self.observed.mu_error, self.synthetic.mu_error)


def read_profile(path: Path) -> Profile:
    metadata = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        lines = stream.readlines()
    for line in lines:
        if line.startswith("#") and " = " in line:
            key, value = line[1:].strip().split(" = ", 1)
            metadata[key] = value
    reader = csv.DictReader(
        line for line in lines if line.strip() and not line.lstrip().startswith("#")
    )
    required = (
        "sma_arcsec",
        "semi_minor_arcsec",
        "annulus_mu_extcorr_ab_arcsec2",
        "annulus_sb_extcorr_nmgy_arcsec2",
        "annulus_sb_extcorr_err_nmgy_arcsec2",
    )
    missing = set(required) - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
    rows = list(reader)
    if not rows:
        raise ValueError(f"{path}: no profile measurements")
    values = {
        key: np.array([float(row[key]) for row in rows], dtype=float)
        for key in required
    }
    if not all(np.isfinite(value).all() for value in values.values()):
        raise ValueError(f"{path}: non-finite required measurements")
    radius = values["sma_arcsec"]
    intensity = values["annulus_sb_extcorr_nmgy_arcsec2"]
    intensity_error = values["annulus_sb_extcorr_err_nmgy_arcsec2"]
    if np.any(radius <= 0) or np.any(np.diff(radius) <= 0):
        raise ValueError(f"{path}: radii must be positive and strictly increasing")
    if np.any(intensity <= 0) or np.any(intensity_error < 0):
        raise ValueError(f"{path}: invalid surface brightness or uncertainty")
    minor = values["semi_minor_arcsec"]
    if np.any(minor <= 0) or np.any(minor > radius):
        raise ValueError(f"{path}: invalid elliptical-aperture minor axes")
    return Profile(
        path,
        radius,
        values["annulus_mu_extcorr_ab_arcsec2"],
        (2.5 / np.log(10.0)) * intensity_error / intensity,
        minor,
        metadata,
    )


def compare(name: str, observed: Path, synthetic: Path) -> Comparison:
    obs, syn = read_profile(observed), read_profile(synthetic)
    for key in ("radius", "minor_axis"):
        a, b = getattr(obs, key), getattr(syn, key)
        if a.shape != b.shape or not np.allclose(a, b, rtol=1e-9, atol=1e-9):
            raise ValueError(f"{name}: observed and synthetic aperture {key} differ")
    for key in ("theta_image_deg", "tractor_geometry_description"):
        if key in obs.metadata and key in syn.metadata:
            if obs.metadata[key] != syn.metadata[key]:
                raise ValueError(f"{name}: aperture metadata {key} differ")
    return Comparison(name, obs, syn)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["STIXGeneral", "DejaVu Serif"],
            "font.size": 9.5,
            "mathtext.fontset": "stix",
            "axes.labelsize": 10.5,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "0.2",
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "legend.fontsize": 9.0,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def residual_extent(comparisons: list[Comparison]) -> float:
    extent = max(np.max(np.abs(c.residual) + c.residual_error) for c in comparisons)
    return max(0.1, np.ceil(extent * 1.12 / 0.05) * 0.05)


def mu_limits(comparisons: list[Comparison]) -> tuple[float, float]:
    measurements = [p for c in comparisons for p in (c.observed, c.synthetic)]
    bright = min(np.min(p.mu - p.mu_error) for p in measurements)
    faint = max(np.max(p.mu + p.mu_error) for p in measurements)
    return (np.floor((bright - 0.45) * 2) / 2, np.ceil((faint + 0.35) * 2) / 2)


def draw_comparison(top, bottom, c: Comparison, label: str, extent: float) -> None:
    common = {"capsize": 1.5, "capthick": 0.7, "elinewidth": 0.7}
    obs, syn = c.observed, c.synthetic
    top.errorbar(
        obs.radius,
        obs.mu,
        yerr=obs.mu_error,
        fmt="o-",
        color="black",
        markerfacecolor="black",
        markersize=3.1,
        markeredgewidth=0.7,
        linewidth=0.7,
        label="Observed",
        zorder=2,
        **common,
    )
    top.errorbar(
        syn.radius,
        syn.mu,
        yerr=syn.mu_error,
        fmt="s--",
        color=MODEL_COLOR,
        markerfacecolor="white",
        markersize=3.8,
        markeredgewidth=0.9,
        linewidth=0.85,
        label="Synthetic",
        zorder=3,
        **common,
    )
    top.text(
        0.04,
        0.96,
        label,
        transform=top.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
    )
    top.legend(loc="lower left", handlelength=2.4, borderaxespad=0.7)
    top.tick_params(labelbottom=False)
    top.yaxis.set_major_locator(MultipleLocator(1.0))
    top.yaxis.set_minor_locator(AutoMinorLocator(2))
    bottom.axhline(0, color="0.55", linewidth=0.7, linestyle=(0, (4, 3)), zorder=1)
    bottom.errorbar(
        obs.radius,
        c.residual,
        yerr=c.residual_error,
        fmt="o",
        color="0.15",
        markerfacecolor="0.15",
        markersize=3.0,
        markeredgewidth=0.6,
        zorder=2,
        **common,
    )
    bottom.set_ylim(-extent, extent)
    bottom.yaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
    bottom.yaxis.set_minor_locator(AutoMinorLocator(2))
    bottom.set_xlabel(r"$a\;(\mathrm{arcsec})$", labelpad=3.0)
    bottom.set_xlim(0, obs.radius[-1] + max(0.35, obs.radius[-1] * 0.025))
    bottom.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    bottom.xaxis.set_minor_locator(AutoMinorLocator(2))


def save_figure(fig, output: Path, dpi: int) -> None:
    # Explicit margins keep the intended final print dimensions fixed.
    fig.savefig(
        output.with_suffix(".pdf"),
        metadata={
            "Title": "Observed and synthetic elliptical-annulus r-band profiles",
            "Subject": "Extinction-corrected photometry and synthetic-minus-observed residuals",
            "Creator": "plot_apj_ellipse_profiles.py / Matplotlib",
        },
    )
    fig.savefig(output.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


def plot_combined(
    comparisons: list[Comparison], output: Path, dpi: int, extent: float
) -> None:
    fig = plt.figure(figsize=(7.1, 3.65))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(3, 1),
        left=0.085,
        right=0.982,
        bottom=0.145,
        top=0.975,
        wspace=0.11,
        hspace=0.035,
    )
    for i, c in enumerate(comparisons):
        top = fig.add_subplot(grid[0, i])
        bottom = fig.add_subplot(grid[1, i], sharex=top)
        draw_comparison(top, bottom, c, f"({chr(97 + i)}) {c.name}", extent)
        top.set_ylim(*mu_limits(comparisons)[::-1])
        if i == 0:
            top.set_ylabel(r"$\mu_r\;(\mathrm{mag}\;\mathrm{arcsec}^{-2})$", labelpad=5)
            bottom.set_ylabel(r"$\Delta\mu_r\;(\mathrm{mag})$", labelpad=5)
        else:
            top.tick_params(labelleft=False)
            bottom.tick_params(labelleft=False)
    save_figure(fig, output, dpi)


def plot_single(c: Comparison, output: Path, dpi: int, extent: float) -> None:
    fig = plt.figure(figsize=(3.55, 3.65))
    grid = fig.add_gridspec(
        2,
        1,
        height_ratios=(3, 1),
        left=0.17,
        right=0.965,
        bottom=0.145,
        top=0.975,
        hspace=0.035,
    )
    top = fig.add_subplot(grid[0])
    bottom = fig.add_subplot(grid[1], sharex=top)
    draw_comparison(top, bottom, c, c.name, extent)
    top.set_ylim(*mu_limits([c])[::-1])
    top.set_ylabel(r"$\mu_r\;(\mathrm{mag}\;\mathrm{arcsec}^{-2})$", labelpad=5)
    bottom.set_ylabel(r"$\Delta\mu_r\;(\mathrm{mag})$", labelpad=5)
    save_figure(fig, output, dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for role in ("host", "satellite"):
        for kind in ("observed", "synthetic"):
            parser.add_argument(f"--{role}-{kind}", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    parser.add_argument("--dpi", type=int, default=400)
    args = parser.parse_args()
    if args.dpi < 300:
        parser.error("Use --dpi 300 or greater for the raster exports")
    comparisons = [
        compare("LEDA 175897", args.host_observed, args.host_synthetic),
        compare(
            "2MASS J03072631-6043043", args.satellite_observed, args.satellite_synthetic
        ),
    ]
    set_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    extent = residual_extent(comparisons)
    plot_combined(
        comparisons, args.output_dir / "profile_comparison_apj", args.dpi, extent
    )
    for c, stem in zip(
        comparisons, ("host_profile_comparison", "sat_profile_comparison")
    ):
        plot_single(c, args.output_dir / stem, args.dpi, extent)
        print(
            f"{c.name}: {len(c.observed.radius)} matched annuli; "
            f"residual range {c.residual.min():+.4f} to {c.residual.max():+.4f} mag"
        )
    print(f"Shared residual range: +/-{extent:.2f} mag")
    print(f"Saved PDF and {args.dpi}-dpi PNG figures in {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
