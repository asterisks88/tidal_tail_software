#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elliptical/S0 stellar masses and Chabrier-based SHMR halo references (Astropy).

Usage: mass_from_photometry.py galaxy.json galaxy_profile.magnitudes.json [options]
The galaxy file supplies redshift and structural data; the second file is the
combined g/r Sérsic-total magnitude JSON from legacy_ellipse_profile.py.

Formatted JSON keeps masses/luminosities to 3 significant figures and log10
quantities to 6 significant figures. --json_numeric retains unrounded numbers.

The selected --ml normalization sets the modeled stellar mass. SHMR inversion
always uses its Chabrier-equivalent photometric mass (Bell diet Salpeter minus
0.10 dex; Bernardi et al. 2010, equation 5). Progenitor luminosity adjustments
and --target_ml_sigma affect the model, not the photometric SHMR reference.

UniverseMachine DR1 returns historical peak Bryan--Norman virial masses,
Mpeak, NOT M200c (Behroozi et al. 2019, Appendix J). An equivalent spherical
NFW halo is converted to M200c, solving consistently for c200. This boundary
conversion does not infer mass lost by a satellite. --halo_conversion_z can
specify a known peak epoch; otherwise the observed redshift is an explicit
reference-epoch assumption. The native Mpeak is retained in the output.

Monte Carlo propagation includes g,r,z through the correlated color,
K-corrections, luminosity, both IMF mass scales, SHMR inversion and conversion.
Separate budgets report photometric errors, added M/L systematics, and an
optional assumed halo scatter plus concentration scatter. These are inverted
relation reference intervals, not a Bayesian P(Mhalo | photometry) posterior.
Use --compare_log_m200 13.57 to check a modeled mass against each interval.

Dependencies: numpy, astropy. UMACHINE_PATH supplies the DR1 parameter files.
"""

import argparse
import ast
import copy
import json
import math
import os
from functools import lru_cache
from typing import Any, List, Optional

import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM, Planck18


# ------------------ Formatting helpers ------------------
def sci3(x: Any) -> str:
    """Return number in scientific notation with 3 significant figures, as a string."""
    try:
        return f"{float(x):.2e}"
    except Exception:
        return str(x)


def sig_noexp_str(x: Any, sig: int) -> str:
    """Format to 'sig' significant figures, NO scientific notation, as a string."""
    try:
        xf = float(x)
    except Exception:
        return str(x)
    if xf == 0 or not np.isfinite(xf):
        return "0" if xf == 0 else str(xf)
    p = sig - 1 - int(math.floor(math.log10(abs(xf))))
    val = round(xf, p)
    decimals = max(0, p)
    return f"{val:.{decimals}f}"


def sig3_noexp_str(x: Any) -> str:
    return sig_noexp_str(x, 3)


def mag5_noexp_str(x: Any) -> str:
    """Magnitudes/colors: 6 significant figures, fixed notation."""
    return sig_noexp_str(x, 6)


def key_is_msun_or_lsun(key: Optional[str]) -> bool:
    if key is None:
        return False
    kl = key.lower()
    return ("msun" in kl) or ("lsun" in kl) or ("l_sun" in kl)


def key_is_logarithmic(key: Optional[str]) -> bool:
    return key is not None and ("log10" in key.lower() or key.lower().endswith("_dex"))


_MAG_KEYS_EXPLICIT = {
    "g",
    "r",
    "u",
    "i",
    "z",
    "M_r",
    "M_r_mag",
    "M_r_input_mag",
    "M_r_photometric_mag",
    "M_g_mag",
    "M_i_mag",
    "M_z_mag",
    "M_u_mag",
    "M_r_sun_mag",
    "g0",
    "r0",
    "(g-r)_0",
    "g_minus_r_0",
    "g_minus_r_obs",
    "Kg_mag",
    "Kr_mag",
}


def key_is_magnitude_like(key: Optional[str]) -> bool:
    if key is None:
        return False
    kl = key.lower()
    return ("mag" in kl) or (key in _MAG_KEYS_EXPLICIT)


def key_is_position_like(key: Optional[str]) -> bool:
    if key is None:
        return False
    kl = key.lower()
    return kl in ["ra", "dec"]


def _effective_format_key(
    leaf_key: Optional[str], context_key: Optional[str]
) -> Optional[str]:
    """
    For nested summary dicts like:
      "M_star_Msun": {"p16": ..., "median": ...}
    use the parent/context key to decide formatting.
    """
    if leaf_key is not None:
        if (
            key_is_logarithmic(leaf_key)
            or key_is_msun_or_lsun(leaf_key)
            or key_is_magnitude_like(leaf_key)
            or key_is_position_like(leaf_key)
        ):
            return leaf_key
    return context_key if context_key is not None else leaf_key


def format_numbers_inplace(obj: Any, context_key: Optional[str] = None):
    """
    Recursively format numeric values.
    Nested uncertainty summaries inherit formatting from their parent key.
    """
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, (dict, list)):
                format_numbers_inplace(v, context_key=k)
            else:
                if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
                    use_key = _effective_format_key(k, context_key)
                    if key_is_logarithmic(use_key):
                        obj[k] = f"{float(v):.6g}"
                    elif key_is_msun_or_lsun(use_key):
                        obj[k] = sci3(v)
                    elif key_is_position_like(use_key):
                        obj[k] = str(v)
                    elif key_is_magnitude_like(use_key):
                        obj[k] = mag5_noexp_str(v)
                    else:
                        obj[k] = sig3_noexp_str(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                format_numbers_inplace(v, context_key=context_key)
            else:
                if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool):
                    use_key = context_key
                    if key_is_logarithmic(use_key):
                        obj[i] = f"{float(v):.6g}"
                    elif key_is_msun_or_lsun(use_key):
                        obj[i] = sci3(v)
                    elif key_is_position_like(use_key):
                        obj[i] = str(v)
                    elif key_is_magnitude_like(use_key):
                        obj[i] = mag5_noexp_str(v)
                    else:
                        obj[i] = sig3_noexp_str(v)


def finite_json_values(obj):
    """Represent absent/undefined quantities as null, never nonstandard JSON NaN."""
    if isinstance(obj, dict):
        return {key: finite_json_values(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [finite_json_values(value) for value in obj]
    if isinstance(obj, (float, np.floating)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# ------------------ Chilingarian/Saulder K-corrections ------------------
# From Saulder C., Mieske S., Zeilinger W. W., Chilingarian I., 2013, A&A, 557, A21
B_g: List[List[float]] = [
    [0.0, 0.0, 0.0, 0.0],
    [-2.45204, 4.10188, 10.5258, -13.5889],
    [56.7969, -140.913, 144.572, 57.2155],
    [-466.949, 222.789, -917.46, -78.0591],
    [2906.77, 1500.8, 1689.97, 30.889],
    [-10453.7, -4419.56, -1011.01, 0.0],
    [17568.0, 3236.68, 0.0, 0.0],
    [-10820.7, 0.0, 0.0, 0.0],
]
B_r: List[List[float]] = [
    [0.0, 0.0, 0.0, 0.0],
    [1.83285, -2.71446, 4.97336, -3.66864],
    [-19.7595, 10.5033, 18.8196, 6.07785],
    [33.6059, -120.713, -49.299, 0.0],
    [144.371, 216.453, 0.0, 0.0],
    [-295.39, 0.0, 0.0, 0.0],
]


def kpoly(z: float, color: float, coeffs: List[List[float]]) -> float:
    z_p = 1.0
    total = 0.0
    for row in coeffs:
        c0 = 1.0
        for Bij in row:
            total += Bij * z_p * c0
            c0 *= color
        z_p *= z
    return total


# ------------------ Color–M/L relations ------------------
# All of these return Y_r, the r-band M/L from (g-r)


# From Zibetti S., Charlot S., Rix H.-W., 2009, MNRAS, 400, 1181
def mlr_zibetti2009_gr(color0: float) -> float:
    return 10.0 ** (-0.840 + 1.654 * color0)


# Bell E. F., McIntosh D. H., Katz N., Weinberg M. D., 2003, ApJS, 149, 289
def mlr_bell2003_diet_gr(color0: float) -> float:
    return 10.0 ** (-0.306 + 1.097 * color0)


# Bell adjusted by 0.1 dex for closer match to Chabrier
def mlr_bell2003_chabrier_gr(color0: float) -> float:
    return mlr_bell2003_diet_gr(color0) / (10.0**0.10)


def default_ml_systematic_dex(ml_mode: str) -> float:
    """
    Conservative defaults:
      - zcr09: ~30% => log10(1.30) = 0.11394 dex
      - Bell03 variants: 0.10 dex
    """
    if ml_mode == "zcr09":
        return math.log10(1.30)
    return 0.10


def compute_restframe_quantities(
    g,
    r,
    z,
    ml_mode: str,
    M_r_sun: float,
    cosmo,
):
    """
    Vectorized helper:
      g,r,z -> K_g,K_r -> (g-r)_0
      M_r -> L_r -> M/L_r -> M_star

    M_r is computed from r, K_r, and the distance modulus.
    """
    g = np.asarray(g, dtype=float)
    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)

    gmr_obs = g - r
    Kg = kpoly(z, gmr_obs, B_g)
    Kr = kpoly(z, gmr_obs, B_r)

    g0 = g - Kg
    r0 = r - Kr
    gmr0 = g0 - r0
    mu = cosmo.distmod(z).value
    M_r = r0 - mu

    Lr_Lsun = 10.0 ** (-0.4 * (M_r - M_r_sun))

    if ml_mode == "zcr09":
        ML = mlr_zibetti2009_gr(gmr0)
    elif ml_mode == "bell03_chabrier":
        ML = mlr_bell2003_chabrier_gr(gmr0)
    else:
        ML = mlr_bell2003_diet_gr(gmr0)

    Mstar = ML * Lr_Lsun

    return {
        "gmr_obs": gmr_obs,
        "Kg": Kg,
        "Kr": Kr,
        "gmr0": gmr0,
        "M_r": M_r,
        "Lr_Lsun": Lr_Lsun,
        "ML": ML,
        "Mstar": Mstar,
    }


def summarize_samples(samples, positive_only: bool = False) -> Optional[dict]:
    x = np.asarray(samples, dtype=float)
    good = np.isfinite(x)
    if positive_only:
        good &= x > 0
    x = x[good]
    if x.size == 0:
        return None

    p025, p16, p50, p84, p975 = np.percentile(x, [2.5, 16, 50, 84, 97.5])
    out = {
        "p2_5": float(p025),
        "p16": float(p16),
        "median": float(p50),
        "p84": float(p84),
        "p97_5": float(p975),
        "err_minus": float(p50 - p16),
        "err_plus": float(p84 - p50),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "nsamp_used": int(x.size),
    }

    if positive_only:
        lx = np.log10(x)
        l025, l16, l50, l84, l975 = np.percentile(lx, [2.5, 16, 50, 84, 97.5])
        out["sigma_dex"] = float(np.std(lx, ddof=1)) if x.size > 1 else 0.0
        out.update(
            {
                "log10_p2_5": float(l025),
                "log10_p16": float(l16),
                "log10_median": float(l50),
                "log10_p84": float(l84),
                "log10_p97_5": float(l975),
                "log10_err_minus": float(l50 - l16),
                "log10_err_plus": float(l84 - l50),
            }
        )

    return out


def estimate_photometric_uncertainties_mc(
    g: float,
    r: float,
    z: float,
    ml_mode: str,
    M_r_sun: float,
    cosmo,
    g_err: Optional[float] = None,
    r_err: Optional[float] = None,
    z_err: Optional[float] = None,
    nsamp: int = 5000,
    systematic_dex: float = 0.0,
    seed: Optional[int] = None,
    gr_correlation: float = 0.0,
    photometric_mass_scale: float = 1.0,
    model_bulge_scale: float = 1.0,
    model_disk_scale: float = 0.0,
    model_ml_scale: float = 1.0,
    chabrier_offset_dex: float = 0.0,
    return_samples: bool = False,
) -> dict:
    """
    Propagate g,r,z uncertainties through:
      g,r,z -> K_g,K_r -> (g-r)_0 -> M_r -> L_r -> M/L_r -> M_star

    Returns:
      - photometric_only
      - with_systematic   (if systematic_dex > 0)
    """
    rng = np.random.default_rng(seed)
    nsamp = max(100, int(nsamp))

    g_err = 0.0 if g_err is None else max(0.0, float(g_err))
    r_err = 0.0 if r_err is None else max(0.0, float(r_err))
    z_err = 0.0 if z_err is None else max(0.0, float(z_err))

    # The same draws feed color and luminosity; their covariance is retained.
    if not np.isfinite(gr_correlation) or abs(gr_correlation) > 1:
        raise ValueError("g/r error correlation must lie in [-1, 1].")
    ng, nr = rng.normal(size=(2, nsamp))
    gs = g + g_err * ng
    rs = r + r_err * (gr_correlation * ng + np.sqrt(1 - gr_correlation**2) * nr)
    zs = rng.normal(z, z_err, nsamp) if z_err > 0 else np.full(nsamp, z, dtype=float)

    # Truncated normal, rather than clipping negative draws to an artificial peak.
    while np.any(zs <= 0):
        bad = zs <= 0
        zs[bad] = rng.normal(z, z_err, int(np.sum(bad)))
    base = compute_restframe_quantities(
        gs,
        rs,
        zs,
        ml_mode,
        M_r_sun,
        cosmo,
    )

    def budget(sys_factor):
        model_base = base["Mstar"] * sys_factor * model_ml_scale
        model_total = model_base * (model_bulge_scale + model_disk_scale)
        phot_total = base["Mstar"] * sys_factor * photometric_mass_scale
        chabrier = phot_total * 10.0 ** (-chabrier_offset_dex)
        summary = {
            "g_minus_r_0": summarize_samples(base["gmr0"]),
            "M_r_mag": summarize_samples(base["M_r"]),
            "L_r_Lsun": summarize_samples(base["Lr_Lsun"], positive_only=True),
            "M_over_L_r": summarize_samples(
                base["ML"] * sys_factor * model_ml_scale, True
            ),
            "M_star_Msun": summarize_samples(model_total, True),
            "M_star_Msun_with_missing": summarize_samples(model_total, True),
            "M_bulge_Msun": summarize_samples(
                model_base * model_bulge_scale, positive_only=model_bulge_scale > 0
            ),
            "M_disk_Msun": summarize_samples(
                model_base * model_disk_scale, positive_only=model_disk_scale > 0
            ),
            "M_star_photometric_Msun": summarize_samples(phot_total, True),
            "M_star_Chabrier_for_SHMR_Msun": summarize_samples(chabrier, True),
        }
        return summary, chabrier

    phot, chabrier_phot = budget(np.ones(nsamp))

    out = {
        "method": "monte_carlo",
        "inputs": {
            "g_err_mag": g_err,
            "r_err_mag": r_err,
            "z_err": z_err,
            "systematic_dex": float(systematic_dex),
            "nsamp": nsamp,
            "seed": seed,
            "g_r_error_correlation": gr_correlation,
            "photometric_mass_scale": photometric_mass_scale,
            "model_bulge_scale": model_bulge_scale,
            "model_disk_scale": model_disk_scale,
            "model_ml_scale": model_ml_scale,
            "diet_to_chabrier_offset_dex": chabrier_offset_dex,
        },
        "photometric_only": phot,
    }

    sys_factor = 10.0 ** rng.normal(0.0, systematic_dex, nsamp)
    with_sys, chabrier_sys = budget(sys_factor)
    if systematic_dex > 0:
        out["with_systematic"] = with_sys
    if return_samples:
        out["_samples"] = {
            "z": zs,
            "Mstar_chabrier_photometric": chabrier_phot,
            "Mstar_chabrier_with_systematic": chabrier_sys,
        }

    return out


# ------------------ Moster+2013 SHMR and inversion ------------------
def moster2013_params_z0():
    return dict(N=0.0351, log10M1=11.59, beta=1.376, gamma=0.608)


def mstar_from_mhalo_moster13(
    Mh: float, N: float, M1: float, beta: float, gamma: float
) -> float:
    x = Mh / M1
    return Mh * (2.0 * N) / (x ** (-beta) + x ** (gamma))


def invert_shmr_moster13(Mstar: float) -> float:
    p = moster2013_params_z0()
    N, M1, beta, gamma = p["N"], 10.0 ** p["log10M1"], p["beta"], p["gamma"]
    stars = np.asarray(Mstar, dtype=float)
    if np.any(~np.isfinite(stars) | (stars <= 0)):
        raise ValueError("Moster inversion requires positive finite stellar masses.")
    lo, hi = np.full(stars.shape, 8.0), np.full(stars.shape, 18.0)
    if np.any(
        (stars < mstar_from_mhalo_moster13(10.0**lo, N, M1, beta, gamma))
        | (stars > mstar_from_mhalo_moster13(10.0**hi, N, M1, beta, gamma))
    ):
        raise ValueError("Stellar mass exceeds the Moster inversion bracket.")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        Mh = 10.0**mid
        below = mstar_from_mhalo_moster13(Mh, N, M1, beta, gamma) < stars
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    result = 10.0 ** (0.5 * (lo + hi))
    return float(result) if result.ndim == 0 else result


# Behroozi et al. 2019, MNRAS, 488, 3143, Appendix J, equations J1--J8.
# The abscissa is historical peak Mvir in Msun, not M200c or Msun/h.
UM_PARAMETER_NAMES = (
    "EFF_0 EFF_0_A EFF_0_A2 EFF_0_Z M_1 M_1_A M_1_A2 M_1_Z "
    "ALPHA ALPHA_A ALPHA_A2 ALPHA_Z BETA BETA_A BETA_Z DELTA "
    "GAMMA GAMMA_A GAMMA_Z CHI2"
).split()
UM_COSMOLOGY = FlatLambdaCDM(H0=67.8, Om0=0.307)
MOSTER_COSMOLOGY = FlatLambdaCDM(H0=70.4, Om0=0.272)


def um_parameter_path(galaxy_type: str, root: Optional[str] = None) -> str:
    root = root or os.environ.get("UMACHINE_PATH", os.path.expanduser("~/umachine-dr1"))
    filename = {
        "all": "smhm_med_params.txt",
        "host": "smhm_med_cen_params.txt",
        "sat": "smhm_med_sat_params.txt",
    }[galaxy_type]
    return os.path.join(os.path.expanduser(root), "data", "smhm", "params", filename)


@lru_cache(maxsize=8)
def read_um_parameters(path: str) -> dict:
    """Read the 19 fit coefficients and an optional trailing CHI2 diagnostic.

    Accept named or legacy positional files. Missing named fit coefficients
    are errors: an omitted coefficient must never shift subsequent values.
    CHI2 is not used in evaluating or inverting the stellar--halo relation.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            rows = [line.split("#", 1)[0].split() for line in handle]
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"UniverseMachine parameter file not found: {path}. Set UMACHINE_PATH "
            "or --umachine_path to the DR1 root containing data/smhm/params/."
        ) from exc
    rows = [row for row in rows if row]
    required_names = UM_PARAMETER_NAMES[:-1]  # CHI2 is an optional diagnostic.
    if len(rows) not in (len(required_names), len(UM_PARAMETER_NAMES)):
        raise ValueError(
            f"Expected 19 UniverseMachine fit-coefficient rows, optionally followed "
            f"by CHI2 (20 rows), in {path}; got {len(rows)}."
        )
    labels = [row[0].upper().rstrip(":=") if len(row) >= 2 else None for row in rows]
    values = []
    for index, row in enumerate(rows, start=1):
        token = row[1] if len(row) >= 2 else row[0]
        try:
            values.append(float(token.replace("D", "E").replace("d", "e")))
        except ValueError as exc:
            raise ValueError(
                f"Invalid numeric value in UniverseMachine data row {index} of {path}: {token!r}"
            ) from exc
    if any(label in UM_PARAMETER_NAMES for label in labels):
        named = {}
        for label, value in zip(labels, values):
            if label in named:
                raise ValueError(
                    f"Duplicate UniverseMachine parameter {label!r} in {path}."
                )
            named[label] = value
        missing = [name for name in required_names if name not in named]
        if missing:
            raise ValueError(
                f"Missing required UniverseMachine coefficients in {path}: {', '.join(missing)}. Only CHI2 is optional."
            )
        params = {name: named[name] for name in UM_PARAMETER_NAMES if name in named}
    else:
        # Preserve compatibility with the positional labels used by DR1 files.
        params = dict(zip(UM_PARAMETER_NAMES, values))
    if (
        not all(np.isfinite(params[name]) for name in required_names)
        or params["DELTA"] <= 0
    ):
        raise ValueError(f"Invalid UniverseMachine parameters in {path}.")
    return params


def um_log_mstar(log_mpeak, z, params: dict):
    """Vectorized median log10 Chabrier stellar mass at historical log10 Mpeak."""
    z = np.asarray(z, dtype=float)
    a = 1.0 / (1.0 + z)
    a1, lna = a - 1.0, np.log(a)
    m1 = (
        params["M_1"]
        + a1 * params["M_1_A"]
        - lna * params["M_1_A2"]
        + z * params["M_1_Z"]
    )
    epsilon = (
        params["EFF_0"]
        + a1 * params["EFF_0_A"]
        - lna * params["EFF_0_A2"]
        + z * params["EFF_0_Z"]
    )
    alpha = (
        params["ALPHA"]
        + a1 * params["ALPHA_A"]
        - lna * params["ALPHA_A2"]
        + z * params["ALPHA_Z"]
    )
    beta = params["BETA"] + a1 * params["BETA_A"] + z * params["BETA_Z"]
    gamma = 10.0 ** (params["GAMMA"] + a1 * params["GAMMA_A"] + z * params["GAMMA_Z"])
    dm = np.asarray(log_mpeak, dtype=float) - m1
    log_sum = np.logaddexp(
        -alpha * dm * np.log(10.0), -beta * dm * np.log(10.0)
    ) / np.log(10.0)
    return m1 + epsilon - log_sum + gamma * np.exp(-0.5 * (dm / params["DELTA"]) ** 2)


def load_umachine_smhm(z: float, galaxy_type: str) -> list[list]:
    """Compatibility table: [log Mstar, log Mpeak, log(Mstar/Mpeak)]."""
    params = read_um_parameters(um_parameter_path(galaxy_type))
    mh = np.arange(10.5, 14.5 - 0.35 * z + 0.025, 0.05)
    sm = um_log_mstar(mh, z, params)
    return np.column_stack((sm, mh, sm - mh)).tolist()


def _interp_loglog(x, xy):
    xs = [row[0] for row in xy]
    ys = [row[1] for row in xy]
    lx = np.log10(x)
    if lx <= xs[0]:
        m = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return 10 ** (ys[0] + m * (lx - xs[0]))
    if lx >= xs[-1]:
        m = (ys[-1] - ys[-2]) / (xs[-1] - xs[-2])
        return 10 ** (ys[-1] + m * (lx - xs[-1]))
    return 10 ** (np.interp(lx, xs, ys))


def mpeak_um_median(Mstar, z, galaxy_type: str, params: Optional[dict] = None):
    """Invert the forward median relation; this is not a conditional posterior."""
    if params is None:
        params = read_um_parameters(um_parameter_path(galaxy_type))
    mass, redshift = np.broadcast_arrays(np.asarray(Mstar, float), np.asarray(z, float))
    if np.any(
        ~np.isfinite(mass) | (mass <= 0) | ~np.isfinite(redshift) | (redshift < 0)
    ):
        raise ValueError(
            "UM inversion requires positive finite stellar masses and z >= 0."
        )
    target = np.log10(mass)
    lo, hi = np.full(mass.shape, 8.0), np.full(mass.shape, 18.0)
    if np.any(
        (target < um_log_mstar(lo, redshift, params))
        | (target > um_log_mstar(hi, redshift, params))
    ):
        raise ValueError(
            "Stellar-mass draws exceed the UM inversion bracket (10^8--10^18 Msun)."
        )
    for _ in range(60):
        mid = (lo + hi) / 2.0
        below = um_log_mstar(mid, redshift, params) < target
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    result = 10.0 ** ((lo + hi) / 2.0)
    return float(result) if result.ndim == 0 else result


def m200_um_median(
    Mstar: float,
    z: float,
    galaxy_type: str,
    cosmo=Planck18,
    c_input: Optional[float] = None,
) -> float:
    """M200c equivalent of Mpeak, assuming the halo is evaluated at redshift z.

    Mstar MUST already be on the Chabrier scale. No stripping correction is made.
    """
    native = mpeak_um_median(Mstar, z, galaxy_type)
    converted, _ = virial_to_m200c_nfw(native, z, cosmo, c_input)
    return float(converted)


# ------------------ (R200, M200) helpers ------------------
# From Mo H. J., Mao S., White S. D. M., 1998, MNRAS, 295, 319
def r200_from_m200(M200: float, z: float, cosmo) -> u.Quantity:
    rho_c = cosmo.critical_density(z).to(u.Msun / u.Mpc**3).value
    return ((3.0 * M200) / (4.0 * np.pi * 200.0 * rho_c)) ** (1.0 / 3.0) * u.Mpc


def v200_from_m200(M200: float, z: float, cosmo) -> u.Quantity:
    R200 = r200_from_m200(M200, z, cosmo)
    return (10.0 * cosmo.H(z) * R200).to(u.km / u.s)


# ------------------ NFW mass & concentration; Re-based fit ------------------
def c200_from_m200_dutton2014(M200: float, h: float) -> float:
    # log10 c200 = a + b * log10( M200 / (1e12 h^-1 Msun) )
    return 10.0 ** (0.905 + -0.101 * math.log10(M200 * h / 1.0e12))


def c200_from_m200_wang2022(M200: float, h: float) -> float:
    # Retained legacy function name. Gu et al. 2022, RAA 22, 125015, equation 5:
    # log10 c200 = 2.161 - 0.105 log10(M200 / (h^-1 Msun)), z~0.
    return 10.0 ** (2.161 - 0.105 * np.log10(np.asarray(M200) * h))


def bryan_norman_delta_critical(z, cosmo):
    """Flat-LambdaCDM virial overdensity relative to critical density (BN98 eq. 6)."""
    x = np.asarray(cosmo.Om(z)) - 1.0
    return 18.0 * np.pi**2 + 82.0 * x - 39.0 * x**2


def nfw_f(c):
    c = np.asarray(c, dtype=float)
    return np.log1p(c) - c / (1.0 + c)


def virial_to_m200c_nfw(
    Mvir,
    z,
    cosmo,
    c_input=None,
    concentration_factor=1.0,
    source_cosmo=UM_COSMOLOGY,
    source_delta=None,
):
    """Preserve one spherical NFW density profile while changing its boundary.

    Source Mvir uses BN98 in the native SHMR cosmology; target M200c uses
    the user's cosmology. c_input is c200c, NOT cvir. Otherwise the retained
    z~0 concentration--mass law is evaluated self-consistently at target M200c.
    No historical-to-current mass evolution or tidal stripping is inferred.
    """
    mass, redshift, factor = np.broadcast_arrays(
        np.asarray(Mvir, float),
        np.asarray(z, float),
        np.asarray(concentration_factor, float),
    )
    if np.any(
        ~np.isfinite(mass)
        | (mass <= 0)
        | ~np.isfinite(redshift)
        | (redshift < 0)
        | ~np.isfinite(factor)
        | (factor <= 0)
    ):
        raise ValueError(
            "NFW conversion needs finite positive masses/factors and z >= 0."
        )
    if c_input is not None and (not np.isfinite(c_input) or c_input <= 0):
        raise ValueError("c200 override must be finite and positive.")
    rho_source = source_cosmo.critical_density(redshift).to_value(u.Msun / u.kpc**3)
    rho_target = cosmo.critical_density(redshift).to_value(u.Msun / u.kpc**3)
    delta_source = (
        bryan_norman_delta_critical(redshift, source_cosmo)
        if source_delta is None
        else np.full_like(redshift, source_delta)
    )
    delta_target = delta_source * rho_source / rho_target
    if np.any(delta_target <= 0):
        raise ValueError("Nonpositive virial density threshold.")
    log_native = np.log10(mass)
    lo, hi = log_native - 3.0, log_native + 3.0
    h = cosmo.H0.value / 100.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        trial = 10.0**mid
        c200 = (
            c200_from_m200_wang2022(trial, h) if c_input is None else c_input
        ) * factor
        cvir = c200 * (200.0 * mass / (delta_target * trial)) ** (1.0 / 3.0)
        predicted_log_vir = mid + np.log10(nfw_f(cvir) / nfw_f(c200))
        below = predicted_log_vir < log_native
        lo, hi = np.where(below, mid, lo), np.where(below, hi, mid)
    result = 10.0 ** ((lo + hi) / 2.0)
    c200 = (c200_from_m200_wang2022(result, h) if c_input is None else c_input) * factor
    cvir = c200 * (200.0 * mass / (delta_target * result)) ** (1.0 / 3.0)
    r200 = (3.0 * result / (4.0 * np.pi * 200.0 * rho_target)) ** (1.0 / 3.0)
    rvir = (3.0 * mass / (4.0 * np.pi * delta_source * rho_source)) ** (1.0 / 3.0)
    metadata = {
        "Delta_vir_critical_native": delta_source,
        "virial_density_over_target_critical": delta_target,
        "M200c_over_Mvir": result / mass,
        "c200": c200,
        "cvir": cvir,
        "R200_kpc": r200,
        "Rvir_kpc": rvir,
        "r_s_kpc": r200 / c200,
    }
    metadata = {
        key: float(value) if np.ndim(value) == 0 else value
        for key, value in metadata.items()
    }
    return (float(result) if result.ndim == 0 else result), metadata


# ------------------ ICL prior ------------------
def icl_prior_from_m200(
    M200: float,
    anchor_mass: float = 1.0e14,
    at_anchor: float = 0.20,
    slope_per_dex: float = -0.08,
    fmin: float = 0.05,
    fmax: float = 0.35,
) -> float:
    frac = at_anchor + slope_per_dex * math.log10(max(M200, 1e10) / anchor_mass)
    return float(min(max(frac, fmin), fmax))


# ------------------ Halo/Bulge Hernquist scales ------------------
def a_halo_from_c_rs(rs: u.Quantity, c: float) -> u.Quantity:
    if c <= 0:
        return np.nan * u.kpc
    A = np.log(1.0 + c) - c / (1.0 + c)
    return (rs * np.sqrt(2.0 * A)).to(u.kpc)


def hernquist_a_from_Re(Re_kpc: float) -> float:
    if not np.isfinite(Re_kpc) or Re_kpc <= 0:
        return float("nan")
    return Re_kpc / 1.8153


def halo_structure_from_M200(
    M200: float, z: float, cosmo, c_input: Optional[float]
) -> dict:
    R200 = r200_from_m200(M200, z, cosmo)
    R200_kpc = R200.to_value(u.kpc)
    H0 = cosmo.H0.value
    h = H0 / 100.0
    if (c_input is not None) and (c_input > 0):
        c_used = float(c_input)
    else:
        c_used = float(c200_from_m200_wang2022(M200, h))
    rs = (R200 / c_used).to(u.kpc)
    a_halo = a_halo_from_c_rs(rs, c_used)
    return {
        "R200_kpc": R200_kpc,
        "c_used": c_used,
        "r_s_kpc": rs.to_value(u.kpc),
        "a_halo_kpc": a_halo.to_value(u.kpc),
    }


def estimate_halo_reference_mc(
    samples: dict,
    cosmo,
    galaxy_type: str,
    c_input=None,
    params=None,
    shmr_model: str = "umachine",
    icl_fraction: float = 0.0,
    halo_scatter_dex: float = 0.25,
    c_scatter_dex: float = 0.10,
    native_shift_dex: float = 0.0,
    conversion_z=None,
    seed: int = 12346,
    model_mass_per_chabrier_mass: float = 1.0,
) -> dict:
    """Propagate shared draws all the way to native and M200c reference masses.

    Halo scatter is an explicitly assumed lognormal prior at fixed stellar
    mass, applied to native halo mass. It is not UM's scatter in Mstar at fixed
    Mpeak and does not implement a Bayesian inversion or a halo mass function.
    """
    zs = samples["z"]
    zconvert = zs if conversion_z is None else np.full_like(zs, conversion_z)
    rng = np.random.default_rng(seed)
    source_cosmo = UM_COSMOLOGY if shmr_model == "umachine" else MOSTER_COSMOLOGY
    native = []
    for key in ("Mstar_chabrier_photometric", "Mstar_chabrier_with_systematic"):
        stars = samples[key] * (1.0 + icl_fraction)
        mass = (
            mpeak_um_median(stars, zs, galaxy_type, params)
            if shmr_model == "umachine"
            else invert_shmr_moster13(stars)
        )
        native.append(mass * 10.0**native_shift_dex)
    c_factor = 10.0 ** rng.normal(0.0, c_scatter_dex, len(zs))
    halo_factor = 10.0 ** rng.normal(0.0, halo_scatter_dex, len(zs))
    stage_specs = [
        ("photometric_only", native[0], 1.0, "Mstar_chabrier_photometric"),
        ("with_ml_systematic", native[1], 1.0, "Mstar_chabrier_with_systematic"),
        (
            "with_ml_systematic_and_concentration",
            native[1],
            c_factor,
            "Mstar_chabrier_with_systematic",
        ),
        (
            "with_ml_systematic_and_concentration_and_halo_scatter",
            native[1] * halo_factor,
            c_factor,
            "Mstar_chabrier_with_systematic",
        ),
    ]
    budgets, mass_draws = {}, {}
    for name, native_draws, concentration_factor, stellar_key in stage_specs:
        m200, converted = virial_to_m200c_nfw(
            native_draws,
            zconvert,
            cosmo,
            c_input,
            concentration_factor=concentration_factor,
            source_cosmo=(source_cosmo if shmr_model == "umachine" else cosmo),
            source_delta=(None if shmr_model == "umachine" else 200.0),
        )
        r200 = converted["R200_kpc"]
        v200 = (10.0 * cosmo.H(zconvert) * r200 * u.kpc).to_value(u.km / u.s)
        model_stars = samples[stellar_key] * model_mass_per_chabrier_mass
        budgets[name] = {
            "native_halo_mass_Msun": summarize_samples(native_draws, True),
            "M200c_Msun": summarize_samples(m200, True),
            "V200_kms": summarize_samples(v200, True),
            "R200_kpc": summarize_samples(r200, True),
            "c200": summarize_samples(converted["c200"], True),
            "Mstar_model_over_M200c": summarize_samples(model_stars / m200, True),
        }
        mass_draws[name] = m200
    base_logs = np.log10(np.concatenate(native)) - native_shift_dex
    outside_fit = (
        float(np.mean((base_logs < 10.5) | (base_logs > 15.0)))
        if shmr_model == "umachine"
        else None
    )
    return {
        "method": "shared_draw_monte_carlo_inversion_of_forward_median_relation",
        "interpretation": "Reference intervals, not a Bayesian conditional halo-mass posterior.",
        "assumed_halo_scatter_dex_at_fixed_stellar_mass": halo_scatter_dex,
        "concentration_scatter_dex": c_scatter_dex,
        "uncertainties_not_included": [
            "UM fit-parameter covariance and cosmological abundance recalibration",
            "Unspecified photometric systematics, D/T and progenitor adjustment errors",
            "IMF-conversion uncertainty, halo profile/shape uncertainty and satellite stripping",
        ],
        "stellar_driven_fraction_outside_log10_native_mass_10_5_to_15": outside_fit,
        "budgets": budgets,
        "_mass_draws": mass_draws,
    }


def compare_model_mass(log_mass: float, mass_draws: dict) -> dict:
    """Report interval membership without changing or fitting the uncertainty budget."""
    model_mass = 10.0**log_mass
    intervals = {}
    for name, draws in mass_draws.items():
        summary = summarize_samples(draws, True)
        percentile = 100.0 * (
            np.mean(draws < model_mass) + 0.5 * np.mean(draws == model_mass)
        )
        intervals[name] = {
            "within_68_percent": bool(summary["p16"] <= model_mass <= summary["p84"]),
            "within_95_percent": bool(
                summary["p2_5"] <= model_mass <= summary["p97_5"]
            ),
            "reference_sample_percentile": float(percentile),
            "offset_from_median_dex": float(log_mass - summary["log10_median"]),
        }
    return {
        "model_log10_M200c_Msun": log_mass,
        "model_M200c_Msun": model_mass,
        "interval_checks": intervals,
        "interpretation": "Membership in an assumed reference distribution; not a formal significance.",
    }


# ------------------ Sérsic bulge & exponential disk fractions ------------------
def b_n_of_n(n: float) -> float:
    n = max(0.36, float(n))
    return 2.0 * n - (1.0 / 3.0) + (0.009876 / n)


def sersic_frac_enclosed(R_over_Re: float, n: float) -> float:
    if R_over_Re <= 0:
        return 0.0
    bn = b_n_of_n(n)
    x = bn * (R_over_Re ** (1.0 / max(1e-6, n)))
    a = max(1, int(round(2.0 * n)))
    s = 0.0
    term = 1.0
    for k in range(a):
        if k == 0:
            term = 1.0
        else:
            term *= x / k
        s += term
    P = 1.0 - math.exp(-x) * s
    return float(min(max(P, 0.0), 1.0))


def expdisk_frac_enclosed(R_over_h: float) -> float:
    if R_over_h <= 0:
        return 0.0
    x = float(R_over_h)
    return float(1.0 - (1.0 + x) * math.exp(-x))


# ------------------ I/O ------------------
def load_input_dict(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read().strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        result = ast.literal_eval(text)
    if not isinstance(result, dict):
        raise ValueError(f"Input must contain a dictionary: {path}")
    return result


def load_profile_magnitudes(path: str) -> dict:
    """Read the compact g/r JSON written by legacy_ellipse_profile.py."""
    with open(path, "r", encoding="utf-8") as f:
        values = json.load(f)
    if not isinstance(values, dict):
        raise ValueError(f"Magnitude JSON must contain an object: {path}")
    for band in ("g", "r"):
        value = as_float_or_none(values.get(band))
        if value is None:
            raise ValueError(
                f"Magnitude JSON needs a finite '{band}' value; run both "
                f"g and r profiles into the same magnitude file: {path}"
            )
        values[band] = value
        error_key = f"{band}_err"
        error = values.get(error_key)
        if error is not None:
            error = as_float_or_none(error)
            if error is None or error < 0:
                raise ValueError(f"Magnitude JSON has invalid '{error_key}': {path}")
        values[error_key] = error
    radius = values.get("r_reff_arcsec")
    if radius is not None:
        radius = as_float_or_none(radius)
        if radius is None or radius <= 0:
            raise ValueError(f"Magnitude JSON has invalid 'r_reff_arcsec': {path}")
    values["r_reff_arcsec"] = radius
    return values


def as_float_or_none(x):
    if x is None:
        return None
    try:
        xf = float(x)
    except Exception:
        return None
    return xf if np.isfinite(xf) else None


def choose_redshift_from_input(d: dict):
    """
    Prefer spectroscopic redshift for all distance-dependent quantities.

    Supported keys, in priority order:
      zspec, z_spec, z_spectroscopic, z

    Liao photometric redshift is retained as provenance/diagnostic only.
    """
    priority = [
        ("zspec", "spectroscopic redshift from input key 'zspec'"),
        ("z_spec", "spectroscopic redshift from input key 'z_spec'"),
        ("z_spectroscopic", "spectroscopic redshift from input key 'z_spectroscopic'"),
        ("z", "redshift from input key 'z'"),
        ("z_phot_liao", "Liao+2023 photometric redshift"),
    ]

    for key, desc in priority:
        zval = as_float_or_none(d.get(key, None))
        if zval is not None and zval > 0:
            return zval, key, desc

    raise KeyError(
        "No usable redshift found. Provide one of: zspec, z_spec, "
        "z_spectroscopic, z, or z_phot_liao."
    )


def choose_redshift_error_from_input(cli_value, d: dict, z_key_used: str):
    """
    Pick the uncertainty corresponding to the redshift actually used.
    CLI --z_err overrides everything.
    """
    if cli_value is not None:
        return float(cli_value)

    if z_key_used == "zspec":
        keys = ["zspec_err", "z_spec_err", "z_err"]
    elif z_key_used == "z_spec":
        keys = ["z_spec_err", "zspec_err", "z_err"]
    elif z_key_used == "z_spectroscopic":
        keys = ["z_spectroscopic_err", "zspec_err", "z_spec_err", "z_err"]
    elif z_key_used == "z_phot_liao":
        keys = ["z_phot_liao_err", "z_err"]
    else:
        keys = ["z_err", "zspec_err", "z_spec_err", "z_phot_liao_err"]

    for key in keys:
        val = as_float_or_none(d.get(key, None))
        if val is not None and val >= 0:
            return val

    return None


def liao_redshift_diagnostic(d: dict, z_used: float):
    """
    Return a simple diagnostic comparing the adopted redshift to the Liao photo-z.
    """
    z_phot = as_float_or_none(d.get("z_phot_liao", None))
    z_phot_err = as_float_or_none(d.get("z_phot_liao_err", None))

    if z_phot is None:
        return None

    out = {
        "z_phot_liao": z_phot,
        "z_phot_liao_err": z_phot_err,
        "z_used_minus_z_phot_liao": z_used - z_phot,
    }

    if z_phot_err is not None and z_phot_err > 0:
        nsig = abs(z_used - z_phot) / z_phot_err
        out["difference_in_liao_sigma"] = nsig
        if nsig > 5:
            out["warning"] = (
                "Adopted redshift differs from Liao photo-z by >5 sigma. "
                "Check whether zspec has a missing digit or decimal-place error."
            )

    return out


# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser(
        description="Keep the selected model IMF; invert Chabrier stellar mass, convert native virial/peak mass to M200c, and propagate uncertainties.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "input_file", type=str, help="Galaxy data JSON/dict (without magnitudes)"
    )
    ap.add_argument(
        "magnitudes_file",
        type=str,
        help="Combined g/r .magnitudes.json from legacy_ellipse_profile.py",
    )

    # Photometry/IMF
    ap.add_argument(
        "--ml",
        type=str,
        default="zcr09",
        choices=["zcr09", "bell03_diet", "bell03_chabrier"],
        help="Color–M/L relation",
    )
    ap.add_argument("--M_r_sun", type=float, default=4.64, help="Solar |M_r| (AB)")
    ap.add_argument(
        "--photometry_component",
        choices=["bulge", "total"],
        default=None,
        help="What the supplied magnitudes measure; use 'total' for whole-galaxy photometry. Otherwise read the galaxy JSON photometry_component, falling back to 'bulge' for legacy compatibility.",
    )
    ap.add_argument(
        "--gr_correlation",
        type=float,
        default=None,
        help="Correlation of g/r magnitude errors, from -1 to 1; otherwise use magnitude JSON g_r_corr, or 0.",
    )
    # Cosmology
    ap.add_argument(
        "--planck18", action="store_true", help="Use astropy.cosmology.Planck18"
    )
    ap.add_argument(
        "--H0", type=float, default=None, help="H0 [km/s/Mpc]; ignored if --planck18"
    )
    ap.add_argument(
        "--Om0", type=float, default=None, help="Omega_m; ignored if --planck18"
    )

    # Output formatting
    ap.add_argument("--json", action="store_true", help="Print JSON payload")
    ap.add_argument(
        "--json_numeric",
        action="store_true",
        help="Print JSON with unrounded numeric values rather than formatted strings.",
    )

    # SHMR + ICL prior
    ap.add_argument(
        "--shmr_model",
        type=str,
        default="umachine",
        choices=["moster13", "umachine"],
        help="SHMR for halo mass from M*",
    )
    ap.add_argument(
        "--icl_frac",
        type=float,
        default=0.0,
        help="Manual ICL fraction added to M* for SHMR",
    )
    ap.add_argument(
        "--gal_type",
        type=str,
        default="all",
        choices=["host", "sat", "all"],
        help="Type of galaxy, a host, satellite, or generic",
    )
    ap.add_argument(
        "--shmr_shift_dex",
        type=float,
        default=0.0,
        help="Explicit shift of native log10 halo mass before conversion (UM: Mpeak; Moster: M200c); 0 gives the unshifted reference.",
    )
    ap.add_argument(
        "--um_scatter_dex",
        type=float,
        default=0.25,
        help="Assumed log10 Mpeak scatter at fixed stellar mass, added in the final reference budget. This is not UM's measured stellar-mass scatter.",
    )
    ap.add_argument(
        "--c200",
        type=float,
        default=None,
        help="CDM halo concentration override",
    )
    ap.add_argument(
        "--c_scatter_dex",
        type=float,
        default=0.10,
        help="log10 scatter in concentration (MC)",
    )
    ap.add_argument(
        "--target_halo_sigma",
        type=float,
        default=0,
        help="Select a Gaussian-equivalent quantile of the combined halo reference draws: 0=median, -1~p16, +1~p84.",
    )
    ap.add_argument(
        "--target_ml_sigma",
        type=float,
        default=0,
        help="Shift the model M/L by this many M/L-systematic sigma; the photometric SHMR reference stays unshifted.",
    )
    ap.add_argument(
        "--umachine_path",
        type=str,
        default=None,
        help="UniverseMachine DR1 root; overrides UMACHINE_PATH.",
    )
    ap.add_argument(
        "--halo_conversion_z",
        type=float,
        default=None,
        help="Epoch for the Mvir-to-M200c boundary conversion, e.g. known peak-mass redshift. Default: adopted observed redshift, explicitly an equivalent-halo assumption.",
    )
    ap.add_argument(
        "--compare_log_m200",
        type=float,
        default=None,
        help="Check a model log10(M200c/Msun), e.g. 13.57, against each 68%% and 95%% reference interval.",
    )
    # Optional photometric / redshift uncertainties
    ap.add_argument(
        "--g_err",
        type=float,
        default=None,
        help="1σ uncertainty in g magnitude [mag]. Overrides magnitude JSON g_err.",
    )
    ap.add_argument(
        "--r_err",
        type=float,
        default=None,
        help="1σ uncertainty in r magnitude [mag]. Overrides magnitude JSON r_err.",
    )
    ap.add_argument(
        "--z_err",
        type=float,
        default=None,
        help="1σ uncertainty in redshift. If omitted, use input dict key 'z_err' if present.",
    )
    ap.add_argument(
        "--ml_mc_samples",
        type=int,
        default=5000,
        help="Number of Monte Carlo draws for uncertainty propagation.",
    )
    ap.add_argument(
        "--ml_mc_seed",
        type=int,
        default=12345,
        help="Random seed for Monte Carlo uncertainty propagation.",
    )
    ap.add_argument(
        "--ml_systematic_dex",
        type=float,
        default=None,
        help=(
            "Additional systematic scatter in log10(M/L). "
            "Default: 0.114 dex for zcr09, 0.10 dex for Bell03 variants. "
            "Set to 0 to disable."
        ),
    )
    ap.add_argument(
        "--disk_luminosity_adj",
        type=float,
        default=1.0,
        help=("Manual adjustment to luminosity to account for mass stripped away"),
    )
    ap.add_argument(
        "--bulge_luminosity_adj",
        type=float,
        default=1.0,
        help=("Manual adjustment to luminosity to account for mass stripped away"),
    )

    args = ap.parse_args()

    # --- Load input dict ---
    d = load_input_dict(args.input_file)
    phot = load_profile_magnitudes(args.magnitudes_file)

    def pick_optional_float(cli_value, dict_key):
        v = cli_value if cli_value is not None else phot.get(dict_key, None)
        return None if v is None else float(v)

    z, z_key_used, z_used_desc = choose_redshift_from_input(d)

    g = phot["g"]
    r = phot["r"]
    ra = float(d.get("ra", np.nan))
    dec = float(d.get("dec", np.nan))

    z_phot_liao = as_float_or_none(d.get("z_phot_liao", None))
    z_phot_liao_err = as_float_or_none(d.get("z_phot_liao_err", None))
    z_source_label = d.get("z_source", z_used_desc)

    target_halo_sigma = float(d.get("target_halo_sigma", args.target_halo_sigma))
    target_ml_sigma = float(d.get("target_ml_sigma", args.target_ml_sigma))

    disk_luminosity_adj = float(d.get("disk_luminosity_adj", args.disk_luminosity_adj))
    bulge_luminosity_adj = float(
        d.get("bulge_luminosity_adj", args.bulge_luminosity_adj)
    )
    photometry_component = (
        args.photometry_component
        if args.photometry_component is not None
        else d.get("photometry_component", "bulge")
    )

    g_err = pick_optional_float(args.g_err, "g_err")
    r_err = pick_optional_float(args.r_err, "r_err")
    z_err_in = choose_redshift_error_from_input(args.z_err, d, z_key_used)
    gr_correlation = (
        args.gr_correlation
        if args.gr_correlation is not None
        else float(phot.get("g_r_corr", 0.0))
    )

    z_diagnostic = liao_redshift_diagnostic(d, z)

    ml_systematic_dex = (
        default_ml_systematic_dex(args.ml)
        if args.ml_systematic_dex is None
        else float(args.ml_systematic_dex)
    )

    # The profile radius is circularized; retain the old data-file radius as a
    # fallback for magnitude files generated before this field was added.
    Re_arcsec = phot["r_reff_arcsec"]
    if Re_arcsec is None:
        Re_arcsec = as_float_or_none(d.get("Reff_r_arcsec"))
    Re_arcsec = float("nan") if Re_arcsec is None else Re_arcsec
    sersic_n = float(d.get("sersic_n", 4.0))
    b_over_a_raw = d.get("b/a", None)
    b_over_a = None if b_over_a_raw is None else float(b_over_a_raw)
    a_over_b = None if (b_over_a is None or b_over_a <= 0) else 1.0 / b_over_a

    # Disk to Total ratio
    DT_raw = d.get("D/T", d.get("D_over_T", 0.0))
    DT = float(DT_raw if DT_raw is not None else 0.0)

    # Disk to Bulge ratio
    DB_raw = d.get("D/B", d.get("D_over_B", 0.0))
    DB = float(DB_raw if DB_raw is not None else 0.0)

    if args.c200 is None:
        c_input = d.get("c", d.get("c200", None))
        if c_input is not None:
            try:
                c_input = float(c_input)
            except Exception:
                c_input = None
    else:
        c_input = args.c200

    for label, value in (
        ("g_err", g_err),
        ("r_err", r_err),
        ("z_err", z_err_in),
        ("ml_systematic_dex", ml_systematic_dex),
        ("um_scatter_dex", args.um_scatter_dex),
        ("c_scatter_dex", args.c_scatter_dex),
        ("icl_frac", args.icl_frac),
    ):
        if value is not None and (not np.isfinite(value) or value < 0):
            ap.error(f"{label} must be finite and nonnegative.")
    if not np.isfinite(DT) or not 0 <= DT <= 1:
        ap.error("D/T must lie in [0, 1].")
    if photometry_component not in ("bulge", "total"):
        ap.error("photometry_component must be 'bulge' or 'total'.")
    if photometry_component == "bulge" and DT == 1:
        ap.error(
            "Bulge-only photometry cannot normalize a D/T=1 galaxy; use --photometry_component total."
        )
    if (not np.isfinite(gr_correlation)) or abs(gr_correlation) > 1:
        ap.error("gr_correlation must lie in [-1, 1].")
    if any(
        not np.isfinite(x) or x <= 0
        for x in (disk_luminosity_adj, bulge_luminosity_adj)
    ):
        ap.error("Disk and bulge luminosity adjustments must be finite and positive.")
    if c_input is not None and (not np.isfinite(c_input) or c_input <= 0):
        ap.error("c200 must be finite and positive.")
    if args.ml_mc_samples < 100:
        ap.error("ml_mc_samples must be at least 100.")
    if args.halo_conversion_z is not None and (
        not np.isfinite(args.halo_conversion_z) or args.halo_conversion_z < 0
    ):
        ap.error("halo_conversion_z must be finite and nonnegative.")
    for label, value in (
        ("target_ml_sigma", target_ml_sigma),
        ("target_halo_sigma", target_halo_sigma),
        ("shmr_shift_dex", args.shmr_shift_dex),
        ("compare_log_m200", args.compare_log_m200),
    ):
        if value is not None and not np.isfinite(float(value)):
            ap.error(f"{label} must be finite.")

    Re_bulge_arcsec = d.get("Re_bulge_arcsec", None)

    # --- Cosmology & H(z) ---
    cosmo = (
        Planck18
        if args.planck18
        else FlatLambdaCDM(
            H0=(70.0 if args.H0 is None else float(args.H0)) * u.km / u.s / u.Mpc,
            Om0=(0.3 if args.Om0 is None else float(args.Om0)),
        )
    )
    Hz = cosmo.H(z).to(u.km / u.s / u.Mpc)
    Hz_val = Hz.value

    # --- K-corrections & rest-frame colour ---
    gmr_obs = g - r
    Kg = kpoly(z, gmr_obs, B_g)
    Kr = kpoly(z, gmr_obs, B_r)

    g0 = g - Kg
    r0 = r - Kr
    gmr0 = g0 - r0

    # Diagnostics:
    #   M_r_photometric uses the script's rest-frame K-corrected convention.
    #   M_r_noK uses the closer Liao convention: dereddened apparent r minus DM.
    mu = cosmo.distmod(z).value
    M_r = r0 - mu
    M_r_noK = r - mu
    M_r_source = "computed_from_r_z_Kr"

    # --- Luminosity & stellar mass ---
    Lr_Lsun = 10.0 ** (-0.4 * (M_r - args.M_r_sun))
    if args.ml == "zcr09":
        ML = mlr_zibetti2009_gr(gmr0)
        ml_name = "Zibetti+2009 (Chabrier)"
    elif args.ml == "bell03_chabrier":
        ML = mlr_bell2003_chabrier_gr(gmr0)
        ml_name = "Bell+2003 (~Chabrier)"
    else:
        ML = mlr_bell2003_diet_gr(gmr0)
        ml_name = "Bell+2003 (diet Salpeter)"

    # Model and empirical SHMR comparison use explicitly separate mass scales.
    # The supplied magnitudes may describe a bulge or the whole galaxy.
    if photometry_component == "bulge":
        # Bulge scaling to photometrically determined mass
        model_bulge_scale = bulge_luminosity_adj
        # Disk scale found from Disk/Bulge ratio
        model_disk_scale = DB * bulge_luminosity_adj
        # Total photometric scale
        photometric_total_mass_scale = 1 + DB
    else:
        photometric_total_mass_scale = 1.0
        model_bulge_scale = (1.0 - DT) * bulge_luminosity_adj
        model_disk_scale = DT * disk_luminosity_adj
    chabrier_offset_dex = 0.10 if args.ml == "bell03_diet" else 0.0
    model_ml_scale = 10.0 ** (ml_systematic_dex * target_ml_sigma)
    Mstar_photometric_Msun = ML * Lr_Lsun * photometric_total_mass_scale
    Mstar_chabrier_Msun = Mstar_photometric_Msun * 10.0 ** (-chabrier_offset_dex)
    ML_chabrier = ML * 10.0 ** (-chabrier_offset_dex)

    uncertainty_mc = estimate_photometric_uncertainties_mc(
        g=g,
        r=r,
        z=z,
        ml_mode=args.ml,
        M_r_sun=args.M_r_sun,
        cosmo=cosmo,
        g_err=g_err,
        r_err=r_err,
        z_err=z_err_in,
        nsamp=args.ml_mc_samples,
        systematic_dex=ml_systematic_dex,
        seed=args.ml_mc_seed,
        gr_correlation=gr_correlation,
        photometric_mass_scale=photometric_total_mass_scale,
        model_bulge_scale=model_bulge_scale,
        model_disk_scale=model_disk_scale,
        model_ml_scale=model_ml_scale,
        chabrier_offset_dex=chabrier_offset_dex,
        return_samples=True,
    )
    photometric_samples = uncertainty_mc.pop("_samples")
    ML *= model_ml_scale

    # --- Disk/Bulge split ---
    Mdisk_Msun = ML * Lr_Lsun * model_disk_scale
    Mbulge_Msun = ML * Lr_Lsun * model_bulge_scale
    Mstar_Msun = Mbulge_Msun + Mdisk_Msun
    DT_init = Mdisk_Msun / Mstar_Msun

    # --- Angular → physical scales ---
    kpc_per_arcsec = cosmo.angular_diameter_distance(z).to_value(u.kpc) * (
        np.pi / 648000.0
    )

    # --- Component sizes (arcsec) with heuristics if missing ---
    Re_bulge_arcsec = Re_arcsec if np.isfinite(Re_arcsec) else float("nan")

    Re_ap_arcsec = Re_bulge_arcsec
    Re_ap_reason = "as-provided"
    Re_ap_kpc = Re_ap_arcsec * kpc_per_arcsec

    # --- Bulge Hernquist a_bulge ---
    Re_bulge_kpc = (
        Re_bulge_arcsec * kpc_per_arcsec
        if np.isfinite(Re_bulge_arcsec)
        else float("nan")
    )
    a_bulge_kpc = hernquist_a_from_Re(Re_bulge_kpc)

    # ------------------ Decide ICL fraction for SHMR path ----------------
    icl_used = max(0.0, float(args.icl_frac))

    # ------------------ SHMR path (with icl_used) ------------------
    Mstar_eff = Mstar_chabrier_Msun * (1.0 + icl_used)
    z_conversion = z if args.halo_conversion_z is None else args.halo_conversion_z
    params = None
    if args.shmr_model == "umachine":
        params_path = um_parameter_path(args.gal_type, args.umachine_path)
        params = read_um_parameters(params_path)
        native_reference = mpeak_um_median(Mstar_eff, z, args.gal_type, params)
        source_cosmo = UM_COSMOLOGY
    else:
        native_reference = invert_shmr_moster13(Mstar_eff)
        source_cosmo = MOSTER_COSMOLOGY
    native_shifted = native_reference * 10.0**args.shmr_shift_dex
    reference_m200, conversion_details = virial_to_m200c_nfw(
        native_shifted,
        z_conversion,
        cosmo,
        c_input,
        source_cosmo=(source_cosmo if args.shmr_model == "umachine" else cosmo),
        source_delta=(None if args.shmr_model == "umachine" else 200.0),
    )
    halo_uncertainty = estimate_halo_reference_mc(
        photometric_samples,
        cosmo,
        args.gal_type,
        c_input=c_input,
        params=params,
        shmr_model=args.shmr_model,
        icl_fraction=icl_used,
        halo_scatter_dex=(
            args.um_scatter_dex if args.shmr_model == "umachine" else 0.0
        ),
        c_scatter_dex=args.c_scatter_dex,
        native_shift_dex=args.shmr_shift_dex,
        conversion_z=args.halo_conversion_z,
        seed=args.ml_mc_seed + 1,
        model_mass_per_chabrier_mass=Mstar_Msun / Mstar_chabrier_Msun,
    )
    halo_draws = halo_uncertainty.pop("_mass_draws")
    combined_budget_name = "with_ml_systematic_and_concentration_and_halo_scatter"
    combined_mass = halo_uncertainty["budgets"][combined_budget_name]["M200c_Msun"]
    model_comparison = (
        compare_model_mass(args.compare_log_m200, halo_draws)
        if args.compare_log_m200 is not None
        else None
    )
    shmr = {
        "icl_used": icl_used,
        "stellar_IMF_for_inversion": "Chabrier",
        "M_star_Chabrier_for_inversion_Msun": Mstar_eff,
        "M_star_photometric_selected_IMF_Msun": Mstar_photometric_Msun,
        "diet_to_chabrier_offset_dex": chabrier_offset_dex,
        "native_halo_mass_definition": (
            "Mpeak_BryanNorman1998"
            if args.shmr_model == "umachine"
            else "M200c_Moster2013_z0_relation"
        ),
        "converted_halo_mass_definition": (
            "M200c_spherical_NFW_equivalent"
            if args.shmr_model == "umachine"
            else "native_M200c_retained_no_BN_conversion"
        ),
        "native_unshifted_reference_Msun": native_reference,
        "native_shifted_reference_Msun": native_shifted,
        "native_shift_dex": args.shmr_shift_dex,
        "M200c_reference_Msun": reference_m200,
        "log10_M200c_reference_Msun": math.log10(reference_m200),
        "halo_conversion_z": z_conversion,
        "conversion_epoch_assumption": (
            "specified_epoch"
            if args.halo_conversion_z is not None
            else "observed_redshift_used_as_reference_epoch"
        ),
        "conversion": conversion_details,
        "native_halo_cosmology": {"H0": source_cosmo.H0.value, "Om0": source_cosmo.Om0},
        "concentration_method": (
            "user_c200_override" if c_input is not None else "Gu2022_eq5_z0_legacy_fit"
        ),
        "uncertainty_mc": halo_uncertainty,
        "model_mass_comparison": model_comparison,
        "interpretation": (
            (
                "Equivalent peak-halo reference, not a measurement of the satellite's bound model-start mass."
                if args.gal_type == "sat"
                else "Assumes peak virial mass approximates virial mass at the reference epoch; no mass evolution inferred."
            )
            if args.shmr_model == "umachine"
            else "Native M200c from the z~0 Moster relation retained; no Bryan--Norman boundary conversion or cosmological abundance recalibration."
        ),
        "progenitor_adjustments_applied_to_SHMR": False,
    }
    if args.shmr_model == "umachine":
        shmr["UM_parameter_file"] = params_path
        shmr["Mpeak_reference_Msun"] = native_reference
    analysis_warnings = []
    missing_errors = [
        key
        for key, value in (("g_err", g_err), ("r_err", r_err), ("z_err", z_err_in))
        if value is None
    ]
    if missing_errors:
        analysis_warnings.append(
            "Unspecified errors treated as zero: " + ", ".join(missing_errors)
        )
    if args.gal_type == "sat":
        analysis_warnings.append(shmr["interpretation"])
    if args.shmr_model == "umachine" and icl_used > 0:
        analysis_warnings.append(
            "Selected smhm_med* UM files exclude IHL; the requested added ICL changes the stellar-mass definition."
        )
    extrapolated = halo_uncertainty[
        "stellar_driven_fraction_outside_log10_native_mass_10_5_to_15"
    ]
    if extrapolated is not None and extrapolated > 0:
        analysis_warnings.append(
            "Some stellar-driven halo draws extrapolate beyond the published log10 native mass fit range 10.5--15."
        )
    shmr["warnings"] = analysis_warnings

    if args.shmr_model == "moster13":
        M200 = reference_m200
        V200 = v200_from_m200(M200, z_conversion, cosmo).to_value(u.km / u.s)
        shmr.update(
            {
                "model": "Moster+2013 (z~0)",
                "M200c_Msun": M200,
                "V200_kms": V200,
                "Mstar_over_M200": ((Mstar_Msun / M200) if M200 > 0 else None),
                "Mbulge_over_M200": (Mbulge_Msun / M200) if M200 > 0 else None,
                "Mdisk_over_M200": (Mdisk_Msun / M200) if M200 > 0 else None,
            }
        )
        hs_target = halo_structure_from_M200(M200, z_conversion, cosmo, c_input)
        shmr["halo_structure"] = hs_target
        shmr["a_bulge_kpc"] = a_bulge_kpc
        shmr["a_bulge_over_a_halo"] = (
            (a_bulge_kpc / hs_target["a_halo_kpc"])
            if np.isfinite(a_bulge_kpc)
            else None
        )

    else:
        med = reference_m200
        scatter = max(0.0, args.um_scatter_dex)
        p16, p84 = combined_mass["p16"], combined_mass["p84"]
        ratio_interval = halo_uncertainty["budgets"][combined_budget_name][
            "Mstar_model_over_M200c"
        ]
        target_quantile = 0.5 * (1.0 + math.erf(target_halo_sigma / math.sqrt(2.0)))
        target_p = float(
            np.percentile(halo_draws[combined_budget_name], 100 * target_quantile)
        )

        shmr.update(
            {
                "model": "UniverseMachine DR1 (Chabrier -> Mpeak -> M200c equivalent)",
                "M200c_median_Msun": med,
                "target_halo_sigma": target_halo_sigma,
                "scatter_dex": scatter,
                "M200c_target_Msun": target_p,
                "M200c_p16_Msun": p16,
                "M200c_p84_Msun": p84,
                "V200_target_kms": v200_from_m200(
                    target_p, z_conversion, cosmo
                ).to_value(u.km / u.s),
                "V200_p16_kms": v200_from_m200(p16, z_conversion, cosmo).to_value(
                    u.km / u.s
                ),
                "V200_p84_kms": v200_from_m200(p84, z_conversion, cosmo).to_value(
                    u.km / u.s
                ),
                "Mstar_over_M200_target": (
                    (Mstar_Msun / target_p) if target_p > 0 else None
                ),
                "Mstar_over_M200_p16": ratio_interval["p16"],
                "Mstar_over_M200_p84": ratio_interval["p84"],
                "Mbulge_over_M200_target": (
                    (Mbulge_Msun / target_p) if target_p > 0 else None
                ),
                "Mbulge_over_M200_p16": ratio_interval["p16"]
                * Mbulge_Msun
                / Mstar_Msun,
                "Mbulge_over_M200_p84": ratio_interval["p84"]
                * Mbulge_Msun
                / Mstar_Msun,
                "Mdisk_over_M200_target": (
                    (Mdisk_Msun / target_p) if target_p > 0 else None
                ),
                "Mdisk_over_M200_p16": ratio_interval["p16"] * Mdisk_Msun / Mstar_Msun,
                "Mdisk_over_M200_p84": ratio_interval["p84"] * Mdisk_Msun / Mstar_Msun,
            }
        )
        hs_target = halo_structure_from_M200(target_p, z_conversion, cosmo, c_input)
        hs_p16 = halo_structure_from_M200(p16, z_conversion, cosmo, c_input)
        hs_p84 = halo_structure_from_M200(p84, z_conversion, cosmo, c_input)
        shmr["halo_structure_target"] = hs_target
        shmr["halo_structure_p16"] = hs_p16
        shmr["halo_structure_p84"] = hs_p84
        shmr["a_bulge_kpc"] = a_bulge_kpc
        shmr["Re_bulge_kpc"] = Re_bulge_kpc
        shmr["a_bulge_over_a_halo_target"] = (
            (a_bulge_kpc / hs_target["a_halo_kpc"])
            if np.isfinite(a_bulge_kpc)
            else None
        )
        shmr["a_bulge_over_a_halo_p16"] = (
            (a_bulge_kpc / hs_p16["a_halo_kpc"]) if np.isfinite(a_bulge_kpc) else None
        )
        shmr["a_bulge_over_a_halo_p84"] = (
            (a_bulge_kpc / hs_p84["a_halo_kpc"]) if np.isfinite(a_bulge_kpc) else None
        )

    # ------------------ Output payload ------------------
    cosmo_label = (
        "Planck18"
        if cosmo is Planck18
        else f"FlatLambdaCDM(H0={cosmo.H0.value:.3f}, Om0={cosmo.Om0:.3f})"
    )

    out = {
        "schema_version": "2.0_IMF_and_halo_definition_corrected",
        "inputs": {
            "ra": ra,
            "dec": dec,
            "name": d.get("name", None),
            "ra_icrs": d.get("ra_icrs", None),
            "dec_icrs": d.get("dec_icrs", None),
            "z": z,
            "z_key_used": z_key_used,
            "z_used_description": z_used_desc,
            "z_source": z_source_label,
            "z_phot_liao": z_phot_liao,
            "z_phot_liao_err": z_phot_liao_err,
            "z_diagnostic": z_diagnostic,
            "g": g,
            "r": r,
            "g_err": g_err,
            "r_err": r_err,
            "g_r_error_correlation": gr_correlation,
            "photometry_component": photometry_component,
            "magnitudes_file": args.magnitudes_file,
            "z_err": z_err_in,
            "dereddened": True,
            "M_r_source": M_r_source,
            "Reff_r_arcsec": (
                None if not np.isfinite(Re_bulge_arcsec) else float(Re_bulge_arcsec)
            ),
            "Re_bulge_arcsec": (
                None if not np.isfinite(Re_bulge_arcsec) else float(Re_bulge_arcsec)
            ),
            "sersic_n": sersic_n,
            "b_over_a": b_over_a,
            "a_over_b": a_over_b,
            "D_over_T": DT,
            "c_input": c_input,
        },
        "cosmology": {"model": cosmo_label, "H_z_km_s_Mpc": Hz_val},
        "k_corrections": {"Kg_mag": Kg, "Kr_mag": Kr, "g_minus_r_obs": gmr_obs},
        "rest_frame": {
            "g0": g0,
            "r0": r0,
            "(g-r)_0": gmr0,
            "M_r_mag": M_r,
            "M_r_source": M_r_source,
            "M_r_noK_mag": M_r_noK,
            "distance_modulus_mag": mu,
        },
        "luminosity": {"L_r_Lsun": Lr_Lsun, "M_r_sun_mag": args.M_r_sun},
        "sizes": {
            "Reff_r_arcsec": (
                None if not np.isfinite(Re_bulge_arcsec) else float(Re_bulge_arcsec)
            ),
            "Reff_r_kpc": None if not np.isfinite(Re_ap_kpc) else float(Re_ap_kpc),
            "Reff_definition": (
                "Sérsic-corrected circularized r-band radius from legacy_ellipse_profile"
                if phot["r_reff_arcsec"] is not None
                else d.get("Reff_definition", "Galaxy input Reff_r_arcsec")
            ),
            "kpc_per_arcsec": kpc_per_arcsec,
            "Re_bulge_kpc": (
                None if not np.isfinite(Re_bulge_kpc) else float(Re_bulge_kpc)
            ),
            "a_bulge_kpc": None if not np.isfinite(a_bulge_kpc) else float(a_bulge_kpc),
        },
        "stellar_mass": {
            "target_ml_sigma": target_ml_sigma,
            "bulge_luminosity_adj": bulge_luminosity_adj,
            "disk_luminosity_adj": disk_luminosity_adj,
            "D_over_T_init": DT_init,
            "M_over_L_r": ML,
            "M_bulge_Msun": Mbulge_Msun,
            "M_disk_Msun": Mdisk_Msun,
            "M_star_Msun": Mstar_Msun,
            "ML_method": ml_name,
            "model_IMF": ("diet_Salpeter" if args.ml == "bell03_diet" else "Chabrier"),
            "M_star_photometric_selected_IMF_Msun": Mstar_photometric_Msun,
            "M_star_photometric_Chabrier_Msun": Mstar_chabrier_Msun,
            "M_over_L_r_Chabrier_for_SHMR": ML_chabrier,
            "SHMR_stellar_mass_note": "Present-day photometric total; no model M/L target shift or progenitor luminosity adjustments.",
            "D_over_T": DT,
            "M_bulge_Msun": Mbulge_Msun,
            "M_disk_Msun": Mdisk_Msun,
            "uncertainty_mc": uncertainty_mc,
        },
        "halo_mass_SHMR": shmr,
    }

    # ------------------ Print (human-readable) ------------------
    s3 = sig3_noexp_str
    s5 = mag5_noexp_str

    if args.json or args.json_numeric:
        out_fmt = finite_json_values(copy.deepcopy(out))
        if not args.json_numeric:
            format_numbers_inplace(out_fmt)
        print(json.dumps(out_fmt, indent=2, allow_nan=False))
    else:
        print("\n=== Inputs ===")
        print(f"z = {s3(z)} [{z_key_used}], g = {s5(g)}, r = {s5(r)}  (dereddened)")
        if z_phot_liao is not None:
            msg = f"Liao photo-z = {s3(z_phot_liao)}"
            if z_phot_liao_err is not None:
                msg += f" ± {s3(z_phot_liao_err)}"
            print(msg)
        if z_diagnostic is not None and "warning" in z_diagnostic:
            print(f"WARNING: {z_diagnostic['warning']}")
        if g_err is not None or r_err is not None or z_err_in is not None:
            print(
                f"input uncertainties: g_err = {s3(g_err) if g_err is not None else 'None'} mag, "
                f"r_err = {s3(r_err) if r_err is not None else 'None'} mag, "
                f"z_err = {s3(z_err_in) if z_err_in is not None else 'None'}"
            )
        if np.isfinite(Re_bulge_arcsec):
            print(f"Reff_r (circularized) = {s3(Re_bulge_arcsec)} arcsec")
        if np.isfinite(Re_bulge_arcsec):
            print(f"Re_bulge = {s3(Re_bulge_arcsec)} arcsec")
        print(f"Sérsic n (bulge) = {s3(sersic_n)}")
        if b_over_a is not None:
            print(f"b/a = {s3(b_over_a)}  →  a/b = {s3(a_over_b)}")
        print(f"D/T (disk-to-total luminous mass) = {s3(DT)}")
        print(f"Cosmology: {cosmo_label}")
        print(f"H(z) = {s3(Hz_val)} km s^-1 Mpc^-1")

        print("\n=== K-corrections (Chilingarian+2010; Saulder+2013) ===")
        print(f"K_g = {s5(Kg)} mag,  K_r = {s5(Kr)} mag   (obs (g-r)={s5(gmr_obs)})")

        print("\n=== Rest-frame & luminosity ===")
        print(f"g0 = {g0} mag")
        print(f"r0 = {r0} mag")
        print(f"(g-r)_0 = {s5(gmr0)}")
        print(f"M_r = {s5(M_r)} mag [{M_r_source}]")
        print(f"M_r from r,z without K-correction = {s5(M_r_noK)} mag")
        print(f"L_r = {sci3(Lr_Lsun)} L_sun,r  (M_r,Sun={s5(args.M_r_sun)})")

        print("\n=== Stellar mass (luminous) & bulge/disk split ===")
        print(f"M/L_r ({ml_name}) = {s3(ML)}")
        print(f"M_star = {sci3(Mstar_Msun)} Msun")
        print(f"Photometry describes: {photometry_component}")
        print(f"Photometric total (selected IMF) = {sci3(Mstar_photometric_Msun)} Msun")
        print(
            f"Chabrier-equivalent total for SHMR = {sci3(Mstar_chabrier_Msun)} Msun "
            f"(log10={math.log10(Mstar_chabrier_Msun):.5f}; IMF offset={chabrier_offset_dex:.3f} dex)"
        )
        print(
            f"  M_bulge = {sci3(Mbulge_Msun)} Msun,  M_disk = {sci3(Mdisk_Msun)} Msun"
        )
        if np.isfinite(a_bulge_kpc):
            print(f"  a_bulge (Hernquist) = {s3(a_bulge_kpc)} kpc  (from Re_bulge)")
        print(
            f"Re_ap for dynamics = {s3(Re_ap_arcsec)} arcsec ({s3(Re_ap_kpc)} kpc) [{Re_ap_reason}]"
        )

        umc = out["stellar_mass"].get("uncertainty_mc", None)
        if umc is not None:
            print("\n=== Monte Carlo uncertainty propagation ===")
            up = umc.get("photometric_only", {})
            cp = up.get("g_minus_r_0", None)
            lp = up.get("L_r_Lsun", None)
            mlp = up.get("M_over_L_r", None)
            mp = up.get("M_star_Msun", None)

            if cp is not None:
                print(
                    f"(g-r)_0 photometric 68% = {s5(cp['p16'])} – {s5(cp['p84'])}  "
                    f"(median={s5(cp['median'])})"
                )
            if lp is not None:
                print(
                    f"L_r photometric 68% = {sci3(lp['p16'])} – {sci3(lp['p84'])}  "
                    f"(median={sci3(lp['median'])})"
                )
            if mlp is not None:
                print(
                    f"M/L_r photometric 68% = {s3(mlp['p16'])} – {s3(mlp['p84'])}  "
                    f"(median={s3(mlp['median'])}, sigma_log10={s3(mlp['sigma_dex'])} dex)"
                )
            if mp is not None:
                print(
                    f"M_star photometric 68% = {sci3(mp['p16'])} – {sci3(mp['p84'])}  "
                    f"(median={sci3(mp['median'])}, sigma_log10={s3(mp['sigma_dex'])} dex)"
                )

            uts = umc.get("with_systematic", {})
            mls = uts.get("M_over_L_r", None)
            ms = uts.get("M_star_Msun", None)
            if mls is not None:
                print(
                    f"M/L_r with systematic ({s3(ml_systematic_dex)} dex) 68% = "
                    f"{s3(mls['p16'])} – {s3(mls['p84'])}  "
                    f"(median={s3(mls['median'])}, sigma_log10={s3(mls['sigma_dex'])} dex)"
                )
            if ms is not None:
                print(
                    f"M_star with systematic ({s3(ml_systematic_dex)} dex) 68% = "
                    f"{sci3(ms['p16'])} – {sci3(ms['p84'])}  "
                    f"(median={sci3(ms['median'])}, sigma_log10={s3(ms['sigma_dex'])} dex)"
                )

        print("\n=== Halo mass from SHMR ===")
        s = out["halo_mass_SHMR"]
        print(f"ICL fraction used for SHMR = {s3(s['icl_used'])}")
        native_name = "Mpeak" if args.shmr_model == "umachine" else "M200c"
        print(
            f"Native {native_name} reference = {sci3(native_reference)} Msun "
            f"(log10={math.log10(native_reference):.5f})"
        )
        print(
            f"M200c reference = {sci3(reference_m200)} Msun "
            f"(log10={math.log10(reference_m200):.5f}; conversion z={z_conversion:.6f})"
        )
        print(
            f"Native virial overdensity = {conversion_details['Delta_vir_critical_native']:.3f} rho_crit(native cosmology); "
            f"M200c/native_mass={conversion_details['M200c_over_Mvir']:.5f}"
        )
        for message in analysis_warnings:
            print(f"NOTE: {message}")

        if s["model"].startswith("Moster"):
            print(
                f"M200c = {sci3(s['M200c_Msun'])} Msun,  V200 = {s3(s['V200_kms'])} km/s"
            )
            print(
                f"  M*/M200 = {s3(s['Mstar_over_M200'])},  "
                f"M_bulge/M200 = {s3(s['Mbulge_over_M200'])},  M_disk/M200 = {s3(s['Mdisk_over_M200'])}"
            )
            if "halo_structure" in s:
                hs = s["halo_structure"]
                print(
                    f"  R200 = {s3(hs['R200_kpc'])} kpc,  c = {s3(hs['c_used'])},  "
                    f"r_s = {s3(hs['r_s_kpc'])} kpc,  a_halo = {s3(hs['a_halo_kpc'])} kpc"
                )
                if np.isfinite(a_bulge_kpc):
                    print(f"  a_bulge/a_halo = {s3(s['a_bulge_over_a_halo'])}")
        else:
            print(
                f"{s['model']} (target): M200c = {sci3(s['M200c_target_Msun'])} Msun,  "
                f"V200 = {s3(s['V200_target_kms'])} km/s"
            )
            print(
                f"  M*/M200 = {s3(s['Mstar_over_M200_target'])}  "
                f"(68%: {s3(s['Mstar_over_M200_p16'])}–{s3(s['Mstar_over_M200_p84'])})"
            )
            print(
                f"  M_bulge/M200 = {s3(s['Mbulge_over_M200_target'])}  "
                f"(68%: {s3(s['Mbulge_over_M200_p16'])}–{s3(s['Mbulge_over_M200_p84'])})"
            )
            print(
                f"  M_disk/M200 = {s3(s['Mdisk_over_M200_target'])}  "
                f"(68%: {s3(s['Mdisk_over_M200_p16'])}–{s3(s['Mdisk_over_M200_p84'])})"
            )
            hs = s["halo_structure_target"]
            print(
                f"  R200 (target) = {s3(hs_target['R200_kpc'])} kpc,  c = {s3(hs_target['c_used'])},  "
                f"r_s = {s3(hs_target['r_s_kpc'])} kpc,  a_halo = {s3(hs_target['a_halo_kpc'])} kpc"
            )
            if np.isfinite(a_bulge_kpc):
                print(
                    f"  a_bulge/a_halo (target) = {s3(s['a_bulge_over_a_halo_target'])}"
                )
        print("\nHalo reference intervals in log10(M200c/Msun):")
        budget_labels = {
            "photometric_only": "Photometry + redshift only",
            "with_ml_systematic": "+ M/L systematic",
            "with_ml_systematic_and_concentration": "+ concentration scatter",
            "with_ml_systematic_and_concentration_and_halo_scatter": "+ assumed halo scatter",
        }
        for key, label in budget_labels.items():
            summary = halo_uncertainty["budgets"][key]["M200c_Msun"]
            print(
                f"  {label}: {summary['log10_median']:.5f} "
                f"(-{summary['log10_err_minus']:.5f}, +{summary['log10_err_plus']:.5f}); "
                f"68% [{summary['log10_p16']:.5f}, {summary['log10_p84']:.5f}]; "
                f"95% [{summary['log10_p2_5']:.5f}, {summary['log10_p97_5']:.5f}]"
            )
        if args.shmr_model == "umachine":
            print(
                f"  Halo scatter is an assumed {args.um_scatter_dex:.3f}-dex prior in native halo mass at fixed stellar mass."
            )
        print("  These are reference intervals, not a Bayesian halo-mass posterior.")
        if model_comparison is not None:
            print(f"\nModel comparison: log10(M200c/Msun)={args.compare_log_m200:.5f}")
            for key, label in budget_labels.items():
                check = model_comparison["interval_checks"][key]
                print(
                    f"  {label}: inside 68% = {check['within_68_percent']}; "
                    f"inside 95% = {check['within_95_percent']}; "
                    f"reference percentile = {check['reference_sample_percentile']:.2f}%"
                )
        print()


if __name__ == "__main__":
    # UniverseMachine DR1 is required if you use --shmr_model umachine
    main()
