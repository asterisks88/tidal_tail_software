#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elliptical/S0 luminous mass & M200 from SDSS/DES g,r & z (Astropy)

Usage: mass_from_photometry.py galaxy.json galaxy_profile.magnitudes.json [options]
The galaxy file supplies redshift and structural data; the second file is the
combined g/r Sérsic-total magnitude JSON from legacy_ellipse_profile.py.

Formatting rules:
- Msun/Lsun fields: scientific notation strings with 3 significant figures.
- Magnitudes & colors: 5 significant figures, fixed notation.
- All other numbers: 3 significant figures, fixed notation.

This version adds Monte Carlo uncertainty propagation for:
  g,r,z -> K_g,K_r -> (g-r)_0 -> M_r -> L_r -> M/L_r -> M_star

It reports:
  - photometric_only uncertainties
  - with_systematic uncertainties (optional M/L systematic floor)

Also fixes:
  - bad halo_structure_from_M200(...) call in the Moster branch
  - human-readable print block referencing missing icl_details
"""

import argparse
import ast
import copy
import json
import math
import os
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
            key_is_msun_or_lsun(leaf_key)
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
                    if key_is_msun_or_lsun(use_key):
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
                    if key_is_msun_or_lsun(use_key):
                        obj[i] = sci3(v)
                    elif key_is_position_like(use_key):
                        obj[i] = str(v)
                    elif key_is_magnitude_like(use_key):
                        obj[i] = mag5_noexp_str(v)
                    else:
                        obj[i] = sig3_noexp_str(v)


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

    If M_r_input is supplied, it is used directly for the luminosity.
    Otherwise M_r is computed from r, K_r, and the distance modulus.
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

    p16, p50, p84 = np.percentile(x, [16, 50, 84])
    out = {
        "p16": float(p16),
        "median": float(p50),
        "p84": float(p84),
        "err_minus": float(p50 - p16),
        "err_plus": float(p84 - p50),
        "std": float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "nsamp_used": int(x.size),
    }

    if positive_only:
        lx = np.log10(x)
        out["sigma_dex"] = float(np.std(lx, ddof=1)) if x.size > 1 else 0.0

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

    gs = rng.normal(g, g_err, nsamp) if g_err > 0 else np.full(nsamp, g, dtype=float)
    rs = rng.normal(r, r_err, nsamp) if r_err > 0 else np.full(nsamp, r, dtype=float)
    zs = rng.normal(z, z_err, nsamp) if z_err > 0 else np.full(nsamp, z, dtype=float)

    # Avoid z <= 0 causing pathological distance moduli
    zs = np.clip(zs, 1.0e-8, None)
    base = compute_restframe_quantities(
        gs,
        rs,
        zs,
        ml_mode,
        M_r_sun,
        cosmo,
    )

    phot = {
        "g_minus_r_0": summarize_samples(base["gmr0"], positive_only=False),
        "M_r_mag": summarize_samples(base["M_r"], positive_only=False),
        "L_r_Lsun": summarize_samples(base["Lr_Lsun"], positive_only=True),
        "M_over_L_r": summarize_samples(base["ML"], positive_only=True),
        "M_star_Msun": summarize_samples(base["Mstar"], positive_only=True),
        "M_star_Msun_with_missing": summarize_samples(
            base["Mstar"], positive_only=True
        ),
    }

    out = {
        "method": "monte_carlo",
        "inputs": {
            "g_err_mag": g_err,
            "r_err_mag": r_err,
            "z_err": z_err,
            "systematic_dex": float(systematic_dex),
            "nsamp": nsamp,
            "seed": seed,
        },
        "photometric_only": phot,
    }

    if systematic_dex > 0:
        sys_factor = 10.0 ** rng.normal(0.0, systematic_dex, nsamp)
        ML_sys = base["ML"] * sys_factor
        Mstar_sys = ML_sys * base["Lr_Lsun"]

        out["with_systematic"] = {
            "g_minus_r_0": summarize_samples(base["gmr0"], positive_only=False),
            "M_r_mag": summarize_samples(base["M_r"], positive_only=False),
            "L_r_Lsun": summarize_samples(base["Lr_Lsun"], positive_only=True),
            "M_over_L_r": summarize_samples(ML_sys, positive_only=True),
            "M_star_Msun": summarize_samples(Mstar_sys, positive_only=True),
            "M_star_Msun_with_missing": summarize_samples(
                Mstar_sys, positive_only=True
            ),
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
    lo, hi = 10.0, 16.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        Mh = 10.0**mid
        if mstar_from_mhalo_moster13(Mh, N, M1, beta, gamma) < Mstar:
            lo = mid
        else:
            hi = mid
    return 10.0 ** (0.5 * (lo + hi))


# Load data from Figure 9, 10 in Behroozi P., Wechsler R. H., Hearin A. P., Conroy C., 2019, MNRAS, 488, 289
def load_umachine_smhm(z: float, galaxy_type: str) -> list[list]:
    umachine_smhm = []
    upath_str = os.environ.get(
        "UMACHINE_PATH", os.path.join(os.environ.get("HOME", ""), "umachine-dr1")
    )
    param_files = {
        "all": os.path.join(upath_str, "data/smhm/params/smhm_med_params.txt"),
        "host": os.path.join(upath_str, "data/smhm/params/smhm_med_cen_params.txt"),
        "sat": os.path.join(upath_str, "data/smhm/params/smhm_med_sat_params.txt"),
    }

    with open(param_files[galaxy_type], "r") as param_file:
        param_list = []
        for line in param_file:
            param_list.append(float((line.split(" "))[1]))

    names = "EFF_0 EFF_0_A EFF_0_A2 EFF_0_Z M_1 M_1_A M_1_A2 M_1_Z ALPHA ALPHA_A ALPHA_A2 ALPHA_Z BETA BETA_A BETA_Z DELTA GAMMA GAMMA_A GAMMA_Z CHI2".split(
        " "
    )
    params = dict(zip(names, param_list))
    a = 1.0 / (1.0 + z)
    a1 = a - 1.0
    lna = math.log(a)
    zparams = {}
    zparams["m_1"] = (
        params["M_1"]
        + a1 * params["M_1_A"]
        - lna * params["M_1_A2"]
        + z * params["M_1_Z"]
    )
    zparams["sm_0"] = (
        zparams["m_1"]
        + params["EFF_0"]
        + a1 * params["EFF_0_A"]
        - lna * params["EFF_0_A2"]
        + z * params["EFF_0_Z"]
    )
    zparams["alpha"] = (
        params["ALPHA"]
        + a1 * params["ALPHA_A"]
        - lna * params["ALPHA_A2"]
        + z * params["ALPHA_Z"]
    )
    zparams["beta"] = params["BETA"] + a1 * params["BETA_A"] + z * params["BETA_Z"]
    zparams["delta"] = params["DELTA"]
    zparams["gamma"] = 10 ** (
        params["GAMMA"] + a1 * params["GAMMA_A"] + z * params["GAMMA_Z"]
    )

    smhm_max = 14.5 - 0.35 * z
    for m in [x * 0.05 for x in range(int(10.5 * 20), int(smhm_max * 20 + 1), 1)]:
        dm = m - zparams["m_1"]
        dm2 = dm / zparams["delta"]
        sm = (
            zparams["sm_0"]
            - math.log10(10 ** (-zparams["alpha"] * dm) + 10 ** (-zparams["beta"] * dm))
            + zparams["gamma"] * math.exp(-0.5 * (dm2 * dm2))
        )
        umachine_smhm.append([sm, m, sm - m])

    return umachine_smhm


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


def m200_um_median(Mstar: float, z: float, galaxy_type: str) -> float:
    umachine_smhm = load_umachine_smhm(z, galaxy_type)
    return _interp_loglog(Mstar, umachine_smhm)


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
    # log10 c200 = a + b * log10( M200 / (h^-1 Msun) )
    return 10.0 ** (2.161 + -0.105 * math.log10(M200 * h))


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
        description="Estimate stellar mass and M200 (Astropy). Includes MC uncertainty propagation for color, M/L, luminosity, and stellar mass.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input_file", type=str, help="Galaxy data JSON/dict (without magnitudes)")
    ap.add_argument(
        "magnitudes_file", type=str,
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
        help="Add shift to log10(M200) from SHMR",
    )
    ap.add_argument(
        "--um_scatter_dex",
        type=float,
        default=0.25,
        help="1σ log10 scatter for 'umachine' (report only)",
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
        help="Target percentage of CDM halo mass from SHMR inversion, -1sigma = p16 0=median +1sigma=p84",
    )
    ap.add_argument(
        "--target_ml_sigma",
        type=float,
        default=0,
        help="Target percentage of median computed M/L, -1sigma = p16 0=median +1sigma=p84",
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

    target_halo_sigma = d.get("target_halo_sigma", args.target_halo_sigma)
    target_ml_sigma = d.get("target_ml_sigma", args.target_ml_sigma)

    disk_luminosity_adj = d.get("disk_luminosity_adj", args.disk_luminosity_adj)
    bulge_luminosity_adj = d.get("bulge_luminosity_adj", args.disk_luminosity_adj)

    g_err = pick_optional_float(args.g_err, "g_err")
    r_err = pick_optional_float(args.r_err, "r_err")
    z_err_in = choose_redshift_error_from_input(args.z_err, d, z_key_used)

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

    DT_raw = d.get("D/T", d.get("D_over_T", 0.0))
    DT = float(max(0.0, min(1.0, DT_raw if DT_raw is not None else 0.0)))

    if args.c200 is None:
        c_input = d.get("c", d.get("c200", None))
        if c_input is not None:
            try:
                c_input = float(c_input)
            except Exception:
                c_input = None
    else:
        c_input = args.c200

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
    )
    ML *= 10 ** (ml_systematic_dex * target_ml_sigma)
    Mbulge_Msun = ML * Lr_Lsun

    # --- Disk/Bulge split ---
    # Assume photometry is for the bulge
    Mdisk_Msun = Mbulge_Msun * DT / (1 - DT) * disk_luminosity_adj
    Mbulge_Msun *= bulge_luminosity_adj
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
    Mstar_eff = Mstar_Msun * (1.0 + icl_used)
    shmr = {"icl_used": icl_used}

    if args.shmr_model == "moster13":
        M200 = invert_shmr_moster13(Mstar_eff) * (10.0**args.shmr_shift_dex)
        V200 = v200_from_m200(M200, z, cosmo).to_value(u.km / u.s)
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
        hs_target = halo_structure_from_M200(M200, z, cosmo, c_input)
        shmr["halo_structure"] = hs_target
        shmr["a_bulge_kpc"] = a_bulge_kpc
        shmr["a_bulge_over_a_halo"] = (
            (a_bulge_kpc / hs_target["a_halo_kpc"])
            if np.isfinite(a_bulge_kpc)
            else None
        )

    else:
        med = m200_um_median(Mstar_eff, z, args.gal_type) * (10.0**args.shmr_shift_dex)
        scatter = max(0.0, args.um_scatter_dex)
        p16, p84 = med * (10.0**-scatter), med * (10.0**scatter)
        target_p = med * 10 ** (scatter * target_halo_sigma)

        shmr.update(
            {
                "model": "UM/lensing (z~0)",
                "target_halo_sigma": target_halo_sigma,
                "scatter_dex": scatter,
                "M200c_target_Msun": target_p,
                "M200c_p16_Msun": p16,
                "M200c_p84_Msun": p84,
                "V200_target_kms": v200_from_m200(target_p, z, cosmo).to_value(
                    u.km / u.s
                ),
                "V200_p16_kms": v200_from_m200(p16, z, cosmo).to_value(u.km / u.s),
                "V200_p84_kms": v200_from_m200(p84, z, cosmo).to_value(u.km / u.s),
                "Mstar_over_M200_target": (
                    (Mstar_Msun / target_p) if target_p > 0 else None
                ),
                "Mstar_over_M200_p16": ((Mstar_Msun / p16) if p16 > 0 else None),
                "Mstar_over_M200_p84": ((Mstar_Msun / p84) if p84 > 0 else None),
                "Mbulge_over_M200_target": (
                    (Mbulge_Msun / target_p) if target_p > 0 else None
                ),
                "Mbulge_over_M200_p16": (Mbulge_Msun / p16) if p16 > 0 else None,
                "Mbulge_over_M200_p84": (Mbulge_Msun / p84) if p84 > 0 else None,
                "Mdisk_over_M200_target": (
                    (Mdisk_Msun / target_p) if target_p > 0 else None
                ),
                "Mdisk_over_M200_p16": (Mdisk_Msun / p16) if p16 > 0 else None,
                "Mdisk_over_M200_p84": (Mdisk_Msun / p84) if p84 > 0 else None,
            }
        )
        hs_target = halo_structure_from_M200(target_p, z, cosmo, c_input)
        hs_p16 = halo_structure_from_M200(p16, z, cosmo, c_input)
        hs_p84 = halo_structure_from_M200(p84, z, cosmo, c_input)
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

    if args.json:
        out_fmt = copy.deepcopy(out)
        format_numbers_inplace(out_fmt)
        print(json.dumps(out_fmt, indent=2))
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
        print()


if __name__ == "__main__":
    # UniverseMachine DR1 is required if you use --shmr_model umachine
    main()
