#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Elliptical/S0 luminous mass & M200 from SDSS g,r & z (Astropy)

Formatting rules:
- Msun/Lsun fields: scientific notation strings with 3 significant figures (e.g., "1.23e+12").
- Magnitudes & colors (g, r, K-corr, M_r, solar M_r, (g-r), etc.): 5 significant figures, fixed notation.
- All other numbers: 3 significant figures, fixed notation.

All physics features from prior version are preserved.
"""

import argparse
import ast
import copy
import json
import math
import os
from pathlib import Path
from typing import Any, List, Optional

import astropy.units as u
import numpy as np
from astropy.constants import G
from astropy.cosmology import FlatLambdaCDM, Planck18


# ------------------ Formatting helpers ------------------
def sci3(x: Any) -> str:
    """Return number in scientific notation with 3 significant figures, as a string."""
    try:
        return f"{float(x):.2e}"
    except Exception:
        return str(x)


def _round_sig(x: float, sig: int) -> float:
    if not np.isfinite(x) or x == 0:
        return 0.0 if x == 0 else float(x)
    p = sig - 1 - int(math.floor(math.log10(abs(x))))
    return round(x, p)


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
    """Magnitudes/colors: 5 significant figures, fixed notation."""
    return sig_noexp_str(x, 5)


def key_is_msun_or_lsun(key: str) -> bool:
    kl = key.lower()
    return ("msun" in kl) or ("lsun" in kl) or ("l_sun" in kl)


# Treat these as magnitude/color fields
_MAG_KEYS_EXPLICIT = {
    "g",
    "r",
    "u",
    "i",
    "z",
    "M_r_mag",
    "M_g_mag",
    "M_i_mag",
    "M_z_mag",
    "M_u_mag",
    "M_r_sun_mag",
    "(g-r)_0",
    "g_minus_r_obs",
    "Kg_mag",
    "Kr_mag",
}


def key_is_magnitude_like(key: str) -> bool:
    kl = key.lower()
    return ("mag" in kl) or (key in _MAG_KEYS_EXPLICIT)


def key_is_position_like(key: str) -> bool:
    kl = key.lower()
    return kl in ["ra", "dec"]


def format_msun_lsun_inplace(obj: Any):
    """Convert all Msun/Lsun fields to sci-3 strings."""
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, (dict, list)):
                format_msun_lsun_inplace(v)
            else:
                if (
                    key_is_msun_or_lsun(k)
                    and isinstance(v, (int, float, np.floating))
                    and not isinstance(v, bool)
                ):
                    obj[k] = sci3(v)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                format_msun_lsun_inplace(v)


def format_other_numbers_inplace(obj: Any):
    """
    For all other numeric fields:
      - magnitudes/colors: 5-sig fixed
      - ra/dec: use input
      - everything else: 3-sig fixed
    """
    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            if isinstance(v, (dict, list)):
                format_other_numbers_inplace(v)
            else:
                if (
                    (not key_is_msun_or_lsun(k))
                    and isinstance(v, (int, float, np.floating))
                    and not isinstance(v, bool)
                ):
                    if key_is_position_like(k):
                        obj[k] = str(v)
                    else:
                        obj[k] = (
                            mag5_noexp_str(v)
                            if key_is_magnitude_like(k)
                            else sig3_noexp_str(v)
                        )
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            if isinstance(v, (dict, list)):
                format_other_numbers_inplace(v)


# ------------------ Chilingarian/Saulder K-corrections ------------------
# From Saulder C., Mieske S., Zeilinger W. W., Chilingarian I., 2013, A&A, 557,A21
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
# All of these return Yr, the r band M/L from (g-r)


# From Zibetti S., Charlot S., Rix H.-W., 2009, MNRAS, 400, 1181
def mlr_zibetti2009_gr(color0: float) -> float:
    return 10.0 ** (-0.840 + 1.654 * color0)


# Bell E. F., McIntosh D. H., Katz N., Weinberg M. D., 2003, Astrophysical Journal Supplement Series, 149, 289
def mlr_bell2003_diet_gr(color0: float) -> float:
    return 10.0 ** (-0.306 + 1.097 * color0)


# Bell adjusted by 0.1dex for closer match to Chabrier
def mlr_bell2003_chabrier_gr(color0: float) -> float:
    return mlr_bell2003_diet_gr(color0) / (10.0**0.10)


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
    # These files come from the UniverseMachine DR1
    param_files = {
        "all": os.path.join(upath_str, "data/smhm/params/smhm_med_params.txt"),
        "host": os.path.join(upath_str, "data/smhm/params/smhm_med_cen_params.txt"),
        "sat": os.path.join(upath_str, "data/smhm/params/smhm_med_sat_params.txt"),
    }
    # This code from gen_smhm.py from UniverseMachine DR1
    # -----------------------------------------------------
    param_file = open(param_files[galaxy_type], "r")
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
    xs = [x[0] for x in xy]
    ys = [x[1] for x in xy]
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
    # a = 0.905 and b = -0.101 from Dutton A. A., Macciò A. V., 2014, MNRAS, 441, 3359
    # log10 c200 = a + b * log10( M200 / (1e12 h^-1 Msun) )
    return 10.0 ** (0.905 + -0.101 * math.log10(M200 * h / 1.0e12))


def c200_from_m200_wang2022(M200: float, h: float) -> float:
    # Gu Q., Guo Q., Zhang T., Wang W., Guo Q., Gao L., 2022, Research in
    # Astronomy and Astrophysics, 22, 125015
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
    R200 = r200_from_m200(M200, z, cosmo)  # Mpc
    R200_kpc = R200.to_value(u.kpc)
    H0 = cosmo.H0.value
    h = H0 / 100.0
    if (c_input is not None) and (c_input > 0):
        c_used = float(c_input)
    else:
        c_used = float(c200_from_m200_wang2022(M200, h))
    rs = (R200 / c_used).to(u.kpc)
    a_halo = a_halo_from_c_rs(rs, c_used)  # kpc
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
    text = open(path, "r").read().strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


# ------------------ Main ------------------
def main():
    ap = argparse.ArgumentParser(
        description="Estimate stellar mass and M200 (Astropy). ICL prior; D/T; bulge+disk M(<Re). Msun/Lsun sci-3; mags 5-sig; others 3-sig.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("input_file", type=str, help="Path to file with the input dict")

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
        default="moster13",
        choices=["moster13", "umachine", "fixed"],
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
        default=0.25,  # Adopted Figure 12, Behroozi P., Wechsler R. H., Hearin A. P., Conroy C., 2019, MNRAS, 488, 289
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
        "--missing_stellar",
        type=float,
        default=None,
        help="Missing Stellar Mass Overide",
    )
    ap.add_argument(
        "--missing_halo",
        type=float,
        default=None,
        help="Missing Halo Mass Overide",
    )
    args = ap.parse_args()

    # --- Load input dict ---
    d = load_input_dict(args.input_file)
    z = float(d["z"])
    g = float(d["g"])
    r = float(d["r"])
    ra = float(d["ra"])
    dec = float(d["dec"])
    Re_arcsec = float(d.get("Reff_r_arcsec", np.nan))
    sersic_n = float(d.get("sersic_n", 4.0))
    a_pix = d.get("A_pix", None)
    b_pix = d.get("B_pix", None)
    b_over_a_raw = None if a_pix is None or b_pix is None else b_pix / a_pix
    b_over_a = None if b_over_a_raw is None else float(b_over_a_raw)
    a_over_b = None if (b_over_a is None or b_over_a <= 0) else 1.0 / b_over_a
    DT_raw = d.get("D/T", d.get("D_over_T", 0.0))
    DT = float(max(0.0, min(1.0, DT_raw if DT_raw is not None else 0.0)))
    if args.missing_stellar is None:
        mstar_missing = d.get("mstar_missing", 1.0)
    else:
        mstar_missing = args.missing_stellar
    if args.missing_halo is None:
        mhalo_missing = d.get("mhalo_missing", 1.0)
    else:
        mhalo_missing = args.missing_halo

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
    Re_disk_arcsec = d.get("Re_disk_arcsec", None)
    Rd_exp_arcsec = d.get("Rd_exp_arcsec", None)

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

    # --- K-corrections & rest-frame photometry ---
    gmr_obs = g - r
    Kg = kpoly(z, gmr_obs, B_g)
    Kr = kpoly(z, gmr_obs, B_r)
    g0 = g - Kg
    r0 = r - Kr
    gmr0 = g0 - r0
    mu = cosmo.distmod(z).value
    M_r = r0 - mu

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
    Mstar_Msun = ML * Lr_Lsun
    # Make mstar_missing = mhalo_missing to make median halo mass match luminous estimate
    # Mstar figure used to compute luminous mass
    Mstar_Msun_with_stripped = Mstar_Msun * mstar_missing
    # Mstar figure used to compute halo mass
    Mstar_Msun = Mstar_Msun * mhalo_missing

    # --- Disk/Bulge split ---
    Mdisk_Msun = Mstar_Msun_with_stripped * DT
    Mbulge_Msun = Mstar_Msun_with_stripped * (1.0 - DT)

    # --- Angular → physical scales ---
    kpc_per_arcsec = cosmo.angular_diameter_distance(z).to_value(u.kpc) * (
        np.pi / 648000.0
    )

    # --- Component sizes (arcsec) with heuristics if missing ---
    Re_gal_arcsec = Re_arcsec if np.isfinite(Re_arcsec) else float("nan")
    if Re_bulge_arcsec is None:
        Re_bulge_arcsec = (
            (0.6 * Re_gal_arcsec)
            if (np.isfinite(Re_gal_arcsec) and DT > 0.0)
            else Re_gal_arcsec
        )
    Re_bulge_arcsec = (
        float(Re_bulge_arcsec) if Re_bulge_arcsec is not None else float("nan")
    )
    if Rd_exp_arcsec is not None:
        h_disk_arcsec = float(Rd_exp_arcsec)
        Re_disk_arcsec_used = 1.678 * h_disk_arcsec
    elif Re_disk_arcsec is not None:
        Re_disk_arcsec_used = float(Re_disk_arcsec)
        h_disk_arcsec = Re_disk_arcsec_used / 1.678
    else:
        if np.isfinite(Re_gal_arcsec) and DT > 0.0:
            Re_disk_arcsec_used = 1.3 * Re_gal_arcsec
            h_disk_arcsec = Re_disk_arcsec_used / 1.678
        else:
            Re_disk_arcsec_used = float("nan")
            h_disk_arcsec = float("nan")

    Re_ap_arcsec = Re_gal_arcsec
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
                "Mstar_over_M200": (
                    (Mstar_Msun_with_stripped / M200) if M200 > 0 else None
                ),
                "Mbulge_over_M200": (Mbulge_Msun / M200) if M200 > 0 else None,
                "Mdisk_over_M200": (Mdisk_Msun / M200) if M200 > 0 else None,
            }
        )
        hs = halo_structure_from_M200(M200, z, cosmo, c_input, args.c_a, args.c_b)
        shmr["halo_structure"] = hs
        shmr["a_bulge_kpc"] = a_bulge_kpc
        shmr["a_bulge_over_a_halo"] = (
            (a_bulge_kpc / hs["a_halo_kpc"]) if np.isfinite(a_bulge_kpc) else None
        )
    elif args.shmr_model == "umachine":
        med = m200_um_median(Mstar_eff, z, args.gal_type) * (10.0**args.shmr_shift_dex)
        scatter = max(0.0, args.um_scatter_dex)
        p16, p84 = med / (10.0**scatter), med * (10.0**scatter)
        shmr.update(
            {
                "model": "UM/lensing (z~0)",
                "M200c_median_Msun": med,
                "M200c_p16_Msun": p16,
                "M200c_p84_Msun": p84,
                "V200_median_kms": v200_from_m200(med, z, cosmo).to_value(u.km / u.s),
                "V200_p16_kms": v200_from_m200(p16, z, cosmo).to_value(u.km / u.s),
                "V200_p84_kms": v200_from_m200(p84, z, cosmo).to_value(u.km / u.s),
                "Mstar_over_M200_median": (
                    (Mstar_Msun_with_stripped / med) if med > 0 else None
                ),
                "Mstar_over_M200_p16": (
                    (Mstar_Msun_with_stripped / p16) if p16 > 0 else None
                ),
                "Mstar_over_M200_p84": (
                    (Mstar_Msun_with_stripped / p84) if p84 > 0 else None
                ),
                "Mbulge_over_M200_median": (Mbulge_Msun / med) if med > 0 else None,
                "Mbulge_over_M200_p16": (Mbulge_Msun / p16) if p16 > 0 else None,
                "Mbulge_over_M200_p84": (Mbulge_Msun / p84) if p84 > 0 else None,
                "Mdisk_over_M200_median": (Mdisk_Msun / med) if med > 0 else None,
                "Mdisk_over_M200_p16": (Mdisk_Msun / p16) if p16 > 0 else None,
                "Mdisk_over_M200_p84": (Mdisk_Msun / p84) if p84 > 0 else None,
                "scatter_dex": scatter,
            }
        )
        hs_med = halo_structure_from_M200(med, z, cosmo, c_input)
        hs_p16 = halo_structure_from_M200(p16, z, cosmo, c_input)
        hs_p84 = halo_structure_from_M200(p84, z, cosmo, c_input)
        shmr["halo_structure_median"] = hs_med
        shmr["halo_structure_p16"] = hs_p16
        shmr["halo_structure_p84"] = hs_p84
        shmr["a_bulge_kpc"] = a_bulge_kpc
        shmr["Re_bulge_kpc"] = Re_bulge_kpc
        shmr["a_bulge_over_a_halo_median"] = (
            (a_bulge_kpc / hs_med["a_halo_kpc"]) if np.isfinite(a_bulge_kpc) else None
        )
        shmr["a_bulge_over_a_halo_p16"] = (
            (a_bulge_kpc / hs_p16["a_halo_kpc"]) if np.isfinite(a_bulge_kpc) else None
        )
        shmr["a_bulge_over_a_halo_p84"] = (
            (a_bulge_kpc / hs_p84["a_halo_kpc"]) if np.isfinite(a_bulge_kpc) else None
        )
    else:
        med = 1.5e12
        # TODO: This is all code the same as umachine, need to consolidate
        scatter = max(0.0, args.um_scatter_dex)
        p16, p84 = med / (10.0**scatter), med * (10.0**scatter)
        shmr.update(
            {
                "model": "UM/lensing (z~0)",
                "M200c_median_Msun": med,
                "M200c_p16_Msun": p16,
                "M200c_p84_Msun": p84,
                "V200_median_kms": v200_from_m200(med, z, cosmo).to_value(u.km / u.s),
                "V200_p16_kms": v200_from_m200(p16, z, cosmo).to_value(u.km / u.s),
                "V200_p84_kms": v200_from_m200(p84, z, cosmo).to_value(u.km / u.s),
                "Mstar_over_M200_median": (Mstar_Msun / med) if med > 0 else None,
                "Mstar_over_M200_p16": (Mstar_Msun / p16) if p16 > 0 else None,
                "Mstar_over_M200_p84": (Mstar_Msun / p84) if p84 > 0 else None,
                "Mbulge_over_M200_median": (Mbulge_Msun / med) if med > 0 else None,
                "Mbulge_over_M200_p16": (Mbulge_Msun / p16) if p16 > 0 else None,
                "Mbulge_over_M200_p84": (Mbulge_Msun / p84) if p84 > 0 else None,
                "Mdisk_over_M200_median": (Mdisk_Msun / med) if med > 0 else None,
                "Mdisk_over_M200_p16": (Mdisk_Msun / p16) if p16 > 0 else None,
                "Mdisk_over_M200_p84": (Mdisk_Msun / p84) if p84 > 0 else None,
                "scatter_dex": scatter,
            }
        )
        hs_med = halo_structure_from_M200(med, z, cosmo, c_input)
        hs_p16 = halo_structure_from_M200(p16, z, cosmo, c_input)
        hs_p84 = halo_structure_from_M200(p84, z, cosmo, c_input)
        shmr["halo_structure_median"] = hs_med
        shmr["halo_structure_p16"] = hs_p16
        shmr["halo_structure_p84"] = hs_p84
        shmr["a_bulge_kpc"] = a_bulge_kpc
        shmr["Re_bulge_kpc"] = Re_bulge_kpc
        shmr["a_bulge_over_a_halo_median"] = (
            (a_bulge_kpc / hs_med["a_halo_kpc"]) if np.isfinite(a_bulge_kpc) else None
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
            "z": z,
            "g": g,
            "r": r,
            "dereddened": True,
            "Reff_r_arcsec": (
                None if not np.isfinite(Re_gal_arcsec) else float(Re_gal_arcsec)
            ),
            "Re_bulge_arcsec": (
                None if not np.isfinite(Re_bulge_arcsec) else float(Re_bulge_arcsec)
            ),
            "Re_disk_arcsec": (
                None
                if not np.isfinite(
                    Re_disk_arcsec_used := (
                        Re_disk_arcsec if Re_disk_arcsec is not None else float("nan")
                    )
                )
                else float(Re_disk_arcsec_used)
            ),
            "Rd_exp_arcsec": (
                None if not np.isfinite(h_disk_arcsec) else float(h_disk_arcsec)
            ),
            "sersic_n": sersic_n,
            "b_over_a": b_over_a,
            "a_over_b": a_over_b,
            "D_over_T": DT,
            "c_input": c_input,
        },
        "cosmology": {"model": cosmo_label, "H_z_km_s_Mpc": Hz_val},
        "k_corrections": {"Kg_mag": Kg, "Kr_mag": Kr, "g_minus_r_obs": gmr_obs},
        "rest_frame": {"(g-r)_0": gmr0, "M_r_mag": M_r},
        "luminosity": {"L_r_Lsun": Lr_Lsun, "M_r_sun_mag": args.M_r_sun},
        "stellar_mass": {
            "M_over_L_r": ML,
            "M_star_Msun": Mstar_Msun,
            "M_star_Msun_with_missing": Mstar_Msun_with_stripped,
            "Stellar Missing": mstar_missing,
            "Halo Missing": mhalo_missing,
            "ML_method": ml_name,
            "D_over_T": DT,
            "M_bulge_Msun": Mbulge_Msun,
            "M_disk_Msun": Mdisk_Msun,
        },
        "halo_mass_SHMR": shmr,
    }

    # ------------------ Print (human-readable) ------------------
    # Formatters
    s3 = sig3_noexp_str
    s5 = mag5_noexp_str

    if args.json:
        out_fmt = copy.deepcopy(out)
        # First convert Msun/Lsun fields
        format_msun_lsun_inplace(out_fmt)
        # Then format the rest: magnitudes 5-sig, others 3-sig
        format_other_numbers_inplace(out_fmt)
        print(json.dumps(out_fmt, indent=2))
    else:
        print("\n=== Inputs ===")
        print(f"z = {s3(z)}, g = {s5(g)}, r = {s5(r)}  (dereddened)")
        if np.isfinite(Re_gal_arcsec):
            print(f"Reff_r (galaxy) = {s3(Re_gal_arcsec)} arcsec")
        if np.isfinite(Re_bulge_arcsec):
            print(f"Re_bulge = {s3(Re_bulge_arcsec)} arcsec")
        if np.isfinite(h_disk_arcsec):
            print(
                f"Rd_exp (disk scale h) = {s3(h_disk_arcsec)} arcsec  (Re_disk ≈ {s3(1.678 * h_disk_arcsec)} arcsec)"
            )
        elif Re_disk_arcsec is not None:
            print(f"Re_disk = {s3(float(Re_disk_arcsec))} arcsec")
        print(f"Sérsic n (bulge) = {s3(sersic_n)}")
        if b_over_a is not None:
            print(f"b/a = {s3(b_over_a)}  →  a/b = {s3(a_over_b)}")
        print(f"D/T (disk-to-total luminous mass) = {s3(DT)}")
        print(f"Cosmology: {cosmo_label}")
        print(f"H(z) = {s3(Hz_val)} km s^-1 Mpc^-1")

        print("\n=== K-corrections (Chilingarian+2010; Saulder+2013) ===")
        print(f"K_g = {s5(Kg)} mag,  K_r = {s5(Kr)} mag   (obs (g-r)={s5(gmr_obs)})")

        print("\n=== Rest-frame & luminosity ===")
        print(f"(g-r)_0 = {s5(gmr0)}")
        print(f"M_r = {s5(M_r)} mag")
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

        print("\n=== Halo mass from SHMR ===")
        s = out["halo_mass_SHMR"]
        if s["icl_details"]["mode"] == "manual":
            print(f"ICL: manual icl_frac = {s3(s['icl_used'])}")
        elif s["icl_details"]["mode"] == "auto":
            baseM = s["icl_details"].get("baseM200_for_prior", float("nan"))
            i1 = s["icl_details"].get("icl1", float("nan"))
            i2 = s["icl_details"].get("icl2", float("nan"))
            print(
                f"ICL: auto prior → icl_frac = {s3(s['icl_used'])}  (baseM={sci3(baseM)}, iter: {s3(i1)}→{s3(i2)})"
            )
        else:
            print("ICL: off (icl_frac = 0.000)")

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
                    f"  R200 = {s3(hs['R200_kpc'])} kpc,  c = {s3(hs['c_used'])},  r_s = {s3(hs['r_s_kpc'])} kpc,  a_halo = {s3(hs['a_halo_kpc'])} kpc"
                )
                if np.isfinite(a_bulge_kpc):
                    print(f"  a_bulge/a_halo = {s3(s['a_bulge_over_a_halo'])}")
        else:
            print(
                f"UM/lensing (median): M200c = {sci3(s['M200c_median_Msun'])} Msun,  V200 = {s3(s['V200_median_kms'])} km/s"
            )
            print(
                f"  M*/M200 = {s3(s['Mstar_over_M200_median'])}  "
                f"(68%: {s3(s['Mstar_over_M200_p16'])}–{s3(s['Mstar_over_M200_p84'])})"
            )
            print(
                f"  M_bulge/M200 = {s3(s['Mbulge_over_M200_median'])}  "
                f"(68%: {s3(s['Mbulge_over_M200_p16'])}–{s3(s['Mbulge_over_M200_p84'])})"
            )
            print(
                f"  M_disk/M200 = {s3(s['Mdisk_over_M200_median'])}  "
                f"(68%: {s3(s['Mdisk_over_M200_p16'])}–{s3(s['Mdisk_over_M200_p84'])})"
            )
            hs = s["halo_structure_median"]
            print(
                f"  R200 (median) = {s3(hs['R200_kpc'])} kpc,  c = {s3(hs['c_used'])},  r_s = {s3(hs['r_s_kpc'])} kpc,  a_halo = {s3(hs['a_halo_kpc'])} kpc"
            )
            if np.isfinite(a_bulge_kpc):
                print(
                    f"  a_bulge/a_halo (median) = {s3(s['a_bulge_over_a_halo_median'])}"
                )
        print()


if __name__ == "__main__":
    # There is a dependency on UniverseMachine DR1 if you use this as the shmr described in:
    # Behroozi P., Wechsler R. H., Hearin A. P., Conroy C., 2019, MNRAS, 488, 289
    #
    # Download this from: https://www.peterbehroozi.com/data.html
    #
    main()
