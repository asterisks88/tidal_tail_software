#!/usr/bin/env python3
"""
build_leda175897_agama_host_schwarzschild_v10_spin.py

Construct a bounded, two-component Schwarzschild AGAMA model of LEDA175897:

  * flattened deprojected Sérsic stellar spheroid;
  * flattened, smoothly truncated Hernquist-like dark-matter halo;
  * bounded stellar and halo orbit libraries;
  * direct GADGET-4 HDF5 output.

GalIC-compatible halo mass normalization
----------------------------------------
The final GalIC host startup output reported

    M200 = 3723.97 code mass units = 3.72397e13 Msun,

and GalIC treated this as the total live galaxy mass. Its bulge mass was the
stellar fraction MB*M200, leaving the remainder as the asymptotic Hernquist
halo mass.

The default v9 normalization therefore sets

    M_halo,finite + M_star = M_galaxy,GalIC,

where M_halo,finite is the total mass of the smoothly truncated AGAMA halo.
For M_star=2.08e11 Msun, the target finite halo mass is

    3.72397e13 - 2.08e11 = 3.70317e13 Msun.

Three normalization modes are available:

  galic-live-total (default)
      Match the finite HDF5 halo mass plus stellar mass to the old GalIC total
      live mass. This is the closest match in total particle mass.

  galic-profile
      Match the density normalization of an untruncated Hernquist-like halo
      whose asymptotic halo mass is M_galaxy,GalIC-M_star, then apply the finite
      cutoff without renormalizing. This best preserves the old GalIC inner
      Hernquist force profile, but the finite HDF5 total mass is lower.

  m200-enclosed
      Retain the previous v6 convention in which the halo alone is normalized
      so M_halo(<r200)=M200 and the stars are added separately.

The mode is selected with --halo-mass-normalization.

Optional GalIC parameter input
------------------------------
A host GalIC parameter file may be supplied as the optional second positional
argument after model_json (or with --galic-param-file).  Explicit command-line
overrides take precedence.  A valid LEDA175897 GalIC file supplies CC, V200,
LAMBDA, MB, HaloStretch, BulgeStretch, particle counts, velocity-structure
metadata, streaming parameters, and softening.  V200 is converted using
--galic-hubble-kms-kpc (default 0.0699) and the GalIC internal gravitational
constant corresponding to the reported startup units.


Run configuration
-----------------
All positional inputs and command-line options may be supplied through

    --config run_config.json

The file may be JSON or a Python-literal dictionary. It may be flat or contain
a nested ``parameters`` mapping. Configuration keys use argparse destination
names, e.g. ``star_orbits`` or ``star-orbits``. Explicit command-line values
override configuration values. Relative input and output paths are resolved
from the configuration file's directory.

Halo spin
---------
The optional halo-spin calibration matches the global angular momentum to
GalIC's Peebles-lambda normalization. Enable it with either

    --halo-spin-from-galic-lambda

or

    --halo-spin-lambda VALUE

The method changes only prograde/retrograde signs of v_phi in the final
axisymmetric halo sample. It does not change particle positions, speed
magnitudes, energies, density, or even velocity moments.

GalIC's LAMBDA and HaloStreamingVelocityParameter are distinct quantities.
LAMBDA supplies the global angular-momentum scale and disk scale length,
whereas the streaming parameter controls the local mean-v_phi field.

Extreme-apocentre filtering
---------------------------
Candidate orbit initial conditions are rejected before integration when their
estimated apocentre or energy-based maximum radius exceeds the component
limit. Final particles are validated after sampling and are never clipped.

Production example matching the old GalIC total live mass
---------------------------------------------------------
python build_leda175897_agama_host_schwarzschild_v10_spin.py \
    LEDA175897_Legacy_test_model.json \
    LEDA175897_20260228_legacy_dered_med_super_oblate_bulge_zerostream_oblate_halo_lambda035_ml_0pt0_lum-1pt0_halo0pt0_zerostream_medium-hdf5.param \
    --photometry-json LEDA175897_Legacy_param.json \
    --output-dir agama_leda175897_schwarzschild_v10_spin \
    --halo-mass-normalization galic-live-total \
    --galic-total-galaxy-mass-msun 3.72397e13 \
    --halo-q 0.666666 \
    --n-star 500000 \
    --n-halo 1000000 \
    --star-orbits 20000 \
    --halo-orbits 40000 \
    --star-periods 100 \
    --halo-periods 100 \
    --star-target-bins 50 \
    --star-target-lmax 8 \
    --halo-target-bins 50 \
    --halo-target-lmax 6 \
    --star-max-apocentre-kpc 300 \
    --halo-max-apocentre-kpc 2760 \
    --density-constraint-mode soft \
    --halo-density-constraint-mode soft \
    --density-tolerance 0.01 \
    --halo-density-tolerance 0.015 \
    --orbit-regularization 1 \
    --halo-orbit-regularization 10

Units
-----
AGAMA: kpc, km/s, Msun.
GADGET HDF5 mass unit: --gadget-mass-unit-msun, default 1e10 Msun.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np


GALIC_G_KPC_KMS2_MSUN = 4.30187e-6
DEFAULT_GALIC_HUBBLE_KMS_KPC = 0.0699
DEFAULT_GALIC_TOTAL_GALAXY_MASS_MSUN = 3.72397e13


@dataclass
class HostParameters:
    name: str
    redshift: float

    stellar_mass_msun: float
    sersic_n: float
    re_circularized_kpc: float
    re_major_kpc: float
    projected_q: float
    intrinsic_q: float
    inclination_deg: float

    m200_msun: float
    r200_kpc: float
    concentration: float
    halo_scale_radius_nfw_kpc: float
    halo_scale_radius_hernquist_kpc: float

    halo_mass_normalization: str
    galic_total_galaxy_mass_msun: float
    target_halo_mass_msun: float
    halo_total_mass_hernquist_msun: float
    halo_q: float

    galic_parameter_file: str | None
    galic_file_usable_for_host: bool
    galic_mismatch_reason: str | None
    galic_hubble_kms_kpc: float
    galic_v200_kms: float | None
    galic_concentration: float | None
    galic_lambda: float | None
    galic_md: float | None
    galic_mb: float | None
    galic_mbh: float | None
    galic_halo_stretch: float | None
    galic_bulge_stretch: float | None
    galic_halo_velocity_structure: int | None
    galic_bulge_velocity_structure: int | None
    galic_halo_streaming: float | None
    galic_bulge_streaming: float | None
    galic_softening_kpc: float | None
    stellar_mass_source: str
    stellar_q_source: str
    halo_q_source: str

    n_star: int
    n_halo: int
    random_seed: int
    gadget_mass_unit_msun: float


def read_mapping(path: str | Path) -> dict[str, Any]:
    """
    Read either strict JSON or a Python-style dictionary containing comments.

    The user's photometry parameter file is a Python dictionary literal with
    '#' comments even though its filename ends in .json.
    """
    path = Path(path)
    text = path.read_text()

    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    try:
        value = ast.literal_eval(text)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Could not parse {path} as JSON or a dictionary literal") from exc

    if not isinstance(value, dict):
        raise ValueError(f"{path} does not contain a dictionary")
    return value



def read_galic_parameters(path: str | Path) -> dict[str, Any]:
    """Parse GalIC's whitespace-delimited parameter format with '%' comments."""
    result: dict[str, Any] = {}
    for raw in Path(path).read_text().splitlines():
        line = raw.split("%", 1)[0].strip()
        if not line:
            continue
        fields = line.split(None, 1)
        if len(fields) != 2:
            continue
        key, value_text = fields
        value_text = value_text.strip()
        try:
            if any(ch in value_text.lower() for ch in (".", "e")):
                value: Any = float(value_text)
            else:
                value = int(value_text)
        except ValueError:
            value = value_text
        result[key] = value
    return result


def as_optional_float(mapping: dict[str, Any], key: str) -> float | None:
    if key not in mapping:
        return None
    try:
        value = float(mapping[key])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def as_optional_int(mapping: dict[str, Any], key: str) -> int | None:
    if key not in mapping:
        return None
    try:
        return int(mapping[key])
    except (TypeError, ValueError):
        return None


def stretch_to_axis_ratio(stretch: float | None, fallback: float) -> float:
    """Map GalIC Stretch>1 oblate convention to AGAMA c/a~=1/Stretch."""
    if stretch is None or not np.isfinite(stretch) or stretch <= 0:
        return fallback
    return min(1.0, 1.0 / stretch) if stretch >= 1.0 else min(1.0, stretch)


def resolve_galic_path(args: argparse.Namespace) -> str | None:
    positional = getattr(args, "galic_param", None)
    option = getattr(args, "galic_param_file", None)
    if positional and option:
        if Path(positional).expanduser().resolve() != Path(option).expanduser().resolve():
            raise ValueError(
                "GalIC parameter file was supplied both positionally and with "
                "--galic-param-file, but the paths differ"
            )
    selected = option or positional
    return str(Path(selected).expanduser().resolve()) if selected else None


def validate_host_galic_parameters(
    galic: dict[str, Any],
) -> tuple[bool, str | None]:
    if not galic:
        return False, "no GalIC parameter file supplied"
    reasons: list[str] = []
    output_dir = str(galic.get("OutputDir", ""))
    if "LEDA175897" not in output_dir.upper():
        reasons.append("OutputDir does not identify LEDA175897")
    md = as_optional_float(galic, "MD")
    mb = as_optional_float(galic, "MB")
    nh = as_optional_int(galic, "N_HALO")
    nb = as_optional_int(galic, "N_BULGE")
    if md is None or abs(md) > 1e-12:
        reasons.append("MD is absent or nonzero")
    if mb is None or mb <= 0:
        reasons.append("MB is absent or non-positive")
    if nh is None or nh <= 0:
        reasons.append("N_HALO is absent or non-positive")
    if nb is None or nb <= 0:
        reasons.append("N_BULGE is absent or non-positive")
    if as_optional_float(galic, "CC") in (None, 0.0):
        reasons.append("CC is absent or zero")
    if as_optional_float(galic, "V200") in (None, 0.0):
        reasons.append("V200 is absent or zero")
    return (not reasons), ("; ".join(reasons) if reasons else None)

def as_float(value: Any, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value for {description}: {value!r}") from exc
    if not np.isfinite(result):
        raise ValueError(f"Non-finite value for {description}: {result}")
    return result


def get_nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            raise KeyError(".".join(keys))
        value = value[key]
    return value


def oblate_intrinsic_q(projected_q: float, inclination_deg: float) -> float:
    """
    Infer intrinsic c/a for an oblate axisymmetric body:

        q_obs^2 = cos(i)^2 + q_int^2 sin(i)^2.

    i=90 deg is edge-on.
    """
    inclination = math.radians(inclination_deg)
    sin2 = math.sin(inclination) ** 2
    cos2 = math.cos(inclination) ** 2
    numerator = projected_q**2 - cos2

    if sin2 <= 0 or numerator <= 0:
        minimum = math.degrees(math.acos(projected_q))
        raise ValueError(
            f"projected q={projected_q:.6g} requires inclination > "
            f"{minimum:.3f} deg for an oblate deprojection"
        )

    q_intrinsic = math.sqrt(numerator / sin2)
    if not (0 < q_intrinsic <= 1):
        raise ValueError(f"Derived invalid intrinsic q={q_intrinsic}")
    return q_intrinsic


def derive_parameters(args: argparse.Namespace) -> HostParameters:
    model = read_mapping(args.model_json)
    phot = read_mapping(args.photometry_json) if args.photometry_json else {}

    galic_path = resolve_galic_path(args)
    galic = read_galic_parameters(galic_path) if galic_path else {}
    galic_usable, galic_reason = validate_host_galic_parameters(galic)
    if galic_path and not galic_usable:
        print(
            f"Warning: GalIC parameter file will not provide defaults: {galic_reason}",
            file=sys.stderr,
        )

    name = str(get_nested(model, "inputs", "name")).strip().rstrip(",")
    redshift = as_float(get_nested(model, "inputs", "z"), "redshift")

    model_stellar_mass = as_float(
        get_nested(model, "stellar_mass", "M_star_Msun"), "stellar mass"
    )

    galic_hubble = float(args.galic_hubble_kms_kpc)
    galic_v200 = as_optional_float(galic, "V200") if galic_usable else None
    galic_cc = as_optional_float(galic, "CC") if galic_usable else None
    galic_md = as_optional_float(galic, "MD") if galic_usable else None
    galic_mb = as_optional_float(galic, "MB") if galic_usable else None
    galic_mbh = as_optional_float(galic, "MBH") if galic_usable else None

    derived_galic_total_mass: float | None = None
    derived_galic_r200: float | None = None
    if galic_v200 is not None:
        derived_galic_total_mass = galic_v200**3 / (
            10.0 * GALIC_G_KPC_KMS2_MSUN * galic_hubble
        )
        derived_galic_r200 = galic_v200 / (10.0 * galic_hubble)

    galic_total_galaxy_mass = (
        float(args.galic_total_galaxy_mass_msun)
        if args.galic_total_galaxy_mass_msun is not None
        else derived_galic_total_mass
        if derived_galic_total_mass is not None
        else DEFAULT_GALIC_TOTAL_GALAXY_MASS_MSUN
    )

    stellar_mass_source = str(args.stellar_mass_source)
    if stellar_mass_source == "auto":
        stellar_mass_source = "galic" if galic_usable else "model"
    if stellar_mass_source == "galic":
        if not galic_usable or galic_mb is None:
            raise ValueError(
                "--stellar-mass-source=galic requires a valid host GalIC file"
            )
        stellar_mass = galic_mb * galic_total_galaxy_mass
    else:
        stellar_mass = model_stellar_mass

    re_circ = (
        float(args.re_circularized_kpc)
        if args.re_circularized_kpc is not None
        else as_float(get_nested(model, "sizes", "Reff_r_kpc"), "circularized Re")
    )

    if args.sersic_n is not None:
        sersic_n = float(args.sersic_n)
    elif "sersic_n" in phot:
        sersic_n = as_float(phot["sersic_n"], "photometry Sérsic index")
    else:
        sersic_n = as_float(get_nested(model, "inputs", "sersic_n"), "Sérsic index")

    if args.projected_q is not None:
        projected_q = float(args.projected_q)
    elif "b/a" in phot:
        projected_q = as_float(phot["b/a"], "photometry projected b/a")
    else:
        projected_q = as_float(get_nested(model, "inputs", "b_over_a"), "projected b/a")

    if not 0 < projected_q <= 1:
        raise ValueError(f"Projected b/a must be in (0,1], got {projected_q}")
    if sersic_n <= 0:
        raise ValueError(f"Sérsic n must be positive, got {sersic_n}")
    if re_circ <= 0:
        raise ValueError(f"Circularized Re must be positive, got {re_circ}")

    re_major = (
        float(args.re_major_kpc)
        if args.re_major_kpc is not None
        else re_circ / math.sqrt(projected_q)
    )

    inclination = float(args.inclination_deg)
    photometric_intrinsic_q = oblate_intrinsic_q(projected_q, inclination)
    galic_bulge_stretch = (
        as_optional_float(galic, "BulgeStretch") if galic_usable else None
    )
    if args.stellar_q is not None:
        intrinsic_q = float(args.stellar_q)
        stellar_q_source = "command-line --stellar-q"
    elif galic_usable and args.use_galic_stellar_stretch:
        intrinsic_q = stretch_to_axis_ratio(
            galic_bulge_stretch, photometric_intrinsic_q
        )
        stellar_q_source = "GalIC BulgeStretch"
    else:
        intrinsic_q = photometric_intrinsic_q
        stellar_q_source = "photometric oblate deprojection"
    if not 0 < intrinsic_q <= 1:
        raise ValueError(f"Intrinsic stellar q must be in (0,1], got {intrinsic_q}")

    halo_info = get_nested(model, "halo_mass_SHMR")
    halo_structure = get_nested(halo_info, "halo_structure_target")
    model_m200 = as_float(halo_info["M200c_target_Msun"], "M200")
    model_r200 = as_float(halo_structure["R200_kpc"], "r200")
    model_concentration = as_float(halo_structure["c_used"], "halo concentration")
    model_rs = as_float(halo_structure["r_s_kpc"], "NFW scale radius")
    model_a = as_float(halo_structure["a_halo_kpc"], "Hernquist halo scale radius")

    m200 = (
        float(args.m200_msun)
        if args.m200_msun is not None
        else galic_total_galaxy_mass
        if galic_usable or args.galic_total_galaxy_mass_msun is not None
        else model_m200
    )
    r200 = (
        float(args.r200_kpc)
        if args.r200_kpc is not None
        else derived_galic_r200
        if galic_usable and derived_galic_r200 is not None
        else model_r200
    )
    concentration = (
        float(args.concentration)
        if args.concentration is not None
        else galic_cc
        if galic_usable and galic_cc is not None
        else model_concentration
    )
    rs_nfw = r200 / concentration
    calculated_a = rs_nfw * math.sqrt(
        2.0 * (math.log1p(concentration) - concentration / (1.0 + concentration))
    )
    a_halo = (
        float(args.halo_scale_radius_kpc)
        if args.halo_scale_radius_kpc is not None
        else calculated_a
        if galic_usable
        else model_a
    )
    if not galic_usable and args.r200_kpc is None and args.concentration is None:
        rs_nfw = model_rs

    galic_halo_stretch = (
        as_optional_float(galic, "HaloStretch") if galic_usable else None
    )
    if args.halo_q is not None:
        halo_q = float(args.halo_q)
        halo_q_source = "command-line --halo-q"
    elif galic_usable:
        halo_q = stretch_to_axis_ratio(galic_halo_stretch, 0.666666)
        halo_q_source = "GalIC HaloStretch"
    else:
        halo_q = 0.666666
        halo_q_source = "fallback"

    def choose_count(cli_value: int | None, galic_key: str, fallback: int) -> int:
        if cli_value is not None:
            return int(cli_value)
        if galic_usable:
            value = as_optional_int(galic, galic_key)
            if value is not None and value > 0:
                return value
        return fallback

    n_star = choose_count(args.n_star, "N_BULGE", 500_000)
    n_halo = choose_count(args.n_halo, "N_HALO", 1_000_000)

    galic_bulge_streaming = (
        as_optional_float(galic, "BulgeStreamingVelocityParameter")
        if galic_usable else None
    )
    galic_halo_streaming = (
        as_optional_float(galic, "HaloStreamingVelocityParameter")
        if galic_usable else None
    )
    if args.star_kappa is None:
        args.star_kappa = galic_bulge_streaming or 0.0
    if args.halo_kappa is None:
        args.halo_kappa = galic_halo_streaming or 0.0
    if args.star_beta is None:
        galic_type = as_optional_int(galic, "TypeOfBulgeVelocityStructure")
        beta = as_optional_float(galic, "BulgeBetaParameter")
        args.star_beta = beta if galic_usable and galic_type == 1 and beta is not None else 0.0
    if args.halo_beta is None:
        galic_type = as_optional_int(galic, "TypeOfHaloVelocityStructure")
        beta = as_optional_float(galic, "HaloBetaParameter")
        args.halo_beta = beta if galic_usable and galic_type == 1 and beta is not None else 0.0
    if args.enforce_nonrotation is None:
        args.enforce_nonrotation = (
            galic_bulge_streaming is None or abs(galic_bulge_streaming) < 1e-12
        )
    if args.enforce_halo_nonrotation is None:
        args.enforce_halo_nonrotation = (
            galic_halo_streaming is None or abs(galic_halo_streaming) < 1e-12
        )

    normalization = str(args.halo_mass_normalization)
    if args.galic_halo_mass_msun is not None:
        galic_halo_mass = float(args.galic_halo_mass_msun)
    else:
        galic_halo_mass = galic_total_galaxy_mass - stellar_mass
    if galic_halo_mass <= 0:
        raise ValueError(
            "The GalIC-compatible halo mass must be positive: "
            f"Mgal={galic_total_galaxy_mass:.6e}, Mstar={stellar_mass:.6e}"
        )

    if normalization in ("galic-live-total", "galic-profile"):
        m_halo_reference = galic_halo_mass
    elif normalization == "m200-enclosed":
        m_halo_reference = m200 * (1.0 + a_halo / r200) ** 2
    else:
        raise ValueError(f"Unknown halo mass normalization mode: {normalization}")

    args._galic_parameters = galic
    args._galic_path = galic_path
    args._galic_usable = galic_usable
    args._galic_reason = galic_reason
    args._model_stellar_mass_msun = model_stellar_mass
    args._model_halo_parameters = {
        "M200_msun": model_m200,
        "r200_kpc": model_r200,
        "concentration": model_concentration,
        "rs_kpc": model_rs,
        "a_halo_kpc": model_a,
    }

    return HostParameters(
        name=name,
        redshift=redshift,
        stellar_mass_msun=stellar_mass,
        sersic_n=sersic_n,
        re_circularized_kpc=re_circ,
        re_major_kpc=re_major,
        projected_q=projected_q,
        intrinsic_q=intrinsic_q,
        inclination_deg=inclination,
        m200_msun=m200,
        r200_kpc=r200,
        concentration=concentration,
        halo_scale_radius_nfw_kpc=rs_nfw,
        halo_scale_radius_hernquist_kpc=a_halo,
        halo_mass_normalization=normalization,
        galic_total_galaxy_mass_msun=galic_total_galaxy_mass,
        target_halo_mass_msun=galic_halo_mass,
        halo_total_mass_hernquist_msun=m_halo_reference,
        halo_q=halo_q,
        galic_parameter_file=galic_path,
        galic_file_usable_for_host=galic_usable,
        galic_mismatch_reason=galic_reason,
        galic_hubble_kms_kpc=galic_hubble,
        galic_v200_kms=galic_v200,
        galic_concentration=galic_cc,
        galic_lambda=as_optional_float(galic, "LAMBDA") if galic_usable else None,
        galic_md=galic_md,
        galic_mb=galic_mb,
        galic_mbh=galic_mbh,
        galic_halo_stretch=galic_halo_stretch,
        galic_bulge_stretch=galic_bulge_stretch,
        galic_halo_velocity_structure=(
            as_optional_int(galic, "TypeOfHaloVelocityStructure") if galic_usable else None
        ),
        galic_bulge_velocity_structure=(
            as_optional_int(galic, "TypeOfBulgeVelocityStructure") if galic_usable else None
        ),
        galic_halo_streaming=galic_halo_streaming,
        galic_bulge_streaming=galic_bulge_streaming,
        galic_softening_kpc=(
            as_optional_float(galic, "Softening") if galic_usable else None
        ),
        stellar_mass_source=stellar_mass_source,
        stellar_q_source=stellar_q_source,
        halo_q_source=halo_q_source,
        n_star=n_star,
        n_halo=n_halo,
        random_seed=int(args.seed),
        gadget_mass_unit_msun=float(args.gadget_mass_unit_msun),
    )


def validate_halo_conversion(par: HostParameters) -> dict[str, float | str]:
    f_c = math.log1p(par.concentration) - par.concentration / (
        1 + par.concentration
    )
    rs_from_c = par.r200_kpc / par.concentration
    a_from_nfw = rs_from_c * math.sqrt(2 * f_c)

    reference_m_r200 = (
        par.halo_total_mass_hernquist_msun
        * par.r200_kpc**2
        / (par.r200_kpc + par.halo_scale_radius_hernquist_kpc) ** 2
    )
    return {
        "halo_mass_normalization": par.halo_mass_normalization,
        "rs_from_r200_over_c_kpc": rs_from_c,
        "a_hernquist_from_nfw_matching_kpc": a_from_nfw,
        "reference_untruncated_halo_mass_msun":
            par.halo_total_mass_hernquist_msun,
        "reference_untruncated_M_within_r200_msun": reference_m_r200,
        "galic_total_galaxy_mass_msun":
            par.galic_total_galaxy_mass_msun,
        "galic_compatible_halo_mass_msun":
            par.target_halo_mass_msun,
        "galic_compatible_stars_plus_halo_msun":
            par.stellar_mass_msun + par.target_halo_mass_msun,
        "provided_vs_derived_rs_fractional_difference":
            (par.halo_scale_radius_nfw_kpc - rs_from_c) / rs_from_c,
        "provided_vs_derived_a_fractional_difference":
            (par.halo_scale_radius_hernquist_kpc - a_from_nfw) / a_from_nfw,
    }


def write_derived_config(
    output_dir: Path,
    par: HostParameters,
    halo_checks: dict[str, Any],
    args: argparse.Namespace,
) -> Path:
    output = output_dir / "LEDA175897_agama_derived_config.json"
    payload = {
        "derived_parameters": asdict(par),
        "halo_conversion_checks": halo_checks,
        "model_choices": {
            "stellar_density": "AGAMA Sersic deprojection",
            "stellar_scale_radius_definition": (
                "observed major-axis half-light radius derived from circularized Re"
            ),
            "halo_density": (
                "finite Spheroid(alpha=1,beta=4,gamma=1) with smooth cutoff"
            ),
            "halo_normalization": par.halo_mass_normalization,
            "galic_total_galaxy_mass_msun":
                par.galic_total_galaxy_mass_msun,
            "target_halo_mass_msun": par.target_halo_mass_msun,
            "stellar_df": "Schwarzschild orbit superposition",
            "halo_df": "Schwarzschild orbit superposition",
            "streaming": (
                "halo GalIC-equivalent Peebles lambda sign bias"
                if getattr(args, "_halo_spin_target_lambda", None) is not None
                else "none"
            ),
            "halo_spin": {
                "source": getattr(args, "_halo_spin_source", "none"),
                "target_lambda_unsigned": getattr(
                    args, "_halo_spin_target_lambda", None
                ),
                "target_lambda_signed": getattr(
                    args, "_halo_spin_target_signed_lambda", None
                ),
                "method": (
                    "phase-independent prograde/retrograde v_phi sign bias"
                ),
            },
            "stellar_mass_source": par.stellar_mass_source,
            "stellar_q_source": par.stellar_q_source,
            "halo_q_source": par.halo_q_source,
        },
        "galic_mapping": {
            "file_usable_for_host": par.galic_file_usable_for_host,
            "mismatch_reason": par.galic_mismatch_reason,
            "hubble_kms_kpc": par.galic_hubble_kms_kpc,
            "parsed_parameters": getattr(args, "_galic_parameters", {}),
            "model_stellar_mass_msun": getattr(args, "_model_stellar_mass_msun", None),
            "model_halo_parameters": getattr(args, "_model_halo_parameters", {}),
        },
        "run_configuration": resolved_run_configuration(args),
        "source_files": {
            "model_json": str(Path(args.model_json).resolve()),
            "photometry_json": (
                str(Path(args.photometry_json).resolve())
                if args.photometry_json
                else None
            ),
            "galic_param": par.galic_parameter_file,
        },
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return output


def print_parameter_summary(
    par: HostParameters,
    checks: dict[str, Any],
    args: argparse.Namespace,
) -> None:
    print("\nDerived LEDA175897 AGAMA model")
    print("--------------------------------")
    print(f"Stellar mass             : {par.stellar_mass_msun:.6e} Msun")
    print(f"Sérsic index             : {par.sersic_n:.6f}")
    print(f"Re,circularized          : {par.re_circularized_kpc:.6f} kpc")
    print(f"Re,major                 : {par.re_major_kpc:.6f} kpc")
    print(f"Projected b/a            : {par.projected_q:.6f}")
    print(f"Intrinsic c/a            : {par.intrinsic_q:.6f}")
    print(f"Inclination assumption   : {par.inclination_deg:.3f} deg")
    print()
    print(f"Reference M200           : {par.m200_msun:.6e} Msun")
    print(f"r200                     : {par.r200_kpc:.6f} kpc")
    print(f"c200                     : {par.concentration:.6f}")
    print(f"NFW rs                   : {par.halo_scale_radius_nfw_kpc:.6f} kpc")
    print(f"Hernquist a              : {par.halo_scale_radius_hernquist_kpc:.6f} kpc")
    print(f"Halo intrinsic c/a       : {par.halo_q:.6f}")
    print()
    print(f"Halo normalization       : {par.halo_mass_normalization}")
    print(
        f"Old GalIC total galaxy M : "
        f"{par.galic_total_galaxy_mass_msun:.6e} Msun"
    )
    print(
        f"Target GalIC halo mass   : "
        f"{par.target_halo_mass_msun:.6e} Msun"
    )
    print(
        f"Reference halo mass      : "
        f"{par.halo_total_mass_hernquist_msun:.6e} Msun"
    )
    print(
        f"Reference M_H(<r200)     : "
        f"{checks['reference_untruncated_M_within_r200_msun']:.6e} Msun"
    )
    print(f"Particles (stars, halo)  : {par.n_star}, {par.n_halo}")
    if getattr(args, "_halo_spin_target_lambda", None) is None:
        print("Halo spin target         : none")
    else:
        print(
            f"Halo spin target         : "
            f"{args._halo_spin_target_signed_lambda:.6f} "
            f"({args._halo_spin_source})"
        )
    print(f"Stellar mass source      : {par.stellar_mass_source}")
    print(f"Stellar q source         : {par.stellar_q_source}")
    print(f"Halo q source            : {par.halo_q_source}")
    if par.galic_parameter_file:
        status = "accepted" if par.galic_file_usable_for_host else "not applied"
        print(f"GalIC parameter file     : {status}")
        print(f"GalIC file path          : {par.galic_parameter_file}")
        if par.galic_file_usable_for_host:
            print(
                f"GalIC V200, H, CC         : {par.galic_v200_kms}, "
                f"{par.galic_hubble_kms_kpc}, {par.galic_concentration}"
            )
            print(
                f"GalIC lambda, MB          : {par.galic_lambda}, {par.galic_mb}"
            )
            print(
                f"GalIC stretches           : halo={par.galic_halo_stretch}, "
                f"bulge={par.galic_bulge_stretch}"
            )
        elif par.galic_mismatch_reason:
            print(f"GalIC mismatch            : {par.galic_mismatch_reason}")


def import_agama():
    try:
        import agama  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "AGAMA is not importable in this Python environment. Install the "
            "GalacticDynamics-Oxford AGAMA package from its official repository, "
            "then rerun this script. The PyPI package named 'agama' is unrelated."
        ) from exc
    return agama


def make_densities(
    agama,
    par: HostParameters,
    args: argparse.Namespace,
):
    density_stars = agama.Density(
        type="Sersic",
        mass=par.stellar_mass_msun,
        scaleRadius=par.re_major_kpc,
        sersicIndex=par.sersic_n,
        axisRatioY=1.0,
        axisRatioZ=par.intrinsic_q,
    )

    cutoff_radius = (
        float(args.halo_cutoff_radius_kpc)
        if args.halo_cutoff_radius_kpc is not None
        else float(args.halo_cutoff_r200_multiple) * par.r200_kpc
    )
    cutoff_strength = float(args.halo_cutoff_strength)

    if cutoff_radius <= par.r200_kpc:
        print(
            "Warning: halo cutoff radius is at or inside r200; the cutoff will "
            "materially modify the density near the virial radius.",
            file=sys.stderr,
        )

    base_params = dict(
        type="Spheroid",
        densityNorm=1.0,
        alpha=1.0,
        beta=4.0,
        gamma=1.0,
        scaleRadius=par.halo_scale_radius_hernquist_kpc,
        axisRatioY=1.0,
        axisRatioZ=par.halo_q,
    )
    cutoff_unit_params = dict(
        base_params,
        outerCutoffRadius=cutoff_radius,
        cutoffStrength=cutoff_strength,
    )

    unit_cutoff_density = agama.Density(**cutoff_unit_params)
    unit_cutoff_total = float(unit_cutoff_density.totalMass())
    unit_cutoff_m200 = float(
        unit_cutoff_density.enclosedMass(par.r200_kpc)
    )

    unit_uncut_density = agama.Density(**base_params)
    unit_uncut_total = float(unit_uncut_density.totalMass())

    for label, value in (
        ("unit cutoff total mass", unit_cutoff_total),
        ("unit cutoff M(<r200)", unit_cutoff_m200),
        ("unit untruncated total mass", unit_uncut_total),
    ):
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError(f"Invalid {label}: {value}")

    mode = par.halo_mass_normalization
    if mode == "galic-live-total":
        # Make the finite HDF5 halo mass equal the old GalIC halo mass.
        density_norm = par.target_halo_mass_msun / unit_cutoff_total
        normalization_target = "finite live halo total mass"
    elif mode == "galic-profile":
        # Preserve the old GalIC Hernquist density amplitude before applying
        # the finite outer cutoff.
        density_norm = par.target_halo_mass_msun / unit_uncut_total
        normalization_target = "untruncated Hernquist-equivalent halo mass"
    elif mode == "m200-enclosed":
        # Previous v6 convention.
        density_norm = par.m200_msun / unit_cutoff_m200
        normalization_target = "finite halo mass enclosed within r200"
    else:
        raise RuntimeError(f"Unsupported halo normalization mode: {mode}")

    halo_params = dict(cutoff_unit_params)
    halo_params["densityNorm"] = density_norm
    density_halo = agama.Density(**halo_params)

    halo_total_mass = float(density_halo.totalMass())
    halo_m200_check = float(density_halo.enclosedMass(par.r200_kpc))
    uncut_equivalent_mass = density_norm * unit_uncut_total
    finite_total_galaxy_mass = halo_total_mass + par.stellar_mass_msun

    profile = {
        "density_type": "Spheroid",
        "alpha": 1.0,
        "beta": 4.0,
        "gamma": 1.0,
        "scale_radius_kpc": par.halo_scale_radius_hernquist_kpc,
        "cutoff_radius_kpc": cutoff_radius,
        "cutoff_radius_over_r200": cutoff_radius / par.r200_kpc,
        "cutoff_strength": cutoff_strength,
        "halo_mass_normalization": mode,
        "normalization_target_description": normalization_target,
        "density_norm_msun_kpc3": density_norm,
        "target_galic_total_galaxy_mass_msun":
            par.galic_total_galaxy_mass_msun,
        "target_galic_halo_mass_msun": par.target_halo_mass_msun,
        "total_live_halo_mass_msun": halo_total_mass,
        "total_live_galaxy_mass_msun": finite_total_galaxy_mass,
        "total_live_galaxy_fractional_difference_from_galic": (
            finite_total_galaxy_mass
            / par.galic_total_galaxy_mass_msun
            - 1.0
        ),
        "m200_reference_msun": par.m200_msun,
        "m_within_r200_msun": halo_m200_check,
        "m_within_r200_over_reference_m200":
            halo_m200_check / par.m200_msun,
        "untruncated_equivalent_total_halo_mass_msun":
            uncut_equivalent_mass,
        "agama_parameters": halo_params,
    }

    print("\nFinite live-halo profile")
    print("------------------------")
    print(f"Normalization mode       : {mode}")
    print(f"Normalization target     : {normalization_target}")
    print(f"Cutoff radius            : {cutoff_radius:.6f} kpc")
    print(f"Cutoff radius / r200     : {cutoff_radius/par.r200_kpc:.6f}")
    print(f"Cutoff strength          : {cutoff_strength:.6f}")
    print(f"Live halo total mass     : {halo_total_mass:.6e} Msun")
    print(
        f"Live galaxy total mass   : "
        f"{finite_total_galaxy_mass:.6e} Msun"
    )
    print(
        f"Old GalIC galaxy mass    : "
        f"{par.galic_total_galaxy_mass_msun:.6e} Msun"
    )
    print(f"Halo M(<r200)            : {halo_m200_check:.6e} Msun")
    print(
        f"Untruncated-equivalent M : "
        f"{uncut_equivalent_mass:.6e} Msun"
    )

    return density_stars, density_halo, profile


def make_grid_parameters(
    par: HostParameters,
    halo_cutoff_radius_kpc: float,
) -> dict[str, dict[str, float | int | bool]]:
    star_rmin = max(0.003, par.re_major_kpc * 1.0e-3)
    star_rmax = max(200.0, par.re_major_kpc * 50.0)

    halo_rmin = max(
        0.02,
        par.halo_scale_radius_hernquist_kpc * 2.5e-4,
    )
    halo_rmax = max(
        2.5 * halo_cutoff_radius_kpc,
        2.0 * par.r200_kpc,
    )

    global_rmin = min(star_rmin, halo_rmin) * 0.5
    global_rmax = max(star_rmax, halo_rmax) * 1.2

    return {
        "global": {
            "rminSph": global_rmin,
            "rmaxSph": global_rmax,
            "sizeRadialSph": 90,
            "lmaxAngularSph": 8,
            "useActionInterpolation": False,
        },
        "stars": {
            "rminSph": star_rmin,
            "rmaxSph": star_rmax,
            "sizeRadialSph": 55,
            "lmaxAngularSph": 8,
        },
        "halo": {
            "rminSph": halo_rmin,
            "rmaxSph": halo_rmax,
            "sizeRadialSph": 65,
            "lmaxAngularSph": 6,
        },
    }

def density_diagnostic_table(
    output: Path,
    target_stars,
    final_stars,
    target_halo,
    final_halo,
    par: HostParameters,
) -> None:
    radii = np.geomspace(0.01, max(2 * par.r200_kpc, 1000.0), 240)
    equatorial = np.column_stack((radii, np.zeros_like(radii), np.zeros_like(radii)))
    polar = np.column_stack((np.zeros_like(radii), np.zeros_like(radii), radii))

    columns = np.column_stack(
        (
            radii,
            target_stars.density(equatorial),
            final_stars.density(equatorial),
            target_stars.density(polar),
            final_stars.density(polar),
            target_halo.density(equatorial),
            final_halo.density(equatorial),
            target_halo.density(polar),
            final_halo.density(polar),
        )
    )
    header = (
        "r_kpc "
        "rho_star_target_equatorial rho_star_final_equatorial "
        "rho_star_target_polar rho_star_final_polar "
        "rho_halo_target_equatorial rho_halo_final_equatorial "
        "rho_halo_target_polar rho_halo_final_polar"
    )
    np.savetxt(output, columns, header=header)


def rotation_curve_table(output: Path, potential, par: HostParameters) -> None:
    radii = np.geomspace(0.02, max(2 * par.r200_kpc, 1500.0), 240)
    points = np.column_stack((radii, np.zeros_like(radii), np.zeros_like(radii)))
    acceleration = potential.force(points)
    vc2 = np.maximum(0.0, -acceleration[:, 0] * radii)
    np.savetxt(
        output,
        np.column_stack((radii, np.sqrt(vc2))),
        header="radius_kpc circular_velocity_km_s",
    )


def _center_one_sample(
    sample: tuple[np.ndarray, np.ndarray],
    label: str,
) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
    xv = np.asarray(sample[0], dtype=np.float64).copy()
    mass = np.asarray(sample[1], dtype=np.float64).copy()

    total_mass = float(np.sum(mass))
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise RuntimeError(f"{label} sample has invalid total mass")

    com_pos = np.sum(xv[:, :3] * mass[:, None], axis=0) / total_mass
    com_vel = np.sum(xv[:, 3:] * mass[:, None], axis=0) / total_mass

    radius_before = np.linalg.norm(xv[:, :3], axis=1)
    percentiles_before = np.percentile(
        radius_before, [50.0, 90.0, 99.0, 99.9, 99.99]
    )

    xv[:, :3] -= com_pos
    xv[:, 3:] -= com_vel

    radius_after = np.linalg.norm(xv[:, :3], axis=1)
    diagnostics = {
        "label": label,
        "center_shift_kpc": com_pos.tolist(),
        "center_shift_magnitude_kpc": float(np.linalg.norm(com_pos)),
        "bulk_velocity_shift_kms": com_vel.tolist(),
        "bulk_velocity_shift_magnitude_kms": float(np.linalg.norm(com_vel)),
        "radius_percentiles_before_kpc": {
            "p50": float(percentiles_before[0]),
            "p90": float(percentiles_before[1]),
            "p99": float(percentiles_before[2]),
            "p99_9": float(percentiles_before[3]),
            "p99_99": float(percentiles_before[4]),
            "maximum": float(np.max(radius_before)),
        },
        "maximum_radius_after_centering_kpc": float(np.max(radius_after)),
    }
    print(
        f"{label} recentering: |dX|="
        f"{diagnostics['center_shift_magnitude_kpc']:.6g} kpc, "
        f"|dV|={diagnostics['bulk_velocity_shift_magnitude_kms']:.6g} km/s, "
        f"rmax={diagnostics['maximum_radius_after_centering_kpc']:.6g} kpc"
    )
    return (xv, mass), diagnostics


def recenter_samples(
    star_sample: tuple[np.ndarray, np.ndarray],
    halo_sample: tuple[np.ndarray, np.ndarray],
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
    dict[str, Any],
]:
    """
    Recenter each component independently.

    The two equilibrium components are analytically centered on the same
    origin. Independent recentering removes Monte-Carlo offsets without
    allowing the much more massive and extended halo sample to displace the
    stellar spheroid.
    """
    star_centered, star_diag = _center_one_sample(star_sample, "Stars")
    halo_centered, halo_diag = _center_one_sample(halo_sample, "Halo")
    return star_centered, halo_centered, {
        "stars": star_diag,
        "halo": halo_diag,
    }

def renormalize_sample_mass(
    sample: tuple[np.ndarray, np.ndarray],
    target_mass: float,
    label: str,
    maximum_fractional_change: float,
) -> tuple[np.ndarray, np.ndarray]:
    xv = np.asarray(sample[0], dtype=np.float64)
    masses = np.asarray(sample[1], dtype=np.float64)
    sampled_mass = float(np.sum(masses))
    if sampled_mass <= 0 or not np.isfinite(sampled_mass):
        raise RuntimeError(f"{label} sample has invalid mass {sampled_mass}")

    factor = target_mass / sampled_mass
    fractional_change = abs(factor - 1.0)
    print(
        f"{label} sampled mass={sampled_mass:.6e} Msun; "
        f"renormalization factor={factor:.8f}"
    )
    if fractional_change > maximum_fractional_change:
        raise RuntimeError(
            f"{label} DF mass differs from the requested mass by "
            f"{fractional_change:.3%}, exceeding "
            f"--max-mass-renormalization={maximum_fractional_change:.3%}. "
            "Inspect the self-consistent model rather than forcing the mass."
        )
    masses *= factor
    return xv, masses


def write_gadget_hdf5(
    path: Path,
    stars: tuple[np.ndarray, np.ndarray],
    halo: tuple[np.ndarray, np.ndarray],
    par: HostParameters,
    halo_spin_diag: dict[str, Any],
) -> None:
    """
    Write a single-file, non-cosmological GADGET-4 HDF5 IC.

    PartType1: live dark-matter halo
    PartType3: stellar Sérsic spheroid
    """
    star_xv, star_mass_msun = stars
    halo_xv, halo_mass_msun = halo

    n_halo = len(halo_xv)
    n_star = len(star_xv)
    n_total = n_halo + n_star

    if n_halo != par.n_halo or n_star != par.n_star:
        raise RuntimeError("Sample sizes differ from requested particle counts")

    counts = np.zeros(6, dtype=np.uint32)
    counts[1] = n_halo
    counts[3] = n_star

    halo_ids = np.arange(1, n_halo + 1, dtype=np.uint64)
    star_ids = np.arange(n_halo + 1, n_total + 1, dtype=np.uint64)

    with h5py.File(path, "w") as handle:
        header = handle.create_group("Header")
        attrs = header.attrs
        attrs["NumPart_ThisFile"] = counts
        attrs["NumPart_Total"] = counts
        attrs["NumPart_Total_HighWord"] = np.zeros(6, dtype=np.uint32)
        attrs["MassTable"] = np.zeros(6, dtype=np.float64)
        attrs["Time"] = 0.0
        attrs["Redshift"] = 0.0
        attrs["BoxSize"] = 0.0
        attrs["NumFilesPerSnapshot"] = 1
        attrs["Omega0"] = 0.0
        attrs["OmegaLambda"] = 0.0
        attrs["HubbleParam"] = 1.0

        for flag in (
            "Flag_Sfr",
            "Flag_Cooling",
            "Flag_StellarAge",
            "Flag_Metals",
            "Flag_Feedback",
            "Flag_DoublePrecision",
            "Flag_IC_Info",
        ):
            attrs[flag] = 0

        attrs["UnitLength_in_cm"] = 3.0856775814913673e21
        attrs["UnitMass_in_g"] = par.gadget_mass_unit_msun * 1.988409870698051e33
        attrs["UnitVelocity_in_cm_per_s"] = 1.0e5

        attrs["GalaxyID_Name"] = np.asarray(
            [par.name.encode("utf-8")], dtype="S64"
        )
        attrs["GalaxyID_IDStart"] = np.asarray([1], dtype=np.uint64)
        attrs["GalaxyID_IDEnd"] = np.asarray([n_total], dtype=np.uint64)
        attrs["GalaxyID_IDMin"] = np.asarray([1], dtype=np.uint64)
        attrs["GalaxyID_IDMax"] = np.asarray([n_total], dtype=np.uint64)
        attrs["GalaxyID_IDOffset"] = np.asarray([0], dtype=np.uint64)
        attrs["GalaxyID_Infile"] = np.asarray(
            [b"AGAMA_SERSIC_HOST"], dtype="S64"
        )

        attrs["AGAMA_StellarMass_Msun"] = par.stellar_mass_msun
        attrs["AGAMA_SersicIndex"] = par.sersic_n
        attrs["AGAMA_ReMajor_kpc"] = par.re_major_kpc
        attrs["AGAMA_ReCirc_kpc"] = par.re_circularized_kpc
        attrs["AGAMA_StellarQ"] = par.intrinsic_q
        attrs["AGAMA_M200_Msun"] = par.m200_msun
        attrs["AGAMA_R200_kpc"] = par.r200_kpc
        attrs["AGAMA_HaloMassNormalization"] = (
            par.halo_mass_normalization.encode("utf-8")
        )
        attrs["AGAMA_GalICGalaxyMass_Msun"] = (
            par.galic_total_galaxy_mass_msun
        )
        attrs["AGAMA_TargetHaloMass_Msun"] = par.target_halo_mass_msun
        attrs["AGAMA_HaloA_kpc"] = par.halo_scale_radius_hernquist_kpc
        attrs["AGAMA_HaloQ"] = par.halo_q
        attrs["AGAMA_StellarMassSource"] = par.stellar_mass_source.encode("utf-8")
        attrs["AGAMA_StellarQSource"] = par.stellar_q_source.encode("utf-8")
        attrs["AGAMA_HaloQSource"] = par.halo_q_source.encode("utf-8")
        if par.galic_parameter_file is not None:
            attrs["GalIC_ParameterFile"] = par.galic_parameter_file.encode("utf-8")
        if par.galic_v200_kms is not None:
            attrs["GalIC_V200_kms"] = par.galic_v200_kms
        attrs["GalIC_Hubble_kms_kpc"] = par.galic_hubble_kms_kpc
        if par.galic_concentration is not None:
            attrs["GalIC_CC"] = par.galic_concentration
        if par.galic_lambda is not None:
            attrs["GalIC_LAMBDA"] = par.galic_lambda
        if par.galic_mb is not None:
            attrs["GalIC_MB"] = par.galic_mb
        if par.galic_halo_stretch is not None:
            attrs["GalIC_HaloStretch"] = par.galic_halo_stretch
        if par.galic_bulge_stretch is not None:
            attrs["GalIC_BulgeStretch"] = par.galic_bulge_stretch
        if par.galic_softening_kpc is not None:
            attrs["GalIC_Softening_kpc"] = par.galic_softening_kpc
        spin_after = halo_spin_diag.get("after", {})
        attrs["AGAMA_HaloSpinApplied"] = int(bool(halo_spin_diag.get("applied", False)))
        attrs["AGAMA_HaloSpinDefinition"] = np.asarray(
            [b"GalIC_Peebles_equivalent"], dtype="S32"
        )
        if "target_lambda_signed" in halo_spin_diag:
            attrs["AGAMA_HaloSpinTargetLambda"] = float(
                halo_spin_diag["target_lambda_signed"]
            )
        if "lambda_galic_equivalent" in spin_after:
            attrs["AGAMA_HaloSpinAchievedLambda"] = float(
                spin_after["lambda_galic_equivalent"]
            )
            attrs["AGAMA_HaloSpinJz_Msun_kpc_kms"] = float(
                spin_after["Jz_msun_kpc_kms"]
            )
            attrs["AGAMA_HaloSpinProgradeMassFraction"] = float(
                spin_after["prograde_mass_fraction"]
            )
        attrs["AGAMA_HaloLiveMass_Msun"] = float(np.sum(halo_mass_msun))
        attrs["AGAMA_StellarSampleRmax_kpc"] = float(
            np.max(np.linalg.norm(star_xv[:, :3], axis=1))
        )
        attrs["AGAMA_HaloSampleRmax_kpc"] = float(
            np.max(np.linalg.norm(halo_xv[:, :3], axis=1))
        )

        group_halo = handle.create_group("PartType1")
        group_halo.create_dataset("Coordinates", data=halo_xv[:, :3])
        group_halo.create_dataset("Velocities", data=halo_xv[:, 3:])
        group_halo.create_dataset("ParticleIDs", data=halo_ids)
        group_halo.create_dataset(
            "Masses", data=halo_mass_msun / par.gadget_mass_unit_msun
        )

        group_star = handle.create_group("PartType3")
        group_star.create_dataset("Coordinates", data=star_xv[:, :3])
        group_star.create_dataset("Velocities", data=star_xv[:, 3:])
        group_star.create_dataset("ParticleIDs", data=star_ids)
        group_star.create_dataset(
            "Masses", data=star_mass_msun / par.gadget_mass_unit_msun
        )



def make_component_potentials(
    agama,
    par: HostParameters,
    halo_profile: dict[str, Any],
):
    """
    Return the target stellar potential, finite halo potential, and their sum.
    """
    pot_stars = agama.Potential(
        type="Sersic",
        mass=par.stellar_mass_msun,
        scaleRadius=par.re_major_kpc,
        sersicIndex=par.sersic_n,
        axisRatioY=1.0,
        axisRatioZ=par.intrinsic_q,
    )
    pot_halo = agama.Potential(**halo_profile["agama_parameters"])
    return pot_stars, pot_halo, agama.Potential(pot_stars, pot_halo)

def component_target_grid(
    agama,
    *,
    rmin_kpc: float,
    rmax_kpc: float,
    bins: int,
    lmax: int,
):
    if not np.isfinite(rmin_kpc) or not np.isfinite(rmax_kpc):
        raise ValueError("Target-grid radii must be finite")
    if rmin_kpc <= 0 or rmax_kpc <= rmin_kpc:
        raise ValueError(
            f"Invalid target-grid range: rmin={rmin_kpc}, rmax={rmax_kpc}"
        )
    if bins < 10:
        raise ValueError("At least 10 target-grid points are required")
    if lmax < 0 or lmax % 2:
        raise ValueError("lmax must be a non-negative even integer")

    gridr = agama.nonuniformGrid(int(bins), float(rmin_kpc), float(rmax_kpc))
    return agama.Target(
        type="DensitySphHarm",
        gridr=gridr,
        lmax=int(lmax),
        mmax=0,
    )



def build_quasispherical_halo(
    agama,
    *,
    density_stars,
    density_halo,
    par: HostParameters,
    target_halo_total_mass_msun: float,
    halo_cutoff_radius_kpc: float,
    args: argparse.Namespace,
    output_dir: Path,
):
    """
    Construct and iterate a QuasiSpherical halo DF while holding the requested
    flattened Sérsic stellar density fixed.

    Returns
    -------
    scm
        Converged AGAMA SelfConsistentModel. Its potential is subsequently
        used for the stellar Schwarzschild orbit library.
    df_halo
        QuasiSpherical halo DF defined during the initial SCM iteration.
    diagnostics
        Dictionary describing the halo mass and convergence history.
    """
    grids = make_grid_parameters(par, halo_cutoff_radius_kpc)
    global_grid = dict(grids["global"])
    halo_grid = dict(grids["halo"])

    if args.halo_scm_global_rmin_kpc is not None:
        global_grid["rminSph"] = float(args.halo_scm_global_rmin_kpc)
    if args.halo_scm_global_rmax_kpc is not None:
        global_grid["rmaxSph"] = float(args.halo_scm_global_rmax_kpc)
    if args.halo_scm_global_size_radial is not None:
        global_grid["sizeRadialSph"] = int(args.halo_scm_global_size_radial)
    if args.halo_scm_global_lmax is not None:
        global_grid["lmaxAngularSph"] = int(args.halo_scm_global_lmax)

    if args.halo_scm_rmin_kpc is not None:
        halo_grid["rminSph"] = float(args.halo_scm_rmin_kpc)
    if args.halo_scm_rmax_kpc is not None:
        halo_grid["rmaxSph"] = float(args.halo_scm_rmax_kpc)
    if args.halo_scm_size_radial is not None:
        halo_grid["sizeRadialSph"] = int(args.halo_scm_size_radial)
    if args.halo_scm_lmax is not None:
        halo_grid["lmaxAngularSph"] = int(args.halo_scm_lmax)

    print("\n[Halo] SelfConsistentModel grid")
    print(
        json.dumps(
            {"global": global_grid, "halo_component": halo_grid},
            indent=2,
            sort_keys=True,
        )
    )

    scm = agama.SelfConsistentModel(**global_grid)

    # Component 0 remains the prescribed flattened Sérsic density throughout
    # the halo iteration. Component 1 begins as the target Hernquist density.
    scm.components.append(
        agama.Component(density=density_stars, disklike=False)
    )
    scm.components.append(
        agama.Component(density=density_halo, disklike=False)
    )

    print("\n[Halo] Computing initial stellar-plus-target-halo potential")
    scm.iterate()

    print("[Halo] Constructing QuasiSpherical DF in the initial total potential")
    df_halo = agama.DistributionFunction(
        type="QuasiSpherical",
        potential=scm.potential,
        density=density_halo,
    )

    df_mass = float(df_halo.totalMass())
    print(
        f"[Halo] QuasiSpherical DF total mass={df_mass:.6e} Msun "
        f"(target={target_halo_total_mass_msun:.6e} Msun)"
    )

    scm.components[1] = agama.Component(
        df=df_halo,
        disklike=False,
        **halo_grid,
    )

    history = []
    for iteration in range(1, int(args.halo_scm_iterations) + 1):
        print(
            f"[Halo] Self-consistent iteration "
            f"{iteration}/{args.halo_scm_iterations}"
        )
        scm.iterate()

        final_density = scm.components[1].density
        final_mass = float(final_density.totalMass())

        # A few density samples provide a lightweight convergence diagnostic.
        a = par.halo_scale_radius_hernquist_kpc
        sample_radii = np.asarray([0.1 * a, a, 5.0 * a], dtype=float)
        equatorial = np.column_stack(
            (sample_radii, np.zeros_like(sample_radii), np.zeros_like(sample_radii))
        )
        polar = np.column_stack(
            (np.zeros_like(sample_radii), np.zeros_like(sample_radii), sample_radii)
        )
        rho_eq = np.asarray(final_density.density(equatorial), dtype=float)
        rho_pol = np.asarray(final_density.density(polar), dtype=float)

        record = {
            "iteration": iteration,
            "halo_density_total_mass_msun": final_mass,
            "halo_mass_fractional_difference_from_target": (
                final_mass / target_halo_total_mass_msun - 1.0
            ),
            "sample_radii_kpc": sample_radii.tolist(),
            "rho_equatorial": rho_eq.tolist(),
            "rho_polar": rho_pol.tolist(),
        }
        history.append(record)

        print(
            f"[Halo] iteration={iteration}; "
            f"density mass={final_mass:.6e} Msun; "
            f"fractional difference="
            f"{record['halo_mass_fractional_difference_from_target']:+.3e}"
        )

    final_density = scm.components[1].density
    final_mass = float(final_density.totalMass())

    final_density.export(str(output_dir / "density_halo_quasispherical_final"))
    scm.potential.export(str(output_dir / "potential_total_hybrid_final"))

    density_diagnostic_table(
        output_dir / "target_vs_quasispherical_halo_density.txt",
        density_stars,
        density_stars,
        density_halo,
        final_density,
        par,
    )

    diagnostics = {
        "method": "QuasiSpherical DF with SelfConsistentModel",
        "df_total_mass_msun": df_mass,
        "target_halo_total_mass_msun": target_halo_total_mass_msun,
        "final_halo_density_total_mass_msun": final_mass,
        "final_halo_mass_fractional_difference_from_target": (
            final_mass / target_halo_total_mass_msun - 1.0
        ),
        "iterations": int(args.halo_scm_iterations),
        "global_grid": global_grid,
        "halo_component_grid": halo_grid,
        "history": history,
    }
    (output_dir / "halo_quasispherical_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )

    return scm, df_halo, diagnostics


def _coerce_rperiapo_result(
    result,
    number_points: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize the possible AGAMA Rperiapo return layouts to two 1-D arrays."""
    if isinstance(result, (tuple, list)) and len(result) == 2:
        rperi = np.asarray(result[0], dtype=np.float64).reshape(-1)
        rapo = np.asarray(result[1], dtype=np.float64).reshape(-1)
    else:
        array = np.asarray(result, dtype=np.float64)
        if array.shape == (number_points, 2):
            rperi = array[:, 0]
            rapo = array[:, 1]
        elif array.shape == (2, number_points):
            rperi = array[0]
            rapo = array[1]
        elif number_points == 1 and array.size == 2:
            flat = array.reshape(-1)
            rperi = flat[:1]
            rapo = flat[1:2]
        else:
            raise RuntimeError(
                f"{label}: Potential.Rperiapo returned unexpected shape "
                f"{array.shape}; expected ({number_points},2), "
                f"(2,{number_points}), or a two-element tuple"
            )

    if len(rperi) != number_points or len(rapo) != number_points:
        raise RuntimeError(
            f"{label}: Potential.Rperiapo returned {len(rperi)} pericentres "
            f"and {len(rapo)} apocentres for {number_points} points"
        )
    return rperi, rapo


def sample_bounded_orbit_initial_conditions(
    *,
    source_component,
    total_potential,
    number_orbits: int,
    beta: float,
    kappa: float,
    maximum_apocentre_kpc: float,
    batch_factor: float,
    maximum_batches: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Repeatedly sample Jeans initial conditions and retain only finite, bound
    orbits whose estimated apocentre does not exceed the requested limit.
    """
    if maximum_apocentre_kpc <= 0 or not np.isfinite(maximum_apocentre_kpc):
        raise ValueError(
            f"{label}: maximum apocentre must be finite and positive"
        )
    if batch_factor <= 1:
        raise ValueError(f"{label}: orbit-sampling batch factor must exceed 1")
    if maximum_batches < 1:
        raise ValueError(f"{label}: maximum sampling batches must be positive")

    accepted_ic: list[np.ndarray] = []
    accepted_prior: list[np.ndarray] = []
    accepted_rperi: list[np.ndarray] = []
    accepted_rapo: list[np.ndarray] = []
    accepted_energy_rmax: list[np.ndarray] = []

    total_drawn = 0
    total_accepted_before_truncation = 0
    rejected_nonfinite = 0
    rejected_nonpositive = 0
    rejected_over_apocentre = 0
    rejected_over_energy_rmax = 0

    for batch_index in range(1, maximum_batches + 1):
        accepted_so_far = sum(len(chunk) for chunk in accepted_ic)
        remaining = number_orbits - accepted_so_far
        if remaining <= 0:
            break

        batch_size = max(
            1000,
            int(math.ceil(float(batch_factor) * remaining)),
        )
        candidate_ic, candidate_prior = source_component.sample(
            batch_size,
            potential=total_potential,
            beta=float(beta),
            kappa=float(kappa),
        )
        candidate_ic = np.asarray(candidate_ic, dtype=np.float64)
        candidate_prior = np.asarray(candidate_prior, dtype=np.float64).reshape(-1)

        if candidate_ic.shape != (batch_size, 6):
            raise RuntimeError(
                f"{label}: source sample returned initial-condition shape "
                f"{candidate_ic.shape}, expected ({batch_size}, 6)"
            )
        if len(candidate_prior) != batch_size:
            raise RuntimeError(
                f"{label}: source sample returned {len(candidate_prior)} prior "
                f"weights for {batch_size} initial conditions"
            )

        result = total_potential.Rperiapo(candidate_ic)
        rperi, rapo = _coerce_rperiapo_result(
            result, batch_size, label
        )

        # Rperiapo is exact for spherical potentials but only approximate for
        # out-of-plane orbits in nonspherical potentials. Add the energy-only
        # maximum accessible radius Rmax(E) as a conservative second guard.
        potential_values = np.asarray(
            total_potential.potential(candidate_ic[:, :3]),
            dtype=np.float64,
        ).reshape(-1)
        kinetic = 0.5 * np.sum(candidate_ic[:, 3:] ** 2, axis=1)
        energy = potential_values + kinetic
        energy_rmax = np.asarray(
            total_potential.Rmax(energy), dtype=np.float64
        ).reshape(-1)
        if len(energy_rmax) != batch_size:
            raise RuntimeError(
                f"{label}: Potential.Rmax returned {len(energy_rmax)} values "
                f"for {batch_size} candidate orbits"
            )

        finite = (
            np.isfinite(rperi)
            & np.isfinite(rapo)
            & np.isfinite(energy_rmax)
        )
        positive = (
            (rperi >= 0)
            & (rapo > 0)
            & (rapo >= rperi)
            & (energy_rmax > 0)
        )
        within_rapo = rapo <= float(maximum_apocentre_kpc)
        within_energy = energy_rmax <= float(maximum_apocentre_kpc)
        keep = finite & positive & within_rapo & within_energy

        rejected_nonfinite += int(np.count_nonzero(~finite))
        rejected_nonpositive += int(np.count_nonzero(finite & ~positive))
        rejected_over_apocentre += int(
            np.count_nonzero(finite & positive & ~within_rapo)
        )
        rejected_over_energy_rmax += int(
            np.count_nonzero(finite & positive & ~within_energy)
        )
        total_drawn += batch_size
        total_accepted_before_truncation += int(np.count_nonzero(keep))

        if np.any(keep):
            accepted_ic.append(candidate_ic[keep])
            accepted_prior.append(candidate_prior[keep])
            accepted_rperi.append(rperi[keep])
            accepted_rapo.append(rapo[keep])
            accepted_energy_rmax.append(energy_rmax[keep])

        accepted_now = sum(len(chunk) for chunk in accepted_ic)
        print(
            f"[{label}] bounded-orbit sampling batch {batch_index}: "
            f"accepted {int(np.count_nonzero(keep))}/{batch_size}; "
            f"accumulated {min(accepted_now, number_orbits)}/{number_orbits}"
        )

    accepted_so_far = sum(len(chunk) for chunk in accepted_ic)
    if accepted_so_far < number_orbits:
        acceptance = accepted_so_far / total_drawn if total_drawn else 0.0
        raise RuntimeError(
            f"{label}: obtained only {accepted_so_far}/{number_orbits} bounded "
            f"orbits after {maximum_batches} batches and {total_drawn} draws "
            f"(acceptance={acceptance:.3%}). Increase "
            f"--{label.lower()}-max-apocentre-kpc, increase "
            "--orbit-sampling-max-batches, or inspect the target potential."
        )

    initcond = np.concatenate(accepted_ic, axis=0)[:number_orbits]
    weightprior = np.concatenate(accepted_prior)[:number_orbits]
    rperi = np.concatenate(accepted_rperi)[:number_orbits]
    rapo = np.concatenate(accepted_rapo)[:number_orbits]
    energy_rmax = np.concatenate(accepted_energy_rmax)[:number_orbits]

    diagnostics = {
        "requested_orbits": int(number_orbits),
        "maximum_apocentre_kpc": float(maximum_apocentre_kpc),
        "batch_factor": float(batch_factor),
        "maximum_batches": int(maximum_batches),
        "total_candidates_drawn": int(total_drawn),
        "accepted_candidates_before_truncation": int(
            total_accepted_before_truncation
        ),
        "acceptance_fraction_before_truncation": float(
            total_accepted_before_truncation / total_drawn
        ),
        "rejected_nonfinite_rperiapo": int(rejected_nonfinite),
        "rejected_nonpositive_or_inconsistent_rperiapo": int(
            rejected_nonpositive
        ),
        "rejected_over_Rperiapo_apocentre_limit": int(
            rejected_over_apocentre
        ),
        "rejected_over_energy_Rmax_limit": int(rejected_over_energy_rmax),
        "accepted_rperi_min_kpc": float(np.min(rperi)),
        "accepted_rperi_median_kpc": float(np.median(rperi)),
        "accepted_rperi_max_kpc": float(np.max(rperi)),
        "accepted_rapo_min_kpc": float(np.min(rapo)),
        "accepted_rapo_median_kpc": float(np.median(rapo)),
        "accepted_rapo_p99_kpc": float(np.percentile(rapo, 99.0)),
        "accepted_rapo_max_kpc": float(np.max(rapo)),
        "accepted_energy_rmax_min_kpc": float(np.min(energy_rmax)),
        "accepted_energy_rmax_median_kpc": float(np.median(energy_rmax)),
        "accepted_energy_rmax_p99_kpc": float(
            np.percentile(energy_rmax, 99.0)
        ),
        "accepted_energy_rmax_max_kpc": float(np.max(energy_rmax)),
    }

    print(
        f"[{label}] accepted bounded library: "
        f"rperi={diagnostics['accepted_rperi_min_kpc']:.6g}--"
        f"{diagnostics['accepted_rperi_max_kpc']:.6g} kpc; "
        f"rapo={diagnostics['accepted_rapo_min_kpc']:.6g}--"
        f"{diagnostics['accepted_rapo_max_kpc']:.6g} kpc; "
        f"Rmax(E)<={diagnostics['accepted_energy_rmax_max_kpc']:.6g} kpc; "
        f"candidate acceptance="
        f"{diagnostics['acceptance_fraction_before_truncation']:.2%}"
    )

    return (
        initcond,
        weightprior,
        rperi,
        rapo,
        energy_rmax,
        diagnostics,
    )

def solve_orbit_component(
    agama,
    *,
    label: str,
    source_component,
    total_potential,
    component_mass_msun: float,
    target,
    number_orbits: int,
    integration_periods: float,
    beta: float,
    kappa: float,
    orbit_accuracy: float,
    regularization: float,
    density_constraint_mode: str,
    density_tolerance: float,
    density_scale_floor: float,
    maximum_apocentre_kpc: float,
    orbit_sampling_batch_factor: float,
    orbit_sampling_max_batches: int,
    output_dir: Path,
    save_orbit_table: bool,
):
    """
    Build one Schwarzschild orbit library and solve its non-negative weights.

    The official AGAMA example solves in physical component-mass units and
    treats the mass and density constraints as exact.  This routine follows
    that convention for its first attempt.

    Exact density matching can fail for a reduced or poorly sampled orbit
    library.  In ``auto`` mode, a failed exact solve is retried using
    row-scaled quadratic density penalties while retaining an exact total-mass
    constraint.  The scaled residual of each retained density coefficient is
    then measured in units of ``density_tolerance``.
    """
    if number_orbits < 100:
        raise ValueError(f"{label}: number of orbits must be at least 100")
    if integration_periods <= 0:
        raise ValueError(f"{label}: integration periods must be positive")
    if density_constraint_mode not in ("auto", "exact", "soft"):
        raise ValueError(
            f"{label}: density_constraint_mode must be auto, exact, or soft"
        )
    if density_tolerance <= 0:
        raise ValueError(f"{label}: density_tolerance must be positive")
    if not 0 < density_scale_floor < 1:
        raise ValueError(
            f"{label}: density_scale_floor must lie strictly between 0 and 1"
        )

    print(
        f"\n[{label}] Sampling {number_orbits} bounded orbit-library "
        f"initial conditions (rapo <= {maximum_apocentre_kpc:g} kpc)"
    )
    (
        initcond,
        weightprior,
        orbit_rperi,
        orbit_rapo,
        orbit_energy_rmax,
        bounded_sampling_diag,
    ) = sample_bounded_orbit_initial_conditions(
        source_component=source_component,
        total_potential=total_potential,
        number_orbits=int(number_orbits),
        beta=float(beta),
        kappa=float(kappa),
        maximum_apocentre_kpc=float(maximum_apocentre_kpc),
        batch_factor=float(orbit_sampling_batch_factor),
        maximum_batches=int(orbit_sampling_max_batches),
        label=label,
    )

    periods = np.asarray(total_potential.Tcirc(initcond), dtype=np.float64)
    valid = np.isfinite(periods) & (periods > 0)
    if not np.all(valid):
        bad = int(np.count_nonzero(~valid))
        raise RuntimeError(
            f"{label}: {bad} orbit-library initial conditions have invalid "
            "circular-period estimates. Increase the target radial bounds or "
            "inspect the sampled initial conditions."
        )
    inttimes = float(integration_periods) * periods

    print(
        f"[{label}] Integrating orbit library for "
        f"{integration_periods:g} circular periods"
    )
    data, trajectories = agama.orbit(
        potential=total_potential,
        ic=initcond,
        time=inttimes,
        dtype=object,
        targets=target,
        method="dprkn8",
        accuracy=float(orbit_accuracy),
    )
    data = np.asarray(data, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] != len(initcond):
        raise RuntimeError(
            f"{label}: orbit-target matrix has unexpected shape {data.shape}"
        )
    if np.any(~np.isfinite(data)):
        raise RuntimeError(f"{label}: orbit-target matrix contains non-finite values")

    # Normalize the optimization problem to unit component mass.
    #
    # The orbit-target matrix records occupation fractions and is independent
    # of the physical mass assigned to each orbit. Hence the unknowns can be
    # dimensionless mass fractions w_i with sum(w_i)=1. The physical target
    # coefficients are divided by the component mass to use the same units.
    #
    # This normalization is essential when xpenq is nonzero: solving directly
    # for weights of order 1e7--1e8 Msun would make 0.5*sum(w_i^2) enormous
    # and badly condition the quadratic optimization.
    rhs_density_physical = np.asarray(
        target(source_component), dtype=np.float64
    )
    if (
        rhs_density_physical.ndim != 1
        or data.shape[1] != len(rhs_density_physical)
    ):
        raise RuntimeError(
            f"{label}: target vector shape {rhs_density_physical.shape} is "
            f"incompatible with orbit matrix shape {data.shape}"
        )
    if np.any(~np.isfinite(rhs_density_physical)):
        raise RuntimeError(
            f"{label}: target density vector contains non-finite values"
        )

    rhs_density = rhs_density_physical / float(component_mass_msun)
    mass_matrix = np.ones((len(initcond), 1), dtype=np.float64)

    # Row scaling is used only in the soft problem. It prevents the largest
    # low-order coefficients from dominating while also keeping tiny outer or
    # high-order coefficients from destabilizing the KKT system.
    max_rhs = float(np.max(np.abs(rhs_density)))
    if not np.isfinite(max_rhs) or max_rhs <= 0:
        raise RuntimeError(f"{label}: density target has no nonzero coefficients")
    row_floor = float(density_scale_floor) * max_rhs
    row_scale = np.maximum(np.abs(rhs_density), row_floor)
    data_scaled = data / row_scale[None, :]
    rhs_scaled = rhs_density / row_scale

    # The unknown weights are dimensionless fractions, so an order-unity
    # quadratic regularization is now numerically sensible.
    xpenq = np.full(len(initcond), float(regularization), dtype=np.float64)

    print(
        f"[{label}] Optimization uses unit-mass orbit fractions "
        f"(physical component mass={component_mass_msun:.6e} Msun)"
    )

    def solve_exact():
        print(f"[{label}] Solving with exact unit mass and density constraints")
        # With no RHS penalties, solveOpt treats all rows as exact.
        return agama.solveOpt(
            matrix=(mass_matrix.T, data.T),
            rhs=([1.0], rhs_density),
            xpenq=xpenq,
        )

    def solve_soft():
        # For scaled residual e=(model-target)/row_scale, use the documented
        # quadratic coefficient 2/error^2. The total mass row is kept exact
        # with an infinite *linear* RHS penalty, while density rows have zero
        # linear penalty and finite quadratic penalties.
        density_penalty = np.full(
            len(rhs_density),
            2.0 / float(density_tolerance) ** 2,
            dtype=np.float64,
        )
        print(
            f"[{label}] Solving with exact unit total mass and soft, "
            f"row-scaled density constraints "
            f"(tolerance={density_tolerance:g})"
        )
        return agama.solveOpt(
            matrix=(mass_matrix.T, data_scaled.T),
            rhs=([1.0], rhs_scaled),
            rpenl=(
                [np.inf],
                np.zeros(len(rhs_density), dtype=np.float64),
            ),
            rpenq=(
                [0.0],
                density_penalty,
            ),
            xpenq=xpenq,
        )

    solver_mode_used = density_constraint_mode
    if density_constraint_mode == "exact":
        weights = solve_exact()
    elif density_constraint_mode == "soft":
        weights = solve_soft()
    else:
        try:
            weights = solve_exact()
            solver_mode_used = "exact"
        except (RuntimeError, ArithmeticError) as exc:
            print(
                f"[{label}] Exact solve failed ({type(exc).__name__}: {exc}).\n"
                f"[{label}] Retrying with soft scaled density constraints.",
                file=sys.stderr,
            )
            weights = solve_soft()
            solver_mode_used = "soft"

    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != (len(initcond),):
        raise RuntimeError(
            f"{label}: optimizer returned weights with shape {weights.shape}"
        )
    if np.any(~np.isfinite(weights)) or np.any(weights < -1e-12):
        raise RuntimeError(f"{label}: optimizer returned invalid orbit weights")
    weights = np.maximum(weights, 0.0)

    model_density = data.T @ weights
    residual = model_density - rhs_density
    scaled_residual = residual / row_scale

    # Relative residuals are useful only where the coefficient itself is not
    # close to zero; otherwise the scaled residual is the meaningful measure.
    relative_floor = max_rhs * float(density_scale_floor)
    nz = np.abs(rhs_density) >= relative_floor
    relative = np.full_like(residual, np.nan)
    relative[nz] = residual[nz] / rhs_density[nz]

    positive = weights[weights > 0]
    effective_orbits = (
        float(weights.sum() ** 2 / np.sum(weights**2))
        if np.sum(weights**2) > 0
        else 0.0
    )
    diagnostics = {
        "label": label,
        "solver_mode_requested": density_constraint_mode,
        "solver_mode_used": solver_mode_used,
        "density_tolerance": float(density_tolerance),
        "density_scale_floor_fraction": float(density_scale_floor),
        "number_orbits": int(number_orbits),
        "bounded_orbit_sampling": bounded_sampling_diag,
        "positive_weight_orbits": int(len(positive)),
        "effective_number_orbits": effective_orbits,
        "weight_sum_fraction": float(weights.sum()),
        "weight_sum_fractional_mass_error": float(weights.sum() - 1.0),
        "weight_max_fraction": float(np.max(weights)),
        "weight_max_msun": float(np.max(weights) * component_mass_msun),
        "weight_median_positive_fraction": (
            float(np.median(positive)) if len(positive) else float("nan")
        ),
        "density_absolute_residual_max_unit_mass": float(
            np.max(np.abs(residual))
        ),
        "density_scaled_residual_rms": float(
            np.sqrt(np.mean(scaled_residual**2))
        ),
        "density_scaled_residual_max": float(np.max(np.abs(scaled_residual))),
        "density_relative_residual_rms_above_floor": (
            float(np.sqrt(np.mean(relative[nz] ** 2)))
            if np.any(nz)
            else float("nan")
        ),
        "density_relative_residual_max_above_floor": (
            float(np.max(np.abs(relative[nz])))
            if np.any(nz)
            else float("nan")
        ),
    }

    print(
        f"[{label}] mode={solver_mode_used}; "
        f"positive-weight orbits={diagnostics['positive_weight_orbits']}; "
        f"N_eff={effective_orbits:.1f}; "
        f"scaled RMS residual={diagnostics['density_scaled_residual_rms']:.3e}; "
        f"scaled max residual={diagnostics['density_scaled_residual_max']:.3e}"
    )

    np.savetxt(
        output_dir / f"{label.lower()}_density_constraint_residuals.txt",
        np.column_stack(
            (
                rhs_density,
                model_density,
                residual,
                row_scale,
                scaled_residual,
                relative,
                rhs_density_physical,
                model_density * component_mass_msun,
            )
        ),
        header=(
            "target_unit_mass model_unit_mass residual_unit_mass "
            "row_scale_unit_mass scaled_residual "
            "relative_residual_above_floor target_msun model_msun"
        ),
    )

    if save_orbit_table:
        np.savetxt(
            output_dir / f"{label.lower()}_orbit_library.txt",
            np.column_stack(
                (
                    initcond,
                    weights,
                    weights * component_mass_msun,
                    weightprior,
                    inttimes,
                    orbit_rperi,
                    orbit_rapo,
                    orbit_energy_rmax,
                )
            ),
            header=(
                "x y z vx vy vz optimized_weight_fraction "
                "optimized_weight_msun prior_weight integration_time "
                "rperi_kpc rapo_kpc energy_Rmax_kpc"
            ),
        )

    return trajectories, weights, diagnostics

def scale_orbit_library_sample(
    sample: tuple[np.ndarray, np.ndarray],
    target_mass_msun: float,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    xv = np.asarray(sample[0], dtype=np.float64)
    mass = np.asarray(sample[1], dtype=np.float64)
    total = float(np.sum(mass))
    if not np.isfinite(total) or total <= 0:
        raise RuntimeError(f"{label}: sampled orbit library has invalid mass {total}")
    mass *= float(target_mass_msun) / total
    return xv, mass


def symmetrize_azimuthal_velocity(
    sample: tuple[np.ndarray, np.ndarray],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove net streaming while retaining the orbit-library density and speed
    distribution by assigning equally probable signs to |v_phi|.

    In an axisymmetric potential, changing the sign of v_phi maps a phase-space
    point onto the corresponding counter-rotating orbit.
    """
    xv = np.asarray(sample[0], dtype=np.float64).copy()
    mass = np.asarray(sample[1], dtype=np.float64)

    x = xv[:, 0]
    y = xv[:, 1]
    vx = xv[:, 3]
    vy = xv[:, 4]
    radius = np.hypot(x, y)

    good = radius > 0
    v_radial = np.zeros_like(radius)
    v_phi = np.zeros_like(radius)
    v_radial[good] = (x[good] * vx[good] + y[good] * vy[good]) / radius[good]
    v_phi[good] = (x[good] * vy[good] - y[good] * vx[good]) / radius[good]

    signs = np.ones(len(xv), dtype=np.float64)
    signs[: len(signs) // 2] = -1.0
    rng.shuffle(signs)
    v_phi = np.abs(v_phi) * signs

    xv[good, 3] = (
        v_radial[good] * x[good] / radius[good]
        - v_phi[good] * y[good] / radius[good]
    )
    xv[good, 4] = (
        v_radial[good] * y[good] / radius[good]
        + v_phi[good] * x[good] / radius[good]
    )
    return xv, mass





# -----------------------------------------------------------------------------
# GalIC-equivalent halo spin
# -----------------------------------------------------------------------------

def galic_spin_energy_factor(concentration: float) -> float:
    """
    GalIC's f_c(c) factor used in its Peebles-spin angular-momentum scale.

    GalIC computes

        J_lambda = lambda * sqrt(G) * M200^(3/2)
                   * sqrt(2 R200 / f_c).

    This is a global angular-momentum normalization. It is distinct from
    GalIC's component StreamingVelocityParameter, which controls the local
    Satoh-like streaming field.
    """
    c = float(concentration)
    if not np.isfinite(c) or c <= 0:
        raise ValueError(f"concentration must be positive, got {c}")
    denominator = math.log1p(c) - c / (1.0 + c)
    numerator = c * (
        0.5
        - 0.5 / (1.0 + c) ** 2
        - math.log1p(c) / (1.0 + c)
    )
    value = numerator / denominator**2
    if not np.isfinite(value) or value <= 0:
        raise RuntimeError(f"invalid GalIC f_c({c})={value}")
    return value


def galic_spin_angular_momentum_unit(
    m200_msun: float,
    r200_kpc: float,
    concentration: float,
) -> float:
    """
    Return J corresponding to GalIC lambda=1 in Msun kpc km/s.
    """
    m200 = float(m200_msun)
    r200 = float(r200_kpc)
    if m200 <= 0 or r200 <= 0:
        raise ValueError("M200 and r200 must be positive")
    fc = galic_spin_energy_factor(concentration)
    return (
        math.sqrt(GALIC_G_KPC_KMS2_MSUN)
        * m200**1.5
        * math.sqrt(2.0 * r200 / fc)
    )


def resolve_halo_spin_target(
    args: argparse.Namespace,
    galic_lambda: float | None,
) -> None:
    """
    Resolve the requested halo spin and store it on args.

    An explicit --halo-spin-lambda takes precedence over
    --halo-spin-from-galic-lambda.
    """
    if args.halo_spin_lambda is not None:
        target = float(args.halo_spin_lambda)
        source = "explicit --halo-spin-lambda"
    elif args.halo_spin_from_galic_lambda:
        if galic_lambda is None:
            raise ValueError(
                "--halo-spin-from-galic-lambda was requested, but no LAMBDA "
                "was available from the GalIC parameter file"
            )
        target = float(galic_lambda)
        source = "GalIC LAMBDA"
    else:
        target = None
        source = "none"

    if target is not None:
        if not np.isfinite(target) or target < 0:
            raise ValueError("halo spin lambda must be finite and non-negative")
        if args.halo_spin_sign not in (-1, 1):
            raise ValueError("--halo-spin-sign must be -1 or +1")
        if args.halo_spin_relative_tolerance <= 0:
            raise ValueError("--halo-spin-relative-tolerance must be positive")

    args._halo_spin_target_lambda = target
    args._halo_spin_target_signed_lambda = (
        None if target is None else float(args.halo_spin_sign) * target
    )
    args._halo_spin_source = source


def halo_spin_diagnostics(
    sample: tuple[np.ndarray, np.ndarray],
    *,
    m200_msun: float,
    r200_kpc: float,
    concentration: float,
) -> dict[str, Any]:
    """
    Measure total halo angular momentum and GalIC-equivalent spin.

    The primary lambda diagnostic uses all live halo particles, matching the
    global angular-momentum target imposed by apply_galic_equivalent_halo_spin.
    Values restricted to r<=r200 are also reported for interpretation.
    """
    xv = np.asarray(sample[0], dtype=np.float64)
    mass = np.asarray(sample[1], dtype=np.float64)
    if xv.ndim != 2 or xv.shape[1] != 6 or mass.shape != (len(xv),):
        raise ValueError("halo sample must contain an (N,6) phase-space array")
    if not np.all(np.isfinite(xv)) or not np.all(np.isfinite(mass)):
        raise ValueError("halo sample contains non-finite values")
    if np.any(mass <= 0):
        raise ValueError("halo particle masses must be positive")

    position = xv[:, :3]
    velocity = xv[:, 3:]
    angular_momentum_particles = np.cross(position, velocity) * mass[:, None]
    angular_momentum = np.sum(angular_momentum_particles, axis=0)
    lz_specific = position[:, 0] * velocity[:, 1] - position[:, 1] * velocity[:, 0]
    jz_max_sign_bias = float(np.sum(mass * np.abs(lz_specific)))
    j_unit = galic_spin_angular_momentum_unit(
        m200_msun, r200_kpc, concentration
    )

    cylindrical_radius = np.hypot(position[:, 0], position[:, 1])
    good = cylindrical_radius > 0
    vphi = np.zeros(len(xv), dtype=np.float64)
    vphi[good] = (
        position[good, 0] * velocity[good, 1]
        - position[good, 1] * velocity[good, 0]
    ) / cylindrical_radius[good]

    total_mass = float(np.sum(mass))
    prograde_mass = float(np.sum(mass[lz_specific > 0]))
    retrograde_mass = float(np.sum(mass[lz_specific < 0]))
    zero_lz_mass = total_mass - prograde_mass - retrograde_mass

    radius = np.linalg.norm(position, axis=1)
    inside = radius <= float(r200_kpc)
    j_inside = np.sum(angular_momentum_particles[inside], axis=0)
    mass_inside = float(np.sum(mass[inside]))

    v200 = math.sqrt(
        GALIC_G_KPC_KMS2_MSUN * float(m200_msun) / float(r200_kpc)
    )
    bullock_denominator = (
        math.sqrt(2.0)
        * float(m200_msun)
        * v200
        * float(r200_kpc)
    )

    center = np.sum(position * mass[:, None], axis=0) / total_mass
    bulk = np.sum(velocity * mass[:, None], axis=0) / total_mass
    kinetic = 0.5 * float(np.sum(mass * np.sum(velocity**2, axis=1)))

    return {
        "definition": (
            "GalIC Peebles-equivalent lambda: "
            "Jz/[sqrt(G) M200^(3/2) sqrt(2 R200/f_c(c))]"
        ),
        "m200_reference_msun": float(m200_msun),
        "r200_reference_kpc": float(r200_kpc),
        "concentration_reference": float(concentration),
        "galic_fc": galic_spin_energy_factor(concentration),
        "angular_momentum_unit_lambda1_msun_kpc_kms": j_unit,
        "angular_momentum_vector_msun_kpc_kms": angular_momentum.tolist(),
        "Jz_msun_kpc_kms": float(angular_momentum[2]),
        "lambda_galic_equivalent": float(angular_momentum[2] / j_unit),
        "lambda_bullock_equivalent": float(
            angular_momentum[2] / bullock_denominator
        ),
        "maximum_abs_Jz_from_sign_bias_msun_kpc_kms": jz_max_sign_bias,
        "maximum_abs_lambda_from_sign_bias": float(jz_max_sign_bias / j_unit),
        "total_live_halo_mass_msun": total_mass,
        "mass_within_r200_msun": mass_inside,
        "Jz_within_r200_msun_kpc_kms": float(j_inside[2]),
        "lambda_galic_equivalent_using_Jz_within_r200": float(
            j_inside[2] / j_unit
        ),
        "prograde_mass_fraction": prograde_mass / total_mass,
        "retrograde_mass_fraction": retrograde_mass / total_mass,
        "zero_Lz_mass_fraction": zero_lz_mass / total_mass,
        "mass_weighted_mean_vphi_kms": float(
            np.sum(mass * vphi) / total_mass
        ),
        "center_of_mass_kpc": center.tolist(),
        "bulk_velocity_kms": bulk.tolist(),
        "kinetic_energy_msun_kms2": kinetic,
    }


def apply_galic_equivalent_halo_spin(
    sample: tuple[np.ndarray, np.ndarray],
    *,
    target_lambda: float,
    spin_sign: int,
    m200_msun: float,
    r200_kpc: float,
    concentration: float,
    rng: np.random.Generator,
    relative_tolerance: float,
) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, Any]]:
    """
    Impose a global GalIC-equivalent halo spin by changing only signs of v_phi.

    The random ordering is independent of phase-space coordinates. A prefix of
    that ordering is assigned prograde and the remainder retrograde, choosing
    the split that most closely matches the requested total Jz. Since only the
    sign of v_phi is changed, particle positions, speeds, energies, density,
    and all even velocity moments are preserved in the axisymmetric potential.
    """
    target_lambda = float(target_lambda)
    spin_sign = int(spin_sign)
    if target_lambda < 0 or not np.isfinite(target_lambda):
        raise ValueError("target halo lambda must be finite and non-negative")
    if spin_sign not in (-1, 1):
        raise ValueError("halo spin sign must be -1 or +1")
    if relative_tolerance <= 0:
        raise ValueError("halo spin relative tolerance must be positive")

    xv = np.asarray(sample[0], dtype=np.float64).copy()
    mass = np.asarray(sample[1], dtype=np.float64).copy()
    before = halo_spin_diagnostics(
        (xv, mass),
        m200_msun=m200_msun,
        r200_kpc=r200_kpc,
        concentration=concentration,
    )

    x, y = xv[:, 0], xv[:, 1]
    vx, vy = xv[:, 3], xv[:, 4]
    cylindrical_radius = np.hypot(x, y)
    good = cylindrical_radius > 0
    good_indices = np.flatnonzero(good)
    if good_indices.size == 0:
        raise RuntimeError("cannot impose halo spin: all particles have R=0")

    v_radial = np.zeros(len(xv), dtype=np.float64)
    v_phi = np.zeros(len(xv), dtype=np.float64)
    v_radial[good] = (
        x[good] * vx[good] + y[good] * vy[good]
    ) / cylindrical_radius[good]
    v_phi[good] = (
        x[good] * vy[good] - y[good] * vx[good]
    ) / cylindrical_radius[good]

    contribution = mass[good_indices] * cylindrical_radius[good_indices] * np.abs(
        v_phi[good_indices]
    )
    jmax = float(np.sum(contribution))
    j_unit = galic_spin_angular_momentum_unit(
        m200_msun, r200_kpc, concentration
    )
    signed_target_lambda = spin_sign * target_lambda
    target_jz = signed_target_lambda * j_unit

    if abs(target_jz) > jmax * (1.0 + 1e-12):
        raise RuntimeError(
            f"Requested halo lambda={signed_target_lambda:.6g} requires "
            f"|Jz|={abs(target_jz):.6e}, but sign-biasing this sample can "
            f"provide at most {jmax:.6e} "
            f"(lambda_max={jmax/j_unit:.6g}). Increase rotational support in "
            "the source DF/orbit library or request a smaller spin."
        )

    random_order = rng.permutation(good_indices.size)
    ordered_contribution = contribution[random_order]
    candidate_jz = np.concatenate(
        (
            np.asarray([-jmax]),
            -jmax + 2.0 * np.cumsum(ordered_contribution),
        )
    )
    number_prograde = int(np.argmin(np.abs(candidate_jz - target_jz)))

    signs_good = np.full(good_indices.size, -1.0)
    if number_prograde:
        signs_good[random_order[:number_prograde]] = 1.0

    desired_vphi = np.abs(v_phi[good_indices]) * signs_good
    xv[good_indices, 3] = (
        v_radial[good_indices] * x[good_indices] / cylindrical_radius[good_indices]
        - desired_vphi * y[good_indices] / cylindrical_radius[good_indices]
    )
    xv[good_indices, 4] = (
        v_radial[good_indices] * y[good_indices] / cylindrical_radius[good_indices]
        + desired_vphi * x[good_indices] / cylindrical_radius[good_indices]
    )

    # Remove the tiny finite-sampling bulk velocity introduced by sign choices.
    total_mass = float(np.sum(mass))
    bulk_after_sign_bias = np.sum(xv[:, 3:] * mass[:, None], axis=0) / total_mass
    xv[:, 3:] -= bulk_after_sign_bias

    after = halo_spin_diagnostics(
        (xv, mass),
        m200_msun=m200_msun,
        r200_kpc=r200_kpc,
        concentration=concentration,
    )
    achieved = float(after["lambda_galic_equivalent"])
    if target_lambda > 0:
        relative_error = abs(
            (achieved - signed_target_lambda) / signed_target_lambda
        )
        if relative_error > relative_tolerance:
            raise RuntimeError(
                f"Achieved halo lambda={achieved:.8g}, target="
                f"{signed_target_lambda:.8g}; relative error "
                f"{relative_error:.3%} exceeds "
                f"{relative_tolerance:.3%}"
            )
    else:
        relative_error = abs(achieved)
        if relative_error > relative_tolerance:
            raise RuntimeError(
                f"Achieved residual halo lambda={achieved:.8g} exceeds "
                f"the absolute zero-spin tolerance {relative_tolerance:.8g}"
            )

    speed2_before = np.sum(np.asarray(sample[0])[:, 3:] ** 2, axis=1)
    speed2_after = np.sum(xv[:, 3:] ** 2, axis=1)
    speed2_fractional_rms = float(
        np.sqrt(
            np.mean(
                ((speed2_after - speed2_before) / np.maximum(speed2_before, 1e-30))
                ** 2
            )
        )
    )

    diagnostics = {
        "applied": True,
        "method": (
            "random phase-independent prograde/retrograde v_phi sign bias"
        ),
        "target_lambda_unsigned": target_lambda,
        "spin_sign": spin_sign,
        "target_lambda_signed": signed_target_lambda,
        "target_Jz_msun_kpc_kms": target_jz,
        "number_particles": int(len(xv)),
        "number_particles_with_R_gt_0": int(good_indices.size),
        "number_assigned_prograde": number_prograde,
        "number_assigned_retrograde": int(good_indices.size - number_prograde),
        "bulk_velocity_removed_after_sign_bias_kms":
            bulk_after_sign_bias.tolist(),
        "relative_lambda_error": relative_error,
        "speed_squared_fractional_rms_change_after_final_recentering":
            speed2_fractional_rms,
        "before": before,
        "after": after,
    }

    print(
        "\nHalo spin calibration\n"
        "---------------------\n"
        f"Target GalIC-equivalent lambda : {signed_target_lambda:.8f}\n"
        f"Achieved lambda                : {achieved:.8f}\n"
        f"Maximum sign-bias lambda       : "
        f"{after['maximum_abs_lambda_from_sign_bias']:.8f}\n"
        f"Prograde mass fraction         : "
        f"{after['prograde_mass_fraction']:.6f}\n"
        f"Jz                             : "
        f"{after['Jz_msun_kpc_kms']:.6e} Msun kpc km/s"
    )

    return (xv, mass), diagnostics

def validate_sample_radius(
    sample: tuple[np.ndarray, np.ndarray],
    *,
    label: str,
    maximum_apocentre_kpc: float,
    tolerance_factor: float,
) -> dict[str, Any]:
    """Abort if the sampled realization exceeds the bounded orbit library."""
    xv = np.asarray(sample[0], dtype=np.float64)
    radius = np.linalg.norm(xv[:, :3], axis=1)
    allowed = float(maximum_apocentre_kpc) * float(tolerance_factor)
    percentiles = np.percentile(radius, [50.0, 90.0, 99.0, 99.9, 99.99])
    maximum = float(np.max(radius))
    number_over_nominal = int(
        np.count_nonzero(radius > float(maximum_apocentre_kpc))
    )
    number_over_allowed = int(np.count_nonzero(radius > allowed))

    diagnostics = {
        "maximum_apocentre_kpc": float(maximum_apocentre_kpc),
        "tolerance_factor": float(tolerance_factor),
        "allowed_sample_radius_kpc": allowed,
        "radius_percentiles_kpc": {
            "p50": float(percentiles[0]),
            "p90": float(percentiles[1]),
            "p99": float(percentiles[2]),
            "p99_9": float(percentiles[3]),
            "p99_99": float(percentiles[4]),
            "maximum": maximum,
        },
        "particles_over_nominal_apocentre_limit": number_over_nominal,
        "particles_over_allowed_radius": number_over_allowed,
    }

    print(
        f"{label} sampled-radius check: rmax={maximum:.6g} kpc; "
        f"allowed={allowed:.6g} kpc; "
        f"N(r>{maximum_apocentre_kpc:g})={number_over_nominal}"
    )
    if number_over_allowed:
        raise RuntimeError(
            f"{label} sampled realization contains {number_over_allowed} "
            f"particle(s) beyond {allowed:.6g} kpc, even though the orbit "
            f"library was limited to rapo <= {maximum_apocentre_kpc:.6g} "
            "kpc. Inspect Potential.Rperiapo, the stored trajectories, and "
            "sampleOrbitLibrary before using this IC."
        )
    return diagnostics

def intrinsic_shape_half_mass(
    sample: tuple[np.ndarray, np.ndarray],
) -> dict[str, float]:
    xv = np.asarray(sample[0], dtype=np.float64)
    mass = np.asarray(sample[1], dtype=np.float64)
    pos = xv[:, :3]

    center = np.average(pos, axis=0, weights=mass)
    pos = pos - center
    radius = np.linalg.norm(pos, axis=1)
    order = np.argsort(radius)
    cumulative = np.cumsum(mass[order])
    half_index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1]))
    half_index = min(max(half_index, 0), len(order) - 1)
    rhalf = float(radius[order[half_index]])

    selected = radius <= rhalf
    p = pos[selected]
    m = mass[selected]
    tensor = (p.T * m) @ p / np.sum(m)
    eigenvalues = np.linalg.eigvalsh(tensor)[::-1]

    return {
        "half_mass_radius_kpc": rhalf,
        "b_over_a_half_mass": float(np.sqrt(eigenvalues[1] / eigenvalues[0])),
        "c_over_a_half_mass": float(np.sqrt(eigenvalues[2] / eigenvalues[0])),
    }

def build_model(args: argparse.Namespace, par: HostParameters, output_dir: Path) -> None:
    agama = import_agama()
    agama.setUnits(length=1.0, velocity=1.0, mass=1.0)
    agama.setRandomSeed(par.random_seed)
    rng = np.random.default_rng(par.random_seed)

    density_stars, density_halo, halo_profile = make_densities(
        agama, par, args
    )
    pot_stars, pot_halo, total_potential = make_component_potentials(
        agama, par, halo_profile
    )

    density_stars.export(str(output_dir / "density_stars_target"))
    density_halo.export(str(output_dir / "density_halo_target"))
    pot_stars.export(str(output_dir / "potential_stars_target"))
    pot_halo.export(str(output_dir / "potential_halo_target"))
    total_potential.export(str(output_dir / "potential_total_target"))
    rotation_curve_table(output_dir / "rotation_curve.txt", total_potential, par)

    (output_dir / "finite_halo_profile.json").write_text(
        json.dumps(
            {
                key: value
                for key, value in halo_profile.items()
                if key != "agama_parameters"
            },
            indent=2,
            sort_keys=True,
        ) + "\n"
    )

    # Stellar density target.
    star_rmin = (
        float(args.star_target_rmin_kpc)
        if args.star_target_rmin_kpc is not None
        else 0.02 * par.re_major_kpc
    )
    star_rmax = (
        float(args.star_target_rmax_kpc)
        if args.star_target_rmax_kpc is not None
        else 20.0 * par.re_major_kpc
    )
    target_stars = component_target_grid(
        agama,
        rmin_kpc=star_rmin,
        rmax_kpc=star_rmax,
        bins=args.star_target_bins,
        lmax=args.star_target_lmax,
    )

    # Halo target covers the resolved inner halo through the smooth cutoff.
    halo_rmin = (
        float(args.halo_target_rmin_kpc)
        if args.halo_target_rmin_kpc is not None
        else 0.01 * par.halo_scale_radius_hernquist_kpc
    )
    halo_rmax = (
        float(args.halo_target_rmax_kpc)
        if args.halo_target_rmax_kpc is not None
        else halo_profile["cutoff_radius_kpc"]
    )
    target_halo = component_target_grid(
        agama,
        rmin_kpc=halo_rmin,
        rmax_kpc=halo_rmax,
        bins=args.halo_target_bins,
        lmax=args.halo_target_lmax,
    )

    star_max_apocentre = (
        float(args.star_max_apocentre_kpc)
        if args.star_max_apocentre_kpc is not None
        else 2.0 * star_rmax
    )
    halo_max_apocentre = (
        float(args.halo_max_apocentre_kpc)
        if args.halo_max_apocentre_kpc is not None
        else 2.0 * halo_profile["cutoff_radius_kpc"]
    )
    if star_max_apocentre < star_rmax:
        raise ValueError(
            f"--star-max-apocentre-kpc={star_max_apocentre:g} is smaller "
            f"than the stellar target outer radius {star_rmax:g} kpc"
        )
    if halo_max_apocentre < halo_rmax:
        raise ValueError(
            f"--halo-max-apocentre-kpc={halo_max_apocentre:g} is smaller "
            f"than the halo target outer radius {halo_rmax:g} kpc"
        )
    print(
        "\nBounded orbit-library limits\n"
        "----------------------------\n"
        f"Stellar maximum apocentre: {star_max_apocentre:.6f} kpc\n"
        f"Halo maximum apocentre:    {halo_max_apocentre:.6f} kpc"
    )

    star_traj, star_weights, star_diag = solve_orbit_component(
        agama,
        label="Stars",
        source_component=pot_stars,
        total_potential=total_potential,
        component_mass_msun=par.stellar_mass_msun,
        target=target_stars,
        number_orbits=args.star_orbits,
        integration_periods=args.star_periods,
        beta=args.star_beta,
        kappa=args.star_kappa,
        orbit_accuracy=args.orbit_accuracy,
        regularization=args.orbit_regularization,
        density_constraint_mode=args.density_constraint_mode,
        density_tolerance=args.density_tolerance,
        density_scale_floor=args.density_scale_floor,
        maximum_apocentre_kpc=star_max_apocentre,
        orbit_sampling_batch_factor=args.orbit_sampling_batch_factor,
        orbit_sampling_max_batches=args.orbit_sampling_max_batches,
        output_dir=output_dir,
        save_orbit_table=args.save_orbit_tables,
    )

    print(f"\n[Stars] Sampling {par.n_star} N-body particles")
    star_sample = agama.sampleOrbitLibrary(
        par.n_star, star_traj, star_weights
    )
    star_sample = scale_orbit_library_sample(
        star_sample, par.stellar_mass_msun, "Stars"
    )
    if args.enforce_nonrotation:
        star_sample = symmetrize_azimuthal_velocity(star_sample, rng)
    del star_traj, star_weights

    halo_traj, halo_weights, halo_diag = solve_orbit_component(
        agama,
        label="Halo",
        source_component=pot_halo,
        total_potential=total_potential,
        component_mass_msun=halo_profile["total_live_halo_mass_msun"],
        target=target_halo,
        number_orbits=args.halo_orbits,
        integration_periods=args.halo_periods,
        beta=args.halo_beta,
        kappa=args.halo_kappa,
        orbit_accuracy=args.orbit_accuracy,
        regularization=args.halo_orbit_regularization,
        density_constraint_mode=args.halo_density_constraint_mode,
        density_tolerance=args.halo_density_tolerance,
        density_scale_floor=args.halo_density_scale_floor,
        maximum_apocentre_kpc=halo_max_apocentre,
        orbit_sampling_batch_factor=args.orbit_sampling_batch_factor,
        orbit_sampling_max_batches=args.orbit_sampling_max_batches,
        output_dir=output_dir,
        save_orbit_table=args.save_orbit_tables,
    )

    print(f"\n[Halo] Sampling {par.n_halo} N-body particles")
    halo_sample = agama.sampleOrbitLibrary(
        par.n_halo, halo_traj, halo_weights
    )
    halo_sample = scale_orbit_library_sample(
        halo_sample,
        halo_profile["total_live_halo_mass_msun"],
        "Halo",
    )
    if args.enforce_halo_nonrotation:
        halo_sample = symmetrize_azimuthal_velocity(halo_sample, rng)
    del halo_traj, halo_weights

    star_sample, halo_sample, centering_diag = recenter_samples(
        star_sample, halo_sample
    )

    if getattr(args, "_halo_spin_target_lambda", None) is not None:
        halo_spin_rng = np.random.default_rng(
            par.random_seed + int(args.halo_spin_seed_offset)
        )
        halo_sample, halo_spin_diag = apply_galic_equivalent_halo_spin(
            halo_sample,
            target_lambda=args._halo_spin_target_lambda,
            spin_sign=args.halo_spin_sign,
            m200_msun=par.m200_msun,
            r200_kpc=par.r200_kpc,
            concentration=par.concentration,
            rng=halo_spin_rng,
            relative_tolerance=args.halo_spin_relative_tolerance,
        )
    else:
        halo_spin_diag = {
            "applied": False,
            "source": getattr(args, "_halo_spin_source", "none"),
            "after": halo_spin_diagnostics(
                halo_sample,
                m200_msun=par.m200_msun,
                r200_kpc=par.r200_kpc,
                concentration=par.concentration,
            ),
        }

    star_radius_diag = validate_sample_radius(
        star_sample,
        label="Stars",
        maximum_apocentre_kpc=star_max_apocentre,
        tolerance_factor=args.sample_radius_tolerance_factor,
    )
    halo_radius_diag = validate_sample_radius(
        halo_sample,
        label="Halo",
        maximum_apocentre_kpc=halo_max_apocentre,
        tolerance_factor=args.sample_radius_tolerance_factor,
    )

    star_shape = intrinsic_shape_half_mass(star_sample)
    halo_shape = intrinsic_shape_half_mass(halo_sample)
    print(
        "\nSampled intrinsic half-mass shapes:\n"
        f"  stars: b/a={star_shape['b_over_a_half_mass']:.5f}, "
        f"c/a={star_shape['c_over_a_half_mass']:.5f}\n"
        f"  halo:  b/a={halo_shape['b_over_a_half_mass']:.5f}, "
        f"c/a={halo_shape['c_over_a_half_mass']:.5f}"
    )

    diagnostics = {
        "architecture": {
            "stars": "Schwarzschild orbit superposition",
            "halo": "Schwarzschild orbit superposition",
        },
        "halo_profile": {
            key: value
            for key, value in halo_profile.items()
            if key != "agama_parameters"
        },
        "stellar_orbit_library": star_diag,
        "halo_orbit_library": halo_diag,
        "sample_centering": centering_diag,
        "sample_radius_validation": {
            "stars": star_radius_diag,
            "halo": halo_radius_diag,
        },
        "sampled_stellar_shape": star_shape,
        "sampled_halo_shape": halo_shape,
        "halo_spin": halo_spin_diag,
        "target_stellar_intrinsic_q": par.intrinsic_q,
        "target_halo_intrinsic_q": par.halo_q,
        "galic_mapping": {
            "parameter_file": par.galic_parameter_file,
            "file_usable_for_host": par.galic_file_usable_for_host,
            "mismatch_reason": par.galic_mismatch_reason,
            "hubble_kms_kpc": par.galic_hubble_kms_kpc,
            "V200_kms": par.galic_v200_kms,
            "CC": par.galic_concentration,
            "LAMBDA": par.galic_lambda,
            "MD": par.galic_md,
            "MB": par.galic_mb,
            "MBH": par.galic_mbh,
            "HaloStretch": par.galic_halo_stretch,
            "BulgeStretch": par.galic_bulge_stretch,
            "halo_velocity_structure": par.galic_halo_velocity_structure,
            "bulge_velocity_structure": par.galic_bulge_velocity_structure,
            "halo_streaming": par.galic_halo_streaming,
            "bulge_streaming": par.galic_bulge_streaming,
            "softening_kpc": par.galic_softening_kpc,
            "stellar_mass_source": par.stellar_mass_source,
            "stellar_q_source": par.stellar_q_source,
            "halo_q_source": par.halo_q_source,
        },
        "stellar_nonrotation_symmetrization_applied": bool(
            args.enforce_nonrotation
        ),
        "halo_pre_spin_nonrotation_symmetrization_applied": bool(
            args.enforce_halo_nonrotation
        ),
        "target_ranges_kpc": {
            "stars": [star_rmin, star_rmax],
            "halo": [halo_rmin, halo_rmax],
        },
        "maximum_apocentres_kpc": {
            "stars": star_max_apocentre,
            "halo": halo_max_apocentre,
        },
        "target_grids": {
            "stars": {
                "bins": int(args.star_target_bins),
                "lmax": int(args.star_target_lmax),
            },
            "halo": {
                "bins": int(args.halo_target_bins),
                "lmax": int(args.halo_target_lmax),
            },
        },
    }
    (output_dir / "schwarzschild_v10_spin_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2, sort_keys=True) + "\n"
    )

    hdf5_path = output_dir / args.output_hdf5
    write_gadget_hdf5(
        hdf5_path, star_sample, halo_sample, par, halo_spin_diag
    )
    print(f"\nWrote GADGET-4 HDF5 host IC: {hdf5_path}")

    if args.write_agama_gadget:
        for label, sample in (("stars", star_sample), ("halo", halo_sample)):
            try:
                agama.writeSnapshot(
                    str(output_dir / f"agama_{label}_gadget"),
                    sample,
                    "gadget",
                )
            except Exception as exc:
                print(
                    f"Warning: AGAMA native Gadget output for {label} failed: "
                    f"{exc}",
                    file=sys.stderr,
                )


# -----------------------------------------------------------------------------
# Run-configuration support
# -----------------------------------------------------------------------------

CONFIG_METADATA_KEYS = {
    "config_version",
    "description",
    "notes",
    "parameters",
}


def _load_mapping_file(path: Path) -> dict[str, Any]:
    """Read a JSON or Python-literal dictionary."""
    text = path.read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        try:
            value = ast.literal_eval(text)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Could not parse run configuration {path} as JSON or a "
                "Python dictionary"
            ) from exc
    if not isinstance(value, dict):
        raise ValueError(f"Run configuration {path} must contain a dictionary")
    return value


def _normalize_config_key(key: str) -> str:
    normalized = str(key).strip()
    while normalized.startswith("-"):
        normalized = normalized[1:]
    return normalized.replace("-", "_")


def _resolved_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _resolved_json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_resolved_json_value(v) for v in value]
    return value


def load_run_configuration(
    parser: argparse.ArgumentParser,
    config_filename: str,
    relative_path_keys: set[str],
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    """
    Load config defaults and validate them against argparse destinations.

    Accepted layouts are either a flat mapping of argparse destination names,
    or a document with a nested ``parameters`` mapping. Keys may use
    underscores (``star_orbits``) or CLI spelling (``star-orbits`` or
    ``--star-orbits``).
    """
    config_path = Path(config_filename).expanduser().resolve()
    raw = _load_mapping_file(config_path)

    if "parameters" in raw:
        parameters = raw["parameters"]
        if not isinstance(parameters, dict):
            raise ValueError(
                f"{config_path}: 'parameters' must contain a dictionary"
            )
    else:
        parameters = {
            key: value
            for key, value in raw.items()
            if key not in CONFIG_METADATA_KEYS
        }

    normalized: dict[str, Any] = {}
    for key, value in parameters.items():
        destination = _normalize_config_key(key)
        if destination in normalized:
            raise ValueError(
                f"{config_path}: duplicate setting for {destination!r}"
            )
        normalized[destination] = value

    valid_destinations = {
        action.dest
        for action in parser._actions
        if action.dest not in ("help", argparse.SUPPRESS)
    }
    unknown = sorted(set(normalized) - valid_destinations)
    if unknown:
        raise ValueError(
            f"{config_path}: unknown configuration parameter(s): "
            + ", ".join(unknown)
        )

    # Paths in a portable config file are interpreted relative to that file.
    for destination in relative_path_keys:
        value = normalized.get(destination)
        if value is None or not isinstance(value, str) or value.strip() == "":
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = config_path.parent / candidate
        normalized[destination] = str(candidate.resolve())

    return config_path, normalized, raw


def parse_args_with_configuration(
    parser: argparse.ArgumentParser,
    relative_path_keys: set[str],
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """
    Parse a possible --config file, then parse the full command line.

    Precedence:
      explicit command-line value > config value > parser default.
    """
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config")
    preliminary, _ = pre_parser.parse_known_args(argv)

    config_path: Path | None = None
    loaded_parameters: dict[str, Any] = {}
    raw_config: dict[str, Any] = {}

    if preliminary.config:
        config_path, loaded_parameters, raw_config = load_run_configuration(
            parser,
            preliminary.config,
            relative_path_keys,
        )
        parser.set_defaults(**loaded_parameters)

    args = parser.parse_args(argv)
    args._run_config_file = str(config_path) if config_path else None
    args._run_config_loaded_parameters = loaded_parameters
    args._run_config_raw = raw_config
    return args


def resolved_run_configuration(args: argparse.Namespace) -> dict[str, Any]:
    parameters = {
        key: _resolved_json_value(value)
        for key, value in vars(args).items()
        if not key.startswith("_") and key != "config"
    }
    return {
        "config_file": getattr(args, "_run_config_file", None),
        "loaded_parameters": _resolved_json_value(
            getattr(args, "_run_config_loaded_parameters", {})
        ),
        "resolved_parameters": parameters,
        "precedence": "command line > config file > script defaults",
    }


def write_resolved_run_configuration(
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    path = output_dir / "resolved_run_configuration.json"
    path.write_text(
        json.dumps(
            resolved_run_configuration(args),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return path

def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build flattened Sérsic stars and a flattened finite halo using "
            "Schwarzschild orbit superposition for both components."
        )
    )
    parser.add_argument(
        "--config",
        help=(
            "JSON or Python-dictionary run configuration. Explicit CLI "
            "arguments override values from this file."
        ),
    )
    parser.add_argument(
        "model_json",
        nargs="?",
        help=(
            "Mass/halo model JSON produced from the photometric calculation. "
            "May be supplied by --config."
        ),
    )
    parser.add_argument(
        "galic_param",
        nargs="?",
        help=(
            "Optional host GalIC parameter file. It may instead be supplied "
            "with --galic-param-file."
        ),
    )
    parser.add_argument(
        "--galic-param-file",
        help="Named alternative to the optional galic_param positional input.",
    )
    parser.add_argument(
        "--photometry-json",
        help=(
            "Optional photometry parameter dictionary. Its exact Sérsic n and "
            "projected b/a take precedence over rounded values in model_json."
        ),
    )
    parser.add_argument(
        "--output-dir", default="agama_leda175897_schwarzschild_v10_spin"
    )
    parser.add_argument(
        "--output-hdf5",
        default="LEDA175897_agama_host_schwarzschild_v10_spin_ic.hdf5",
    )

    parser.add_argument("--n-star", type=int)
    parser.add_argument("--n-halo", type=int)
    parser.add_argument("--seed", type=int, default=12345)

    parser.add_argument(
        "--inclination-deg",
        type=float,
        default=90.0,
        help="Inclination of an oblate stellar host; 90 degrees is edge-on.",
    )
    parser.add_argument("--stellar-q", type=float)
    parser.add_argument(
        "--halo-q",
        type=float,
        help=(
            "Intrinsic oblate halo c/a. Default: GalIC HaloStretch when a "
            "valid file is supplied, otherwise 0.666666."
        ),
    )
    parser.add_argument(
        "--use-galic-stellar-stretch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use GalIC BulgeStretch for the intrinsic stellar q when no "
            "explicit --stellar-q is supplied. Default: true."
        ),
    )
    parser.add_argument(
        "--stellar-mass-source",
        choices=("auto", "model", "galic"),
        default="auto",
        help=(
            "auto uses MB times the GalIC total mass when a valid file is "
            "supplied, otherwise the model JSON stellar mass."
        ),
    )
    parser.add_argument("--m200-msun", type=float)
    parser.add_argument("--r200-kpc", type=float)
    parser.add_argument("--concentration", type=float)
    parser.add_argument("--halo-scale-radius-kpc", type=float)
    parser.add_argument("--projected-q", type=float)
    parser.add_argument("--sersic-n", type=float)
    parser.add_argument("--re-circularized-kpc", type=float)
    parser.add_argument("--re-major-kpc", type=float)

    # Halo mass normalization.
    parser.add_argument(
        "--halo-mass-normalization",
        choices=("galic-live-total", "galic-profile", "m200-enclosed"),
        default="galic-live-total",
        help=(
            "galic-live-total matches the finite HDF5 galaxy mass to the old "
            "GalIC total; galic-profile preserves the old untruncated "
            "Hernquist density amplitude before the cutoff; m200-enclosed "
            "retains the previous v6 convention."
        ),
    )
    parser.add_argument(
        "--galic-total-galaxy-mass-msun",
        type=float,
        help=(
            "Explicit GalIC total live galaxy mass. Default: derive from "
            "V200 in the GalIC file; without a file use 3.72397e13 Msun."
        ),
    )
    parser.add_argument(
        "--galic-hubble-kms-kpc",
        type=float,
        default=DEFAULT_GALIC_HUBBLE_KMS_KPC,
        help=(
            "H used to convert GalIC V200 into R200 and total M200. "
            "Default: 0.0699 km/s/kpc, matching the final host startup output."
        ),
    )
    parser.add_argument(
        "--galic-halo-mass-msun",
        type=float,
        help=(
            "Optional direct old-GalIC halo mass. Default is "
            "GalIC total galaxy mass minus the AGAMA stellar mass."
        ),
    )

    # Optional GalIC-equivalent halo spin.
    parser.add_argument(
        "--halo-spin-from-galic-lambda",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Impose a halo angular momentum matching LAMBDA from the GalIC "
            "parameter file. This is not enabled automatically because GalIC "
            "halo streaming is controlled separately."
        ),
    )
    parser.add_argument(
        "--halo-spin-lambda",
        type=float,
        help=(
            "Explicit non-negative GalIC-equivalent Peebles lambda for the "
            "live halo. Overrides --halo-spin-from-galic-lambda."
        ),
    )
    parser.add_argument(
        "--halo-spin-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help="Sign of halo Jz. Default: +1.",
    )
    parser.add_argument(
        "--halo-spin-relative-tolerance",
        type=float,
        default=0.002,
        help="Maximum relative lambda calibration error. Default: 0.002.",
    )
    parser.add_argument(
        "--halo-spin-seed-offset",
        type=int,
        default=61027,
        help=(
            "Offset added to the model seed for reproducible halo spin sign "
            "assignment. Default: 61027."
        ),
    )

    # Finite halo profile.
    parser.add_argument("--halo-cutoff-radius-kpc", type=float)
    parser.add_argument(
        "--halo-cutoff-r200-multiple",
        type=float,
        default=2.0,
    )
    parser.add_argument(
        "--halo-cutoff-strength",
        type=float,
        default=4.0,
    )

    # Stellar Schwarzschild controls.
    parser.add_argument("--star-orbits", type=int, default=20_000)
    parser.add_argument("--star-periods", type=float, default=100.0)
    parser.add_argument("--star-beta", type=float)
    parser.add_argument("--star-kappa", type=float)
    parser.add_argument("--star-target-bins", type=int, default=50)
    parser.add_argument("--star-target-lmax", type=int, default=8)
    parser.add_argument("--star-target-rmin-kpc", type=float)
    parser.add_argument("--star-target-rmax-kpc", type=float)

    # Halo Schwarzschild controls.
    parser.add_argument("--halo-orbits", type=int, default=40_000)
    parser.add_argument("--halo-periods", type=float, default=100.0)
    parser.add_argument("--halo-beta", type=float)
    parser.add_argument("--halo-kappa", type=float)
    parser.add_argument("--halo-target-bins", type=int, default=50)
    parser.add_argument("--halo-target-lmax", type=int, default=6)
    parser.add_argument("--halo-target-rmin-kpc", type=float)
    parser.add_argument("--halo-target-rmax-kpc", type=float)

    # Bounded orbit-library controls.
    parser.add_argument(
        "--star-max-apocentre-kpc",
        type=float,
        help=(
            "Reject stellar orbit-library candidates with larger apocentre. "
            "Default: twice the stellar target outer radius."
        ),
    )
    parser.add_argument(
        "--halo-max-apocentre-kpc",
        type=float,
        help=(
            "Reject halo orbit-library candidates with larger apocentre. "
            "Default: twice the finite halo cutoff radius."
        ),
    )
    parser.add_argument(
        "--orbit-sampling-batch-factor",
        type=float,
        default=1.5,
        help=(
            "Candidate oversampling factor used while assembling each bounded "
            "orbit library. Must exceed 1. Default: 1.5."
        ),
    )
    parser.add_argument(
        "--orbit-sampling-max-batches",
        type=int,
        default=50,
        help="Maximum candidate-sampling batches per component. Default: 50.",
    )
    parser.add_argument(
        "--sample-radius-tolerance-factor",
        type=float,
        default=1.05,
        help=(
            "Final sampled particles may extend this factor beyond the nominal "
            "apocentre limit before the run aborts. Default: 1.05."
        ),
    )

    # Shared and component-specific optimizer controls.
    parser.add_argument("--orbit-accuracy", type=float, default=1e-6)
    parser.add_argument(
        "--orbit-regularization",
        type=float,
        default=1.0,
        help="Stellar orbit-weight regularization.",
    )
    parser.add_argument(
        "--halo-orbit-regularization",
        type=float,
        default=10.0,
        help="Halo orbit-weight regularization. Default: 10.",
    )
    parser.add_argument(
        "--density-constraint-mode",
        choices=("auto", "exact", "soft"),
        default="soft",
        help="Stellar density-constraint mode.",
    )
    parser.add_argument(
        "--halo-density-constraint-mode",
        choices=("auto", "exact", "soft"),
        default="soft",
        help="Halo density-constraint mode.",
    )
    parser.add_argument("--density-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--halo-density-tolerance",
        type=float,
        default=0.015,
    )
    parser.add_argument("--density-scale-floor", type=float, default=1e-6)
    parser.add_argument(
        "--halo-density-scale-floor",
        type=float,
        default=1e-5,
        help=(
            "Halo target-row floor. A slightly larger value prevents tiny "
            "outer/high-order coefficients from dominating."
        ),
    )

    parser.add_argument(
        "--enforce-nonrotation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default is inferred from GalIC bulge streaming when available.",
    )
    parser.add_argument(
        "--enforce-halo-nonrotation",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Default is inferred from GalIC halo streaming when available.",
    )
    parser.add_argument("--save-orbit-tables", action="store_true")

    parser.add_argument(
        "--gadget-mass-unit-msun",
        type=float,
        default=1.0e10,
    )
    parser.add_argument("--write-agama-gadget", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    parser = make_parser()
    try:
        args = parse_args_with_configuration(
            parser,
            relative_path_keys={
                "model_json",
                "photometry_json",
                "galic_param",
                "galic_param_file",
                "output_dir",
            },
        )
    except ValueError as exc:
        parser.error(str(exc))

    if not args.model_json:
        parser.error(
            "model_json is required either positionally or in --config"
        )

    try:
        par = derive_parameters(args)
        resolve_halo_spin_target(args, par.galic_lambda)
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))

    if par.n_star <= 0 or par.n_halo <= 0:
        parser.error("resolved stellar and halo particle counts must be positive")
    if args.star_orbits < 100 or args.halo_orbits < 100:
        parser.error("--star-orbits and --halo-orbits must be at least 100")
    if args.star_periods <= 0 or args.halo_periods <= 0:
        parser.error("--star-periods and --halo-periods must be positive")
    if args.orbit_accuracy <= 0:
        parser.error("--orbit-accuracy must be positive")
    if args.orbit_regularization < 0 or args.halo_orbit_regularization < 0:
        parser.error("orbit regularization values cannot be negative")
    if args.density_tolerance <= 0 or args.halo_density_tolerance <= 0:
        parser.error("density tolerances must be positive")
    if not 0 < args.density_scale_floor < 1:
        parser.error("--density-scale-floor must lie between 0 and 1")
    if not 0 < args.halo_density_scale_floor < 1:
        parser.error("--halo-density-scale-floor must lie between 0 and 1")
    if not 0 < par.halo_q <= 1:
        parser.error("resolved halo q must be in (0,1]")
    if par.galic_total_galaxy_mass_msun <= 0:
        parser.error("resolved GalIC total galaxy mass must be positive")
    if args.galic_hubble_kms_kpc <= 0:
        parser.error("--galic-hubble-kms-kpc must be positive")
    if args.galic_halo_mass_msun is not None and args.galic_halo_mass_msun <= 0:
        parser.error("--galic-halo-mass-msun must be positive")
    for value, option in (
        (args.m200_msun, "--m200-msun"),
        (args.r200_kpc, "--r200-kpc"),
        (args.concentration, "--concentration"),
        (args.halo_scale_radius_kpc, "--halo-scale-radius-kpc"),
    ):
        if value is not None and value <= 0:
            parser.error(f"{option} must be positive")
    if args.halo_cutoff_radius_kpc is not None and args.halo_cutoff_radius_kpc <= 0:
        parser.error("--halo-cutoff-radius-kpc must be positive")
    if args.halo_cutoff_r200_multiple <= 0:
        parser.error("--halo-cutoff-r200-multiple must be positive")
    if args.halo_cutoff_strength <= 0:
        parser.error("--halo-cutoff-strength must be positive")
    if args.star_max_apocentre_kpc is not None and args.star_max_apocentre_kpc <= 0:
        parser.error("--star-max-apocentre-kpc must be positive")
    if args.halo_max_apocentre_kpc is not None and args.halo_max_apocentre_kpc <= 0:
        parser.error("--halo-max-apocentre-kpc must be positive")
    if args.orbit_sampling_batch_factor <= 1:
        parser.error("--orbit-sampling-batch-factor must exceed 1")
    if args.orbit_sampling_max_batches < 1:
        parser.error("--orbit-sampling-max-batches must be at least 1")
    if args.sample_radius_tolerance_factor < 1:
        parser.error("--sample-radius-tolerance-factor must be at least 1")
    if args.halo_spin_lambda is not None and args.halo_spin_lambda < 0:
        parser.error("--halo-spin-lambda must be non-negative")
    if args.halo_spin_relative_tolerance <= 0:
        parser.error("--halo-spin-relative-tolerance must be positive")

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = write_resolved_run_configuration(output_dir, args)
    print(f"Wrote resolved run configuration: {resolved_config_path}")

    checks = validate_halo_conversion(par)
    print_parameter_summary(par, checks, args)
    config_path = write_derived_config(output_dir, par, checks, args)
    print(f"\nWrote derived configuration: {config_path}")

    if args.dry_run:
        print("\nDry run complete; AGAMA was not imported.")
        return 0

    build_model(args, par, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
