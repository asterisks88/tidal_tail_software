#!/usr/bin/env python3
"""
combine_galaxies_nbody_hdf5_v2.py

Combine isolated galaxy HDF5 initial conditions and initialize a nominal
Keplerian encounter using masses measured directly from the input snapshots.

Major changes from the original combiner
----------------------------------------
1. The configured ``M`` and ``c`` values are no longer used to infer the host
   mass profile.
2. Each input snapshot is inspected directly:
     * total mass is summed from particle ``Masses``/``Mass`` datasets or
       ``Header/MassTable``;
     * mass-weighted position and velocity centres are measured;
     * enclosed mass is measured from particle masses at the actual initial
       separation.
3. The default orbital mass preserves the original initializer's
   convention, but measures both terms from the HDF5 files:

       M_orbit(r) = M_host(<r) + M_satellite,total.

   The ``both-enclosed`` and ``both-total`` alternatives are also available.
4. Full measured snapshot masses, not enclosed masses, are used to place the
   galaxy centres in the zero-centre-of-mass and zero-total-momentum frame.
5. Input internal centres and bulk velocities are removed before rotations
   and offsets are applied.
6. The combined file always contains per-particle ``Masses`` and has a zero
   ``MassTable``. This avoids mixing incompatible constant particle masses
   from different input files.
7. Missing optional datasets are padded to the correct particle count instead
   of producing inconsistent dataset lengths.
8. Detailed mass, orbit, centring, and ParticleID diagnostics are written to
   JSON sidecars and HDF5 Header attributes.

Assumed physical coordinate units
---------------------------------
Coordinates are converted to kpc, velocities to km/s, and masses to Msun
using per-galaxy configuration overrides or recognized HDF5 metadata.
The combined output stores:

    Coordinates : kpc
    Velocities  : km/s
    Masses      : output code mass units

The default output mass unit is the host input mass unit, normally 1e10 Msun.

Configuration
-------------
The existing Python-list configuration format remains valid. ``M`` and ``c``
may remain for provenance, but they are ignored by the orbital calculation.

Optional per-galaxy keys:

    mass_unit_msun     : override input code mass unit
    length_unit_kpc    : kpc represented by one coordinate unit, default 1
    velocity_unit_kms  : km/s represented by one velocity unit, default 1
    center             : explicit internal centre in input coordinate units
    bulk_velocity      : explicit internal bulk velocity in input units

The first galaxy is the host. Subsequent galaxies require the existing orbit
keys ``rp``, ``e``, ``f0``, ``i``, ``w``, and optionally ``omega``.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np

G = 4.3009e-6  # kpc (km/s)^2 Msun^-1
MSUN_IN_G = 1.98847e33
DEFAULT_GALIC_PATH = "~/galic_models"
DEFAULT_GADGET_TEMPLATE = "galaxy_collision_elliptical-gadget4_template.param"


@dataclass
class SnapshotInfo:
    name: str
    path: str
    mass_unit_msun: float
    length_unit_kpc: float
    velocity_unit_kms: float
    total_mass_msun: float
    center_kpc: list[float]
    bulk_velocity_kms: list[float]
    particle_counts: list[int]
    particle_masses_msun: list[float]
    id_min: int
    id_max: int
    maximum_radius_kpc: float
    mass_unit_source: str


@dataclass
class OrbitRecord:
    satellite_name: str
    separation_kpc: float
    host_total_mass_msun: float
    host_enclosed_mass_msun: float
    satellite_total_mass_msun: float
    satellite_enclosed_mass_msun: float
    orbit_mass_mode: str
    orbit_mass_msun: float
    relative_position_kpc: list[float]
    relative_velocity_kms: list[float]
    relative_speed_kms: float


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _attrs_sources(f: h5py.File) -> list[h5py.AttributeManager]:
    sources = [f.attrs]
    if "Header" in f:
        sources.insert(0, f["Header"].attrs)
    return sources


def _find_attr(f: h5py.File, names: tuple[str, ...]) -> Any | None:
    for attrs in _attrs_sources(f):
        for name in names:
            if name in attrs:
                value = attrs[name]
                arr = np.asarray(value)
                if arr.size == 1:
                    return arr.reshape(-1)[0]
                return value
    return None


def infer_mass_unit_msun(
    f: h5py.File,
    galaxy: dict[str, Any],
    default_mass_unit_msun: float,
) -> tuple[float, str]:
    if "mass_unit_msun" in galaxy:
        value = float(galaxy["mass_unit_msun"])
        if value <= 0:
            raise ValueError("mass_unit_msun must be positive")
        return value, "configuration mass_unit_msun"

    direct = _find_attr(
        f,
        (
            "AGAMA_GadgetMassUnit_Msun",
            "Combiner_MassUnit_Msun",
            "GadgetMassUnit_Msun",
            "UnitMass_in_Msun",
            "MassUnit_Msun",
            "UnitMass_Msun",
        ),
    )
    if direct is not None:
        value = float(direct)
        if np.isfinite(value) and value > 0:
            return value, "HDF5 mass-unit metadata"

    grams = _find_attr(
        f,
        (
            "UnitMass_in_g",
            "UnitMassInG",
            "UnitMass_in_grams",
        ),
    )
    if grams is not None:
        value = float(grams) / MSUN_IN_G
        if np.isfinite(value) and value > 0:
            return value, "HDF5 UnitMass_in_g metadata"

    if default_mass_unit_msun <= 0:
        raise ValueError("default mass unit must be positive")
    return float(default_mass_unit_msun), "command-line fallback"


def infer_length_unit_kpc(
    f: h5py.File,
    galaxy: dict[str, Any],
) -> tuple[float, str]:
    if "length_unit_kpc" in galaxy:
        value = float(galaxy["length_unit_kpc"])
        if value <= 0:
            raise ValueError("length_unit_kpc must be positive")
        return value, "configuration length_unit_kpc"

    value = _find_attr(
        f,
        (
            "AGAMA_GadgetLengthUnit_kpc",
            "Combiner_LengthUnit_kpc",
            "LengthUnit_kpc",
            "UnitLength_in_kpc",
        ),
    )
    if value is not None and float(value) > 0:
        return float(value), "HDF5 length-unit metadata"
    return 1.0, "default kpc coordinates"


def infer_velocity_unit_kms(
    f: h5py.File,
    galaxy: dict[str, Any],
) -> tuple[float, str]:
    if "velocity_unit_kms" in galaxy:
        value = float(galaxy["velocity_unit_kms"])
        if value <= 0:
            raise ValueError("velocity_unit_kms must be positive")
        return value, "configuration velocity_unit_kms"

    value = _find_attr(
        f,
        (
            "AGAMA_GadgetVelocityUnit_kms",
            "Combiner_VelocityUnit_kms",
            "VelocityUnit_kms",
            "UnitVelocity_in_km_per_s",
        ),
    )
    if value is not None and float(value) > 0:
        return float(value), "HDF5 velocity-unit metadata"

    cms = _find_attr(
        f,
        (
            "UnitVelocity_in_cm_per_s",
            "UnitVelocityInCmPerS",
        ),
    )
    if cms is not None and float(cms) > 0:
        return float(cms) / 1.0e5, "HDF5 UnitVelocity_in_cm_per_s metadata"
    return 1.0, "default km/s velocities"


def _mass_table(f: h5py.File) -> np.ndarray:
    if "Header" not in f:
        return np.zeros(6, dtype=float)
    return np.asarray(
        f["Header"].attrs.get("MassTable", np.zeros(6)),
        dtype=float,
    )


def particle_count(group: h5py.Group | None) -> int:
    if group is None:
        return 0
    for key in ("Coordinates", "Velocities", "ParticleIDs", "Masses", "Mass"):
        if key in group:
            return int(group[key].shape[0])
    return 0


def read_particle_masses_code(
    f: h5py.File,
    ptype: int,
    count: int | None = None,
) -> np.ndarray:
    group = f.get(f"PartType{ptype}")
    if group is None:
        return np.empty(0, dtype=np.float64)
    if count is None:
        count = particle_count(group)

    if "Masses" in group:
        masses = np.asarray(group["Masses"][()], dtype=np.float64)
    elif "Mass" in group:
        masses = np.asarray(group["Mass"][()], dtype=np.float64)
    else:
        table = _mass_table(f)
        if table[ptype] <= 0:
            raise RuntimeError(
                f"{f.filename}: PartType{ptype} has no Masses dataset and "
                f"MassTable[{ptype}]={table[ptype]}"
            )
        masses = np.full(count, table[ptype], dtype=np.float64)

    if masses.shape != (count,):
        raise RuntimeError(
            f"{f.filename}: PartType{ptype} mass array shape {masses.shape} "
            f"does not match particle count {count}"
        )
    if not np.all(np.isfinite(masses)) or np.any(masses <= 0):
        raise RuntimeError(
            f"{f.filename}: PartType{ptype} contains invalid particle masses"
        )
    return masses


def inspect_snapshot(
    galaxy: dict[str, Any],
    default_mass_unit_msun: float,
) -> SnapshotInfo:
    path = str(Path(galaxy["ic_file"]).expanduser())
    name = str(galaxy.get("name", Path(path).stem))

    with h5py.File(path, "r") as f:
        mass_unit, mass_unit_source = infer_mass_unit_msun(
            f, galaxy, default_mass_unit_msun
        )
        length_unit, _ = infer_length_unit_kpc(f, galaxy)
        velocity_unit, _ = infer_velocity_unit_kms(f, galaxy)

        counts = np.zeros(6, dtype=np.int64)
        component_masses = np.zeros(6, dtype=np.float64)
        total_mass = 0.0
        weighted_position = np.zeros(3, dtype=np.float64)
        weighted_velocity = np.zeros(3, dtype=np.float64)
        id_min: int | None = None
        id_max: int | None = None

        for ptype in range(6):
            group = f.get(f"PartType{ptype}")
            if group is None:
                continue
            count = particle_count(group)
            if count == 0:
                continue
            if "Coordinates" not in group or "Velocities" not in group:
                raise RuntimeError(
                    f"{path}: PartType{ptype} must contain Coordinates and Velocities"
                )

            coords = np.asarray(group["Coordinates"][()], dtype=np.float64)
            velocities = np.asarray(group["Velocities"][()], dtype=np.float64)
            if coords.shape != (count, 3) or velocities.shape != (count, 3):
                raise RuntimeError(
                    f"{path}: PartType{ptype} has invalid coordinate/velocity shape"
                )

            masses = read_particle_masses_code(f, ptype, count) * mass_unit
            coords *= length_unit
            velocities *= velocity_unit

            component_mass = float(np.sum(masses, dtype=np.float64))
            counts[ptype] = count
            component_masses[ptype] = component_mass
            total_mass += component_mass
            weighted_position += np.sum(coords * masses[:, None], axis=0)
            weighted_velocity += np.sum(velocities * masses[:, None], axis=0)

            if "ParticleIDs" in group:
                ids = np.asarray(group["ParticleIDs"][()], dtype=np.int64)
                if ids.shape != (count,):
                    raise RuntimeError(
                        f"{path}: PartType{ptype} ParticleIDs shape mismatch"
                    )
                if ids.size:
                    lo, hi = int(ids.min()), int(ids.max())
                    id_min = lo if id_min is None else min(id_min, lo)
                    id_max = hi if id_max is None else max(id_max, hi)

        if not np.isfinite(total_mass) or total_mass <= 0:
            raise RuntimeError(f"{path}: non-positive total snapshot mass")

        if "center" in galaxy:
            center = np.asarray(galaxy["center"], dtype=float) * length_unit
            if center.shape != (3,):
                raise ValueError(f"{name}: center must contain three values")
        else:
            center = weighted_position / total_mass

        if "bulk_velocity" in galaxy:
            bulk_velocity = (
                np.asarray(galaxy["bulk_velocity"], dtype=float) * velocity_unit
            )
            if bulk_velocity.shape != (3,):
                raise ValueError(f"{name}: bulk_velocity must contain three values")
        else:
            bulk_velocity = weighted_velocity / total_mass

        maximum_radius = 0.0
        for ptype in range(6):
            group = f.get(f"PartType{ptype}")
            if group is None or "Coordinates" not in group:
                continue
            coords = (
                np.asarray(group["Coordinates"][()], dtype=np.float64) * length_unit
            )
            if coords.size:
                maximum_radius = max(
                    maximum_radius,
                    float(np.max(np.linalg.norm(coords - center, axis=1))),
                )

        if id_min is None:
            id_min, id_max = 0, -1

    return SnapshotInfo(
        name=name,
        path=path,
        mass_unit_msun=float(mass_unit),
        length_unit_kpc=float(length_unit),
        velocity_unit_kms=float(velocity_unit),
        total_mass_msun=float(total_mass),
        center_kpc=center.tolist(),
        bulk_velocity_kms=bulk_velocity.tolist(),
        particle_counts=counts.astype(int).tolist(),
        particle_masses_msun=component_masses.tolist(),
        id_min=int(id_min),
        id_max=int(id_max),
        maximum_radius_kpc=float(maximum_radius),
        mass_unit_source=mass_unit_source,
    )


def enclosed_mass_msun(
    info: SnapshotInfo,
    radius_kpc: float,
) -> float:
    if radius_kpc < 0:
        raise ValueError("enclosed-mass radius cannot be negative")

    center = np.asarray(info.center_kpc, dtype=np.float64)
    enclosed = 0.0
    with h5py.File(info.path, "r") as f:
        for ptype in range(6):
            group = f.get(f"PartType{ptype}")
            if group is None or "Coordinates" not in group:
                continue
            count = particle_count(group)
            if count == 0:
                continue
            coords = (
                np.asarray(group["Coordinates"][()], dtype=np.float64)
                * info.length_unit_kpc
            )
            radii = np.linalg.norm(coords - center, axis=1)
            masses = read_particle_masses_code(f, ptype, count) * info.mass_unit_msun
            enclosed += float(np.sum(masses[radii <= radius_kpc]))
    return enclosed


def rotate_about_axis(
    coords: np.ndarray,
    axis: np.ndarray,
    theta_rad: float,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if theta_rad == 0.0:
        return coords

    axis = np.asarray(axis, dtype=np.float64)
    if axis.shape != (3,):
        raise ValueError("rotation axis must be a length-three vector")
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise ValueError("nonzero rotation angle requires a nonzero axis")

    k = axis / norm
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    cross = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ]
    )
    matrix = c * np.eye(3) + (1.0 - c) * np.outer(k, k) + s * cross
    return coords @ matrix.T


def rotate_by_tilt_and_azimuth(
    coords: np.ndarray,
    tilt_deg: float,
    direction_deg: float,
) -> np.ndarray:
    coords = np.asarray(coords, dtype=np.float64)
    if float(tilt_deg) == 0.0:
        return coords

    direction = np.deg2rad(direction_deg)
    target = np.array([np.cos(direction), np.sin(direction), 0.0])
    world_z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(world_z, target)
    return rotate_about_axis(coords, axis, np.deg2rad(tilt_deg))


def apply_galaxy_rotation(
    values: np.ndarray,
    galaxy: dict[str, Any],
) -> np.ndarray:
    rotated = rotate_by_tilt_and_azimuth(
        values,
        float(galaxy.get("tilt", 0.0)),
        float(galaxy.get("azimuth", 0.0)),
    )
    extra_angle = float(galaxy.get("rot", 0.0))
    if extra_angle != 0.0:
        rotated = rotate_about_axis(
            rotated,
            np.asarray(galaxy.get("axis", [0.0, 0.0, 1.0]), dtype=float),
            np.deg2rad(extra_angle),
        )
    return rotated


def _R1(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _R3(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def relative_orbit_geometry(
    galaxy: dict[str, Any],
) -> tuple[np.ndarray, float, float]:
    e = float(galaxy["e"])
    rp = float(galaxy["rp"])
    f = np.deg2rad(float(galaxy["f0"]))
    if not 0 <= e < 1:
        raise ValueError("this initializer requires 0 <= e < 1")
    if rp <= 0:
        raise ValueError("rp must be positive")

    p = rp * (1.0 + e)
    denominator = 1.0 + e * np.cos(f)
    if denominator <= 0:
        raise ValueError("orbit geometry produced a non-positive radius")
    radius = p / denominator

    r_pf = np.array([radius * np.cos(f), radius * np.sin(f), 0.0])
    w = np.deg2rad(float(galaxy.get("w", 0.0)))
    inc = np.deg2rad(float(galaxy.get("i", 0.0)))
    omega = np.deg2rad(float(galaxy.get("omega", 0.0)))
    transform = _R3(omega) @ _R1(inc) @ _R3(w)
    return transform @ r_pf, p, f


def relative_orbit_velocity(
    galaxy: dict[str, Any],
    p_kpc: float,
    f_rad: float,
    orbit_mass_msun: float,
) -> np.ndarray:
    if orbit_mass_msun <= 0:
        raise ValueError("orbit mass must be positive")
    e = float(galaxy["e"])
    speed_scale = math.sqrt(G * orbit_mass_msun / p_kpc)
    v_pf = speed_scale * np.array([-np.sin(f_rad), e + np.cos(f_rad), 0.0])
    w = np.deg2rad(float(galaxy.get("w", 0.0)))
    inc = np.deg2rad(float(galaxy.get("i", 0.0)))
    omega = np.deg2rad(float(galaxy.get("omega", 0.0)))
    transform = _R3(omega) @ _R1(inc) @ _R3(w)
    return transform @ v_pf


def prepare_config(galaxies: list[dict[str, Any]]) -> None:
    if len(galaxies) < 2:
        raise ValueError("configuration must contain a host and at least one satellite")
    for index, galaxy in enumerate(galaxies):
        galaxy.setdefault("tilt", 0.0)
        galaxy.setdefault("azimuth", 0.0)
        galaxy.setdefault("axis", [0.0, 0.0, 1.0])
        galaxy.setdefault("rot", 0.0)
        if index > 0 and "omega" not in galaxy:
            galaxy["omega"] = (float(galaxy.get("w", 0.0)) - 90.0) % 360.0


def initialize_orbits(
    galaxies: list[dict[str, Any]],
    infos: list[SnapshotInfo],
    orbit_mass_mode: str,
) -> tuple[list[np.ndarray], list[np.ndarray], list[OrbitRecord]]:
    host = infos[0]
    relative_positions: list[np.ndarray] = []
    relative_velocities: list[np.ndarray] = []
    records: list[OrbitRecord] = []

    for index, galaxy in enumerate(galaxies[1:], start=1):
        satellite = infos[index]
        r_rel, p, f = relative_orbit_geometry(galaxy)
        separation = float(np.linalg.norm(r_rel))

        host_enclosed = enclosed_mass_msun(host, separation)
        satellite_enclosed = enclosed_mass_msun(satellite, separation)

        if orbit_mass_mode == "both-enclosed":
            orbit_mass = host_enclosed + satellite_enclosed
        elif orbit_mass_mode == "host-enclosed-satellite-total":
            orbit_mass = host_enclosed + satellite.total_mass_msun
        elif orbit_mass_mode == "both-total":
            orbit_mass = host.total_mass_msun + satellite.total_mass_msun
        else:
            raise ValueError(f"unsupported orbit mass mode: {orbit_mass_mode}")

        v_rel = relative_orbit_velocity(galaxy, p, f, orbit_mass)
        relative_positions.append(r_rel)
        relative_velocities.append(v_rel)
        records.append(
            OrbitRecord(
                satellite_name=satellite.name,
                separation_kpc=separation,
                host_total_mass_msun=host.total_mass_msun,
                host_enclosed_mass_msun=host_enclosed,
                satellite_total_mass_msun=satellite.total_mass_msun,
                satellite_enclosed_mass_msun=satellite_enclosed,
                orbit_mass_mode=orbit_mass_mode,
                orbit_mass_msun=orbit_mass,
                relative_position_kpc=r_rel.tolist(),
                relative_velocity_kms=v_rel.tolist(),
                relative_speed_kms=float(np.linalg.norm(v_rel)),
            )
        )

    total_live_mass = float(sum(info.total_mass_msun for info in infos))
    host_position = (
        -sum(
            (
                infos[index].total_mass_msun * relative_positions[index - 1]
                for index in range(1, len(infos))
            ),
            start=np.zeros(3, dtype=np.float64),
        )
        / total_live_mass
    )
    host_velocity = (
        -sum(
            (
                infos[index].total_mass_msun * relative_velocities[index - 1]
                for index in range(1, len(infos))
            ),
            start=np.zeros(3, dtype=np.float64),
        )
        / total_live_mass
    )

    positions = [host_position]
    velocities = [host_velocity]
    for r_rel, v_rel in zip(relative_positions, relative_velocities):
        positions.append(host_position + r_rel)
        velocities.append(host_velocity + v_rel)

    position_com = (
        sum(
            (
                info.total_mass_msun * position
                for info, position in zip(infos, positions)
            ),
            start=np.zeros(3, dtype=np.float64),
        )
        / total_live_mass
    )
    velocity_com = (
        sum(
            (
                info.total_mass_msun * velocity
                for info, velocity in zip(infos, velocities)
            ),
            start=np.zeros(3, dtype=np.float64),
        )
        / total_live_mass
    )

    if np.linalg.norm(position_com) > 1e-8:
        raise RuntimeError(f"nonzero galaxy-centre COM position: {position_com}")
    if np.linalg.norm(velocity_com) > 1e-8:
        raise RuntimeError(f"nonzero galaxy-centre COM velocity: {velocity_com}")

    return positions, velocities, records


def _compression_kwargs(reference: h5py.Dataset) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if reference.compression is not None:
        kwargs["compression"] = reference.compression
        if reference.compression_opts is not None:
            kwargs["compression_opts"] = reference.compression_opts
    if reference.shuffle:
        kwargs["shuffle"] = True
    return kwargs


def determine_id_offsets(infos: list[SnapshotInfo]) -> list[int]:
    offsets: list[int] = []
    next_id = 1
    for info in infos:
        if info.id_max < info.id_min:
            raise RuntimeError(f"{info.path}: no ParticleIDs found")
        offset = next_id - info.id_min
        offsets.append(offset)
        next_id = info.id_max + offset + 1
    return offsets


def combine_hdf5(
    galaxies: list[dict[str, Any]],
    infos: list[SnapshotInfo],
    outfile: str,
    position_offsets: list[np.ndarray],
    velocity_offsets: list[np.ndarray],
    output_mass_unit_msun: float,
    orbit_records: list[OrbitRecord],
) -> dict[str, Any]:
    if output_mass_unit_msun <= 0:
        raise ValueError("output mass unit must be positive")

    files = [h5py.File(info.path, "r") for info in infos]
    id_offsets = determine_id_offsets(infos)

    try:
        with h5py.File(outfile, "w") as fout:
            if "Header" in files[0]:
                files[0].copy("Header", fout)
            else:
                fout.create_group("Header")
            for key, value in files[0].attrs.items():
                fout.attrs[key] = value

            header = fout["Header"].attrs
            names = [info.name for info in infos]
            str_dtype = h5py.string_dtype("utf-8")

            header["GalaxyID_Name"] = np.asarray(names, dtype=str_dtype)
            header["GalaxyID_Infile"] = np.asarray(
                [info.path for info in infos], dtype=str_dtype
            )
            header["GalaxyID_IDOffset"] = np.asarray(id_offsets, dtype=np.int64)
            header["GalaxyID_IDMin"] = np.asarray(
                [info.id_min for info in infos], dtype=np.int64
            )
            header["GalaxyID_IDMax"] = np.asarray(
                [info.id_max for info in infos], dtype=np.int64
            )
            header["GalaxyID_IDStart"] = np.asarray(
                [info.id_min + off for info, off in zip(infos, id_offsets)],
                dtype=np.int64,
            )
            header["GalaxyID_IDEnd"] = np.asarray(
                [info.id_max + off for info, off in zip(infos, id_offsets)],
                dtype=np.int64,
            )
            header["Galaxy_TotalMass_Msun"] = np.asarray(
                [info.total_mass_msun for info in infos], dtype=np.float64
            )
            header["Galaxy_InternalCenter_kpc"] = np.asarray(
                [info.center_kpc for info in infos], dtype=np.float64
            )
            header["Galaxy_InternalBulkVelocity_kms"] = np.asarray(
                [info.bulk_velocity_kms for info in infos], dtype=np.float64
            )
            header["Galaxy_InitialPosition_kpc"] = np.asarray(
                position_offsets, dtype=np.float64
            )
            header["Galaxy_InitialVelocity_kms"] = np.asarray(
                velocity_offsets, dtype=np.float64
            )
            header["Combiner_MassUnit_Msun"] = float(output_mass_unit_msun)
            header["AGAMA_GadgetMassUnit_Msun"] = float(output_mass_unit_msun)
            header["UnitMass_in_g"] = float(output_mass_unit_msun * MSUN_IN_G)
            header["Combiner_LengthUnit_kpc"] = 1.0
            header["Combiner_VelocityUnit_kms"] = 1.0
            header["HubbleParam"] = 1.0
            header["NumFilesPerSnapshot"] = 1

            if orbit_records:
                header["Orbit_InitialSeparation_kpc"] = np.asarray(
                    [record.separation_kpc for record in orbit_records],
                    dtype=np.float64,
                )
                header["Orbit_HostEnclosedMass_Msun"] = np.asarray(
                    [record.host_enclosed_mass_msun for record in orbit_records],
                    dtype=np.float64,
                )
                header["Orbit_SatelliteEnclosedMass_Msun"] = np.asarray(
                    [record.satellite_enclosed_mass_msun for record in orbit_records],
                    dtype=np.float64,
                )
                header["Orbit_MassUsed_Msun"] = np.asarray(
                    [record.orbit_mass_msun for record in orbit_records],
                    dtype=np.float64,
                )
                header["Orbit_RelativeSpeed_kms"] = np.asarray(
                    [record.relative_speed_kms for record in orbit_records],
                    dtype=np.float64,
                )
                header["Orbit_MassMode"] = np.asarray(
                    [record.orbit_mass_mode for record in orbit_records],
                    dtype=str_dtype,
                )

            combined_counts = np.zeros(6, dtype=np.uint64)

            for ptype in range(6):
                groups = [f.get(f"PartType{ptype}") for f in files]
                counts = np.asarray(
                    [particle_count(group) for group in groups], dtype=np.int64
                )
                if int(np.sum(counts)) == 0:
                    continue
                combined_counts[ptype] = np.uint64(np.sum(counts))

                output_group = fout.create_group(f"PartType{ptype}")
                union_keys: set[str] = set()
                for group in groups:
                    if group is not None:
                        union_keys.update(group.keys())
                union_keys.discard("Mass")
                union_keys.discard("Masses")
                union_keys.update(("Coordinates", "Velocities", "ParticleIDs"))

                # Always materialize physical particle masses in the selected
                # output code mass unit.
                mass_chunks: list[np.ndarray] = []
                for index, (f, group, count) in enumerate(zip(files, groups, counts)):
                    if group is None or count == 0:
                        continue
                    masses = (
                        read_particle_masses_code(f, ptype, int(count))
                        * infos[index].mass_unit_msun
                        / output_mass_unit_msun
                    )
                    mass_chunks.append(masses)
                output_group.create_dataset(
                    "Masses",
                    data=np.concatenate(mass_chunks).astype(np.float64),
                    compression="gzip",
                )

                for key in sorted(union_keys):
                    reference = next(
                        (
                            group[key]
                            for group in groups
                            if group is not None and key in group
                        ),
                        None,
                    )
                    if reference is None:
                        if key == "ParticleIDs":
                            reference_shape = ()
                            reference_dtype = np.dtype(np.int64)
                        else:
                            raise RuntimeError(f"No reference dataset found for {key}")
                    else:
                        reference_shape = reference.shape[1:]
                        reference_dtype = reference.dtype

                    chunks: list[np.ndarray] = []
                    for index, (f, group, count) in enumerate(
                        zip(files, groups, counts)
                    ):
                        count = int(count)
                        if count == 0:
                            continue
                        info = infos[index]
                        galaxy = galaxies[index]

                        if group is not None and key in group:
                            data = np.asarray(group[key][()])
                            if data.shape[0] != count:
                                raise RuntimeError(
                                    f"{info.path}: PartType{ptype}/{key} "
                                    "has the wrong leading dimension"
                                )
                        elif key == "ParticleIDs":
                            # A file without IDs was already rejected by
                            # determine_id_offsets, so this is defensive only.
                            raise RuntimeError(f"{info.path}: missing ParticleIDs")
                        else:
                            data = np.zeros(
                                (count,) + tuple(reference_shape),
                                dtype=reference_dtype,
                            )

                        if key == "ParticleIDs":
                            data = np.asarray(data, dtype=np.int64) + np.int64(
                                id_offsets[index]
                            )
                        elif key == "Coordinates":
                            data = (
                                np.asarray(data, dtype=np.float64)
                                * info.length_unit_kpc
                            )
                            data -= np.asarray(info.center_kpc)
                            data = apply_galaxy_rotation(data, galaxy)
                            data += np.asarray(position_offsets[index])
                        elif key == "Velocities":
                            data = (
                                np.asarray(data, dtype=np.float64)
                                * info.velocity_unit_kms
                            )
                            data -= np.asarray(info.bulk_velocity_kms)
                            data = apply_galaxy_rotation(data, galaxy)
                            data += np.asarray(velocity_offsets[index])

                        chunks.append(data)

                    combined = np.concatenate(chunks, axis=0)
                    kwargs = (
                        _compression_kwargs(reference) if reference is not None else {}
                    )
                    dataset = output_group.create_dataset(
                        key,
                        data=combined,
                        **kwargs,
                    )
                    if reference is not None:
                        for attr_key, attr_value in reference.attrs.items():
                            dataset.attrs[attr_key] = attr_value

            low = (combined_counts & np.uint64(0xFFFFFFFF)).astype(np.uint32)
            high = (combined_counts >> np.uint64(32)).astype(np.uint32)
            header["NumPart_ThisFile"] = low
            header["NumPart_Total"] = low
            header["NumPart_Total_HighWord"] = high
            header["MassTable"] = np.zeros(6, dtype=np.float64)

        galaxy_map = {
            "GalaxyID_Name": [info.name for info in infos],
            "GalaxyID_Infile": [info.path for info in infos],
            "GalaxyID_IDOffset": id_offsets,
            "GalaxyID_IDMin": [info.id_min for info in infos],
            "GalaxyID_IDMax": [info.id_max for info in infos],
            "GalaxyID_IDStart": [
                info.id_min + off for info, off in zip(infos, id_offsets)
            ],
            "GalaxyID_IDEnd": [
                info.id_max + off for info, off in zip(infos, id_offsets)
            ],
            "outfile": str(outfile),
        }
        map_path = str(outfile) + ".galaxyid.json"
        Path(map_path).write_text(
            json.dumps(galaxy_map, indent=2, sort_keys=True) + "\n"
        )
        print(f"Wrote GalaxyID sidecar map: {map_path}")
        return galaxy_map
    finally:
        for f in files:
            f.close()


def validate_combined_snapshot(
    path: str,
    expected_total_mass_msun: float,
    mass_unit_msun: float,
    mass_relative_tolerance: float = 1e-10,
    com_position_tolerance_kpc: float = 1e-7,
    com_velocity_tolerance_kms: float = 1e-7,
) -> dict[str, Any]:
    total_mass = 0.0
    position_moment = np.zeros(3, dtype=np.float64)
    momentum = np.zeros(3, dtype=np.float64)

    with h5py.File(path, "r") as f:
        mass_table = np.asarray(
            f["Header"].attrs.get("MassTable", np.zeros(6)),
            dtype=float,
        )
        if np.any(mass_table != 0):
            raise RuntimeError(
                "combined snapshot MassTable is nonzero even though "
                "per-particle Masses were materialized"
            )

        for ptype in range(6):
            group = f.get(f"PartType{ptype}")
            if group is None:
                continue
            for required in ("Coordinates", "Velocities", "ParticleIDs", "Masses"):
                if required not in group:
                    raise RuntimeError(
                        f"combined PartType{ptype} is missing {required}"
                    )

            count = len(group["Coordinates"])
            if any(len(group[key]) != count for key in group.keys()):
                lengths = {key: len(group[key]) for key in group.keys()}
                raise RuntimeError(
                    f"combined PartType{ptype} dataset lengths differ: {lengths}"
                )

            masses = np.asarray(group["Masses"][()], dtype=np.float64) * mass_unit_msun
            coordinates = np.asarray(group["Coordinates"][()], dtype=np.float64)
            velocities = np.asarray(group["Velocities"][()], dtype=np.float64)
            total_mass += float(np.sum(masses))
            position_moment += np.sum(coordinates * masses[:, None], axis=0)
            momentum += np.sum(velocities * masses[:, None], axis=0)

    mass_error = total_mass / expected_total_mass_msun - 1.0
    center = position_moment / total_mass
    bulk_velocity = momentum / total_mass

    if abs(mass_error) > mass_relative_tolerance:
        raise RuntimeError(
            f"combined mass differs from input mass sum by {mass_error:+.3e}"
        )
    if np.linalg.norm(center) > com_position_tolerance_kpc:
        raise RuntimeError(f"combined particle COM is not zero: {center} kpc")
    if np.linalg.norm(bulk_velocity) > com_velocity_tolerance_kms:
        raise RuntimeError(
            f"combined particle bulk velocity is not zero: " f"{bulk_velocity} km/s"
        )

    diagnostics = {
        "total_mass_msun": total_mass,
        "expected_total_mass_msun": expected_total_mass_msun,
        "fractional_mass_error": mass_error,
        "center_of_mass_position_kpc": center.tolist(),
        "bulk_velocity_kms": bulk_velocity.tolist(),
        "total_momentum_msun_kms": momentum.tolist(),
    }
    print(
        "Validated combined snapshot: "
        f"M={total_mass:.8e} Msun; "
        f"|COM|={np.linalg.norm(center):.3e} kpc; "
        f"|VCOM|={np.linalg.norm(bulk_velocity):.3e} km/s"
    )
    return diagnostics


def mass_mismatch_warnings(
    galaxies: list[dict[str, Any]],
    infos: list[SnapshotInfo],
    tolerance: float,
) -> None:
    for galaxy, info in zip(galaxies, infos):
        if "M" not in galaxy:
            continue
        configured = float(galaxy["M"])
        if configured <= 0:
            continue
        difference = info.total_mass_msun / configured - 1.0
        if abs(difference) > tolerance:
            print(
                f"Warning: {info.name} configured M={configured:.6e} Msun "
                f"but snapshot mass={info.total_mass_msun:.6e} Msun "
                f"({difference:+.2%}). Configured M is ignored.",
                file=sys.stderr,
            )


def write_orbit_diagnostics(
    path: str,
    infos: list[SnapshotInfo],
    records: list[OrbitRecord],
    positions: list[np.ndarray],
    velocities: list[np.ndarray],
) -> None:
    total_mass = float(sum(info.total_mass_msun for info in infos))
    total_position_moment = sum(
        (info.total_mass_msun * position for info, position in zip(infos, positions)),
        start=np.zeros(3),
    )
    total_momentum = sum(
        (info.total_mass_msun * velocity for info, velocity in zip(infos, velocities)),
        start=np.zeros(3),
    )

    payload = {
        "snapshots": [asdict(info) for info in infos],
        "orbits": [asdict(record) for record in records],
        "galaxy_initial_positions_kpc": [position.tolist() for position in positions],
        "galaxy_initial_velocities_kms": [velocity.tolist() for velocity in velocities],
        "combined_total_mass_msun": total_mass,
        "center_of_mass_position_kpc": (total_position_moment / total_mass).tolist(),
        "center_of_mass_velocity_kms": (total_momentum / total_mass).tolist(),
        "total_momentum_msun_kms": total_momentum.tolist(),
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Wrote mass/orbit diagnostics: {path}")


def make_gadget_parameter_file(
    galaxies: list[dict[str, Any]],
    infos: list[SnapshotInfo],
    records: list[OrbitRecord],
    positions: list[np.ndarray],
    velocities: list[np.ndarray],
    config_file: str,
    combined_ic_base: str,
    gadget_template: str,
    gadget_path: str,
    gadget_output_dir: str,
    gadget_parameter_file: str,
) -> None:
    leading_lines = [
        "% Galaxy Collision Simulation\n",
        "% Initial masses measured directly from input HDF5 snapshots\n",
        "%\n",
        f"% Config file: {config_file}\n",
        "%\n",
        f"% {infos[0].name} is host\n",
        f"% Input file = {infos[0].path}\n",
        f"% Total snapshot mass = {infos[0].total_mass_msun:.8E} Msun\n",
        f"% Internal center removed = {infos[0].center_kpc} kpc\n",
        f"% Internal bulk velocity removed = " f"{infos[0].bulk_velocity_kms} km/s\n",
        f"% Tilt = {galaxies[0]['tilt']} deg\n",
        f"% Azimuth = {galaxies[0]['azimuth']} deg\n",
        f"% Axis = {galaxies[0]['axis']}\n",
        f"% Rotation = {galaxies[0]['rot']} deg\n",
        f"% r = {positions[0]} kpc\n",
        f"% v = {velocities[0]} km/s\n",
        "%-----------------\n",
    ]

    for index, (galaxy, info, record) in enumerate(
        zip(galaxies[1:], infos[1:], records), start=1
    ):
        leading_lines.extend(
            [
                f"% {info.name} is satellite\n",
                f"% Input file = {info.path}\n",
                f"% Total snapshot mass = {info.total_mass_msun:.8E} Msun\n",
                f"% Initial separation = {record.separation_kpc:.8f} kpc\n",
                f"% Host enclosed mass = "
                f"{record.host_enclosed_mass_msun:.8E} Msun\n",
                f"% Satellite enclosed mass = "
                f"{record.satellite_enclosed_mass_msun:.8E} Msun\n",
                f"% Orbit mass mode = {record.orbit_mass_mode}\n",
                f"% Orbit mass used = {record.orbit_mass_msun:.8E} Msun\n",
                f"% Relative speed = {record.relative_speed_kms:.8f} km/s\n",
                f"% Tilt = {galaxy['tilt']} deg\n",
                f"% Azimuth = {galaxy['azimuth']} deg\n",
                f"% Axis = {galaxy['axis']}\n",
                f"% Rotation = {galaxy['rot']} deg\n",
                f"% i = {galaxy['i']} deg\n",
                f"% e = {galaxy['e']}\n",
                f"% w = {galaxy['w']} deg\n",
                f"% Omega = {galaxy['omega']} deg\n",
                f"% TA = {galaxy['f0']} deg\n",
                f"% rp = {galaxy['rp']} kpc\n",
                f"% r = {positions[index]} kpc\n",
                f"% v = {velocities[index]} km/s\n",
                "%-----------------\n",
                "%\n",
            ]
        )

    lines = Path(gadget_template).read_text().splitlines(keepends=True)
    init_line = f"InitCondFile     {combined_ic_base}\n"
    output_line = f"OutputDir     {gadget_path}/{gadget_output_dir}\n"
    replaced = [
        (
            init_line
            if line.startswith("InitCondFile")
            else output_line if line.startswith("OutputDir") else line
        )
        for line in lines
    ]
    output_path = Path(gadget_path) / gadget_parameter_file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(leading_lines + replaced))
    print(f"Wrote Gadget-4 parameter file: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Combine galaxy HDF5 ICs and initialize an encounter using "
            "snapshot-measured total and enclosed masses."
        )
    )
    parser.add_argument("config_file")
    parser.add_argument(
        "--orbit-mass-mode",
        choices=(
            "both-enclosed",
            "host-enclosed-satellite-total",
            "both-total",
        ),
        default="host-enclosed-satellite-total",
        help=(
            "Mass used in the nominal Keplerian relative velocity. "
            "Default: measured host enclosed mass plus measured satellite "
            "total mass."
        ),
    )
    parser.add_argument(
        "--default-mass-unit-msun",
        type=float,
        default=1.0e10,
        help="Fallback input code mass unit when metadata are absent.",
    )
    parser.add_argument(
        "--output-mass-unit-msun",
        type=float,
        help="Combined-file code mass unit; default is the host input unit.",
    )
    parser.add_argument(
        "--configured-mass-warning-fraction",
        type=float,
        default=0.05,
        help="Warn when legacy config M differs from measured mass by this fraction.",
    )
    parser.add_argument(
        "--galic-path",
        default=DEFAULT_GALIC_PATH,
        help="Directory used for the default combined IC output.",
    )
    parser.add_argument("--output-file")
    parser.add_argument(
        "--gadget-template",
        default=DEFAULT_GADGET_TEMPLATE,
    )
    parser.add_argument("--gadget-path")
    parser.add_argument(
        "--skip-gadget-parameter-file",
        action="store_true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config_path = Path(args.config_file).expanduser().resolve()
    galaxies = ast.literal_eval(config_path.read_text())
    if not isinstance(galaxies, list):
        raise TypeError("configuration must evaluate to a list of dictionaries")
    prepare_config(galaxies)

    infos = [
        inspect_snapshot(galaxy, args.default_mass_unit_msun) for galaxy in galaxies
    ]
    mass_mismatch_warnings(
        galaxies,
        infos,
        args.configured_mass_warning_fraction,
    )

    print("\nMeasured input snapshot masses")
    print("------------------------------")
    for info in infos:
        print(
            f"{info.name}: M={info.total_mass_msun:.8e} Msun; "
            f"center={np.asarray(info.center_kpc)} kpc; "
            f"bulk velocity={np.asarray(info.bulk_velocity_kms)} km/s; "
            f"rmax={info.maximum_radius_kpc:.6g} kpc; "
            f"mass unit={info.mass_unit_msun:.6e} Msun "
            f"({info.mass_unit_source})"
        )

    positions, velocities, records = initialize_orbits(
        galaxies,
        infos,
        args.orbit_mass_mode,
    )

    print("\nOrbit initialization")
    print("--------------------")
    for record in records:
        print(
            f"{record.satellite_name}: r={record.separation_kpc:.6f} kpc; "
            f"Mhost(<r)={record.host_enclosed_mass_msun:.8e} Msun; "
            f"Msat(<r)={record.satellite_enclosed_mass_msun:.8e} Msun; "
            f"Msat,total={record.satellite_total_mass_msun:.8e} Msun; "
            f"Morbit={record.orbit_mass_msun:.8e} Msun; "
            f"|vrel|={record.relative_speed_kms:.6f} km/s"
        )

    config_base = config_path.stem
    date_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = (
        f"{config_base}_rp"
        f"{str(galaxies[1]['rp']).replace('.', 'pt')}_e"
        f"{str(galaxies[1]['e']).replace('.', 'pt')}_{date_string}"
    )
    combined_base = f"{prefix}-galic"

    galic_path = Path(args.galic_path).expanduser().resolve()
    galic_path.mkdir(parents=True, exist_ok=True)
    output_file = (
        Path(args.output_file).expanduser().resolve()
        if args.output_file
        else galic_path / f"{combined_base}.hdf5"
    )
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_mass_unit = (
        float(args.output_mass_unit_msun)
        if args.output_mass_unit_msun is not None
        else infos[0].mass_unit_msun
    )

    combine_hdf5(
        galaxies,
        infos,
        str(output_file),
        positions,
        velocities,
        output_mass_unit,
        records,
    )

    combined_validation = validate_combined_snapshot(
        str(output_file),
        expected_total_mass_msun=sum(info.total_mass_msun for info in infos),
        mass_unit_msun=output_mass_unit,
    )

    diagnostics_path = str(output_file) + ".orbit.json"
    write_orbit_diagnostics(
        diagnostics_path,
        infos,
        records,
        positions,
        velocities,
    )
    validation_path = str(output_file) + ".validation.json"
    Path(validation_path).write_text(
        json.dumps(combined_validation, indent=2, sort_keys=True) + "\n"
    )
    print(f"Wrote combined-snapshot validation: {validation_path}")

    if not args.skip_gadget_parameter_file:
        gadget_path = (
            Path(args.gadget_path).expanduser().resolve()
            if args.gadget_path
            else Path(f"/home/jbn/{infos[0].name}_gadget4/collision_sim")
        )
        gadget_output_dir = prefix
        gadget_parameter_file = f"{prefix}-gadget4.param"
        template = Path(args.gadget_template)
        if not template.is_absolute():
            template = galic_path / template
        make_gadget_parameter_file(
            galaxies,
            infos,
            records,
            positions,
            velocities,
            str(config_path),
            str(output_file.with_suffix("")),
            str(template),
            str(gadget_path),
            gadget_output_dir,
            gadget_parameter_file,
        )

    print(f"\nWrote combined HDF5 IC: {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
