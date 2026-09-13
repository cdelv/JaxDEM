# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Utility functions used to set up simulations and analyze the output."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .angles import angle, angle_x, signed_angle, signed_angle_x
from .clumps import compute_clump_properties
from .contacts import (
    compute_clump_pair_friction,
    compute_contact_pressure,
    compute_contact_stress_tensor,
    compute_group_pair_friction,
    count_clump_contacts,
    count_vertex_contacts,
    get_clump_rattler_ids,
    get_pair_forces_and_ids,
    get_sphere_rattler_ids,
    remove_rattlers,
)
from .dispersity import get_polydisperse_radii
from .dynamical_matrix import (
    bonded_hessian,
    clump_non_bonded_hessian,
    non_bonded_hessian,
    pair_non_bonded_hessian,
    zero_mode_mask,
)
from .dynamics_routines import run_packing_fraction_protocol
from .environment import (
    cross_lidar_2d,
    cross_lidar_3d,
    env_step,
    env_trajectory_rollout,
    lidar_2d,
    lidar_3d,
)
from .grid_state import grid_state
from .jamming import (
    JammingInfo,
    JamReason,
    JamResult,
    bisection_jam,
    pe_band_jam,
    pressure_bisection_jam,
)
from .linalg import cross, cross_3X3D_1X2D, dot, norm, norm2, unit, unit_and_norm
from .meshes import (
    generate_arclength_mesh,
    generate_faceted_mesh,
    generate_fibonacci_sphere_mesh,
    generate_helix_mesh,
    generate_icosphere_mesh,
    generate_thomson_mesh,
    generate_torus_mesh,
)
from .neighbor_list_sizing import (
    CandidateStatistics,
    NeighborCapacityEstimate,
    estimate_neighbor_capacity,
    measure_neighbor_candidates,
)
from .packing_utils import (
    CompressionReason,
    CompressionResult,
    compute_packing_fraction,
    compute_particle_volume,
    quasistatic_compress_to_packing_fraction,
    scale_to_packing_fraction,
)
from .quaternion import Quaternion
from .random_state import random_state
from .randomize_orientations import randomize_orientations
from .rollout_schedules import make_save_steps_linear, make_save_steps_pseudolog
from .serialization import decode_callable, encode_callable
from .thermal import (
    compute_energy,
    compute_potential_energy,
    compute_rotational_kinetic_energy,
    compute_rotational_kinetic_energy_per_particle,
    compute_temperature,
    compute_translational_kinetic_energy,
    compute_translational_kinetic_energy_per_particle,
    scale_to_temperature,
    set_temperature,
)

__all__ = [
    "CandidateStatistics",
    "CompressionReason",
    "CompressionResult",
    "JamReason",
    "JamResult",
    "JammingInfo",
    "NeighborCapacityEstimate",
    "Quaternion",
    "angle",
    "angle_x",
    "bisection_jam",
    "bonded_hessian",
    "clump_non_bonded_hessian",
    "compute_clump_pair_friction",
    "compute_clump_properties",
    "compute_contact_pressure",
    "compute_contact_stress_tensor",
    "compute_energy",
    "compute_group_pair_friction",
    "compute_packing_fraction",
    "compute_particle_volume",
    "compute_potential_energy",
    "compute_rotational_kinetic_energy",
    "compute_rotational_kinetic_energy_per_particle",
    "compute_temperature",
    "compute_translational_kinetic_energy",
    "compute_translational_kinetic_energy_per_particle",
    "count_clump_contacts",
    "count_vertex_contacts",
    "cross",
    "cross_3X3D_1X2D",
    "cross_lidar_2d",
    "cross_lidar_3d",
    "decode_callable",
    "dot",
    "encode_callable",
    "env_step",
    "env_trajectory_rollout",
    "estimate_neighbor_capacity",
    "generate_arclength_mesh",
    "generate_faceted_mesh",
    "generate_fibonacci_sphere_mesh",
    "generate_helix_mesh",
    "generate_icosphere_mesh",
    "generate_thomson_mesh",
    "generate_torus_mesh",
    "get_clump_rattler_ids",
    "get_pair_forces_and_ids",
    "get_polydisperse_radii",
    "get_sphere_rattler_ids",
    "grid_state",
    "lidar_2d",
    "lidar_3d",
    "load",
    "load_legacy_dp",
    "load_legacy_simulation",
    "load_legacy_state",
    "load_legacy_system",
    "make_save_steps_linear",
    "make_save_steps_pseudolog",
    "measure_neighbor_candidates",
    "non_bonded_hessian",
    "norm",
    "norm2",
    "pair_non_bonded_hessian",
    "pe_band_jam",
    "pressure_bisection_jam",
    "quasistatic_compress_to_packing_fraction",
    "random_state",
    "randomize_orientations",
    "remove_rattlers",
    "run_packing_fraction_protocol",
    "save",
    "scale_to_packing_fraction",
    "scale_to_temperature",
    "set_temperature",
    "signed_angle",
    "signed_angle_x",
    "unit",
    "unit_and_norm",
    "zero_mode_mask",
]


def __getattr__(name: str) -> Any:
    module_name = {
        "h5": ".h5",
        "load_legacy": ".load_legacy",
        "load": ".h5",
        "save": ".h5",
        "load_legacy_dp": ".load_legacy",
        "load_legacy_simulation": ".load_legacy",
        "load_legacy_state": ".load_legacy",
        "load_legacy_system": ".load_legacy",
    }.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = import_module(module_name, __name__)
        value = module if name in {"h5", "load_legacy"} else getattr(module, name)
    except ModuleNotFoundError as exc:
        if exc.name == "h5py":
            raise ImportError(
                "HDF5 support requires pip install 'JaxDEM[io]'."
            ) from exc
        raise
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
