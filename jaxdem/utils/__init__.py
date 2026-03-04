# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""
Utility functions used to set up simulations and analyze the output.
"""

from __future__ import annotations

from .linalg import unit, cross, cross_3X3D_1X2D
from .angles import signed_angle, signed_angle_x, angle, angle_x
from .gridState import grid_state
from .randomState import random_state
from .serialization import encode_callable, decode_callable
from .environment import (
    env_step,
    env_trajectory_rollout,
    lidar_2d,
    lidar_3d,
    cross_lidar_2d,
    cross_lidar_3d,
)
from .quaternion import Quaternion
from .clumps import compute_clump_properties
from .packingUtils import (
    compute_particle_volume,
    compute_packing_fraction,
    scale_to_packing_fraction,
)
from .jamming import bisection_jam
from .dispersity import get_polydisperse_radii
from .h5 import load, save
from .randomizeOrientations import randomize_orientations
from .thermal import (
    compute_translational_kinetic_energy_per_particle,
    compute_rotational_kinetic_energy_per_particle,
    compute_translational_kinetic_energy,
    compute_rotational_kinetic_energy,
    compute_potential_energy_per_particle,
    compute_potential_energy,
    compute_energy,
    compute_temperature,
    scale_to_temperature,
    set_temperature,
)
from .contacts import (
    get_pair_forces_and_ids,
    get_pair_potential_derivatives,
    compute_hessian_spheres,
    compute_hessian_clumps_2d,
    get_clump_rattler_ids,
    get_sphere_rattler_ids,
    count_vertex_contacts,
    count_clump_contacts,
    remove_rattlers_from_state,
)
from .load_legacy import (
    load_legacy_state,
    load_legacy_system,
    load_legacy_dp,
    load_legacy_simulation,
)
from .dynamicsRoutines import control_nvt_density, control_nvt_density_rollout
from .rollout_schedules import make_save_steps_linear, make_save_steps_pseudolog

__all__ = [
    "unit",
    "cross",
    "cross_3X3D_1X2D",
    "signed_angle",
    "signed_angle_x",
    "angle",
    "angle_x",
    "grid_state",
    "random_state",
    "encode_callable",
    "decode_callable",
    "env_step",
    "env_trajectory_rollout",
    "lidar_2d",
    "lidar_3d",
    "cross_lidar_2d",
    "cross_lidar_3d",
    "Quaternion",
    "compute_clump_properties",
    "compute_particle_volume",
    "compute_packing_fraction",
    "scale_to_packing_fraction",
    "jamming",
    "dispersity",
    "randomize_orientations",
    "compute_translational_kinetic_energy_per_particle",
    "compute_rotational_kinetic_energy_per_particle",
    "compute_translational_kinetic_energy",
    "compute_rotational_kinetic_energy",
    "compute_potential_energy_per_particle",
    "compute_potential_energy",
    "compute_energy",
    "compute_temperature",
    "scale_to_temperature",
    "set_temperature",
    "get_pair_forces_and_ids",
    "get_pair_potential_derivatives",
    "compute_hessian_spheres",
    "compute_hessian_clumps_2d",
    "get_clump_rattler_ids",
    "get_sphere_rattler_ids",
    "count_vertex_contacts",
    "count_clump_contacts",
    "remove_rattlers_from_state",
    "load_legacy_state",
    "load_legacy_system",
    "load_legacy_dp",
    "load_legacy_simulation",
    "control_nvt_density",
    "control_nvt_density_rollout",
    "make_save_steps_linear",
    "make_save_steps_pseudolog",
]
