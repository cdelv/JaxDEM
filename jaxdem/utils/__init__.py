# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions used to set up simulations and analyze the output.
"""

from __future__ import annotations

from .linalg import unit, cross_3X3D_1X2D
from .angles import signed_angle, signed_angle_x, angle, angle_x
from .gridState import grid_state
from .randomState import random_state
from .serialization import encode_callable, decode_callable
from .environment import env_step, env_trajectory_rollout, lidar
from .quaternion import Quaternion
from .clumps import compute_clump_properties
from .jamming import bisection_jam
from .dispersity import get_polydisperse_radii
from .h5 import load, save
from .randomizeOrientations import randomize_orientations
from .thermal import (
    calculate_rotational_kinetic_energy,
    calculate_translational_kinetic_energy,
    calculate_temperature,
    scale_to_temperature,
    set_temperature
)

__all__ = [
    "unit",
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
    "lidar",
    "Quaternion",
    "compute_clump_properties",
    "jamming",
    "dispersity",
    "randomize_orientations",
    "calculate_rotational_kinetic_energy",
    "calculate_translational_kinetic_energy",
    "calculate_temperature",
    "scale_to_temperature",
    "set_temperature",
]
