# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions used to set up simulations and analyze the output.
"""

from __future__ import annotations

from .linalg import unit
from .angles import signed_angle, signed_angle_x, angle, angle_x
from .gridState import grid_state
from .randomState import random_state
from .serialization import encode_callable, decode_callable

__all__ = [
    "unit",
    "signed_angle",
    "signed_angle_x",
    "angle",
    "angle_x",
    "grid_state",
    "random_state",
    "encode_callable",
    "decode_callable",
]
