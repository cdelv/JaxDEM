# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Material definitions for Lennard-Jones / WCA-style interactions.
"""

from __future__ import annotations

import jax

from dataclasses import dataclass

from . import Material


@Material.register("lj")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class LJMaterial(Material):
    """
    Minimal material for LJ/WCA interactions.

    Notes
    -----
    - `LJ` and `WCA` force laws use `epsilon_eff` from `MaterialTable` and derives sigma from particle radii,
      so only `epsilon` is required here (plus `density` for mass calculations).
    """

    epsilon: float


__all__ = ["LJMaterial"]
