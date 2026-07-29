# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Material definitions for Lennard-Jones / WCA-style interactions."""

from __future__ import annotations

import jax

from dataclasses import dataclass

from . import Material


@Material.register("lj")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class LJMaterial(Material):
    """Material for LJ/WCA interactions.

    Notes
    -----
    - The `LJ` and `WCA` force laws use `epsilon_eff` from `MaterialTable` and derive sigma from particle radii,
      so this material only needs `epsilon` (plus `density` for mass calculations).

    """

    epsilon: float


__all__ = ["LJMaterial"]
