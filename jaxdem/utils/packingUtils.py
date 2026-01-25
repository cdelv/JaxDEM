# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions for calculating and changing the packing fraction.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import replace
from functools import partial

from typing import TYPE_CHECKING, Tuple

from ..minimizers import minimize
from ..colliders import NeighborList

if TYPE_CHECKING:
    from ..state import State
    from ..system import System

@jax.jit
def compute_particle_volume(state):  # DOES NOT WORK FOR DPS
    seg = jax.ops.segment_max(state.volume, state.clump_ID, num_segments=state.N)
    return jnp.sum(jnp.maximum(seg, 0.0))

@jax.jit
def compute_packing_fraction(state, system):
    return compute_particle_volume(state) / jnp.prod(system.domain.box_size)

@jax.jit
def scale_to_packing_fraction(state, system, new_packing_fraction):
    new_box_size_scalar = (
        compute_particle_volume(state) / new_packing_fraction
    ) ** (1 / state.dim)
    current_box_L = system.domain.box_size[0]
    scale_factor = new_box_size_scalar / current_box_L

    new_box_size = jnp.ones_like(system.domain.box_size) * new_box_size_scalar
    new_domain = replace(system.domain, box_size=new_box_size)

    new_state = replace(state, pos_c=state.pos_c * scale_factor)
    new_system = replace(system, domain=new_domain)

    # force rebuild the neighbor list if using it
    if isinstance(new_system.collider, NeighborList):
        new_system = replace(
            new_system,
            collider=replace(new_system.collider, n_build_times=0),
        )
    
    return new_state, new_system