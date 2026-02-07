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
def compute_particle_volume(
    state: "State",
) -> jax.Array:  # This is not the proper instantaneous volume for DPs
    """Return the total particle volume."""
    seg = jax.ops.segment_max(state.volume, state.clump_ID, num_segments=state.N)
    return jnp.sum(jnp.maximum(seg, 0.0))


@jax.jit
def compute_packing_fraction(state: "State", system: "System") -> jax.Array:
    # this assumes that the domain anchor is 0
    return compute_particle_volume(state) / jnp.prod(system.domain.box_size)


@jax.jit
def scale_to_packing_fraction(state, system, new_packing_fraction):
    # this assumes that the domain anchor is 0
    new_box_size_scalar = (compute_particle_volume(state) / new_packing_fraction) ** (
        1 / state.dim
    )
    current_box_L = system.domain.box_size[0]
    scale_factor = new_box_size_scalar / current_box_L

    new_box_size = jnp.ones_like(system.domain.box_size) * new_box_size_scalar
    new_domain = replace(system.domain, box_size=new_box_size)

    # For spheres and clumps, we can just rescale the positions via state.pos_c * scale_factor
    # But, for DPs, we need to scale the com positions
    # Both behaviors can be generalized by scaling the DP com positions, finding the offset
    # before and after the scaling, and applying the offset to state.pos_c
    # This preserves the size of the DPs, clumps, and spheres, uniformly
    total_pos = jax.ops.segment_sum(
        state.pos_c, state.deformable_ID, num_segments=state.N
    )
    dp_counts = jax.ops.segment_sum(
        jnp.ones((state.N,), dtype=state.pos_c.dtype),
        state.deformable_ID,
        num_segments=state.N,
    )
    dp_com = total_pos / jnp.maximum(
        dp_counts[:, None], 1.0
    )  # avoid divide by zero errors for empty clumps (MAY NOT BE NEEDED)
    offset = dp_com * scale_factor - dp_com

    new_state = replace(
        state, pos_c=state.pos_c + offset[state.deformable_ID]
    )  # broadcast back and apply shift
    new_system = replace(system, domain=new_domain)

    # force rebuild the neighbor list if using it
    if isinstance(new_system.collider, NeighborList):
        new_system = replace(
            new_system,
            collider=replace(new_system.collider, n_build_times=0),
        )

    return new_state, new_system
