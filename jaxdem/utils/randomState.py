# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions to randomly initialize states.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import Optional

from .. import State


def random_state(
    *,
    N: int,
    dim: int,
    box_size: Optional[ArrayLike] = None,
    box_anchor: Optional[ArrayLike] = None,
    radius_range: Optional[ArrayLike] = None,
    mass_range: Optional[ArrayLike] = None,
    vel_range: Optional[ArrayLike] = None,
    seed: int = 0,
) -> State:
    """
    Generate `N` non-overlap-checked particles uniformly in an axis-aligned box.

    Parameters
    ----------
    N
        Number of particles.
    dim
        Spatial dimension (2 or 3).
    box_size
        Edge lengths of the domain.
    box_anchor
        Coordinate of the lower box corner.
    radius_range, mass_range
        min and max values that the radius can take.
    vel_range
        min and max values that the velocity components can take.
    seed
        Integer for reproducibility.

    Returns
    -------
    State
        A fully-initialised `State` instance.
    """

    if box_size is None:
        box_size = 10 * jnp.ones(dim, dtype=float)
    box_size = jnp.asarray(box_size, dtype=float)

    if box_anchor is None:
        box_anchor = jnp.zeros(dim, dtype=float)
    box_anchor = jnp.asarray(box_anchor, dtype=float)

    if radius_range is None:
        radius_range = 10 * jnp.ones(2, dtype=float)
    radius_range = jnp.asarray(radius_range, dtype=float)
    assert radius_range.size == 2, "Rad range should be size == 2"

    if mass_range is None:
        mass_range = jnp.ones(2, dtype=float)
    mass_range = jnp.asarray(mass_range, dtype=float)
    assert mass_range.size == 2, "Mass range should be size == 2"

    if vel_range is None:
        vel_range = jnp.ones(2, dtype=float)
    vel_range = jnp.asarray(vel_range, dtype=float)
    assert vel_range.size == 2, "Vel range should be size == 2"

    box_min = box_anchor
    box_max = box_anchor + box_size

    key = jax.random.PRNGKey(seed)
    key_pos, key_rad, key_mass, key_vel = jax.random.split(key, 4)

    pos = jax.random.uniform(
        key_pos, (N, dim), minval=box_min, maxval=box_max, dtype=float
    )

    rad = jax.random.uniform(
        key_rad, (N,), minval=radius_range[0], maxval=radius_range[1], dtype=float
    )

    mass = jax.random.uniform(
        key_mass, (N,), minval=mass_range[0], maxval=mass_range[1], dtype=float
    )

    vel = jax.random.uniform(
        key_vel, (N, dim), minval=vel_range[0], maxval=vel_range[1], dtype=float
    )

    return State.create(
        pos=pos,
        vel=vel,
        rad=rad,
        mass=mass,
        ID=jnp.arange(N, dtype=int),
    )


__all__ = ["random_state"]
