# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions used to set up simulations and analyze the output.
"""

from __future__ import annotations
from typing import Sequence, Optional

import jax
import jax.numpy as jnp

from .state import State


# ------------------------------------------------------------------ #
# 1. Grid initialiser                                                #
# ------------------------------------------------------------------ #
def grid_state(
    *,
    n_per_axis: Sequence[int],          # e.g. (nx, ny, nz)  or (nx, ny)
    spacing: Sequence[float] | float,   # lattice spacing  (sx, sy, …)
    radius:  float = 0.5,               # same radius for every sphere
    mass:    float = 1.0,
    jitter:  float = 0.0,               # optional small random offset
    key: Optional[jax.Array] = None,
) -> State:
    """
    Create a state where particles sit on a rectangular lattice.

    Parameters
    ----------
    n_per_axis : tuple[int]
        Number of spheres along each axis.
    spacing : tuple[float] | float
        Centre-to-centre distance; scalar is broadcast to every axis.
    radius, mass : float
        Shared radius / mass for all particles.
    jitter : float
        Add a uniform random offset in the range [-jitter, +jitter] for
        non-perfect grids (useful to break symmetry).
    key : PRNG key
        Required when `jitter > 0`.

    Returns
    -------
    State
    """
    n_per_axis = tuple(n_per_axis)
    dim = len(n_per_axis)
    spacing = (dim * (spacing,)    if isinstance(spacing, (int, float))
               else tuple(spacing))
    assert len(spacing) == dim

    # build grid
    axes = [jnp.arange(n) * s for n, s in zip(n_per_axis, spacing)]
    coords = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
    coords = coords.reshape((-1, dim))                      # (N, dim)
    N, dim = coords.shape

    if jitter > 0.0:
        if key is None:
            raise ValueError("`key` must be provided when jitter > 0")
        coords += jax.random.uniform(key, coords.shape, minval=-jitter, maxval=jitter)

    
    key = jax.random.key(0)
    vel = jax.random.uniform(key, shape=coords.shape, minval=-1.0, maxval=1.0)

    return State.create(
        pos=coords,
        vel=vel,
        rad=radius * jnp.ones(N),
        mass=mass * jnp.ones(N),
    )


# ------------------------------------------------------------------ #
# 2. Random initialiser                                              #
# ------------------------------------------------------------------ #
def random_state(
    *,
    N: int,
    box_min: Sequence[float],
    box_max: Sequence[float],
    radius: float = 0.5,
    mass:   float = 1.0,
    key:    jax.Array,
) -> State:
    """
    Uniformly sample N particle centres inside an axis-aligned box
    `[box_min, box_max]`.

    NOTE: no overlap check is performed; choose `radius` + `N`
    accordingly or post-process with your own rejection sampling.

    Returns
    -------
    State
    """
    box_min = jnp.asarray(box_min, dtype=float)
    box_max = jnp.asarray(box_max, dtype=float)
    dim = box_min.size

    coords = jax.random.uniform(
        key, shape=(N, dim), minval=box_min, maxval=box_max
    )
    return State.create(
        pos=coords,
        rad=radius * jnp.ones(N),
        mass=mass * jnp.ones(N),
    )