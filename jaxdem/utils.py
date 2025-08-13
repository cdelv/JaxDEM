# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions used to set up simulations and analyze the output.
"""

import jax
import jax.numpy as jnp
from jax.typing import ArrayLike

from typing import Sequence, Tuple, Union, Optional

from .state import State


# ------------------------------------------------------------------ #
# 1. Grid initialiser                                                #
# ------------------------------------------------------------------ #
def grid_state(
    *,
    n_per_axis: Sequence[int],  # e.g. (nx, ny, nz)  or (nx, ny)
    spacing: ArrayLike | float,  # lattice spacing  (sx, sy, …)
    radius: float = 0.5,  # same radius for every sphere
    mass: float = 1.0,
    jitter: float = 0.0,  # optional small random offset
    vel_range: Optional[ArrayLike] = None,
    radius_range: Optional[ArrayLike] = None,
    mass_range: Optional[ArrayLike] = None,
    seed: int = 0,
    key: Optional[jax.Array] = None,
) -> State:
    """
    Create a state where particles sit on a rectangular lattice.

    Random values can be sampled for particle radii, masses and velocities
    by specifying ``*_range`` arguments, which are interpreted as
    ``(min, max)`` bounds for a uniform distribution.  When a range is not
    provided the corresponding ``radius`` or ``mass`` argument is used for
    all particles and the velocity components are sampled in ``[-1, 1]``.

    Parameters
    ----------
    n_per_axis : tuple[int]
        Number of spheres along each axis.
    spacing : tuple[float] | float
        Centre-to-centre distance; scalar is broadcast to every axis.
    radius, mass : float
        Shared radius / mass for all particles when the corresponding range
        is not provided.
    jitter : float
        Add a uniform random offset in the range [-jitter, +jitter] for
        non-perfect grids (useful to break symmetry).
    vel_range, radius_range, mass_range : ArrayLike | None
        ``(min, max)`` values for the velocity components, radii and masses.
    seed : int
        Integer seed used when ``key`` is not supplied.
    key : PRNG key, optional
        Controls randomness. If ``None`` a key will be created from ``seed``.

    Returns
    -------
    State
    """
    n_per_axis = tuple(n_per_axis)
    dim = len(n_per_axis)
    spacing = dim * (spacing,) if isinstance(spacing, (int, float)) else tuple(spacing)
    assert len(spacing) == dim

    # build grid
    axes = [jnp.arange(n) * s for n, s in zip(n_per_axis, spacing)]
    coords = jnp.stack(jnp.meshgrid(*axes, indexing="ij"), axis=-1)
    coords = coords.reshape((-1, dim))  # (N, dim)
    N, dim = coords.shape

    if key is None:
        key = jax.random.PRNGKey(seed)

    if jitter > 0.0:
        key, key_jitter = jax.random.split(key)
        coords += jax.random.uniform(
            key_jitter, coords.shape, minval=-jitter, maxval=jitter
        )

    if radius_range is not None:
        radius_range = jnp.asarray(radius_range, dtype=float)
        assert radius_range.size == 2, "radius_range should be size == 2"
        key, key_rad = jax.random.split(key)
        rad = jax.random.uniform(
            key_rad, (N,), minval=radius_range[0], maxval=radius_range[1], dtype=float
        )
    else:
        rad = radius * jnp.ones(N)

    if mass_range is not None:
        mass_range = jnp.asarray(mass_range, dtype=float)
        assert mass_range.size == 2, "mass_range should be size == 2"
        key, key_mass = jax.random.split(key)
        mass_arr = jax.random.uniform(
            key_mass, (N,), minval=mass_range[0], maxval=mass_range[1], dtype=float
        )
    else:
        mass_arr = mass * jnp.ones(N)

    if vel_range is None:
        vel_range = jnp.array([-1.0, 1.0], dtype=float)
    else:
        vel_range = jnp.asarray(vel_range, dtype=float)
        assert vel_range.size == 2, "vel_range should be size == 2"
    key, key_vel = jax.random.split(key)
    vel = jax.random.uniform(
        key_vel, shape=coords.shape, minval=vel_range[0], maxval=vel_range[1]
    )

    return State.create(
        pos=coords,
        vel=vel,
        rad=rad,
        mass=mass_arr,
    )


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
