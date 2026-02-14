# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions to initialize states with particles arranged in a grid.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from typing import TYPE_CHECKING, Sequence, Optional, Tuple
from functools import partial

if TYPE_CHECKING:
    from ..state import State


@partial(jax.named_call, name="utils.grid_state")
def grid_state(
    *,
    n_per_axis: Sequence[int],  # e.g. (nx, ny, nz)  or (nx, ny)
    spacing: ArrayLike | float,  # lattice spacing  (sx, sy, …)
    radius: float = 1.0,  # same radius for every sphere
    mass: float = 1.0,
    jitter: float = 0.0,  # optional small random offset
    vel_range: Optional[ArrayLike] = None,
    radius_range: Optional[ArrayLike] = None,
    mass_range: Optional[ArrayLike] = None,
    seed: int = 0,
    key: Optional[jax.Array] = None,
) -> "State":
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
    from ..state import State

    n_per_axis = tuple(n_per_axis)
    dim = len(n_per_axis)
    spacing_vals: Tuple[float, ...]
    if isinstance(spacing, (int, float)):
        spacing_vals = tuple(float(spacing) for _ in range(dim))
    else:
        spacing_vals = tuple(float(s) for s in np.asarray(spacing).tolist())
    assert len(spacing_vals) == dim

    # build grid
    axes = [jnp.arange(n) * s for n, s in zip(n_per_axis, spacing_vals)]
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


__all__ = ["grid_state"]
