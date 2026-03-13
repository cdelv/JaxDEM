# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Utility functions to handle environments and LIDAR sensor."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from typing import TYPE_CHECKING, Any
from collections.abc import Callable
from functools import partial

from .linalg import norm

if TYPE_CHECKING:
    from .. import State, System
    from ..rl.environments import Environment


@partial(jax.jit, static_argnames=("model", "n", "stride"))
@partial(jax.named_call, name="utils.env_trajectory_rollout")
def env_trajectory_rollout(
    env: Environment,
    model: Callable[..., Any],
    key: jax.Array,
    *,
    n: int,
    stride: int = 1,
    **kw: Any,
) -> tuple[Environment, jax.Array, Environment]:
    """Roll out a trajectory by applying `model` in chunks of `stride` steps and
    collecting the environment after each chunk.

    Parameters
    ----------
    env : Environment
        Initial environment pytree.
    model : Callable
        Callable with signature `model(obs, key, **kw) -> action`.
    key : jax.Array
        JAX random key.  The returned key is the advanced version that
        should be used for subsequent calls.
    n : int
        Number of chunks to roll out. Total internal steps = `n * stride`.
    stride : int
        Steps per chunk between recorded snapshots.
    **kw : Any
        Extra keyword arguments passed to `model` on every step.

    Returns
    -------
    Tuple[Environment, jax.Array, Environment]
        Final environment, advanced random key, and a stacked pytree of
        environments with length `n`, each snapshot taken after a chunk
        of `stride` steps.

    Examples
    --------
    >>> env, key, traj = env_trajectory_rollout(env, model, key, n=100, stride=5, objective=goal)

    """

    def body(
        carry: tuple[Environment, jax.Array], _: None
    ) -> tuple[tuple[Environment, jax.Array], Environment]:
        env, key = carry
        key, subkey = jax.random.split(key)
        env, _ = env_step(env, model, subkey, n=stride, **kw)
        return (env, key), env

    (env, key), env_traj = jax.lax.scan(body, (env, key), length=n, xs=None)
    return env, key, env_traj


@partial(jax.jit, static_argnames=("model", "n"))
@partial(jax.named_call, name="utils.env_step")
def env_step(
    env: Environment,
    model: Callable[..., Any],
    key: jax.Array,
    *,
    n: int = 1,
    **kw: Any,
) -> tuple[Environment, jax.Array]:
    """Advance the environment `n` steps using actions from `model`.

    Parameters
    ----------
    env : Environment
        Initial environment pytree (batchable).
    model : Callable
        Callable with signature `model(obs, key, **kw) -> action`.
    key : jax.Array
        JAX random key.  The returned key is the advanced version that
        should be used for subsequent calls.
    n : int
        Number of steps to perform.
    **kw : Any
        Extra keyword arguments forwarded to `model`.

    Returns
    -------
    Tuple[Environment, jax.Array]
        Updated environment and the advanced random key.

    Examples
    --------
    >>> env, key = env_step(env, model, key, n=10, objective=goal)

    """

    def body(
        carry: tuple[Environment, jax.Array], _: None
    ) -> tuple[tuple[Environment, jax.Array], None]:
        env, key = carry
        key, subkey = jax.random.split(key)
        env = _env_step(env, model, subkey, **kw)
        return (env, key), None

    (env, key), _ = jax.lax.scan(body, (env, key), length=n, xs=None)
    return env, key


@partial(jax.jit, static_argnames=("model",))
@partial(jax.named_call, name="utils._env_step")
def _env_step(
    env: Environment,
    model: Callable[..., Any],
    key: jax.Array,
    **kw: Any,
) -> Environment:
    """Single environment step driven by `model`.

    Parameters
    ----------
    env : Environment
        Current environment pytree.
    model : Callable
        Callable with signature `model(obs, key, **kw) -> action`.
    **kw : Any
        Extra keyword arguments passed to `model`.

    Returns
    -------
    Environment
        Updated environment after applying `env.step(env, action)`.

    """
    obs = env.observation(env)
    action = model(obs, key, **kw)
    return env.step(env, action)


# ------------------------------------------------------------------ helpers --


def _bin_azimuth(rij: jax.Array, n_bins: int) -> jax.Array:
    r"""Azimuthal bin index from a displacement vector projected onto XY.

    Maps :math:`\theta \in [-\pi, \pi)` to an integer in ``[0, n_bins)``.
    """
    theta = jnp.arctan2(rij[..., 1], rij[..., 0])
    bins = jnp.floor((theta + jnp.pi) * (n_bins / (2.0 * jnp.pi))).astype(int)
    return bins % n_bins


def _bin_spherical(rij: jax.Array, n_azimuth: int, n_elevation: int) -> jax.Array:
    r"""Flat bin index from azimuth and elevation of a 3-D displacement.

    Azimuth :math:`\phi \in [-\pi, \pi)` is mapped to ``[0, n_azimuth)``.
    Elevation :math:`\theta \in [-\pi/2, \pi/2]` is mapped to
    ``[0, n_elevation)``.  The returned flat index equals
    ``az_bin * n_elevation + el_bin``.
    """
    phi = jnp.arctan2(rij[..., 1], rij[..., 0])
    r_xy = jnp.sqrt(rij[..., 0] ** 2 + rij[..., 1] ** 2)
    theta = jnp.arctan2(rij[..., 2], r_xy)

    az = jnp.floor((phi + jnp.pi) * (n_azimuth / (2.0 * jnp.pi))).astype(int)
    az = az % n_azimuth
    el = jnp.floor((theta + jnp.pi / 2.0) * (n_elevation / jnp.pi)).astype(int)
    el = jnp.clip(el, 0, n_elevation - 1)

    return az * n_elevation + el


def _merge_edges_2d(
    prox: jax.Array,
    ids: jax.Array,
    pos: jax.Array,
    system: System,
    n_bins: int,
    lidar_range: float,
) -> tuple[jax.Array, jax.Array]:
    r"""Merge domain boundary proximity into 2-D lidar bins.

    For each particle, computes the perpendicular distance to the four
    domain walls and updates ``prox`` / ``ids`` wherever a wall is closer
    than the current detection.  Wall detections receive ``id = -1``.
    """
    anchor = system.domain.anchor
    upper = anchor + system.domain.box_size

    def per_particle(
        prox_i: jax.Array, ids_i: jax.Array, pos_i: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        z = jnp.zeros_like(pos_i[0])
        wall_disp = jnp.stack(
            [
                jnp.stack([anchor[0] - pos_i[0], z]),
                jnp.stack([upper[0] - pos_i[0], z]),
                jnp.stack([z, anchor[1] - pos_i[1]]),
                jnp.stack([z, upper[1] - pos_i[1]]),
            ]
        )
        wall_dist = norm(wall_disp)
        wall_prox = jnp.maximum(0.0, lidar_range - wall_dist)
        wall_bins = _bin_azimuth(wall_disp, n_bins)

        def update(
            carry: tuple[jax.Array, jax.Array],
            x: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            p, d = carry
            wp, wb = x
            closer = wp > p[wb]
            p = p.at[wb].set(jnp.where(closer, wp, p[wb]))
            d = d.at[wb].set(jnp.where(closer, -1, d[wb]))
            return (p, d), None

        (prox_i, ids_i), _ = jax.lax.scan(
            update, (prox_i, ids_i), (wall_prox, wall_bins)
        )
        return prox_i, ids_i

    return jax.vmap(per_particle)(prox, ids, pos)


def _merge_edges_3d(
    prox: jax.Array,
    ids: jax.Array,
    pos: jax.Array,
    system: System,
    n_azimuth: int,
    n_elevation: int,
    lidar_range: float,
) -> tuple[jax.Array, jax.Array]:
    r"""Merge domain boundary proximity into 3-D lidar bins.

    Same as :func:`_merge_edges_2d` but for six walls in three dimensions.
    """
    anchor = system.domain.anchor
    upper = anchor + system.domain.box_size

    def per_particle(
        prox_i: jax.Array, ids_i: jax.Array, pos_i: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        z = jnp.zeros_like(pos_i[0])
        wall_disp = jnp.stack(
            [
                jnp.stack([anchor[0] - pos_i[0], z, z]),
                jnp.stack([upper[0] - pos_i[0], z, z]),
                jnp.stack([z, anchor[1] - pos_i[1], z]),
                jnp.stack([z, upper[1] - pos_i[1], z]),
                jnp.stack([z, z, anchor[2] - pos_i[2]]),
                jnp.stack([z, z, upper[2] - pos_i[2]]),
            ]
        )
        wall_dist = norm(wall_disp)
        wall_prox = jnp.maximum(0.0, lidar_range - wall_dist)
        wall_bins = _bin_spherical(wall_disp, n_azimuth, n_elevation)

        def update(
            carry: tuple[jax.Array, jax.Array],
            x: tuple[jax.Array, jax.Array],
        ) -> tuple[tuple[jax.Array, jax.Array], None]:
            p, d = carry
            wp, wb = x
            closer = wp > p[wb]
            p = p.at[wb].set(jnp.where(closer, wp, p[wb]))
            d = d.at[wb].set(jnp.where(closer, -1, d[wb]))
            return (p, d), None

        (prox_i, ids_i), _ = jax.lax.scan(
            update, (prox_i, ids_i), (wall_prox, wall_bins)
        )
        return prox_i, ids_i

    return jax.vmap(per_particle)(prox, ids, pos)


# ----------------------------------------------------------- self variants --


@partial(jax.jit, static_argnames=("n_bins", "max_neighbors", "sense_edges"))
@partial(jax.named_call, name="utils.lidar_2d")
def lidar_2d(
    state: State,
    system: System,
    lidar_range: float,
    n_bins: int,
    max_neighbors: int,
    sense_edges: bool = False,
) -> tuple[State, System, jax.Array, jax.Array, jax.Array]:
    r"""2-D LIDAR proximity readings and neighbor IDs.

    For every particle in ``state`` the displacement vectors to all other
    particles are projected onto the :math:`xy`-plane and binned by
    azimuthal angle into ``n_bins`` uniform sectors spanning
    :math:`[-\pi, \pi)`.  Each bin stores the proximity value and the
    index of the closest neighbor in that sector:

    .. math::
        p_k = \max(0,\; r_{\max} - d_{\min,k})

    This works identically for 2-D and 3-D position data; in the 3-D case
    the :math:`z`-component of the displacement is simply ignored during
    binning while the full Euclidean distance is used for proximity.

    Parameters
    ----------
    state : State
        Simulation state (positions, radii, etc.).
    system : System
        System configuration including domain.
    lidar_range : float
        Maximum detection range and reference distance for proximity.
    n_bins : int
        Number of angular bins (rays) spanning :math:`[-\pi, \pi)`.
    max_neighbors : int
        Unused. Kept for backward compatibility.
    sense_edges : bool, optional
        If ``True``, domain boundaries are included as proximity sources.
        Wall detections receive an ID of ``-1``.  Only meaningful for
        bounded domains.  Default is ``False``.

    Returns
    -------
    Tuple[State, System, jax.Array, jax.Array, jax.Array]
        ``(state, system, proximity, ids, overflow)`` where ``state`` and
        ``system`` are unchanged, ``proximity`` and ``ids`` have shape
        ``(N, n_bins)``, and ``overflow`` is always ``False``.
        Bins with no detection have ``ids`` set to the particle's own
        index.

    Notes
    -----
    This function computes all-pairs displacements directly from
    ``state.pos`` and does **not** invoke the collider.  The returned
    ``ids`` are indices into ``state.pos`` in whatever order it has at
    call time, so results are correct regardless of whether a cell-list
    collider has reordered the state.

    Examples
    --------
    >>> state, system, prox, ids, overflow = lidar_2d(state, system,
    ...     lidar_range=5.0, n_bins=36, max_neighbors=64)

    """
    pos = state.pos
    N = pos.shape[0]

    deltas = system.domain.displacement(pos[:, None, :], pos[None, :, :], system)
    dist = norm(deltas)

    bin_idx = _bin_azimuth(deltas, n_bins)
    one_hot = jax.nn.one_hot(bin_idx, n_bins)
    one_hot = one_hot * (~jnp.eye(N, dtype=bool))[..., None]

    masked_dist = jnp.where(one_hot > 0, dist[..., None], jnp.inf)
    min_dist = jnp.min(masked_dist, axis=1)
    min_idx = jnp.argmin(masked_dist, axis=1)

    proximity = jnp.maximum(0.0, lidar_range - min_dist)
    own_idx = jnp.arange(N)[:, None]
    ids = jnp.where(proximity > 0, min_idx, own_idx)

    if sense_edges:
        proximity, ids = _merge_edges_2d(
            proximity, ids, pos, system, n_bins, lidar_range
        )

    return state, system, proximity, ids, jnp.bool_(False)


@partial(
    jax.jit,
    static_argnames=("n_azimuth", "n_elevation", "max_neighbors", "sense_edges"),
)
@partial(jax.named_call, name="utils.lidar_3d")
def lidar_3d(
    state: State,
    system: System,
    lidar_range: float,
    n_azimuth: int,
    n_elevation: int,
    max_neighbors: int,
    sense_edges: bool = False,
) -> tuple[State, System, jax.Array, jax.Array, jax.Array]:
    r"""3-D LIDAR proximity readings and neighbor IDs.

    Similar to :func:`lidar_2d` but bins neighbors on a spherical grid
    defined by ``n_azimuth`` azimuthal sectors in :math:`[-\pi, \pi)` and
    ``n_elevation`` elevation bands in :math:`[-\pi/2, \pi/2]`.  The
    returned proximity and ID arrays have shape
    ``(N, n_azimuth * n_elevation)`` with flat indexing
    ``az * n_elevation + el``.

    Parameters
    ----------
    state : State
        Simulation state.
    system : System
        System configuration including domain.
    lidar_range : float
        Maximum detection range and reference distance for proximity.
    n_azimuth : int
        Number of azimuthal bins.
    n_elevation : int
        Number of elevation bins.
    max_neighbors : int
        Unused. Kept for backward compatibility.
    sense_edges : bool, optional
        If ``True``, domain boundaries are included as proximity sources.
        Wall detections receive an ID of ``-1``.  Default is ``False``.

    Returns
    -------
    Tuple[State, System, jax.Array, jax.Array, jax.Array]
        ``(state, system, proximity, ids, overflow)`` where ``state`` and
        ``system`` are unchanged, ``proximity`` and ``ids`` have shape
        ``(N, n_azimuth * n_elevation)``, and ``overflow`` is always
        ``False``.

    Notes
    -----
    Uses an all-pairs approach and does **not** invoke the collider.
    Returned ``ids`` index into ``state.pos`` in its current order,
    so results are correct regardless of collider-induced reordering.

    Examples
    --------
    >>> state, system, prox, ids, overflow = lidar_3d(state, system,
    ...     lidar_range=5.0, n_azimuth=36, n_elevation=18, max_neighbors=64)

    """
    n_total = n_azimuth * n_elevation
    pos = state.pos
    N = pos.shape[0]

    deltas = system.domain.displacement(pos[:, None, :], pos[None, :, :], system)
    dist = norm(deltas)

    bin_idx = _bin_spherical(deltas, n_azimuth, n_elevation)
    one_hot = jax.nn.one_hot(bin_idx, n_total)
    one_hot = one_hot * (~jnp.eye(N, dtype=bool))[..., None]

    masked_dist = jnp.where(one_hot > 0, dist[..., None], jnp.inf)
    min_dist = jnp.min(masked_dist, axis=1)
    min_idx = jnp.argmin(masked_dist, axis=1)

    proximity = jnp.maximum(0.0, lidar_range - min_dist)
    own_idx = jnp.arange(N)[:, None]
    ids = jnp.where(proximity > 0, min_idx, own_idx)

    if sense_edges:
        proximity, ids = _merge_edges_3d(
            proximity, ids, pos, system, n_azimuth, n_elevation, lidar_range
        )

    return state, system, proximity, ids, jnp.bool_(False)


# ---------------------------------------------------------- cross variants --


@partial(jax.jit, static_argnames=("n_bins", "max_neighbors"))
@partial(jax.named_call, name="utils.cross_lidar_2d")
def cross_lidar_2d(
    pos_a: jax.Array,
    pos_b: jax.Array,
    system: System,
    lidar_range: float,
    n_bins: int,
    max_neighbors: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""2-D LIDAR proximity and IDs from ``pos_a`` sensing targets in ``pos_b``.

    Computes all-pairs displacements from ``pos_a`` to ``pos_b``, bins by
    azimuthal angle, and returns per-bin proximity and closest target IDs.

    Parameters
    ----------
    pos_a : jax.Array
        Sensor positions, shape ``(N_A, dim)``.
    pos_b : jax.Array
        Target positions, shape ``(N_B, dim)``.
    system : System
        System configuration.
    lidar_range : float
        Maximum detection range and reference distance for proximity.
    n_bins : int
        Number of angular bins spanning :math:`[-\pi, \pi)`.
    max_neighbors : int
        Unused. Kept for backward compatibility.

    Returns
    -------
    Tuple[jax.Array, jax.Array, jax.Array]
        ``(proximity, ids, overflow)`` where ``proximity`` and ``ids``
        have shape ``(N_A, n_bins)`` and ``overflow`` is always ``False``.
        Empty bins get ``ids = -1``.

    Notes
    -----
    Uses an all-pairs approach and does **not** invoke the collider.
    Returned ``ids`` are indices into ``pos_b`` regardless of how
    ``pos_a`` may have been reordered by a cell-list collider.

    Examples
    --------
    >>> prox, ids, overflow = cross_lidar_2d(agents, obstacles, system,
    ...                                      lidar_range=5.0, n_bins=36,
    ...                                      max_neighbors=64)

    """
    deltas = system.domain.displacement(pos_a[:, None, :], pos_b[None, :, :], system)
    dist = norm(deltas)

    bin_idx = _bin_azimuth(deltas, n_bins)
    one_hot = jax.nn.one_hot(bin_idx, n_bins)

    masked_dist = jnp.where(one_hot > 0, dist[..., None], jnp.inf)
    min_dist = jnp.min(masked_dist, axis=1)
    min_idx = jnp.argmin(masked_dist, axis=1)

    proximity = jnp.maximum(0.0, lidar_range - min_dist)
    ids = jnp.where(proximity > 0, min_idx, -1)

    return proximity, ids, jnp.bool_(False)


@partial(jax.jit, static_argnames=("n_azimuth", "n_elevation", "max_neighbors"))
@partial(jax.named_call, name="utils.cross_lidar_3d")
def cross_lidar_3d(
    pos_a: jax.Array,
    pos_b: jax.Array,
    system: System,
    lidar_range: float,
    n_azimuth: int,
    n_elevation: int,
    max_neighbors: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    r"""3-D LIDAR proximity and IDs from ``pos_a`` sensing targets in ``pos_b``.

    Computes all-pairs displacements from ``pos_a`` to ``pos_b``, bins on
    a spherical grid, and returns per-bin proximity and closest target IDs.

    Parameters
    ----------
    pos_a : jax.Array
        Sensor positions, shape ``(N_A, 3)``.
    pos_b : jax.Array
        Target positions, shape ``(N_B, 3)``.
    system : System
        System configuration.
    lidar_range : float
        Maximum detection range and reference distance for proximity.
    n_azimuth : int
        Number of azimuthal bins.
    n_elevation : int
        Number of elevation bins.
    max_neighbors : int
        Unused. Kept for backward compatibility.

    Returns
    -------
    Tuple[jax.Array, jax.Array, jax.Array]
        ``(proximity, ids, overflow)`` where ``proximity`` and ``ids``
        have shape ``(N_A, n_azimuth * n_elevation)`` and ``overflow`` is
        always ``False``.  Empty bins get ``ids = -1``.

    Notes
    -----
    Uses an all-pairs approach and does **not** invoke the collider.
    Returned ``ids`` are indices into ``pos_b`` regardless of how
    ``pos_a`` may have been reordered by a cell-list collider.

    Examples
    --------
    >>> prox, ids, overflow = cross_lidar_3d(agents, obstacles, system,
    ...                                      lidar_range=5.0, n_azimuth=36,
    ...                                      n_elevation=18, max_neighbors=64)

    """
    n_total = n_azimuth * n_elevation

    deltas = system.domain.displacement(pos_a[:, None, :], pos_b[None, :, :], system)
    dist = norm(deltas)

    bin_idx = _bin_spherical(deltas, n_azimuth, n_elevation)
    one_hot = jax.nn.one_hot(bin_idx, n_total)

    masked_dist = jnp.where(one_hot > 0, dist[..., None], jnp.inf)
    min_dist = jnp.min(masked_dist, axis=1)
    min_idx = jnp.argmin(masked_dist, axis=1)

    proximity = jnp.maximum(0.0, lidar_range - min_dist)
    ids = jnp.where(proximity > 0, min_idx, -1)

    return proximity, ids, jnp.bool_(False)


__all__ = [
    "cross_lidar_2d",
    "cross_lidar_3d",
    "env_step",
    "env_trajectory_rollout",
    "lidar_2d",
    "lidar_3d",
]
