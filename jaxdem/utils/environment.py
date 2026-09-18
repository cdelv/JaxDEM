# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Utility functions to handle environments and LIDAR sensor."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from numbers import Integral
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp

from .linalg import norm

if TYPE_CHECKING:
    from .. import State, System
    from ..rl.environments import Environment


def _step_count(name: str, value: int, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _mask_agent_actions(action: jax.Array, mask: jax.Array) -> jax.Array:
    """Zero inactive agents without changing categorical or continuous dtypes."""
    action = jnp.asarray(action)
    mask = mask.reshape(mask.shape + (1,) * (action.ndim - mask.ndim))
    return jnp.where(mask, action, jnp.zeros_like(action))


@partial(jax.jit, static_argnames=("skip_frames",), inline=True)
def advance_action(
    env: Environment, action: jax.Array, *, skip_frames: int = 0,
    terminated: jax.Array | None = None, truncated: jax.Array | None = None,
) -> tuple[Environment, jax.Array, jax.Array]:
    """Snapshot the action baseline once, then advance physics to its endpoint.

    Evaluates ``1 + skip_frames`` physics steps for the full batch. Updates
    after each environment's first termination/truncation are discarded,
    preserving its boundary state independently of the other batch entries.
    Calls ``env.checkpoint(env, action)`` before the first physics step and
    preserves that historical baseline throughout the action. Returns the live
    endpoint and the two boundary flags without a second checkpoint. It does not reset
    episodes or accumulate intermediate rewards. Read endpoint reward before
    starting another action or resetting the environment.
    By default this starts a new transition (the caller must reset completed
    episodes). Optional incoming boundary masks keep already finished members
    frozen when continuing an evaluation rollout without automatic resets.
    Inactive agent actions are zeroed before checkpointing and stepping,
    preserving integer categorical indices and floating-point continuous actions.
    """
    skip_frames = _step_count("skip_frames", skip_frames)
    action = _mask_agent_actions(action, env.agent_mask(env))
    shape = jnp.shape(env.done(env))
    if terminated is None:
        terminated = jnp.zeros(shape, dtype=bool)
    if truncated is None:
        truncated = jnp.zeros(shape, dtype=bool)

    # Refresh history even for frozen entries: their difference reward is zero.
    env = env.checkpoint(env, action)

    def step_fn(carry, _):
        current, terminated, truncated = carry
        done_before = terminated | truncated
        # Slots can disappear between repeated physics steps. They must not
        # keep receiving the action sampled while they were active.
        stepped = current.step(current, _mask_agent_actions(action, current.agent_mask(current)))
        current = jax.tree.map(
            lambda new, old: jnp.where(
                done_before.reshape(done_before.shape + (1,)*(new.ndim-done_before.ndim)),
                old, new,
            ), stepped, current,
        )
        terminated = terminated | current.terminated(current)
        truncated = truncated | current.truncated(current)
        return (current, terminated, truncated), None

    (env, terminated, truncated), _ = jax.lax.scan(
        step_fn, (env, terminated, truncated),
        None, length=1 + skip_frames,
    )
    return env, terminated, truncated


@jax.jit(inline=True, static_argnames=("model", "n", "stride", "skip_frames"))
@partial(jax.named_call, name="utils.env_trajectory_rollout")
def env_trajectory_rollout(
    env: Environment,
    model: Callable[..., Any],
    key: jax.Array,
    graphstate: Any,
    *,
    n: int,
    stride: int = 1,
    skip_frames: int = 0,
    **kw: Any,
) -> tuple[Environment, jax.Array, Any, Environment]:
    """Roll out a trajectory by applying `model` in chunks of `stride` steps and
    collecting the environment after each chunk.

    This performs ``n * stride`` policy decisions with the same action/key
    sequence as one ``env_step(..., n=n*stride)`` call. Recording boundaries
    do not consume additional random keys.

    Parameters
    ----------
    env : Environment
        Initial environment pytree.
    model : Callable
        Callable with signature `model(obs, key, graphstate, **kw) -> (action, graphstate)`.
    key : jax.Array
        Initial random key.
    graphstate : Any
        Initial model state.
    n : int
        Total number of macro-steps.
    stride : int
        Number of steps to perform before recording the environment state.
    skip_frames : int
        Number of *additional* physics frames to repeat each action, so every
        logical step requests ``1 + skip_frames`` physics frames. State
        updates after a boundary are discarded; completed episodes stay
        frozen until the caller resets them. Defaults to 0.
    **kw : Any
        Extra keyword arguments passed to `model` on every step.

    Returns
    -------
    Tuple[Environment, jax.Array, Any, Environment]
        Final environment, advanced random key, final graphstate, and a stacked pytree of
        environments with length `n`, each snapshot taken after a chunk
        of `stride` steps.

    Examples
    --------
    >>> env, key, graphstate, traj = env_trajectory_rollout(env, model, key, graphstate, n=100, stride=5, objective=goal)

    """

    n = _step_count("n", n)
    stride = _step_count("stride", stride, minimum=1)
    skip_frames = _step_count("skip_frames", skip_frames)

    def body(
        carry: tuple[Environment, jax.Array, Any], _: None
    ) -> tuple[tuple[Environment, jax.Array, Any], Environment]:
        env, key, gs = carry
        env, key, gs = env_step(
            env, model, key, gs, n=stride, skip_frames=skip_frames, **kw
        )
        return (env, key, gs), env

    (env, key, graphstate), env_traj = jax.lax.scan(
        body, (env, key, graphstate), length=n, xs=None
    )
    return env, key, graphstate, env_traj


@jax.jit(inline=True, static_argnames=("model", "n", "skip_frames"))
@partial(jax.named_call, name="utils.env_step")
def env_step(
    env: Environment,
    model: Callable[..., Any],
    key: jax.Array,
    graphstate: Any,
    *,
    n: int = 1,
    skip_frames: int = 0,
    **kw: Any,
) -> tuple[Environment, jax.Array, Any]:
    """Advance the environment `n` steps using actions from `model`.

    Each policy action begins with one historical checkpoint. Observations
    and rewards read the resulting live state; no endpoint checkpoint is
    needed. Finished batch entries remain physically frozen, with their
    baseline refreshed for each requested action. This helper does not reset
    episodes or recurrent carry. The model is still called for each requested
    logical step, even for frozen environments; its state may therefore advance.

    Parameters
    ----------
    env : Environment
        Initial environment pytree (batchable).
    model : Callable
        Callable with signature `model(obs, key, graphstate, **kw) -> (action, graphstate)`.
    key : jax.Array
        JAX random key.  Use the returned (advanced) key for subsequent
        calls.
    graphstate : Any
        Initial model state.
    n : int
        Number of steps to perform.
    skip_frames : int
        Number of *additional* physics frames to repeat each action, so every
        logical step requests ``1 + skip_frames`` physics frames. State
        updates after a boundary are discarded; completed episodes stay
        frozen until the caller resets them. Defaults to 0.
    **kw : Any
        Extra keyword arguments forwarded to `model`.

    Returns
    -------
    Tuple[Environment, jax.Array, Any]
        Updated environment, the advanced random key, and updated graphstate.

    Examples
    --------
    >>> env, key, graphstate = env_step(env, model, key, graphstate, n=10, objective=goal)

    """

    n = _step_count("n", n)
    skip_frames = _step_count("skip_frames", skip_frames)

    def body(
        carry: tuple[Environment, jax.Array, Any], _: None
    ) -> tuple[tuple[Environment, jax.Array, Any], None]:
        env, key, gs = carry
        key, subkey = jax.random.split(key)

        env, gs = _env_step(env, model, subkey, gs, skip_frames=skip_frames, **kw)
        return (env, key, gs), None

    (env, key, graphstate), _ = jax.lax.scan(
        body, (env, key, graphstate), length=n, xs=None
    )
    return env, key, graphstate


@jax.jit(inline=True, static_argnames=("model", "skip_frames"))
@partial(jax.named_call, name="utils._env_step")
def _env_step(
    env: Environment,
    model: Callable[..., Any],
    key: jax.Array,
    graphstate: Any,
    *,
    skip_frames: int = 0,
    **kw: Any,
) -> tuple[Environment, Any]:
    """Single environment step driven by `model`.

    Parameters
    ----------
    env : Environment
        Current environment pytree.
    model : Callable
        Callable with signature `model(obs, key, graphstate, **kw) -> (action, graphstate)`.
    key : jax.Array
        Random key for the step.
    graphstate : Any
        Current model state.
    skip_frames : int
        Number of *additional* physics frames to repeat the action per
        observation (``1 + skip_frames`` frames total). Defaults to 0.
    **kw : Any
        Extra keyword arguments passed to `model`.

    Returns
    -------
    Tuple[Environment, Any]
        Updated environment after applying `env.step(env, action)` and the updated graphstate.

    """
    obs = env.observation(env)

    action, graphstate = model(obs, key, graphstate, **kw)

    env, _, _ = advance_action(
        env, action, skip_frames=skip_frames,
        terminated=env.terminated(env), truncated=env.truncated(env),
    )

    return env, graphstate


# ------------------------------------------------------------------ helpers --


def _bin_azimuth(rij: jax.Array, n_bins: int) -> jax.Array:
    r"""Azimuthal bin index from a displacement vector projected onto XY.

    Maps :math:`\theta \in [-\pi, \pi)` to an integer in ``[0, n_bins)``.
    Coincident points use angle zero, independently of floating-point zero signs.
    """
    theta = jnp.arctan2(rij[..., 1], rij[..., 0])
    theta = jnp.where((rij[..., 0] == 0) & (rij[..., 1] == 0), 0.0, theta)
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


def _bin_min_dist_and_idx(
    dist: jax.Array, bin_idx: jax.Array, n_bins: int
) -> tuple[jax.Array, jax.Array]:
    """Per-sensor, per-bin minimum distance and the target index achieving it.

    Scatter-min (``jax.ops.segment_min``) formulation that avoids
    materializing the ``(N_A, N_B, n_bins)`` one-hot tensor.

    Parameters
    ----------
    dist : jax.Array
        ``(N_A, N_B)`` distances. Masked entries must be ``inf``.
    bin_idx : jax.Array
        ``(N_A, N_B)`` integer bin assignment in ``[0, n_bins)``.
    n_bins : int
        Number of bins.

    Returns
    -------
    (min_dist, min_idx)
        Both ``(N_A, n_bins)``. Empty bins have ``min_dist = inf``. Their
        ``min_idx`` is unspecified (callers mask on proximity). Ties pick
        the lowest target index, matching ``jnp.argmin``.
    """
    n_a, n_b = dist.shape
    seg = (jnp.arange(n_a)[:, None] * n_bins + bin_idx).reshape(-1)
    dist_flat = dist.reshape(-1)
    n_seg = n_a * n_bins

    min_dist = jax.ops.segment_min(dist_flat, seg, num_segments=n_seg)
    is_min = dist_flat == min_dist[seg]
    col = jnp.broadcast_to(jnp.arange(n_b)[None, :], (n_a, n_b)).reshape(-1)
    min_idx = jax.ops.segment_min(jnp.where(is_min, col, n_b), seg, num_segments=n_seg)
    min_idx = jnp.clip(min_idx, 0, n_b - 1)
    return min_dist.reshape(n_a, n_bins), min_idx.reshape(n_a, n_bins)


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


@jax.jit(inline=True, static_argnames=("n_bins", "sense_edges"))
@partial(jax.named_call, name="utils.lidar_2d")
def lidar_2d(
    state: State,
    system: System,
    lidar_range: float,
    n_bins: int,
    sense_edges: bool = False,
) -> tuple[State, System, jax.Array, jax.Array, jax.Array]:
    r"""2-D LIDAR proximity readings and neighbor IDs.

    For every particle in ``state`` the function projects the displacement
    vectors to all other particles onto the :math:`xy`-plane and bins them
    by azimuthal angle into ``n_bins`` uniform sectors spanning
    :math:`[-\pi, \pi)`.  Each bin stores the proximity value and the
    index of the closest neighbor in that sector:

    .. math::
        p_k = \max(0,\; r_{\max} - d_{\min,k})

    This works identically for 2-D and 3-D position data. In the 3-D case
    the function ignores the :math:`z`-component of the displacement during
    binning and uses the full Euclidean distance for proximity.

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
    sense_edges : bool, optional
        If ``True``, the function includes domain boundaries as proximity
        sources. Wall detections receive an ID of ``-1``.  Only
        meaningful for bounded domains.  Default is ``False``.

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
    ``ids`` are indices into ``state.pos`` in its current order.

    Examples
    --------
    >>> state, system, prox, ids, overflow = lidar_2d(state, system,
    ...     lidar_range=5.0, n_bins=36)

    """
    pos = state.pos
    N = pos.shape[0]

    # Negated so deltas[i, j] points sensor i -> target j, matching the
    # sensor -> wall convention used by _merge_edges_2d.
    deltas = -system.domain.displacement(pos[:, None, :], pos[None, :, :], system)
    dist = jnp.where(jnp.eye(N, dtype=bool), jnp.inf, norm(deltas))

    bin_idx = _bin_azimuth(deltas, n_bins)
    min_dist, min_idx = _bin_min_dist_and_idx(dist, bin_idx, n_bins)

    proximity = jnp.maximum(0.0, lidar_range - min_dist)
    own_idx = jnp.arange(N)[:, None]
    ids = jnp.where(proximity > 0, min_idx, own_idx)

    if sense_edges:
        proximity, ids = _merge_edges_2d(
            proximity, ids, pos, system, n_bins, lidar_range
        )

    return state, system, proximity, ids, jnp.bool_(False)


@jax.jit(
    inline=True,
    static_argnames=("n_azimuth", "n_elevation", "sense_edges"),
)
@partial(jax.named_call, name="utils.lidar_3d")
def lidar_3d(
    state: State,
    system: System,
    lidar_range: float,
    n_azimuth: int,
    n_elevation: int,
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
    sense_edges : bool, optional
        If ``True``, the function includes domain boundaries as proximity
        sources. Wall detections receive an ID of ``-1``.  Default is
        ``False``.

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
    Returned ``ids`` index into ``state.pos`` in its current order.

    Examples
    --------
    >>> state, system, prox, ids, overflow = lidar_3d(state, system,
    ...     lidar_range=5.0, n_azimuth=36, n_elevation=18)

    """
    n_total = n_azimuth * n_elevation
    pos = state.pos
    N = pos.shape[0]

    # Negated so deltas[i, j] points sensor i -> target j, matching the
    # sensor -> wall convention used by _merge_edges_3d.
    deltas = -system.domain.displacement(pos[:, None, :], pos[None, :, :], system)
    dist = jnp.where(jnp.eye(N, dtype=bool), jnp.inf, norm(deltas))

    bin_idx = _bin_spherical(deltas, n_azimuth, n_elevation)
    min_dist, min_idx = _bin_min_dist_and_idx(dist, bin_idx, n_total)

    proximity = jnp.maximum(0.0, lidar_range - min_dist)
    own_idx = jnp.arange(N)[:, None]
    ids = jnp.where(proximity > 0, min_idx, own_idx)

    if sense_edges:
        proximity, ids = _merge_edges_3d(
            proximity, ids, pos, system, n_azimuth, n_elevation, lidar_range
        )

    return state, system, proximity, ids, jnp.bool_(False)


# ---------------------------------------------------------- cross variants --


@jax.jit(inline=True, static_argnames=("n_bins",))
@partial(jax.named_call, name="utils.cross_lidar_2d")
def cross_lidar_2d(
    pos_a: jax.Array,
    pos_b: jax.Array,
    system: System,
    lidar_range: float,
    n_bins: int,
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

    Returns
    -------
    Tuple[jax.Array, jax.Array, jax.Array]
        ``(proximity, ids, overflow)`` where ``proximity`` and ``ids``
        have shape ``(N_A, n_bins)`` and ``overflow`` is always ``False``.
        Empty bins get ``ids = -1``.

    Notes
    -----
    Uses an all-pairs approach and does **not** invoke the collider.
    Returned ``ids`` are indices into ``pos_b``.

    Examples
    --------
    >>> prox, ids, overflow = cross_lidar_2d(agents, obstacles, system,
    ...                                      lidar_range=5.0, n_bins=36)

    """
    # Negated so deltas[i, j] points sensor i -> target j.
    deltas = -system.domain.displacement(pos_a[:, None, :], pos_b[None, :, :], system)
    dist = norm(deltas)

    bin_idx = _bin_azimuth(deltas, n_bins)
    min_dist, min_idx = _bin_min_dist_and_idx(dist, bin_idx, n_bins)

    proximity = jnp.maximum(0.0, lidar_range - min_dist)
    ids = jnp.where(proximity > 0, min_idx, -1)

    return proximity, ids, jnp.bool_(False)


@jax.jit(inline=True, static_argnames=("n_azimuth", "n_elevation"))
@partial(jax.named_call, name="utils.cross_lidar_3d")
def cross_lidar_3d(
    pos_a: jax.Array,
    pos_b: jax.Array,
    system: System,
    lidar_range: float,
    n_azimuth: int,
    n_elevation: int,
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

    Returns
    -------
    Tuple[jax.Array, jax.Array, jax.Array]
        ``(proximity, ids, overflow)`` where ``proximity`` and ``ids``
        have shape ``(N_A, n_azimuth * n_elevation)`` and ``overflow`` is
        always ``False``.  Empty bins get ``ids = -1``.

    Notes
    -----
    Uses an all-pairs approach and does **not** invoke the collider.
    Returned ``ids`` are indices into ``pos_b``.

    Examples
    --------
    >>> prox, ids, overflow = cross_lidar_3d(agents, obstacles, system,
    ...                                      lidar_range=5.0, n_azimuth=36,
    ...                                      n_elevation=18)

    """
    n_total = n_azimuth * n_elevation

    # Negated so deltas[i, j] points sensor i -> target j.
    deltas = -system.domain.displacement(pos_a[:, None, :], pos_b[None, :, :], system)
    dist = norm(deltas)

    bin_idx = _bin_spherical(deltas, n_azimuth, n_elevation)
    min_dist, min_idx = _bin_min_dist_and_idx(dist, bin_idx, n_total)

    proximity = jnp.maximum(0.0, lidar_range - min_dist)
    ids = jnp.where(proximity > 0, min_idx, -1)

    return proximity, ids, jnp.bool_(False)


__all__ = [
    "advance_action",
    "cross_lidar_2d",
    "cross_lidar_3d",
    "env_step",
    "env_trajectory_rollout",
    "lidar_2d",
    "lidar_3d",
]
