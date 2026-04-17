# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Monte-Carlo-style sampling of the surface of a clump particle with a tracer
clump.

The target ("central") clump is held fixed. A tracer clump is placed at a
sequence of approach directions on a sphere (3D) or circle (2D) surrounding
the target. At every approach direction, the tracer is pushed toward the
target along the center-to-center direction until the two clumps achieve a
user-specified geometric overlap -- defined as the maximum pairwise sphere
overlap ``delta = r_i + r_j - |x_i - x_j|`` over all (central-sphere,
tracer-sphere) pairs. Once the target overlap is reached, the interaction
force is decomposed into normal/tangential components with respect to the
center-to-center axis, yielding an effective friction coefficient
``mu = |F_t| / |F_n|``.

This sweep is repeated over a set of tracer orientations, so the map of
``mu`` across the target surface represents the tracer-accessible surface
area (SASA-like) along with the contact-anisotropy at every sample point.

In 3D, full ``SO(3)`` orientation coverage is obtained by sweeping over
(facing direction on ``S^2``, roll angle about that facing axis), which
correctly handles asymmetric tracers. In 2D the orientation degree of
freedom is a single angle.
"""

from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from ..state import State
from .quaternion import Quaternion

if TYPE_CHECKING:  # pragma: no cover
    from ..system import System


_CENTRAL_ID = 0
_TRACER_ID = 1


# --------------------------------------------------------------------------
# Quaternion helpers
# --------------------------------------------------------------------------


def _quat_from_x_to_3d(d: jax.Array) -> jax.Array:
    """Unit quaternion ``[w, x, y, z]`` rotating ``[1, 0, 0]`` to unit vector ``d``."""
    q = jnp.array([1.0 + d[0], 0.0, -d[2], d[1]])
    norm = jnp.linalg.norm(q)
    return jnp.where(norm < 1e-8, jnp.array([0.0, 0.0, 1.0, 0.0]), q / norm)


def _quat_from_x_to_2d(d: jax.Array) -> jax.Array:
    """Quaternion rotating ``[1, 0]`` to unit vector ``d`` in 2D (rotation about z)."""
    angle = jnp.arctan2(d[1], d[0])
    return jnp.array([jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2)])


def _angle_to_quat_2d(angle: float | jax.Array) -> jax.Array:
    """2D rotation by ``angle`` about z as quaternion ``[cos(a/2), 0, 0, sin(a/2)]``."""
    angle = jnp.asarray(angle, dtype=float)
    half = angle / 2
    return jnp.stack([jnp.cos(half), jnp.zeros_like(half), jnp.zeros_like(half), jnp.sin(half)])


def _make_q_base_2d(angle_base: float | jax.Array) -> Quaternion:
    """Tracer body-frame orientation parametrized by a single rotation angle."""
    angle = jnp.asarray(angle_base, dtype=float)
    probe_dir = jnp.array([jnp.cos(angle), jnp.sin(angle)])
    q_fwd_raw = _quat_from_x_to_2d(-probe_dir)
    q_fwd = Quaternion(w=q_fwd_raw[0:1][None, :], xyz=q_fwd_raw[1:4][None, :])
    return Quaternion.inv(q_fwd)


def _make_q_base_3d(facing_dir: jax.Array, roll: float | jax.Array) -> Quaternion:
    """Tracer body-frame orientation parametrized by (facing direction, roll).

    * ``facing_dir`` : unit vector on ``S^2`` (body-frame direction that should
      end up pointing *at* the central particle before the approach-direction
      quaternion is applied).
    * ``roll``       : rotation about the body-frame ``+X`` axis, in radians.

    The returned quaternion spans all of ``SO(3)`` as ``(facing_dir, roll)``
    ranges over ``S^2 x [0, 2 pi)`` (with the usual two-fold cover).
    """
    q_fwd_raw = _quat_from_x_to_3d(-facing_dir)
    q_fwd = Quaternion(w=q_fwd_raw[0:1][None, :], xyz=q_fwd_raw[1:4][None, :])
    q_facing = Quaternion.inv(q_fwd)

    roll = jnp.asarray(roll, dtype=float)
    half = roll / 2
    zero = jnp.zeros_like(half)
    q_roll = Quaternion(
        w=jnp.cos(half).reshape(1, 1),
        xyz=jnp.stack([jnp.sin(half), zero, zero]).reshape(1, 3),
    )
    return q_facing @ q_roll


def _q_base_3d_vec(facing_dir: jax.Array, roll: jax.Array) -> jax.Array:
    """Flat ``(4,)`` ``[w, x, y, z]`` form of :func:`_make_q_base_3d` for vmap."""
    q = _make_q_base_3d(facing_dir, roll)
    return jnp.concatenate([q.w.reshape(1), q.xyz.reshape(3)])


def _q_base_2d_vec(angle: jax.Array) -> jax.Array:
    """Flat ``(4,)`` ``[w, x, y, z]`` form of :func:`_make_q_base_2d` for vmap."""
    q = _make_q_base_2d(angle)
    return jnp.concatenate([q.w.reshape(1), q.xyz.reshape(3)])


def _compose_pair(q_dir_vec: jax.Array, q_base_vec: jax.Array) -> jax.Array:
    """Unit-quaternion product ``q_dir @ q_base`` for flat ``(4,)`` inputs."""
    q_dir = Quaternion(w=q_dir_vec[0:1][None, :], xyz=q_dir_vec[1:4][None, :])
    q_base = Quaternion(w=q_base_vec[0:1][None, :], xyz=q_base_vec[1:4][None, :])
    q = Quaternion.unit(q_dir @ q_base)
    return jnp.concatenate([q.w.reshape(1), q.xyz.reshape(3)])


# --------------------------------------------------------------------------
# Deterministic near-uniform direction sampling
# --------------------------------------------------------------------------


def _sample_directions(n: int, dim: int) -> jax.Array:
    """Exactly ``n`` deterministic near-uniform unit vectors on ``S^{dim-1}``.

    * 2D (``S^1``): exact equispaced angles via ``linspace`` -- no better
      distribution exists.
    * 3D (``S^2``): Fibonacci (golden-spiral) lattice -- near-uniform with
      spherical-cap discrepancy scaling as ``1/sqrt(n)``, computed in
      closed form with no optimization loop.

    Reproducible across invocations (pure function of ``n`` and ``dim``; no
    RNG state or iteration count to configure). Replaces the earlier
    Thomson-mesh sampler, whose ``O(n^2 * steps)`` cost dominated the
    runtime at even moderate ``n``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}.")
    if dim == 2:
        angles = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
        return jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    if dim == 3:
        golden = (1.0 + math.sqrt(5.0)) / 2.0
        i = jnp.arange(n, dtype=float)
        phi = 2.0 * jnp.pi * i / golden
        # Offset (i + 0.5)/n keeps cos_theta strictly inside (-1, 1), avoiding
        # the degenerate poles that a raw linear mapping would place there.
        cos_theta = 1.0 - 2.0 * (i + 0.5) / n
        sin_theta = jnp.sqrt(jnp.clip(1.0 - cos_theta * cos_theta, 0.0, 1.0))
        return jnp.stack(
            [sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), cos_theta],
            axis=-1,
        )
    raise ValueError(f"dim must be 2 or 3; got {dim}.")


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def _bounding_radius(state: State, clump_id: int) -> float:
    """Outer radius of a clump from its COM: ``max(|pos_p_i| + rad_i)`` within the clump."""
    mask = np.asarray(state.clump_id) == clump_id
    pos_p = np.asarray(state.pos_p)[mask]
    rad = np.asarray(state.rad)[mask]
    return float(np.max(np.linalg.norm(pos_p, axis=-1) + rad))


def _pair_max_overlap(
    state: State, central_mask: jax.Array, tracer_mask: jax.Array
) -> jax.Array:
    """Maximum pairwise overlap between central and tracer spheres.

    ``overlap_ij = r_i + r_j - |x_i - x_j|``; positive values indicate
    penetration. The maximum is taken over ``(i in central, j in tracer)``.
    """
    pos = state.pos
    rad = state.rad
    dist = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    overlap = rad[:, None] + rad[None, :] - dist
    pair_mask = central_mask[:, None] & tracer_mask[None, :]
    return jnp.max(jnp.where(pair_mask, overlap, -jnp.inf))


# --------------------------------------------------------------------------
# Default measurement system
# --------------------------------------------------------------------------


def _create_default_system(state: State, margin: float) -> Any:
    """Minimal static measurement system (spring force, no integration)."""
    import jaxdem as jd  # deferred to avoid circular import

    box_size = jnp.ones(state.dim) * margin
    mats = [jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)]
    matcher = jd.MaterialMatchmaker.create("harmonic")
    mat_table = jd.MaterialTable.from_materials(mats, matcher=matcher)
    return jd.System.create(
        state_shape=state.shape,
        dt=1e-2,
        linear_integrator_type="",
        rotation_integrator_type="",
        domain_type="periodic",
        force_model_type="spring",
        collider_type="naive",
        mat_table=mat_table,
        domain_kw=dict(box_size=box_size, anchor=jnp.zeros_like(box_size)),
    )


# --------------------------------------------------------------------------
# Bisection on geometric overlap
# --------------------------------------------------------------------------


def _find_contact_at_overlap(
    state: State,
    target_overlap: jax.Array,
    separation_tolerance: jax.Array,
    max_separation: jax.Array,
    min_separation: jax.Array,
    tracer_mask: jax.Array,
    central_mask: jax.Array,
) -> tuple[jax.Array, State]:
    """Binary-search the tracer along the center-to-center direction until the
    maximum pairwise overlap with the central clump equals ``target_overlap``.

    Moving the tracer inward (along the center-to-center unit vector) strictly
    decreases every central-to-tracer sphere pair distance, so max overlap is
    monotonically non-decreasing as a function of tracer displacement. The
    bisection is therefore well-defined and converges to the unique separation
    at which ``overlap(sep) == target_overlap``.

    Implementation: the while_loop carries only the scalar bracket
    ``(sep_hi, sep_lo)``. All per-particle arrays (positions, radii, masks)
    are closure-captured constants, which keeps the batched-while kernel small
    under ``jax.vmap`` and short to compile.
    """
    n_central = jnp.sum(central_mask)
    n_tracer = jnp.sum(tracer_mask)
    com_c = jnp.sum(state.pos_c * central_mask[:, None], axis=0) / n_central
    com_t = jnp.sum(state.pos_c * tracer_mask[:, None], axis=0) / n_tracer
    r_ij = com_c - com_t
    separation = jnp.linalg.norm(r_ij)
    direction = r_ij / separation  # tracer moves along +direction to approach

    base_pos = state.pos_c
    base_rad = state.rad
    pair_mask = central_mask[:, None] & tracer_mask[None, :]

    # Clamp the tolerance to a safe floor a few ulps above the dtype noise for
    # the current bracket magnitude. Below this the subtraction
    # ``sep_hi - sep_lo`` saturates at the representable ulp gap and the
    # while_loop would spin forever; clamping guarantees natural termination.
    dtype_eps = jnp.finfo(base_pos.dtype).eps
    tol_floor = 4.0 * dtype_eps * max_separation
    effective_tol = jnp.maximum(separation_tolerance, tol_floor)

    def overlap_at(sep: jax.Array) -> jax.Array:
        delta = separation - sep  # positive -> tracer moves toward central
        pos = base_pos + delta * tracer_mask[:, None] * direction
        dist = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        overlap = base_rad[:, None] + base_rad[None, :] - dist
        return jnp.max(jnp.where(pair_mask, overlap, -jnp.inf))

    def cond(v: tuple[jax.Array, jax.Array]) -> jax.Array:
        sep_hi, sep_lo = v
        return sep_hi - sep_lo > effective_tol

    def body(v: tuple[jax.Array, jax.Array]) -> tuple[jax.Array, jax.Array]:
        sep_hi, sep_lo = v
        sep = 0.5 * (sep_hi + sep_lo)
        too_far = overlap_at(sep) < target_overlap
        sep_hi = jnp.where(too_far, sep, sep_hi)
        sep_lo = jnp.where(too_far, sep_lo, sep)
        return sep_hi, sep_lo

    sep_hi, sep_lo = jax.lax.while_loop(
        cond, body, (max_separation, min_separation)
    )
    final_sep = 0.5 * (sep_hi + sep_lo)
    total_delta = separation - final_sep
    state.pos_c = base_pos + total_delta * tracer_mask[:, None] * direction
    return final_sep, state


# --------------------------------------------------------------------------
# Per-probe measurement
# --------------------------------------------------------------------------


def _measure_probe(
    state: State,
    system: Any,
    tracer_position: jax.Array,
    quat: jax.Array,
    target_overlap: jax.Array,
    separation_tolerance: jax.Array,
    max_separation: jax.Array,
    min_separation: jax.Array,
    tracer_mask: jax.Array,
    central_mask: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Configure a single (approach direction, tracer orientation) probe,
    bisect to ``target_overlap``, compute interaction force and friction.
    """
    state.pos_c = jnp.broadcast_to(
        system.domain.box_size / 2, state.pos_c.shape
    ).copy()
    state.pos_c = state.pos_c + tracer_position * tracer_mask[:, None]

    state.q.w = jnp.where(tracer_mask[:, None], quat[0:1], 1.0)
    state.q.xyz = jnp.where(tracer_mask[:, None], quat[1:4], 0.0)

    _, state = _find_contact_at_overlap(
        state,
        target_overlap,
        separation_tolerance,
        max_separation,
        min_separation,
        tracer_mask,
        central_mask,
    )
    state, system = system.collider.compute_force(state, system)

    n_central = jnp.sum(central_mask)
    n_tracer = jnp.sum(tracer_mask)
    com_c = jnp.sum(state.pos_c * central_mask[:, None], axis=0) / n_central
    com_t = jnp.sum(state.pos_c * tracer_mask[:, None], axis=0) / n_tracer
    r_ij = com_c - com_t
    separation = jnp.linalg.norm(r_ij)
    direction = r_ij / separation

    force = jnp.sum(state.force * tracer_mask[:, None], axis=0)
    force_n_mag = jnp.sum(force * direction)
    force_t_mag = jnp.linalg.norm(force - force_n_mag * direction)
    # Guard against 0/0 when a probe fails to establish contact (e.g. an
    # approach direction that misses all central material). jnp.where on both
    # branches keeps the gradient safe under jit/vmap.
    no_contact = jnp.abs(force_n_mag) < 1e-30
    mu = jnp.where(
        no_contact,
        0.0,
        jnp.abs(force_t_mag / jnp.where(no_contact, 1.0, force_n_mag)),
    )
    return mu, separation


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def compute_surface_properties(
    central_state: State,
    tracer_state: State,
    target_overlap: float,
    *,
    system: "System | None" = None,
    n_points: int = 100,
    n_orientations: int = 1,
    n_rolls: int = 1,
    separation_tolerance: float = 1e-10,
    separation_scale: float = 1.1,
    batch_size: int = 10_000,
) -> dict[str, np.ndarray | jax.Array]:
    """Sample surface friction / accessibility over the surface of ``central_state``
    using ``tracer_state`` as a probe.

    Parameters
    ----------
    central_state, tracer_state : State
        Single-clump :class:`State` instances. Their initial orientations are
        ignored (both are reset per probe: central to identity, tracer to the
        swept orientation), so the sampling directions live in the body frame
        of each clump as encoded by its ``pos_p``.
    target_overlap : float
        Desired maximum pairwise sphere overlap
        (``r_i + r_j - |x_i - x_j|``) between central and tracer at the
        reported contact configuration. Must be positive; small values
        correspond to "just barely indented". The converged state satisfies
        this exactly up to ``separation_tolerance``.
    system : System, optional
        Interaction system used to compute forces (for ``mu``). If ``None``,
        a default static measurement system is built (spring force, elastic
        material, naive collider, periodic box large enough for the pair).
    n_points : int
        Exact number of surface sample points (approach directions). In 2D
        these are equispaced on ``S^1``; in 3D they are placed on ``S^2``
        via a Fibonacci (golden-spiral) lattice.
    n_orientations : int
        Number of tracer orientations. In 2D: this is the count of uniformly
        spaced rotation angles. In 3D: the count of facing directions sampled
        on ``S^2`` via a Fibonacci (golden-spiral) lattice (each facing
        direction is paired with every ``roll`` angle below, so total
        orientations are ``n_orientations * n_rolls``).
    n_rolls : int
        3D only: number of uniformly spaced rolls about the facing axis. Must
        be 1 in 2D (no roll degree of freedom). For asymmetric tracers set
        this > 1 to obtain full ``SO(3)`` coverage.
    separation_tolerance : float
        Bisection convergence tolerance on the tracer center-to-center
        separation. The converged ``max(overlap)`` error shrinks linearly
        with this value.
    separation_scale : float
        Safety factor for the upper bound of the bisection bracket.
    batch_size : int
        Number of probes per ``vmap`` call, over the flat
        ``n_points * n_orientations * n_rolls`` probe grid. Larger is better
        for GPU utilization; smaller cuts peak memory. With the default
        ``10_000`` typical sweeps fit in a single kernel launch.

    Returns
    -------
    dict
        A dictionary of stacked ndarrays:

        **Common**
            - ``mu`` -- friction coefficient ``|F_t| / |F_n|`` per probe,
              shape ``(n_points, *orientation_shape)``.
            - ``separation`` -- center-to-center distance at ``target_overlap``,
              same shape as ``mu``.
            - ``approach_directions`` -- surface sample directions, shape
              ``(n_points, dim)``.
            - ``target_overlap`` -- scalar float; echo of the input.
            - ``dim`` -- int; dimensionality (2 or 3).

        **2D only**
            - ``angle_surface`` -- ``(n_points,)``; polar angle of each
              approach direction.
            - ``tracer_angles`` -- ``(n_orientations,)``; the swept tracer
              rotation angles. Grid shape: ``(n_orientations,)``.

        **3D only**
            - ``theta_surface``, ``phi_surface`` -- ``(n_points,)`` each;
              spherical coordinates of each approach direction.
            - ``tracer_facings`` -- ``(n_orientations, 3)``; body-frame
              directions sampled on ``S^2``.
            - ``tracer_facing_theta``, ``tracer_facing_phi`` --
              ``(n_orientations,)`` each; spherical coords of the facings.
            - ``tracer_rolls`` -- ``(n_rolls,)``; roll angles about each
              facing axis. Grid shape: ``(n_orientations, n_rolls)``.

    Notes
    -----
    The ``orientation_shape`` is ``(n_orientations,)`` in 2D and
    ``(n_orientations, n_rolls)`` in 3D, so ``mu[i, ...]`` gives the full
    orientation map for approach-direction ``i`` and any slice along the
    leading axis gives the surface map for a fixed orientation. To recreate
    the actual probe state at some grid coordinate, read the corresponding
    ``approach_directions[i]`` and either ``tracer_angles[j]`` (2D) or
    ``tracer_facings[j], tracer_rolls[k]`` (3D).
    """
    if central_state.dim != tracer_state.dim:
        raise ValueError(
            f"dim mismatch: central={central_state.dim} vs tracer={tracer_state.dim}."
        )
    if target_overlap <= 0.0:
        raise ValueError(f"target_overlap must be positive; got {target_overlap}.")
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1; got {n_points}.")
    if n_orientations < 1:
        raise ValueError(f"n_orientations must be >= 1; got {n_orientations}.")
    if n_rolls < 1:
        raise ValueError(f"n_rolls must be >= 1; got {n_rolls}.")

    n_central_clumps = int(np.unique(np.asarray(central_state.clump_id)).size)
    n_tracer_clumps = int(np.unique(np.asarray(tracer_state.clump_id)).size)
    if n_central_clumps != 1 or n_tracer_clumps != 1:
        raise ValueError(
            "central_state and tracer_state must each contain exactly one clump; "
            f"got {n_central_clumps} and {n_tracer_clumps}."
        )

    dim = central_state.dim
    if dim == 2 and n_rolls != 1:
        raise ValueError(
            f"n_rolls must be 1 in 2D (no roll degree of freedom); got {n_rolls}."
        )

    start = time.perf_counter()
    # Merge with central as clump 0 and tracer as clump 1.
    state = State.merge(central_state, tracer_state)

    r_central = _bounding_radius(state, _CENTRAL_ID)
    r_tracer = _bounding_radius(state, _TRACER_ID)
    r_sum = r_central + r_tracer
    max_separation = r_sum * separation_scale

    # Warn if the requested tolerance / target_overlap is below the float
    # precision floor of the default JAX dtype. Below this, `sep_hi - sep_lo`
    # saturates at the ulp gap and bisection would otherwise run forever (the
    # kernel caps at a safe max-iter, but the result will just be ulp-noisy).
    dtype_eps = float(np.finfo(state.pos_c.dtype).eps)
    tol_floor = 4.0 * dtype_eps * max_separation
    if separation_tolerance < tol_floor or target_overlap < tol_floor:
        import warnings
        warnings.warn(
            f"separation_tolerance={separation_tolerance:g} / "
            f"target_overlap={target_overlap:g} is below the dtype noise "
            f"floor ({tol_floor:g}) for the current bracket in dtype "
            f"{state.pos_c.dtype}. The bisection will be internally clamped "
            "to this floor; the result will be ulp-noisy. To get tighter "
            "control, enable x64 with "
            "`jax.config.update('jax_enable_x64', True)`.",
            stacklevel=2,
        )
    # Bracket lower bound. Using ``r_sum`` as the "just-contact" separation
    # assumes first contact is at the sum of outer bounding radii -- only
    # true for convex clumps. For clumps with interior spheres (cores,
    # overlapping asperities with concave gaps) the first-contact separation
    # along some approach directions is strictly inside ``r_sum``, and a
    # tight lower bound causes the bisection to exit with no contact -> NaN.
    # A wide lower bound costs only ~log2(width) extra iterations and covers
    # any reasonable clump geometry whose material reaches near the origin.
    min_separation = max(r_sum * 1e-3, 1e-6)

    if system is None:
        system = _create_default_system(state, margin=max_separation * 4.0)

    central_mask = state.clump_id == _CENTRAL_ID
    tracer_mask = state.clump_id == _TRACER_ID

    box_size = system.domain.box_size
    x_hat = jnp.zeros(dim).at[0].set(1.0)
    med_separation = 0.5 * (max_separation + min_separation)
    state.pos_c = jnp.broadcast_to(box_size / 2, state.pos_c.shape).copy()
    state.pos_c = state.pos_c + med_separation * tracer_mask[:, None] * x_hat
    state.q.w = jnp.ones_like(state.q.w)
    state.q.xyz = jnp.zeros_like(state.q.xyz)
    jax.block_until_ready(state.pos_c)
    print("state creation", time.perf_counter() - start)

    start = time.perf_counter()
    # Approach directions (surface sample points).
    approach_dirs = _sample_directions(n_points, dim)
    if dim == 3:
        q_dirs = jax.vmap(_quat_from_x_to_3d)(approach_dirs)
    else:
        q_dirs = jax.vmap(_quat_from_x_to_2d)(approach_dirs)
    tracer_positions = approach_dirs * med_separation

    if dim == 3:
        facings = _sample_directions(n_orientations, 3)
        rolls = jnp.linspace(0.0, 2.0 * jnp.pi, n_rolls, endpoint=False)
        facings_grid = jnp.broadcast_to(
            facings[:, None, :], (n_orientations, n_rolls, 3)
        ).reshape(-1, 3)
        rolls_grid = jnp.broadcast_to(
            rolls[None, :], (n_orientations, n_rolls)
        ).reshape(-1)
        q_bases = jax.vmap(_q_base_3d_vec)(facings_grid, rolls_grid)
    else:
        angles = jnp.linspace(0.0, 2.0 * jnp.pi, n_orientations, endpoint=False)
        q_bases = jax.vmap(_q_base_2d_vec)(angles)

    jax.block_until_ready(tracer_positions)
    jax.block_until_ready(q_bases)
    print("direction sampling", time.perf_counter() - start)

    measure_batch = jax.jit(
        jax.vmap(
            _measure_probe,
            in_axes=(None, None, 0, 0, None, None, None, None, None, None),
        )
    )

    start = time.perf_counter()
    # Build the full (n_points x n_orientations [x n_rolls]) probe grid and
    # flatten it into a single axis so the whole sweep becomes one (or a
    # small number of) large vmap'd kernel launch(es).
    n_orient_total = int(q_bases.shape[0])

    compose_outer = jax.jit(
        jax.vmap(
            jax.vmap(_compose_pair, in_axes=(None, 0)),
            in_axes=(0, None),
        )
    )
    q_grid = compose_outer(q_dirs, q_bases)  # (n_points, n_orient_total, 4)
    flat_q = q_grid.reshape(-1, 4)
    flat_pos = jnp.broadcast_to(
        tracer_positions[:, None, :], (n_points, n_orient_total, dim)
    ).reshape(-1, dim)
    n_total = int(flat_q.shape[0])
    jax.block_until_ready(flat_q)
    print("flattening", time.perf_counter() - start)

    start = time.perf_counter()
    mu_flat = np.zeros(n_total)
    sep_flat = np.zeros(n_total)
    n_batches = math.ceil(n_total / batch_size)

    batch_iter: Any = range(n_batches)
    if n_batches > 1:
        try:
            from tqdm import tqdm
        except ImportError:
            pass
        else:
            batch_iter = tqdm(batch_iter, total=n_batches, desc="surface probes")

    for b in batch_iter:
        bstart = b * batch_size
        bend = min(bstart + batch_size, n_total)
        _mu, _sep = measure_batch(
            state,
            system,
            flat_pos[bstart:bend],
            flat_q[bstart:bend],
            jnp.asarray(target_overlap),
            jnp.asarray(separation_tolerance),
            jnp.asarray(max_separation),
            jnp.asarray(min_separation),
            tracer_mask,
            central_mask,
        )
        mu_flat[bstart:bend] = np.asarray(_mu)
        sep_flat[bstart:bend] = np.asarray(_sep)
    print("sampling", time.perf_counter() - start)

    # --- Unflatten and package the result --------------------------------
    if dim == 3:
        mu_grid = mu_flat.reshape(n_points, n_orientations, n_rolls)
        sep_grid = sep_flat.reshape(n_points, n_orientations, n_rolls)

        theta_surface = np.arccos(np.asarray(approach_dirs[:, 2]))
        phi_surface = np.arctan2(
            np.asarray(approach_dirs[:, 1]), np.asarray(approach_dirs[:, 0])
        )
        facings_np = np.asarray(facings)
        facing_theta = np.arccos(facings_np[:, 2])
        facing_phi = np.arctan2(facings_np[:, 1], facings_np[:, 0])
        return dict(
            mu=mu_grid,
            separation=sep_grid,
            approach_directions=np.asarray(approach_dirs),
            theta_surface=theta_surface,
            phi_surface=phi_surface,
            tracer_facings=facings_np,
            tracer_facing_theta=facing_theta,
            tracer_facing_phi=facing_phi,
            tracer_rolls=np.asarray(rolls),
            target_overlap=float(target_overlap),
            dim=dim,
        )

    mu_grid = mu_flat.reshape(n_points, n_orientations)
    sep_grid = sep_flat.reshape(n_points, n_orientations)
    angle_surface = np.arctan2(
        np.asarray(approach_dirs[:, 1]), np.asarray(approach_dirs[:, 0])
    )
    return dict(
        mu=mu_grid,
        separation=sep_grid,
        approach_directions=np.asarray(approach_dirs),
        angle_surface=angle_surface,
        tracer_angles=np.asarray(angles),
        target_overlap=float(target_overlap),
        dim=dim,
    )

from .particleCreation import placeholder_create

# REMOVE THE TIMING STUFF
# python -m jaxdem.utils.surfaceProperties
# DO NOT USE HOLLOW PARTICLES

tracer_radius = 0.01
asperity_radius = 0.1

central_state = placeholder_create(
    N=1,
    nv=30,
    dim=3,
    particle_radius=0.5,
    asperity_radius=asperity_radius,
    n_steps=10_000,
    n_samples=10_000,
    core_type='solid',
)

for tracer_radius in [0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]:
    tracer_state = placeholder_create(  # tracer with a sphere
        N=1,
        nv=1,
        dim=3,
        particle_radius=tracer_radius,
        asperity_radius=tracer_radius,
        n_steps=10_000,
        n_samples=10_000,
        core_type='hollow',
    )


    results = compute_surface_properties(
        central_state,
        tracer_state,
        target_overlap=1e-10,
        n_points=100_000,
        n_orientations=1,  # not needed for spherical tracer
        n_rolls=1,  # not needed for spherical tracer
        separation_tolerance=1e-12,  # should be 2 orders of magnitude smaller than target_overlap
    )

    np.savez(
        f'delete-this-data/rad-{tracer_radius}.npz',
        **results,
        tracer_radius=tracer_radius,
    )