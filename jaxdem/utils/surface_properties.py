# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Monte-Carlo-style sampling of the surface of a clump particle with a tracer
clump.

The target ("central") clump stays fixed. The sweep places a tracer clump at
a sequence of approach directions on a sphere (3D) or circle (2D) surrounding
the target. At every approach direction, the sweep pushes the tracer toward
the target along the center-to-center direction until the two clumps reach a
user-specified geometric overlap -- the maximum pairwise sphere overlap
``delta = r_i + r_j - |x_i - x_j|`` over all (central-sphere, tracer-sphere)
pairs. At the target overlap, the sweep decomposes the interaction force into
normal/tangential components with respect to the center-to-center axis, which
gives an effective friction coefficient ``mu = |F_t| / |F_n|``.

The sweep repeats over a set of tracer orientations, so the map of ``mu``
across the target surface represents the tracer-accessible surface area
(SASA-like) along with the contact anisotropy at every sample point.

In 3D, sweeping over (facing direction on ``S^2``, roll angle about that
facing axis) gives full ``SO(3)`` orientation coverage, which correctly
handles asymmetric tracers. In 2D the orientation degree of freedom is a
single angle.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

import dataclasses
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from ..state import State
from .quaternion import Quaternion
from .linalg import dot, norm, norm2, unit

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
    return jnp.where(norm(q) < 1e-8, jnp.array([0.0, 0.0, 1.0, 0.0]), unit(q))


def _quat_from_x_to_2d(d: jax.Array) -> jax.Array:
    """Quaternion rotating ``[1, 0]`` to unit vector ``d`` in 2D (rotation about z)."""
    angle = jnp.arctan2(d[1], d[0])
    return jnp.array([jnp.cos(angle / 2), 0.0, 0.0, jnp.sin(angle / 2)])


def _angle_to_quat_2d(angle: float | jax.Array) -> jax.Array:
    """2D rotation by ``angle`` about z as quaternion ``[cos(a/2), 0, 0, sin(a/2)]``."""
    angle = jnp.asarray(angle, dtype=float)
    half = angle / 2
    return jnp.stack(
        [jnp.cos(half), jnp.zeros_like(half), jnp.zeros_like(half), jnp.sin(half)]
    )


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
      end up pointing *at* the central particle before the function applies
      the approach-direction quaternion).
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
# Direction / angle sampling (deterministic lattice or seeded uniform-random)
# --------------------------------------------------------------------------


def _sample_directions(n: int, dim: int, key: jax.Array | None = None) -> jax.Array:
    """Exactly ``n`` unit vectors on ``S^{dim-1}``.

    ``key is None`` (default) -- **deterministic near-uniform lattice**:

    * 2D (``S^1``): exact equispaced angles via ``linspace`` -- no better
      distribution exists.
    * 3D (``S^2``): Fibonacci (golden-spiral) lattice -- near-uniform with
      spherical-cap discrepancy scaling as ``1/sqrt(n)``, computed in
      closed form with no optimization loop.

    Reproducible across invocations (pure function of ``n`` and ``dim``, with
    no RNG state or iteration count to configure). Replaces the earlier
    Thomson-mesh sampler, whose ``O(n^2 * steps)`` cost dominated the
    runtime at even moderate ``n``.

    ``key`` given (a ``jax.random.PRNGKey``) -- **iid uniform-random** points,
    area-uniform on the sphere/circle so the polar/orientation angles are
    themselves uniformly distributed. Unlike the lattice, random points are
    *incommensurate* with any clump symmetry, which breaks the sampling-vs-
    particle aliasing that collapses the friction PDF to a few delta spikes
    for symmetric (uniform-asperity) clumps. Reproducible for a fixed ``key``.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}.")
    if key is None:
        if dim == 2:
            angles = jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
            return jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        if dim == 3:
            golden = (1.0 + math.sqrt(5.0)) / 2.0
            i = jnp.arange(n, dtype=float)
            phi = 2.0 * jnp.pi * i / golden
            # Offset (i + 0.5)/n keeps cos_theta strictly inside (-1, 1),
            # avoiding the degenerate poles a raw linear mapping would place.
            cos_theta = 1.0 - 2.0 * (i + 0.5) / n
            sin_theta = jnp.sqrt(jnp.clip(1.0 - cos_theta * cos_theta, 0.0, 1.0))
            return jnp.stack(
                [sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), cos_theta],
                axis=-1,
            )
        raise ValueError(f"dim must be 2 or 3; got {dim}.")

    if dim == 2:
        theta = jax.random.uniform(key, (n,), minval=0.0, maxval=2.0 * jnp.pi)
        return jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    if dim == 3:
        # Area-uniform sampling: cos(theta) = z ~ U(-1, 1) and phi ~ U(0, 2pi)
        # (Archimedes' hat-box theorem -- equal z-measure is equal area).
        u = jax.random.uniform(key, (n, 2))
        cos_theta = 1.0 - 2.0 * u[:, 0]
        phi = 2.0 * jnp.pi * u[:, 1]
        sin_theta = jnp.sqrt(jnp.clip(1.0 - cos_theta * cos_theta, 0.0, 1.0))
        return jnp.stack(
            [sin_theta * jnp.cos(phi), sin_theta * jnp.sin(phi), cos_theta],
            axis=-1,
        )
    raise ValueError(f"dim must be 2 or 3; got {dim}.")


def _sample_angles(n: int, key: jax.Array | None = None) -> jax.Array:
    """``n`` angles in ``[0, 2 pi)``: equispaced when ``key is None`` (the
    deterministic default), else iid ``U(0, 2 pi)`` for the given PRNG key.

    Used for the single 2D orientation DOF and the 3D roll about the facing
    axis, mirroring :func:`_sample_directions` for the angular grids.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}.")
    if key is None:
        return jnp.linspace(0.0, 2.0 * jnp.pi, n, endpoint=False)
    return jax.random.uniform(key, (n,), minval=0.0, maxval=2.0 * jnp.pi)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def _bounding_radius(state: State, clump_id: int) -> float:
    """Outer radius of a clump from its COM: ``max(|pos_p_i| + rad_i)`` within the clump."""
    mask = np.asarray(state.clump_id) == clump_id
    pos_p = np.asarray(state.pos_p)[mask]
    rad = np.asarray(state.rad)[mask]
    return float(np.max(np.linalg.norm(pos_p, axis=-1) + rad))


def _core_mask(state: State, clump_id: int) -> np.ndarray:
    """Boolean mask over *all* spheres flagging the core of ``clump_id``, if any.

    A "core" is an interior sphere sitting (approximately) at the clump COM
    (body-frame offset ``pos_p ~ 0``), as produced by the ``"solid"`` /
    ``"true-solid"`` core options in the particle builders (the builders
    strip a ``"phantom"`` core from the state, so it is never present here).
    Surface asperities sit at a finite distance ``~ core_radius`` from the COM, so
    the core is unambiguously the sphere whose ``|pos_p|`` is far smaller
    than every other sphere in the clump.

    Detection is scale-free: the function flags the closest-to-COM sphere as
    the core only when its offset is less than half the next-closest sphere's
    offset. Returns an all-``False`` mask when the clump has no such
    distinctly interior sphere or is a single bare sphere.
    """
    mask = np.asarray(state.clump_id) == clump_id
    idx = np.where(mask)[0]
    out = np.zeros(mask.shape, dtype=bool)
    if idx.size < 2:
        return out
    norms = np.linalg.norm(np.asarray(state.pos_p)[idx], axis=-1)
    order = np.argsort(norms)
    smallest, second = norms[order[0]], norms[order[1]]
    if smallest < 0.5 * second:
        out[idx[order[0]]] = True
    return out


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
# Contact along the approach ray
# --------------------------------------------------------------------------


def _find_contact_at_overlap(
    state: State,
    target_overlap: jax.Array,
    max_separation: jax.Array,
    central_idx: jax.Array,
    tracer_idx: jax.Array,
) -> tuple[jax.Array, State]:
    """Find the outermost sphere-pair entry along the tracer approach ray.

    Each sphere pair gives a quadratic contact interval in center separation.
    Its outer root is the first contact when approaching from outside. Taking
    the largest valid root handles cavities and nonconvex unions without a
    monotonic-overlap assumption. Return NaN when no pair reaches the target.

    """
    base_pos_c = state.pos_c
    com_c = base_pos_c[central_idx[0]]
    com_t = base_pos_c[tracer_idx[0]]
    ray = com_t - com_c
    separation = norm(ray)
    direction = ray / separation

    # Work with rotated body offsets, avoiding subtraction of world positions.
    offsets = state._pos_p_rot
    delta = offsets[central_idx][:, None, :] - offsets[tracer_idx][None, :, :]
    parallel = dot(delta, direction)
    perpendicular = delta - parallel[..., None] * direction
    radius = (
        state.rad[central_idx][:, None]
        + state.rad[tracer_idx][None, :]
        - target_overlap
    )
    discriminant = radius * radius - norm2(perpendicular)
    positive = discriminant > 0
    root = jnp.where(positive, jnp.sqrt(jnp.where(positive, discriminant, 1.0)), 0.0)
    entry = parallel + root
    valid = (
        (radius >= 0) & (discriminant >= 0) & (entry >= 0) & (entry <= max_separation)
    )
    outermost = jnp.max(jnp.where(valid, entry, -jnp.inf))
    found = jnp.any(valid)
    final_sep = jnp.where(found, outermost, max_separation)
    state.pos_c = base_pos_c.at[tracer_idx].add((final_sep - separation) * direction)
    return jnp.where(found, final_sep, jnp.nan), state


# --------------------------------------------------------------------------
# Per-probe measurement
# --------------------------------------------------------------------------


def _measure_probe(
    state: State,
    system: Any,
    tracer_position: jax.Array,
    quat: jax.Array,
    target_overlap: jax.Array,
    max_separation: jax.Array,
    tracer_mask: jax.Array,
    central_mask: jax.Array,
    central_core_mask: jax.Array,
    tracer_core_mask: jax.Array,
    central_idx: jax.Array,
    tracer_idx: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Configure a single (approach direction, tracer orientation) probe,
    solve for ``target_overlap``, compute interaction force and friction.
    """
    new_pos_c = (
        jnp.broadcast_to(system.domain.box_size / 2, state.pos_c.shape)
        + tracer_position * tracer_mask[:, None]
    )
    # Set the tracer orientation by REPLACING q, not by mutating state.q.w /
    # state.q.xyz in place. Only a whole-attribute assignment refreshes the
    # cached R(q) @ pos_p, so state.pos reflects the new orientation; an
    # in-place field write leaves state.pos stale at the previous orientation,
    # and the contact solver below then positions the tracer with the wrong pose.
    new_q = Quaternion.create(
        w=jnp.where(tracer_mask[:, None], quat[0:1], 1.0),
        xyz=jnp.where(tracer_mask[:, None], quat[1:4], 0.0),
    )
    state = dataclasses.replace(state, pos_c=new_pos_c, q=new_q)

    contact_separation, state = _find_contact_at_overlap(
        state,
        target_overlap,
        max_separation,
        central_idx,
        tracer_idx,
    )
    # A measurement: evaluate forces without advancing any pair history.
    state, system = system.collider.compute_force(state, system, advance_history=False)

    # Every sphere in a clump shares pos_c = rigid-body COM, so picking any
    # sphere belonging to each clump yields the COM without any averaging.
    com_central = state.pos_c[jnp.argmax(central_mask)]
    com_tracer = state.pos_c[jnp.argmax(tracer_mask)]
    r_ij = com_central - com_tracer
    separation = norm(r_ij)
    direction = r_ij / separation

    force = jnp.sum(state.force * central_mask[:, None], axis=0)
    force_n_mag = dot(force, direction)
    force_t_mag = norm(force - force_n_mag * direction)
    # A probe "makes contact" iff the collider produced a positive normal
    # (repulsive) force along the center-to-center axis. Without one the friction
    # ratio is undefined, so mu and separation are reported as NaN. NaN (rather
    # than 0) is deliberate: it is distinguishable from a genuine frictionless
    # (mu == 0) contact, so callers can spot missed probes. jnp.where on both
    # branches keeps the gradient safe under jit/vmap.
    contact = jnp.isfinite(contact_separation) & (force_n_mag > 0.0)
    safe_force_n = jnp.where(contact, force_n_mag, 1.0)
    mu = jnp.where(contact, jnp.abs(force_t_mag / safe_force_n), jnp.nan)
    separation = jnp.where(contact, separation, jnp.nan)

    # Active asperity counts: a vertex sphere with at least one
    # force-bearing contact has nonzero per-sphere force (the same
    # collider that's already populated state.force has masked out
    # intra-clump pairs, so any nonzero entry must be an external
    # asperity contact).
    has_contact = jnp.any(state.force != 0.0, axis=-1)
    n_central_contacts = jnp.sum(has_contact & central_mask)
    n_tracer_contacts = jnp.sum(has_contact & tracer_mask)

    # Whether the force-bearing contact set touches the (optional) core
    # sphere of each clump. If a clump has no core the corresponding mask
    # is all-False, so these are trivially False.
    central_core_contact = jnp.any(has_contact & central_core_mask)
    tracer_core_contact = jnp.any(has_contact & tracer_core_mask)

    # Per-asperity friction: decompose each central vertex sphere's own force
    # along the same center-to-center axis, rather than the clump total.
    #
    # Unlike the clump total, an individual asperity force need not point
    # outward along the COM axis -- a contact on the far side of a lobe pushes
    # inward -- so the sign of the normal component is not a usable contact
    # test here and force-bearing is used instead. For the same reason |f_n|
    # can approach zero for a near-tangential asperity force, which is a
    # genuine divergence of the ratio rather than a failed measurement, hence
    # inf rather than the no-contact NaN.
    f_asperity = state.force[central_idx]
    f_asperity_n = dot(f_asperity, direction)
    f_asperity_t = norm(f_asperity - f_asperity_n[:, None] * direction)
    asperity_denom = jnp.abs(f_asperity_n)
    resolved = asperity_denom > 0.0
    mu_asperity = jnp.where(
        resolved, f_asperity_t / jnp.where(resolved, asperity_denom, 1.0), jnp.inf
    )
    mu_asperity = jnp.where(has_contact[central_idx], mu_asperity, jnp.nan)
    return (
        mu,
        separation,
        n_central_contacts,
        n_tracer_contacts,
        central_core_contact,
        tracer_core_contact,
        mu_asperity,
    )


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
    sampling: str = "lattice",
    seed: int = 0,
    separation_tolerance: float = 1e-10,
    separation_scale: float = 1.1,
    batch_size: int = 10_000,
) -> dict[str, np.ndarray | jax.Array]:
    """Sample surface friction / accessibility over the surface of ``central_state``
    using ``tracer_state`` as a probe.

    Parameters
    ----------
    central_state, tracer_state : State
        Single-clump :class:`State <jaxdem.state.State>` instances. The function ignores their
        initial orientations (it resets both per probe: central to identity,
        tracer to the swept orientation), so the sampling directions live in
        the body frame of each clump as encoded by its ``pos_p``.
    target_overlap : float
        Desired maximum pairwise sphere overlap
        (``r_i + r_j - |x_i - x_j|``) between central and tracer at the
        reported contact configuration. Must be positive. Small values
        correspond to "just barely indented". The converged state satisfies
        this up to floating-point roundoff.
    system : System, optional
        Interaction system used to compute forces (for ``mu``). If ``None``,
        the function builds a default static measurement system (spring force,
        elastic material, naive collider, periodic box large enough for the
        pair).
    n_points : int
        Exact number of surface sample points (approach directions). With
        ``sampling="lattice"`` the function spaces the points equally on
        ``S^1`` (2D) and places them on a Fibonacci golden-spiral lattice on
        ``S^2`` (3D). With ``sampling="random"`` they are iid uniform on the
        circle/sphere.
    n_orientations : int
        Number of tracer orientations. In 2D: the count of rotation angles
        (equispaced, or iid ``U(0, 2 pi)`` when ``sampling="random"``). In 3D:
        the count of facing directions on ``S^2`` (Fibonacci lattice, or
        iid uniform when random). The function pairs each facing with every
        ``roll`` below, so total orientations are ``n_orientations * n_rolls``.
    n_rolls : int
        3D only: number of rolls about the facing axis (equispaced, or iid
        ``U(0, 2 pi)`` when ``sampling="random"``). Must be 1 in 2D (no roll
        degree of freedom). For asymmetric tracers set this > 1 to get full
        ``SO(3)`` coverage.
    sampling : str
        ``"lattice"`` (default) uses the deterministic equispaced / Fibonacci
        grids described above -- current behavior, unchanged. ``"random"``
        draws every grid (approach directions, facings, rolls / 2D angles) iid
        uniform, seeded by ``seed``. Random sampling is *incommensurate* with
        clump symmetry, so it removes the sampling-vs-particle aliasing that
        makes the friction PDF collapse to a few delta spikes for symmetric
        (uniform-asperity) clumps, and it matches theory that assumes a uniform
        distribution of orientation angles. Reseed (vary ``seed``) for
        independent draws / error bars.
    seed : int
        PRNG seed for ``sampling="random"`` (ignored for ``"lattice"``). Fixed
        by default so random runs are reproducible. Pass distinct values for
        independent samples.
    separation_tolerance : float
        Requested separation accuracy, retained for API compatibility. The
        direct root solution has no iteration tolerance. This value must be
        positive and smaller than ``target_overlap``; requests below the
        dtype precision floor emit a warning.
    separation_scale : float
        Safety factor for the initial separation outside the bounding radii.
    batch_size : int
        Number of probes per ``vmap`` call, over the flat
        ``n_points * n_orientations * n_rolls`` probe grid. Larger values
        improve GPU usage. Smaller values cut peak memory. With the default
        ``10_000`` typical sweeps fit in a single kernel launch.

    Returns
    -------
    dict
        A dictionary of stacked ndarrays:

        **Common**
            - ``mu`` -- friction coefficient ``|F_t| / |F_n|`` per probe,
              shape ``(n_points, *orientation_shape)``. ``F`` is the *total*
              force on the central clump, so this is the clump-level
              coefficient (the same quantity as
              :func:`~jaxdem.utils.contacts.get_group_contacts`).
              Reported as ``NaN`` for any probe that fails to establish
              contact (no normal force). This is deliberately distinct from a
              genuine frictionless ``mu == 0``.
            - ``mu_asperity`` -- per-asperity friction coefficient, shape
              ``(n_points, *orientation_shape, n_central_spheres)``. Entry
              ``[..., s]`` decomposes the force on central vertex sphere ``s``
              alone along the same center-to-center axis used for ``mu``. The
              last axis follows the sphere order of ``central_state``.
              ``NaN`` where sphere ``s`` bears no force (it is not one of the
              contacting asperities), so ``np.isnan`` selects the inactive
              asperities and ``np.nanmax(..., axis=-1)`` gives the worst
              active asperity per probe. ``inf`` in the rare case where the
              asperity force is exactly tangential to the COM axis, which is
              a real divergence of the ratio rather than a failed probe.
            - ``separation`` -- center-to-center distance at ``target_overlap``,
              same shape as ``mu``. Also ``NaN`` for no-contact probes, so
              ``np.isnan(separation)`` (or ``mu``) flags the missed samples.
            - ``n_central_contacts`` -- ``int`` per probe, same shape as
              ``mu``. Number of central-clump vertex spheres with at
              least one force-bearing external contact at the resolved
              configuration.
            - ``n_tracer_contacts`` -- same, for the tracer clump.
            - ``central_core_contact`` -- ``bool`` per probe, same shape as
              ``mu``. ``True`` if any force-bearing contact involves the
              central clump's interior core sphere. Always ``False`` when
              the central clump has no core sphere.
            - ``tracer_core_contact`` -- same, for the tracer clump's core.
            - ``tracer_quaternions`` -- shape ``(n_points, *orientation_shape, 4)``.
              The composed quaternion actually applied to the tracer at
              the resolved configuration (approach-direction rotation
              composed with the per-orientation base).
            - ``central_position`` -- shape ``(dim,)``. COM of the central
              clump (constant across probes, equal to
              ``system.domain.box_size / 2``). Combined with
              ``approach_directions`` and ``separation`` this gives the
              tracer COM as ``central_position + separation * approach_dir``.
            - ``approach_directions`` -- surface sample directions, shape
              ``(n_points, dim)``.
            - ``target_overlap`` -- scalar float, echo of the input.
            - ``dim`` -- int, dimensionality (2 or 3).
            - ``sampling`` -- str, echo of the sampling mode used.
            - ``seed`` -- int or ``None``. The PRNG seed for
              ``sampling="random"`` (``None`` for ``"lattice"``). It records
              exactly how the function drew the grids, for reproducibility.

        **2D only**
            - ``angle_surface`` -- ``(n_points,)``, polar angle of each
              approach direction.
            - ``tracer_angles`` -- ``(n_orientations,)``, the swept tracer
              rotation angles. Grid shape: ``(n_orientations,)``.

        **3D only**
            - ``theta_surface``, ``phi_surface`` -- ``(n_points,)`` each,
              spherical coordinates of each approach direction.
            - ``tracer_facings`` -- ``(n_orientations, 3)``, body-frame
              directions sampled on ``S^2``.
            - ``tracer_facing_theta``, ``tracer_facing_phi`` --
              ``(n_orientations,)`` each, spherical coords of the facings.
            - ``tracer_rolls`` -- ``(n_rolls,)``, roll angles about each
              facing axis. Grid shape: ``(n_orientations, n_rolls)``.

    Notes
    -----
    The ``orientation_shape`` is ``(n_orientations,)`` in 2D and
    ``(n_orientations, n_rolls)`` in 3D, so ``mu[i, ...]`` gives the full
    orientation map for approach-direction ``i`` and any slice along the
    leading axis gives the surface map for a fixed orientation.

    ``mu`` and ``mu_asperity`` differ only in the order of summation and
    decomposition: ``mu`` decomposes the summed force, ``mu_asperity``
    decomposes each asperity force separately. Because the normal axis is
    shared by every asperity of a probe, the decomposition is linear, and for
    probes with a single force-bearing asperity (``n_central_contacts == 1``)
    the two are identical. With several asperities the tangential components
    partially cancel in the sum, so ``mu`` is bounded above by the
    normal-force-weighted mean of the active ``mu_asperity`` values whenever
    those asperities all push outward along the COM axis. When some push
    inward the normal components cancel too and ``mu`` can exceed every
    ``mu_asperity``; those probes are worth inspecting separately.

    To reproduce the exact contact configuration of probe ``(i, j[, k])``
    given the original ``central_state`` and ``tracer_state``::

        idx = (i, j, k) if dim == 3 else (i, j)
        tracer_com = result["central_position"] + (
            result["separation"][idx] * result["approach_directions"][i]
        )
        tracer_quat = result["tracer_quaternions"][idx]   # (4,)

    Apply ``tracer_com`` to the tracer's ``state.pos_c`` and
    ``tracer_quat`` to its ``state.q``, leave the central at
    ``result["central_position"]``, and call
    ``system.collider.compute_force(state, system)`` to get the
    resolved force network.

    With ``sampling="random"`` the returned ``approach_directions`` are **not**
    ordered by angle. Consumers that assume ordering -- e.g. the 2D SASA
    perimeter reconstruction, which connects consecutive samples into a polygon
    -- must sort by ``angle_surface`` first (3D SASA via ``ConvexHull`` is
    order-independent and needs no change).
    """
    if central_state.dim != tracer_state.dim:
        raise ValueError(
            f"dim mismatch: central={central_state.dim} vs tracer={tracer_state.dim}."
        )
    if not math.isfinite(target_overlap) or target_overlap <= 0.0:
        raise ValueError(
            f"target_overlap must be finite and positive; got {target_overlap}."
        )
    if not math.isfinite(separation_tolerance) or separation_tolerance <= 0:
        raise ValueError("separation_tolerance must be finite and positive.")
    if separation_tolerance >= target_overlap:
        raise ValueError(
            "separation_tolerance must be strictly smaller than target_overlap "
            "to resolve the contact; got "
            f"separation_tolerance={separation_tolerance} >= "
            f"target_overlap={target_overlap}."
        )
    if n_points < 1:
        raise ValueError(f"n_points must be >= 1; got {n_points}.")
    if n_orientations < 1:
        raise ValueError(f"n_orientations must be >= 1; got {n_orientations}.")
    if n_rolls < 1:
        raise ValueError(f"n_rolls must be >= 1; got {n_rolls}.")
    if sampling not in ("lattice", "random"):
        raise ValueError(f"sampling must be 'lattice' or 'random'; got {sampling!r}.")

    if not math.isfinite(separation_scale) or separation_scale < 1.0:
        raise ValueError("separation_scale must be finite and at least 1.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")

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

    # Merge with central as clump 0 and tracer as clump 1.
    state = State.merge(central_state, tracer_state)

    r_central = _bounding_radius(state, _CENTRAL_ID)
    r_tracer = _bounding_radius(state, _TRACER_ID)
    r_sum = r_central + r_tracer
    max_separation = r_sum * separation_scale

    # Warn when the requested accuracy is below the geometry precision floor.
    dtype_eps = float(np.finfo(state.pos_c.dtype).eps)
    tol_floor = 4.0 * dtype_eps * max_separation
    if separation_tolerance < tol_floor or target_overlap < tol_floor:
        import warnings

        warnings.warn(
            f"separation_tolerance={separation_tolerance:g} / "
            f"target_overlap={target_overlap:g} is below the dtype noise "
            f"floor ({tol_floor:g}) for the geometry in dtype "
            f"{state.pos_c.dtype}. The direct contact roots may be "
            "ulp-noisy. To get tighter "
            "control, enable x64 with "
            "`jax.config.update('jax_enable_x64', True)`.",
            stacklevel=2,
        )

    if system is None:
        system = _create_default_system(state, margin=max_separation * 4.0)

    central_mask = state.clump_id == _CENTRAL_ID
    tracer_mask = state.clump_id == _TRACER_ID

    # Static (compile-time) sphere indices for each clump. clump_id is concrete
    # here, so we resolve the boolean masks to fixed-length index arrays on the
    # host; the solver uses these to gather the central/tracer sphere blocks
    # (JAX cannot boolean-index a traced mask to a static shape).
    central_idx = jnp.asarray(np.where(np.asarray(central_mask))[0])
    tracer_idx = jnp.asarray(np.where(np.asarray(tracer_mask))[0])
    n_central_spheres = int(central_idx.shape[0])

    # Optional interior "core" sphere of each clump (all-False if none).
    central_core_mask = jnp.asarray(_core_mask(state, _CENTRAL_ID))
    tracer_core_mask = jnp.asarray(_core_mask(state, _TRACER_ID))

    # NOTE: the base state's pos_c / q are intentionally left as-is. Every probe
    # rebuilds them via ``dataclasses.replace(state, pos_c=..., q=...)`` inside
    # ``_measure_probe`` (a whole-attribute replace, which refreshes the cached
    # R(q) @ pos_p), so initializing them here would be dead work.

    # Per-grid PRNG keys for sampling="random" (None -> deterministic lattice).
    # Independent keys per grid so approach directions, facings and rolls are
    # sampled independently; reproducible for a fixed ``seed``.
    k_approach: jax.Array | None
    k_orient: jax.Array | None
    k_roll: jax.Array | None
    if sampling == "random":
        k_approach, k_orient, k_roll = jax.random.split(jax.random.PRNGKey(seed), 3)
    else:
        k_approach = k_orient = k_roll = None

    # Approach directions (surface sample points).
    approach_dirs = _sample_directions(n_points, dim, k_approach)
    if dim == 3:
        q_dirs = jax.vmap(_quat_from_x_to_3d)(approach_dirs)
    else:
        q_dirs = jax.vmap(_quat_from_x_to_2d)(approach_dirs)
    tracer_positions = approach_dirs * max_separation

    if dim == 3:
        facings = _sample_directions(n_orientations, 3, k_orient)
        rolls = _sample_angles(n_rolls, k_roll)
        facings_grid = jnp.broadcast_to(
            facings[:, None, :], (n_orientations, n_rolls, 3)
        ).reshape(-1, 3)
        rolls_grid = jnp.broadcast_to(
            rolls[None, :], (n_orientations, n_rolls)
        ).reshape(-1)
        q_bases = jax.vmap(_q_base_3d_vec)(facings_grid, rolls_grid)
    else:
        angles = _sample_angles(n_orientations, k_orient)
        q_bases = jax.vmap(_q_base_2d_vec)(angles)

    measure_batch = jax.jit(
        jax.vmap(
            _measure_probe,
            # state, system, pos(0), quat(0), then all remaining args broadcast.
            in_axes=(None, None, 0, 0) + (None,) * 8,
        )
    )

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

    mu_flat = np.zeros(n_total)
    sep_flat = np.zeros(n_total)
    n_central_flat = np.zeros(n_total, dtype=int)
    n_tracer_flat = np.zeros(n_total, dtype=int)
    central_core_flat = np.zeros(n_total, dtype=bool)
    tracer_core_flat = np.zeros(n_total, dtype=bool)
    mu_asperity_flat = np.zeros((n_total, n_central_spheres))
    n_batches = math.ceil(n_total / batch_size)

    # Every call to ``measure_batch`` is padded up to this fixed leading size so
    # the jit compiles exactly once. Without padding, a trailing partial batch
    # (n_total not a multiple of batch_size) has a different shape and triggers
    # a second full recompile of the (contact solver + collider) kernel. The pad
    # rows repeat a real probe (safe, non-degenerate) and are sliced off below.
    compile_size = min(batch_size, n_total)

    batch_iter: Any = range(n_batches)
    if n_batches > 1:
        try:
            from tqdm import tqdm  # type: ignore[import-untyped]
        except ImportError:
            pass
        else:
            batch_iter = tqdm(batch_iter, total=n_batches, desc="surface probes")

    for b in batch_iter:
        bstart = b * batch_size
        bend = min(bstart + batch_size, n_total)
        actual = bend - bstart
        pos_b = flat_pos[bstart:bend]
        q_b = flat_q[bstart:bend]
        if actual < compile_size:
            pad = compile_size - actual
            pos_b = jnp.concatenate(
                [pos_b, jnp.broadcast_to(pos_b[-1:], (pad, dim))], axis=0
            )
            q_b = jnp.concatenate([q_b, jnp.broadcast_to(q_b[-1:], (pad, 4))], axis=0)
        _mu, _sep, _nc, _nt, _ccore, _tcore, _mu_asp = measure_batch(
            state,
            system,
            pos_b,
            q_b,
            jnp.asarray(target_overlap),
            jnp.asarray(max_separation),
            tracer_mask,
            central_mask,
            central_core_mask,
            tracer_core_mask,
            central_idx,
            tracer_idx,
        )
        mu_flat[bstart:bend] = np.asarray(_mu[:actual])
        sep_flat[bstart:bend] = np.asarray(_sep[:actual])
        n_central_flat[bstart:bend] = np.asarray(_nc[:actual])
        n_tracer_flat[bstart:bend] = np.asarray(_nt[:actual])
        central_core_flat[bstart:bend] = np.asarray(_ccore[:actual])
        tracer_core_flat[bstart:bend] = np.asarray(_tcore[:actual])
        mu_asperity_flat[bstart:bend] = np.asarray(_mu_asp[:actual])

    # --- Unflatten and package the result --------------------------------
    # Final tracer pose per probe. ``central_position`` is a single dim
    # vector shared by every probe (the central clump never moves);
    # combined with ``approach_directions`` and ``separation`` it lets the
    # user reconstruct the exact tracer COM. ``tracer_quaternions`` is
    # the composed (q_dir @ q_base) orientation actually applied to the
    # tracer at the resolved configuration.
    central_position = np.asarray(system.domain.box_size) / 2.0
    mu_grid: Any
    sep_grid: Any
    n_central_grid: Any
    n_tracer_grid: Any
    central_core_grid: Any
    tracer_core_grid: Any
    tracer_quat_grid: Any
    mu_asperity_grid: Any
    if dim == 3:
        mu_grid = mu_flat.reshape(n_points, n_orientations, n_rolls)
        sep_grid = sep_flat.reshape(n_points, n_orientations, n_rolls)
        n_central_grid = n_central_flat.reshape(n_points, n_orientations, n_rolls)
        n_tracer_grid = n_tracer_flat.reshape(n_points, n_orientations, n_rolls)
        central_core_grid = central_core_flat.reshape(n_points, n_orientations, n_rolls)
        tracer_core_grid = tracer_core_flat.reshape(n_points, n_orientations, n_rolls)
        mu_asperity_grid = mu_asperity_flat.reshape(
            n_points, n_orientations, n_rolls, n_central_spheres
        )
        tracer_quat_grid = np.asarray(q_grid).reshape(
            n_points, n_orientations, n_rolls, 4
        )

        theta_surface = np.arccos(np.asarray(approach_dirs[:, 2]))
        phi_surface = np.arctan2(
            np.asarray(approach_dirs[:, 1]), np.asarray(approach_dirs[:, 0])
        )
        facings_np = np.asarray(facings)
        facing_theta = np.arccos(facings_np[:, 2])
        facing_phi = np.arctan2(facings_np[:, 1], facings_np[:, 0])
        return dict(
            mu=mu_grid,
            mu_asperity=mu_asperity_grid,
            separation=sep_grid,
            n_central_contacts=n_central_grid,
            n_tracer_contacts=n_tracer_grid,
            central_core_contact=central_core_grid,
            tracer_core_contact=tracer_core_grid,
            tracer_quaternions=tracer_quat_grid,
            central_position=central_position,
            approach_directions=np.asarray(approach_dirs),
            theta_surface=theta_surface,
            phi_surface=phi_surface,
            tracer_facings=facings_np,
            tracer_facing_theta=facing_theta,
            tracer_facing_phi=facing_phi,
            tracer_rolls=np.asarray(rolls),
            target_overlap=float(target_overlap),
            dim=dim,
            sampling=sampling,
            seed=(seed if sampling == "random" else None),
        )

    mu_grid = mu_flat.reshape(n_points, n_orientations)
    sep_grid = sep_flat.reshape(n_points, n_orientations)
    n_central_grid = n_central_flat.reshape(n_points, n_orientations)
    n_tracer_grid = n_tracer_flat.reshape(n_points, n_orientations)
    central_core_grid = central_core_flat.reshape(n_points, n_orientations)
    tracer_core_grid = tracer_core_flat.reshape(n_points, n_orientations)
    mu_asperity_grid = mu_asperity_flat.reshape(
        n_points, n_orientations, n_central_spheres
    )
    tracer_quat_grid = np.asarray(q_grid).reshape(n_points, n_orientations, 4)
    angle_surface = np.arctan2(
        np.asarray(approach_dirs[:, 1]), np.asarray(approach_dirs[:, 0])
    )
    return dict(
        mu=mu_grid,
        mu_asperity=mu_asperity_grid,
        separation=sep_grid,
        n_central_contacts=n_central_grid,
        n_tracer_contacts=n_tracer_grid,
        central_core_contact=central_core_grid,
        tracer_core_contact=tracer_core_grid,
        tracer_quaternions=tracer_quat_grid,
        central_position=central_position,
        approach_directions=np.asarray(approach_dirs),
        angle_surface=angle_surface,
        tracer_angles=np.asarray(angles),
        target_overlap=float(target_overlap),
        dim=dim,
        sampling=sampling,
        seed=(seed if sampling == "random" else None),
    )
