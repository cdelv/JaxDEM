# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Utility functions for creating states of GA particles (rigid/DP)."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Literal, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ..factory import _normalize_key
from ..state import State
from .clumps import _compute_uniform_union_properties
from .meshes import (
    generate_arclength_mesh,
    generate_faceted_mesh,
    generate_fibonacci_sphere_mesh,
    generate_helix_mesh,
    generate_icosphere_mesh,
    generate_thomson_mesh,
    generate_torus_mesh,
)
from .quaternion import Quaternion
from .random_sphere_configuration import random_sphere_configuration

MESH_TYPES = (
    "thomson",
    "icosphere",
    "fibonacci",
    "torus",
    "helix",
    "arclength",
    "faceted",
)


ASPERITY_DISPERSITY_TYPES = (
    "constant",
    "bidisperse",
    "uniform",
    "truncated_gaussian",
)


def _generate_asperity_mesh(
    mesh_type: str,
    nv: int,
    N: int,
    dim: int,
    *,
    aspect_ratio: float | Sequence[float] | jax.Array | None = None,
    seed: int | None = None,
    mesh_kwargs: dict[str, Any] | None = None,
) -> jax.Array:
    """Dispatch to the asperity-mesh generator selected by ``mesh_type``.

    Returns positions shaped ``(nv, dim)`` if ``N == 1`` else ``(N, nv, dim)``,
    unit-scaled so the longest axis has extent 1. ``mesh_kwargs`` forwards
    mesh-specific keyword arguments to the underlying generator — e.g.
    ``steps`` / ``alpha`` / ``lr`` for Thomson, ``tube_ratio`` for torus,
    ``n_turns`` / ``helix_radius`` for helix, ``n_facets`` for 2D faceted.
    """
    kw = dict(mesh_kwargs or {})
    if mesh_type == "thomson":
        # Default Thomson to 10_000 steps (the create_ga_state convention;
        # more than the bare generate_thomson_mesh default of 1_000).
        kw.setdefault("steps", 10_000)
        pos, _ = generate_thomson_mesh(
            nv=nv,
            N=N,
            dim=dim,
            aspect_ratio=aspect_ratio,
            seed=seed,
            **kw,
        )
        return pos
    if mesh_type == "icosphere":
        return generate_icosphere_mesh(
            nv=nv, N=N, dim=dim, aspect_ratio=aspect_ratio, **kw
        )
    if mesh_type == "fibonacci":
        return generate_fibonacci_sphere_mesh(
            nv=nv, N=N, dim=dim, aspect_ratio=aspect_ratio, **kw
        )
    if mesh_type == "torus":
        return generate_torus_mesh(nv=nv, N=N, dim=dim, aspect_ratio=aspect_ratio, **kw)
    if mesh_type == "helix":
        return generate_helix_mesh(nv=nv, N=N, dim=dim, aspect_ratio=aspect_ratio, **kw)
    if mesh_type == "arclength":
        # The typical 2D GA model with even spacing between asperities.
        return generate_arclength_mesh(
            nv=nv, N=N, dim=dim, aspect_ratio=aspect_ratio, **kw
        )
    if mesh_type == "faceted":
        return generate_faceted_mesh(
            nv=nv, N=N, dim=dim, aspect_ratio=aspect_ratio, **kw
        )
    raise ValueError(f"mesh_type must be one of {MESH_TYPES}; got {mesh_type!r}.")


def _generate_disperse_asperity_radii(
    shape: tuple[int, ...],
    asperity_radius: float,
    dispersity_type: str,
    dispersity_kwargs: dict[str, Any] | None,
    key: jax.Array,
) -> jax.Array:
    r"""Sample asperity radii from the requested dispersity distribution.

    Every distribution is parameterized so its **theoretical mean equals
    ``asperity_radius``** (independent of the realized samples). This
    keeps the clump's core size deterministic across runs and dispersity
    settings — only the asperity-to-asperity variation changes.

    Distributions
    -------------
    ``"constant"`` (default)
        ``r_i = asperity_radius`` for all asperities. ``dispersity_kwargs``
        is ignored.

    ``"bidisperse"`` — kwargs ``{"size_ratio": 1.4, "fraction_small": 0.5}``
        Two discrete sizes ``r_s`` and ``r_l = r_s * size_ratio`` chosen so
        that ``f_s · r_s + (1 - f_s) · r_l = asperity_radius``, i.e.
        ``r_s = asperity_radius / (f_s + (1 - f_s) · size_ratio)``.
        ``round(f_s · n)`` of the asperities are assigned the small size,
        the rest the large; positions are then randomly permuted so the
        small/large pattern is isotropic on the Thomson mesh.

    ``"uniform"`` — kwargs ``{"size_ratio": 2.0}``
        Each :math:`r_i \sim U(r_{\min}, r_{\max})` with
        ``r_max / r_min = size_ratio`` and
        ``(r_min + r_max) / 2 = asperity_radius``, giving
        ``r_min = 2 · asperity_radius / (1 + size_ratio)`` and
        ``r_max = r_min · size_ratio``.

    ``"truncated_gaussian"`` — kwargs ``{"cv": 0.2, "n_sigma": 2.5}``
        :math:`r_i \sim \mathcal{N}(\mu, \sigma^2)` truncated to
        :math:`[\mu - n_\sigma \sigma,\; \mu + n_\sigma \sigma]`, with
        ``mu = asperity_radius`` and ``sigma = cv · asperity_radius``.
        Symmetric truncation preserves the mean. We require
        ``cv · n_sigma < 1`` so the lower bound stays positive.

    Parameters
    ----------
    shape
        Desired output shape, e.g. ``(n_bodies, nv)`` or ``(nv,)``.
    asperity_radius
        Theoretical mean of the sampled radii.
    dispersity_type
        One of ``ASPERITY_DISPERSITY_TYPES``.
    dispersity_kwargs
        Distribution-specific parameters; see above. ``None`` uses the
        defaults.
    key
        :class:`jax.random.PRNGKey` for the sampler.
    """
    kw = dict(dispersity_kwargs or {})

    if dispersity_type == "constant":
        return jnp.full(shape, asperity_radius, dtype=float)

    if dispersity_type == "bidisperse":
        size_ratio = float(kw.get("size_ratio", 1.4))
        fraction_small = float(kw.get("fraction_small", 0.5))
        if size_ratio <= 0:
            raise ValueError(f"bidisperse size_ratio must be > 0; got {size_ratio}.")
        if not (0.0 <= fraction_small <= 1.0):
            raise ValueError(
                f"bidisperse fraction_small must be in [0, 1]; got {fraction_small}."
            )
        r_small = asperity_radius / (
            fraction_small + (1.0 - fraction_small) * size_ratio
        )
        r_large = r_small * size_ratio
        nv = int(shape[-1])
        n_small = int(round(fraction_small * nv))
        base = jnp.concatenate(
            [jnp.full((n_small,), r_small), jnp.full((nv - n_small,), r_large)]
        )
        # One independent permutation per leading-axis index.
        flat_leading = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
        keys = jax.random.split(key, flat_leading)
        permuted = jax.vmap(lambda k: jax.random.permutation(k, base))(keys)
        return permuted.reshape(shape)

    if dispersity_type == "uniform":
        size_ratio = float(kw.get("size_ratio", 2.0))
        if size_ratio < 1.0:
            raise ValueError(f"uniform size_ratio must be >= 1; got {size_ratio}.")
        r_min = 2.0 * asperity_radius / (1.0 + size_ratio)
        r_max = r_min * size_ratio
        return jax.random.uniform(key, shape, minval=r_min, maxval=r_max, dtype=float)

    if dispersity_type == "truncated_gaussian":
        cv = float(kw.get("cv", 0.2))
        n_sigma = float(kw.get("n_sigma", 2.5))
        if cv <= 0:
            raise ValueError(f"truncated_gaussian cv must be > 0; got {cv}.")
        if n_sigma <= 0:
            raise ValueError(f"truncated_gaussian n_sigma must be > 0; got {n_sigma}.")
        if cv * n_sigma >= 1.0:
            raise ValueError(
                f"cv * n_sigma must be < 1 to keep radii positive; "
                f"got cv={cv}, n_sigma={n_sigma}."
            )
        std = cv * asperity_radius
        z = jax.random.truncated_normal(key, -n_sigma, n_sigma, shape)
        return asperity_radius + std * z

    raise ValueError(
        f"asperity_dispersity_type must be one of {ASPERITY_DISPERSITY_TYPES}; "
        f"got {dispersity_type!r}."
    )


def create_sphere_state(
    radii: float | Sequence[float] | np.ndarray | jax.Array,
    dim: int,
    *,
    pos: Sequence[Sequence[float]] | np.ndarray | jax.Array | None = None,
    mass: float | Sequence[float] | np.ndarray | jax.Array = 1.0,
) -> State:
    """Build a :class:`State` of ``N`` simple spheres.

    No Thomson mesh, no Monte-Carlo union-property integration — this is
    the lightweight sphere primitive. Each sphere is a single-particle
    body with its own ``clump_id``; inertia defaults to the solid-disk
    (2D) / solid-sphere (3D) formula from the given mass and radius
    (handled inside :meth:`State.create`).

    Parameters
    ----------
    radii
        Scalar or length-``N`` per-sphere radii.
    dim
        Spatial dimension (2 or 3).
    pos
        Optional ``(N, dim)`` centers. If ``None``, all spheres are
        stacked at the origin (handy for a single tracer).
    mass
        Scalar or length-``N`` per-sphere masses. Default ``1.0``.
    """
    radii_arr = jnp.atleast_1d(jnp.asarray(radii, dtype=float))
    n_spheres = int(radii_arr.shape[0])
    if pos is None:
        pos_arr = jnp.zeros((n_spheres, int(dim)), dtype=float)
    else:
        pos_arr = jnp.asarray(pos, dtype=float)
        if pos_arr.shape != (n_spheres, dim):
            raise ValueError(
                f"pos must have shape ({n_spheres}, {dim}); got {pos_arr.shape}"
            )
    mass_arr = _broadcast_per_body(mass, n_spheres, "mass")
    return State.create(
        pos=pos_arr,
        rad=radii_arr,
        mass=jnp.asarray(mass_arr, dtype=float),
    )


def create_ga_state(
    N: int,
    nv: int,
    dim: int,
    particle_radius: float,
    asperity_radius: float,
    *,
    particle_type: str = "clump",
    core_type: str = "hollow",
    aspect_ratio: float | Sequence[float] | None = None,
    particle_mass: float = 1.0,
    n_samples: int = 10_000_000,
    seed: int | None = None,
    mesh_type: str = "thomson",
    mesh_kwargs: dict[str, Any] | None = None,
    asperity_dispersity_type: str = "constant",
    asperity_dispersity_kwargs: dict[str, Any] | None = None,
) -> State:
    """Build a :class:`State` of ``N`` geometric-asperity bodies.

    Surface asperities are placed on a unit shape (set by ``mesh_type``) and
    an optional central core sphere is added. The union of asperities (and,
    if present, the core) defines the body volume that the rigid-body /
    deformable-body properties are derived from.

    Parameters
    ----------
    particle_type : {"clump", "dp"}
        "clump" produces rigid bodies with a computed COM / principal-axis
        orientation / inertia. "dp" produces deformable particles whose
        nodes share a common ``bond_id`` and each carry an equal share of
        the body mass and union volume. "sphere"-like bodies are the
        ``nv = 1`` degenerate case of "clump".
    core_type : {"hollow", "solid", "phantom"}
        "hollow" uses only the surface asperities. "solid" adds a central
        core sphere that is kept in the final state. "phantom" adds the
        core only for the property calculation and strips it from the
        returned state.
    mesh_type : {"thomson", "icosphere", "fibonacci", "torus", "helix", "arclength", "faceted"}
        How asperity positions are generated on the unit shape.

        * ``"thomson"`` (default): generalized Thomson problem on a
          hyper-ellipsoid (Riesz-energy minimization). Stochastic,
          near-uniform result seeded by ``seed``.
        * ``"icosphere"``: recursive icosahedron subdivision (3D) or regular
          polygon (2D); deterministic; ``nv`` must be a valid icosphere count.
        * ``"fibonacci"``: golden-angle spiral sphere / evenly-spaced circle;
          deterministic; any ``nv``.
        * ``"torus"``: 3D genus-1 torus surface.
        * ``"helix"``: 3D chiral helix / 2D Archimedean spiral.
        * ``"arclength"``: 2D only — equal arc-length spacing on the
          ellipse/circle perimeter. Closed-form analogue of the converged
          Thomson ground state in 2D; deterministic, any ``nv``.
        * ``"faceted"``: regular polygon (2D) or icosahedron (3D) with
          vertex asperities + face/edge-interior fillers. Produces genuinely
          angular (flat-faced, sharp-cornered) particles rather than smooth
          spheres.
    mesh_kwargs : dict, optional
        Mesh-specific keyword arguments forwarded to the underlying generator.
        See each generator in :mod:`jaxdem.utils.meshes` for accepted keys.
        Examples: ``{"steps": 5_000, "alpha": 1.0}`` (thomson — defaults to
        ``steps=10_000``); ``{"tube_ratio": 0.25}`` (torus); ``{"n_turns": 3,
        "helix_radius": 0.3}`` (helix); ``{"n_facets": 8}`` (faceted, 2D only).
    asperity_dispersity_type : {"constant", "bidisperse", "uniform", "truncated_gaussian"}
        How asperity radii are drawn. Every distribution is parameterized
        so its **theoretical mean is** ``asperity_radius``, which keeps
        ``core_radius = particle_radius - asperity_radius`` deterministic
        across runs and dispersity settings — only the asperity-to-asperity
        variation changes. With dispersity, the largest asperities
        naturally protrude beyond ``particle_radius`` and the smallest
        recede; this is by design. See
        :func:`_generate_disperse_asperity_radii` for the per-distribution
        sampling formulas.
    asperity_dispersity_kwargs : dict, optional
        Distribution-specific parameters. Defaults: ``{}`` (no effect for
        ``"constant"``); ``{"size_ratio": 1.4, "fraction_small": 0.5}``
        for ``"bidisperse"``; ``{"size_ratio": 2.0}`` for ``"uniform"``;
        ``{"cv": 0.2, "n_sigma": 2.5}`` for ``"truncated_gaussian"``
        (where ``cv = std / mean`` and ``cv * n_sigma`` must be ``< 1``
        to keep radii positive).
    """
    if particle_type not in ("clump", "dp"):
        raise ValueError(
            f"particle_type must be one of 'clump' or 'dp'; got {particle_type!r}"
        )
    if core_type not in ("solid", "phantom", "hollow"):
        raise ValueError(
            f"core_type must be one of 'solid', 'phantom', 'hollow'; got {core_type!r}"
        )
    if asperity_radius >= particle_radius:
        raise ValueError(
            f"asperity_radius ({asperity_radius}) must be smaller than "
            f"particle_radius ({particle_radius}) so the core radius "
            "stays positive."
        )
    if seed is None:
        seed = np.random.randint(0, 1000000000)

    pos = _generate_asperity_mesh(
        mesh_type=mesh_type,
        nv=nv,
        N=N,
        dim=dim,
        aspect_ratio=aspect_ratio,
        seed=seed,
        mesh_kwargs=mesh_kwargs,
    )
    if pos.ndim == 2:
        pos = pos[None, :, :]
    n_bodies, nv_eff = pos.shape[0], pos.shape[1]

    core_radius = particle_radius - asperity_radius
    pos *= core_radius
    # Sample asperity radii from the requested dispersity. The
    # distribution's theoretical mean is ``asperity_radius`` regardless
    # of dispersity type, so ``core_radius`` and the average outer
    # diameter stay deterministic across runs.
    rad_key = jax.random.fold_in(jax.random.PRNGKey(int(seed)), 0xA5E)
    rad = _generate_disperse_asperity_radii(
        (n_bodies, nv_eff),
        asperity_radius,
        asperity_dispersity_type,
        asperity_dispersity_kwargs,
        rad_key,
    )

    if core_type in ("solid", "phantom"):
        pos = jnp.concatenate([pos, jnp.zeros((n_bodies, 1, dim))], axis=1)
        rad = jnp.concatenate([rad, jnp.full((n_bodies, 1), core_radius)], axis=1)
        nv_eff = pos.shape[1]

    volume, com, inertia, q, pos_p = _compute_uniform_union_properties(
        pos,
        rad,
        jnp.asarray(particle_mass),
        n_samples=n_samples,
    )

    if core_type == "phantom":
        pos = pos[:, :-1, :]
        rad = rad[:, :-1]
        pos_p = pos_p[:, :-1, :]
        nv_eff = pos.shape[1]

    total = n_bodies * nv_eff
    ang_dim = inertia.shape[-1]  # 1 in 2D, 3 in 3D
    rad_flat = rad.reshape(total)
    body_id = jnp.broadcast_to(
        jnp.arange(n_bodies, dtype=int)[:, None], (n_bodies, nv_eff)
    ).reshape(total)

    if particle_type == "clump":
        sphere_pos_p = pos_p.reshape(total, dim)
        pos_c = jnp.broadcast_to(com[:, None, :], (n_bodies, nv_eff, dim)).reshape(
            total, dim
        )
        q_w = jnp.broadcast_to(q[:, None, 0:1], (n_bodies, nv_eff, 1)).reshape(total, 1)
        q_xyz = jnp.broadcast_to(q[:, None, 1:4], (n_bodies, nv_eff, 3)).reshape(
            total, 3
        )
        q_state = Quaternion.create(w=q_w, xyz=q_xyz)

        volume_flat = jnp.broadcast_to(volume[:, None], (n_bodies, nv_eff)).reshape(
            total
        )
        inertia_flat = jnp.broadcast_to(
            inertia[:, None, :], (n_bodies, nv_eff, ang_dim)
        ).reshape(total, ang_dim)
        mass_flat = jnp.full((total,), particle_mass, dtype=float)

        return State.create(
            pos=pos_c,
            pos_p=sphere_pos_p,
            rad=rad_flat,
            q=q_state,
            volume=volume_flat,
            mass=mass_flat,
            inertia=inertia_flat,
            clump_id=body_id,
        )

    # particle_type == "dp": each node carries an equal share of the body's
    # total mass and union volume; nodes connected through the bond graph
    # make up one deformable body. Unlike the clump path we don't decompose
    # pos into (COM, body-frame offset): pos is stored directly as pos_c, and
    # pos_p / q / inertia fall back to State.create defaults
    # (pos_p=0, q=identity, per-node sphere inertia), so each node's
    # world-frame position is just pos.
    pos_flat = pos.reshape(total, dim)
    volume_flat = jnp.broadcast_to(
        (volume / nv_eff)[:, None], (n_bodies, nv_eff)
    ).reshape(total)
    mass_flat = jnp.full((total,), particle_mass / nv_eff, dtype=float)

    return State.create(
        pos=pos_flat,
        rad=rad_flat,
        volume=volume_flat,
        mass=mass_flat,
        bond_id=_body_bond_adjacency(n_bodies, nv_eff),
    )


def _body_bond_adjacency(n_bodies: int, nv_body: int) -> np.ndarray | None:
    """Intra-body bond adjacency for ``n_bodies`` contiguous bodies of ``nv_body`` nodes.

    ``State.create`` expects ``bond_id`` as an *adjacency list* (per-node
    unique IDs of connected nodes), not body labels. This returns the full
    intra-body clique, shape ``(n_bodies * nv_body, nv_body - 1)``, so that

    * the bond graph's connected components are exactly the bodies (used by
      :func:`_resolve_body_grouping`), and
    * intra-body sphere contacts are disabled by the colliders' bond
      filtering (the DP bonded-force model handles internal mechanics).

    Returns ``None`` when ``nv_body == 1`` (no bonds).
    """
    if nv_body <= 1:
        return None
    node_ids = np.arange(n_bodies * nv_body, dtype=int).reshape(n_bodies, nv_body)
    off_diag = ~np.eye(nv_body, dtype=bool)  # (nv_body, nv_body)
    others = np.broadcast_to(node_ids[:, None, :], (n_bodies, nv_body, nv_body))[
        :, off_diag
    ].reshape(n_bodies * nv_body, nv_body - 1)
    return others


def _bond_graph_components(bond_ids_np: np.ndarray) -> tuple[int, np.ndarray]:
    """Connected components of the bond graph (vectorized).

    Parameters
    ----------
    bond_ids_np
        ``(N, W)`` (or ``(N,)``) integer adjacency lists padded with ``-1``.

    Returns
    -------
    (n_components, labels)
        Number of components and an ``(N,)`` int32 label array.
    """
    bond_ids_np = np.asarray(bond_ids_np)
    if bond_ids_np.ndim == 1:
        bond_ids_np = bond_ids_np[:, None]
    n = int(bond_ids_np.shape[0])
    if n == 0:
        return 0, np.zeros((0,), dtype=np.int32)

    import scipy.sparse as sp  # type: ignore[import-untyped]

    width = int(bond_ids_np.shape[-1])
    rows = np.repeat(np.arange(n), width)
    cols = bond_ids_np.reshape(-1)
    valid = (cols >= 0) & (cols < n)
    rows, cols = rows[valid], cols[valid].astype(int)
    if rows.size == 0:
        return n, np.arange(n, dtype=np.int32)
    adj = sp.coo_matrix((np.ones(rows.size), (rows, cols)), shape=(n, n))
    n_components, labels = sp.csgraph.connected_components(adj, directed=False)
    return int(n_components), labels.astype(np.int32)


def _resolve_body_grouping(state: State, group_by: str) -> tuple[jax.Array, int]:
    """Return ``(group_id, n_bodies)`` identifying each particle's body.

    * Rigid clumps group nodes with shared ``clump_id`` (``bond_id`` is
      unique per node).
    * Deformable particles group nodes with shared connected components of the bond graph.
    * Spheres have both ``clump_id`` and bond graph connected components unique per particle.

    ``group_by="auto"`` picks whichever of ``clump_id`` / bond graph connected components has
    sub-``N`` unique values; if both do (mixed clumps + DPs) the caller
    must disambiguate.
    """
    clump_ids_np = np.asarray(state.clump_id)
    bond_ids_np = np.asarray(state.bond_id)
    n = int(state.N)
    n_unique_clump = int(np.unique(clump_ids_np).size)

    # Connected components of the bond graph (vectorized scipy).
    n_unique_bond, bond_group_id_np = _bond_graph_components(bond_ids_np)
    bond_group_id = jnp.asarray(bond_group_id_np, dtype=int)

    if group_by == "auto":
        has_clumps = n_unique_clump < n
        has_dps = n_unique_bond < n
        if has_clumps and has_dps:
            raise ValueError(
                "Could not auto-detect body grouping: state has both clumps "
                "(sub-N unique clump_id) and DPs (sub-N unique bond connected components). "
                "Pass group_by='clump' or group_by='bond' to disambiguate."
            )
        if has_clumps:
            return state.clump_id, n_unique_clump
        if has_dps:
            return bond_group_id, n_unique_bond
        return state.clump_id, n_unique_clump  # all spheres: either ID works
    if group_by == "clump":
        return state.clump_id, n_unique_clump
    if group_by == "bond":
        return bond_group_id, n_unique_bond
    raise ValueError(
        f"group_by must be one of 'auto', 'clump', 'bond'; got {group_by!r}"
    )


def _random_body_quaternions(n_bodies: int, dim: int, key: jax.Array) -> jax.Array:
    """Uniformly random per-body quaternions ``[w, x, y, z]`` in ``(n_bodies, 4)``."""
    if dim == 3:
        q4 = jax.random.normal(key, (n_bodies, 4))
        return q4 / jnp.linalg.norm(q4, axis=-1, keepdims=True)
    theta = jax.random.uniform(key, (n_bodies,), minval=0.0, maxval=2.0 * jnp.pi)
    half = theta / 2.0
    return jnp.stack(
        [jnp.cos(half), jnp.zeros_like(half), jnp.zeros_like(half), jnp.sin(half)],
        axis=-1,
    )


def _randomize_body_orientations(
    state: State,
    group_id: jax.Array,
    n_bodies: int,
    key: jax.Array,
) -> State:
    """Apply a uniformly-random per-body rotation.

    Combines two ops that compose correctly for every body type:

    1. Rotate each node's ``pos_c`` around the body centroid. For clumps
       every node shares ``pos_c`` so centroid equals ``pos_c`` and the
       rotation is a no-op. For DPs this physically rotates the node
       cloud around its centroid.
    2. Compose the random rotation onto ``state.q``. For clumps this
       rotates the stored body frame so ``state.pos = pos_c + R(q) @ pos_p``
       recomputes the new world positions. For DPs / spheres ``pos_p = 0``
       so this has no visible effect but keeps ``q`` consistent.
    """
    N = state.N

    q4 = _random_body_quaternions(n_bodies, state.dim, key)
    q_per_body = Quaternion(w=q4[:, 0:1], xyz=q4[:, 1:4])
    q_per_node = Quaternion(w=q_per_body.w[group_id], xyz=q_per_body.xyz[group_id])

    counts = jax.ops.segment_sum(
        jnp.ones((N,), dtype=state.pos_c.dtype),
        group_id,
        num_segments=n_bodies,
    )
    centroid = (
        jax.ops.segment_sum(state.pos_c, group_id, num_segments=n_bodies)
        / counts[:, None]
    )

    offset = state.pos_c - centroid[group_id]
    rotated_offset = Quaternion.rotate(q_per_node, offset)
    new_pos_c = centroid[group_id] + rotated_offset
    new_q = q_per_node @ state.q

    return replace(state, pos_c=new_pos_c, q=new_q)


def distribute_bodies(
    state: State,
    phi: float,
    *,
    domain_type: str = "periodic",
    box_aspect: Sequence[float] | None = None,
    seed: int | None = None,
    max_avg_pe: float | None = 1e-16,
    randomize_orientation: bool = True,
    group_by: str = "auto",
    collider_type: str = "naive",
) -> tuple[State, jax.Array]:
    """Randomly distribute each body's centroid in a box sized for ``phi``.

    For each body in ``state`` (sphere, rigid clump, or deformable particle),
    build a bounding sphere from its centroid and the max ``|node - centroid|
    + rad``. Place those bounding spheres uniformly at random in a periodic
    (or reflective) box sized to the target packing fraction, FIRE-minimize
    so nothing overlaps, and translate every body so its centroid lands at
    the minimized bounding-sphere center. Optionally apply a uniformly
    random per-body rotation.

    Note that ``phi`` here is the **bounding-sphere** packing fraction, not
    the true body packing fraction (which is lower for non-convex clumps /
    DPs). The target box volume is ``sum(bounding_sphere_volume) / phi``.

    Parameters
    ----------
    state
        Input state containing one or more bodies. Body grouping is inferred
        from ``clump_id`` / ``bond_id``; use ``group_by`` to force one.
    phi
        Target bounding-sphere packing fraction. Must be in (0, 1).
    domain_type
        ``"periodic"`` or ``"reflect"`` for the analogue sphere
        minimization domain.
    box_aspect
        Aspect ratios for the box, shape ``(dim,)``. Defaults to isotropic.
    seed
        RNG seed for both the initial random centroid placement and the
        per-body rotation. Drawn randomly if ``None``.
    max_avg_pe
        Convergence tolerance for the FIRE minimizer.
    randomize_orientation
        If ``True``, apply a uniformly-random per-body rotation after
        placement.
    group_by
        ``"auto"`` (default), ``"clump"``, or ``"bond"``. See
        :func:`_resolve_body_grouping`.
    collider_type
        Collider for the analogue sphere minimization, ``"naive"`` or
        ``"celllist"``.

    Returns
    -------
    state, box_size
        The input state with per-body translation and (optionally) per-body
        random orientation applied, and the periodic box size vector.
    """
    if seed is None:
        seed = int(np.random.randint(0, int(1e9)))

    group_id, n_bodies = _resolve_body_grouping(state, group_by)

    N = state.N
    counts = jax.ops.segment_sum(
        jnp.ones((N,), dtype=state.pos_c.dtype),
        group_id,
        num_segments=n_bodies,
    )
    # body_center uses pos_c (== COM for clumps; == centroid for DPs; ==
    # the sphere position for spheres). bounding radius uses world-frame
    # pos so it captures the rotated sphere offsets inside a clump.
    body_center = (
        jax.ops.segment_sum(state.pos_c, group_id, num_segments=n_bodies)
        / counts[:, None]
    )
    dists = jnp.linalg.norm(state.pos - body_center[group_id], axis=-1) + state.rad
    body_radius = jax.ops.segment_max(dists, group_id, num_segments=n_bodies)

    new_centers, box_size = random_sphere_configuration(
        particle_radii=np.asarray(body_radius).tolist(),
        phi=float(phi),
        dim=int(state.dim),
        seed=int(seed),
        collider_type=collider_type,
        box_aspect=box_aspect,
        max_avg_pe=max_avg_pe,
        domain_type=domain_type,
    )
    new_centers = jnp.asarray(new_centers)
    if new_centers.ndim == 1:
        # single-body case: random_sphere_configuration squeezes to (dim,).
        new_centers = new_centers[None, :]

    offset = (new_centers - body_center)[group_id]
    state = replace(state, pos_c=state.pos_c + offset)

    if randomize_orientation:
        orient_key = jax.random.PRNGKey(int(seed) + 1)
        state = _randomize_body_orientations(state, group_id, n_bodies, orient_key)

    return state, jnp.asarray(box_size)


def ga_surface_mask(
    state: State,
    *,
    group_by: str = "bond",
) -> jax.Array:
    """Per-body surface mask: ``True`` iff the node lies on the convex hull of its body.

    For each body, runs ``scipy.spatial.ConvexHull`` on the body's node
    positions and flags the hull vertices as surface. Bodies with too few
    nodes to form a hull (``<= dim + 1``) or that trigger a Qhull numerical
    failure are treated conservatively — all their nodes marked surface.

    Exact for convex (or near-convex) bodies: a node is interior iff it
    lies strictly inside the convex hull of its body. For Thomson-mesh
    asperity bodies the surface vs. core separation is unambiguous.
    Non-convex bodies with concave pockets can misclassify pocket nodes
    as interior (they sit off the global hull).

    Parameters
    ----------
    state
        State with one or more bodies.
    group_by
        ``"bond"`` (default), ``"clump"``, or ``"auto"`` — see
        :func:`_resolve_body_grouping`.

    Returns
    -------
    jax.Array
        ``(N,)`` bool mask, ``True`` for surface nodes.
    """
    from scipy.spatial import ConvexHull  # type: ignore[import-untyped]

    group_id, n_bodies = _resolve_body_grouping(state, group_by)
    gid_np = np.asarray(group_id)
    pos_np = np.asarray(state.pos)
    dim = int(state.dim)
    N = int(state.N)

    mask = np.zeros(N, dtype=bool)
    for b in range(n_bodies):
        idxs = np.where(gid_np == b)[0]
        pts = pos_np[idxs]
        if pts.shape[0] <= dim + 1:
            mask[idxs] = True
            continue
        try:
            hull = ConvexHull(pts)
            mask[idxs[hull.vertices]] = True
        except Exception:
            mask[idxs] = True
    return jnp.asarray(mask)


def _ensure_per_body_params(
    x: float | jax.Array | None, n_bodies: int, name: str
) -> jax.Array | None:
    """Coerce a scalar or ``(n_bodies,)`` array to a per-body float array (or None)."""
    if x is None:
        return None
    arr = jnp.asarray(x, dtype=float)
    if arr.ndim == 0:
        return jnp.ones((n_bodies,), dtype=float) * arr
    if arr.shape == (n_bodies,):
        return arr
    raise ValueError(f"{name} must be a scalar or shape ({n_bodies},), got {arr.shape}")


def _order_boundary_2d(pts: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    """Order boundary indices CCW by polar angle around centroid."""
    bpts = pts[idx]
    c = jnp.mean(bpts, axis=0)
    angles = jnp.arctan2(bpts[:, 1] - c[1], bpts[:, 0] - c[0])
    order = jnp.argsort(angles)
    ordered = idx[order]

    # enforce CCW orientation (positive signed area)
    poly = pts[ordered]
    x, y = poly[:, 0], poly[:, 1]
    area2 = jnp.sum(x * jnp.roll(y, -1) - y * jnp.roll(x, -1))
    return jnp.where(area2 < 0, jnp.flip(ordered, axis=0), ordered)


def _initial_bending_2d(
    vertices: jnp.ndarray, elements: jnp.ndarray, element_adjacency: jnp.ndarray
) -> jnp.ndarray:
    """Compute rest bending angles for 2D segments using segment normals."""
    from ..bonded_forces.deformable_particle import angle_between_normals
    from .linalg import norm

    p0 = vertices[elements[:, 0]]
    p1 = vertices[elements[:, 1]]
    edge = p1 - p0
    length = norm(edge)
    normal = jnp.stack([edge[:, 1], -edge[:, 0]], axis=1)
    unit_normal = normal / jnp.where(length[:, None] == 0, 1.0, length[:, None])
    n1 = unit_normal[element_adjacency[:, 0]]
    n2 = unit_normal[element_adjacency[:, 1]]
    return angle_between_normals(n1, n2)


def _initial_bending_3d(
    vertices: jnp.ndarray, faces: jnp.ndarray, face_adjacency: jnp.ndarray
) -> jnp.ndarray:
    """Compute rest bending angles for 3D triangles using face normals."""
    from ..bonded_forces.deformable_particle import angle_between_normals
    from .linalg import cross, norm

    tri = vertices[faces]  # (F,3,3)
    r2 = tri[:, 1] - tri[:, 0]
    r3 = tri[:, 2] - tri[:, 0]
    face_normal = cross(r2, r3)
    nrm = norm(face_normal)
    unit = face_normal / jnp.where(nrm[:, None] == 0, 1.0, nrm[:, None])
    n1 = unit[face_adjacency[:, 0]]
    n2 = unit[face_adjacency[:, 1]]
    return angle_between_normals(n1, n2)


def _body_surface_topology_2d(
    surface_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polar-angle polygon ordering for a single 2D body.

    Returns ``(elements_local, adjacency_local, initial_bending)`` in
    *local* (0..n_surface-1) indices. Adjacency is consecutive segments
    around the ring.
    """
    n_s = surface_pos.shape[0]
    pts_j = jnp.asarray(surface_pos)
    order_local = np.asarray(_order_boundary_2d(pts_j, jnp.arange(n_s, dtype=int)))
    M = order_local.size
    elements_local = np.stack([order_local, np.roll(order_local, -1)], axis=1)
    adjacency_local = np.stack(
        [np.arange(M, dtype=int), (np.arange(M, dtype=int) + 1) % M], axis=1
    )
    init_bend = np.asarray(
        _initial_bending_2d(
            pts_j,
            jnp.asarray(elements_local),
            jnp.asarray(adjacency_local),
        )
    )
    return elements_local, adjacency_local, init_bend


def _body_surface_topology_3d(
    surface_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convex-hull triangulation for a single 3D body.

    Returns ``(elements_local, adjacency_local, initial_bending)``. Faces
    are the hull simplices reoriented so that each triangle's vertex
    ordering produces an outward-pointing normal; adjacency is derived
    from ``hull.neighbors``.

    The re-orientation matters because
    :func:`~jaxdem.bonded_forces.deformable_particle.compute_element_properties_3D`
    computes signed per-face volume contributions (divergence theorem).
    With inconsistent triangle orientation the signed contributions can
    cancel or flip the body's total volume negative, which in turn
    makes ``E_content = ec * (C - C0)^2 / C0`` negative when compressed.
    """
    from scipy.spatial import ConvexHull

    try:
        hull = ConvexHull(surface_pos)
    except Exception:
        return (
            np.empty((0, 3), dtype=int),
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=float),
        )

    simplices = np.array(hull.simplices, dtype=int, copy=True)
    outward_normals = np.asarray(hull.equations)[:, :3]  # (F, 3)
    v0 = surface_pos[simplices[:, 0]]
    v1 = surface_pos[simplices[:, 1]]
    v2 = surface_pos[simplices[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    needs_flip = np.einsum("fi,fi->f", face_normals, outward_normals) < 0.0
    # Swap the last two vertex indices on inwardly-oriented faces so that
    # (v1 - v0) x (v2 - v0) aligns with the hull's outward normal.
    if np.any(needs_flip):
        flipped = simplices.copy()
        flipped[needs_flip, 1] = simplices[needs_flip, 2]
        flipped[needs_flip, 2] = simplices[needs_flip, 1]
        simplices = flipped

    elements_local = simplices
    F = elements_local.shape[0]
    adj_set: set[tuple[int, int]] = set()
    neighbors = np.asarray(hull.neighbors)
    for i in range(F):
        for nb in neighbors[i]:
            nb = int(nb)
            if nb >= 0 and nb != i:
                lo, hi = (i, nb) if i < nb else (nb, i)
                adj_set.add((lo, hi))
    if not adj_set:
        return elements_local, np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)
    adjacency_local = np.array(sorted(adj_set), dtype=int)
    init_bend = np.asarray(
        _initial_bending_3d(
            jnp.asarray(surface_pos),
            jnp.asarray(elements_local),
            jnp.asarray(adjacency_local),
        )
    )
    return elements_local, adjacency_local, init_bend


def _unique_edges_from_triangles(faces_global: np.ndarray) -> np.ndarray:
    """Collect unique undirected edge pairs from a (F, 3) triangle array."""
    if faces_global.size == 0:
        return np.empty((0, 2), dtype=int)
    edge_set: set[tuple[int, int]] = set()
    for face in faces_global:
        a, b, c = int(face[0]), int(face[1]), int(face[2])
        for p, q in ((a, b), (b, c), (c, a)):
            lo, hi = (p, q) if p < q else (q, p)
            edge_set.add((lo, hi))
    return np.array(sorted(edge_set), dtype=int)


def _interior_struts(
    interior_global: np.ndarray,
    surface_global: np.ndarray,
    pos_np: np.ndarray,
    strategy: str,
    k_nearest: int | None,
) -> np.ndarray:
    """Return ``(S, 2)`` global edge pairs connecting interior nodes to surface nodes."""
    if interior_global.size == 0 or surface_global.size == 0:
        return np.empty((0, 2), dtype=int)
    surface_pts = pos_np[surface_global]
    strut_pairs: list[tuple[int, int]] = []
    for i_global in interior_global:
        if strategy == "fan":
            targets = surface_global
        else:  # nearest_k
            dists = np.linalg.norm(surface_pts - pos_np[i_global], axis=-1)
            k_eff = min(int(k_nearest or 0), surface_global.size)
            if k_eff <= 0:
                continue
            nearest_local = np.argsort(dists)[:k_eff]
            targets = surface_global[nearest_local]
        for s_global in targets:
            p, q = int(i_global), int(s_global)
            lo, hi = (p, q) if p < q else (q, p)
            strut_pairs.append((lo, hi))
    return (
        np.array(strut_pairs, dtype=int) if strut_pairs else np.empty((0, 2), dtype=int)
    )


def create_dp_container(
    state: State,
    *,
    em: float | jax.Array | None = None,
    ec: float | jax.Array | None = None,
    eb: float | jax.Array | None = None,
    el: float | jax.Array | None = None,
    gamma: float | jax.Array | None = None,
    tau_s: float | jax.Array | None = None,
    plasticity_type: Literal["edge", "perimeter", "bending", "none", None] = None,
    group_by: str = "bond",
    is_surface: jax.Array | None = None,
    interior_edges: str | int = "fan",
) -> Any:
    """Build a ``DeformableParticleModel`` (or a plastic variant) from a DP state.

    For each body (grouped by ``group_by``) the surface topology is computed
    from the node positions: polar-angle-ordered polygon segments in 2D, or
    ``scipy.spatial.ConvexHull`` triangulation in 3D. Interior nodes (those
    flagged ``False`` in ``is_surface``) are wired to surface nodes as extra
    ``edges`` carrying the body's ``el`` coefficient; they never participate
    in surface ``elements`` / ``element_adjacency`` / ``initial_bendings``.

    Energy coefficients (``em, ec, eb, el, gamma``) are per-body scalars or
    per-body arrays of shape ``(n_bodies,)``, matching the existing
    ``generate_ga_deformable_state`` convention. If ``plasticity_type`` is
    given, ``tau_s`` is required and interpreted per-body (perimeter), per-
    edge (edge), or per-adjacency (bending) as appropriate.

    Parameters
    ----------
    state
        State whose nodes make up one or more DPs. Typically produced by
        ``create_ga_state(..., particle_type="dp")`` and optionally placed
        via ``distribute_bodies``.
    em, ec, eb, el, gamma
        Per-body elastic coefficients; ``None`` disables the term.
    tau_s
        Per-body plastic yield threshold; required whenever
        ``plasticity_type`` is not ``None``.
    plasticity_type
        One of ``"edge"``, ``"perimeter"``, ``"bending"``, or ``None`` /
        ``"none"`` for the elastic container.
    group_by
        Body grouping for topology; default ``"bond"``.
    is_surface
        Optional ``(N,)`` bool mask of surface nodes. When ``None`` (the
        default) the mask is computed automatically via
        :func:`ga_surface_mask` (per-body convex hull), which is correct
        for hollow, solid, and phantom core-type DPs from
        :func:`create_ga_state` and for any other approximately-convex
        body. Pass an explicit mask to override — useful for non-convex
        bodies where the convex-hull test would misclassify pocket nodes.
    interior_edges
        How to connect interior nodes to surface nodes. ``"fan"`` (default)
        connects every interior node to every surface node in its body.
        An ``int`` K connects each interior node to its ``K`` nearest
        surface nodes.

    Returns
    -------
    DeformableParticleModel or one of its plastic subclasses.
    """
    from ..bonded_forces.deformable_particle import DeformableParticleModel
    from ..bonded_forces.plastic_bending_deformable_particle import (
        PlasticBendingDeformableParticleModel,
    )
    from ..bonded_forces.plastic_deformable_particle import (
        PlasticDeformableParticleModel,
    )
    from ..bonded_forces.plastic_perimeter_deformable_particle import (
        PlasticPerimeterDeformableParticleModel,
    )

    valid_plasticity = {"edge", "perimeter", "bending", "none", None}
    if plasticity_type not in valid_plasticity:
        raise ValueError(
            "plasticity_type must be one of None, 'none', 'edge', 'perimeter', "
            f"or 'bending'; got {plasticity_type!r}"
        )
    if plasticity_type == "none":
        plasticity_type = None

    if isinstance(interior_edges, int):
        k_val = int(interior_edges)
        if k_val <= 0:
            raise ValueError(f"interior_edges K must be positive; got {k_val}")
        k_nearest: int | None = k_val
        interior_strategy = "nearest_k"
    elif interior_edges == "fan":
        k_nearest = None
        interior_strategy = "fan"
    else:
        raise ValueError(
            f"interior_edges must be 'fan' or a positive integer K; got {interior_edges!r}"
        )

    group_id, n_bodies = _resolve_body_grouping(state, group_by)
    gid_np = np.asarray(group_id)
    pos_np = np.asarray(state.pos)
    dim = int(state.dim)
    N = int(state.N)

    if is_surface is None:
        surface_mask = np.asarray(ga_surface_mask(state, group_by=group_by), dtype=bool)
    else:
        surface_mask = np.asarray(is_surface, dtype=bool)
        if surface_mask.shape != (N,):
            raise ValueError(
                f"is_surface must have shape ({N},); got {surface_mask.shape}"
            )

    em_b = _ensure_per_body_params(em, n_bodies, "em")
    ec_b = _ensure_per_body_params(ec, n_bodies, "ec")
    eb_b = _ensure_per_body_params(eb, n_bodies, "eb")
    el_b = _ensure_per_body_params(el, n_bodies, "el")
    gamma_b = _ensure_per_body_params(gamma, n_bodies, "gamma")
    tau_s_b = _ensure_per_body_params(tau_s, n_bodies, "tau_s")

    elements_all: list[np.ndarray] = []
    elements_id_all: list[np.ndarray] = []
    edges_all: list[np.ndarray] = []
    edges_id_all: list[np.ndarray] = []
    adjacency_all: list[np.ndarray] = []
    adjacency_id_all: list[np.ndarray] = []
    initial_bending_all: list[np.ndarray] = []
    elem_offset = 0

    for b in range(n_bodies):
        body_idxs = np.where(gid_np == b)[0]
        body_surface_mask = surface_mask[body_idxs]
        surface_global = body_idxs[body_surface_mask]
        interior_global = body_idxs[~body_surface_mask]
        n_s = surface_global.size

        # Surface topology in local indices, then re-indexed to global.
        local_elements = np.empty((0, dim), dtype=int)
        local_adj = np.empty((0, 2), dtype=int)
        local_init_bend = np.empty((0,), dtype=float)

        if dim == 2 and n_s >= 3:
            local_elements, local_adj, local_init_bend = _body_surface_topology_2d(
                pos_np[surface_global]
            )
        elif dim == 3 and n_s >= 4:
            local_elements, local_adj, local_init_bend = _body_surface_topology_3d(
                pos_np[surface_global]
            )

        elements_body = (
            surface_global[local_elements]
            if local_elements.size > 0
            else np.empty((0, dim), dtype=int)
        )

        # Surface edges (global). In 2D elements are already edges; in 3D
        # we uniquify triangle edge pairs.
        if dim == 2:
            surface_edges_body = (
                elements_body if elements_body.size > 0 else np.empty((0, 2), dtype=int)
            )
        else:
            surface_edges_body = _unique_edges_from_triangles(elements_body)

        strut_edges_body = _interior_struts(
            interior_global, surface_global, pos_np, interior_strategy, k_nearest
        )

        if surface_edges_body.size > 0 and strut_edges_body.size > 0:
            edges_body = np.concatenate([surface_edges_body, strut_edges_body], axis=0)
        elif surface_edges_body.size > 0:
            edges_body = surface_edges_body
        else:
            edges_body = strut_edges_body

        if elements_body.size > 0:
            elements_all.append(elements_body)
            elements_id_all.append(np.full(elements_body.shape[0], b, dtype=int))
        if edges_body.size > 0:
            edges_all.append(edges_body)
            edges_id_all.append(np.full(edges_body.shape[0], b, dtype=int))
        if local_adj.size > 0:
            adjacency_all.append(local_adj + elem_offset)
            adjacency_id_all.append(np.full(local_adj.shape[0], b, dtype=int))
            initial_bending_all.append(local_init_bend)

        elem_offset += elements_body.shape[0]

    elements = (
        jnp.asarray(np.concatenate(elements_all, axis=0)) if elements_all else None
    )
    elements_id = (
        jnp.asarray(np.concatenate(elements_id_all, axis=0))
        if elements_id_all
        else None
    )
    edges = jnp.asarray(np.concatenate(edges_all, axis=0)) if edges_all else None
    edges_id = (
        jnp.asarray(np.concatenate(edges_id_all, axis=0)) if edges_id_all else None
    )
    element_adjacency = (
        jnp.asarray(np.concatenate(adjacency_all, axis=0)) if adjacency_all else None
    )
    adjacency_id = (
        jnp.asarray(np.concatenate(adjacency_id_all, axis=0))
        if adjacency_id_all
        else None
    )
    initial_bending = (
        jnp.asarray(np.concatenate(initial_bending_all, axis=0))
        if initial_bending_all
        else None
    )

    em_per_elem = (
        em_b[elements_id] if (em_b is not None and elements_id is not None) else em_b
    )
    gamma_per_elem = (
        gamma_b[elements_id]
        if (gamma_b is not None and elements_id is not None)
        else gamma_b
    )
    eb_per_adj = (
        eb_b[adjacency_id] if (eb_b is not None and adjacency_id is not None) else eb_b
    )
    el_per_edge = (
        el_b[edges_id] if (el_b is not None and edges_id is not None) else el_b
    )

    container: Any = DeformableParticleModel.Create(
        vertices=state.pos,
        elements=elements,
        elements_id=elements_id,
        edges=edges,
        element_adjacency=element_adjacency,
        initial_bendings=initial_bending,
        em=em_per_elem,
        ec=ec_b,
        eb=eb_per_adj,
        el=el_per_edge,
        gamma=gamma_per_elem,
    )

    if plasticity_type is not None:
        if tau_s_b is None:
            raise ValueError(
                f"plasticity_type={plasticity_type!r} requires tau_s to be provided"
            )
        plastic_dp_kwargs = dict(
            vertices=state.pos,
            elements=container.elements,
            edges=container.edges,
            element_adjacency=container.element_adjacency,
            element_adjacency_edges=container.element_adjacency_edges,
            elements_id=container.elements_id,
            initial_body_contents=container.initial_body_contents,
            initial_element_measures=container.initial_element_measures,
            initial_edge_lengths=container.initial_edge_lengths,
            initial_bendings=container.initial_bendings,
            w_b=container.w_b,
            em=container.em,
            ec=container.ec,
            eb=container.eb,
            el=container.el,
            gamma=container.gamma,
        )
        if plasticity_type == "perimeter":
            if container.edges is None or container.el is None:
                raise ValueError(
                    "perimeter plasticity requires edge elasticity el to be provided"
                )
            plastic_dp_kwargs["tau_s"] = tau_s_b
            plastic_dp_kwargs["edges_id"] = edges_id
            container = PlasticPerimeterDeformableParticleModel.Create(
                **plastic_dp_kwargs
            )
        elif plasticity_type == "edge":
            if container.edges is None or container.el is None or edges_id is None:
                raise ValueError(
                    "edge plasticity requires edge elasticity el to be provided"
                )
            plastic_dp_kwargs["tau_s"] = tau_s_b[edges_id]
            container = PlasticDeformableParticleModel.Create(**plastic_dp_kwargs)
        else:  # "bending"
            if (
                container.element_adjacency is None
                or container.initial_bendings is None
                or container.eb is None
                or adjacency_id is None
            ):
                raise ValueError(
                    "bending plasticity requires bending elasticity eb to be provided"
                )
            plastic_dp_kwargs["tau_s"] = tau_s_b[adjacency_id]
            container = PlasticBendingDeformableParticleModel.Create(
                **plastic_dp_kwargs
            )
    elif tau_s is not None:
        raise ValueError(
            "tau_s was provided but plasticity_type is None; set plasticity_type to "
            "'edge', 'perimeter', or 'bending' to enable plasticity"
        )

    return container


def _broadcast_per_body(
    x: float | int | Sequence[float] | Sequence[int] | jax.Array | np.ndarray,
    n_bodies: int,
    name: str,
    dtype: Any = float,
) -> np.ndarray:
    """Broadcast a scalar or (n_bodies,) sequence to a (n_bodies,) numpy array."""
    arr = np.asarray(x, dtype=dtype)
    if arr.ndim == 0:
        return np.full((n_bodies,), arr, dtype=dtype)
    if arr.shape == (n_bodies,):
        return arr
    raise ValueError(f"{name} must be a scalar or shape ({n_bodies},); got {arr.shape}")


def _normalize_aspect_ratio(
    aspect_ratio: Any, n_bodies: int, dim: int
) -> np.ndarray | None:
    """Return a (n_bodies, dim) numpy array or None (isotropic)."""
    if aspect_ratio is None:
        return None
    arr = np.asarray(aspect_ratio, dtype=float)
    if arr.ndim == 0:
        return np.full((n_bodies, dim), float(arr), dtype=float)
    if arr.shape == (dim,):
        return np.broadcast_to(arr[None, :], (n_bodies, dim)).copy()
    if arr.shape == (n_bodies,):
        # scalar per body -> isotropic per body (still valid, degenerates to sphere)
        return np.broadcast_to(arr[:, None], (n_bodies, dim)).copy()
    if arr.shape == (n_bodies, dim):
        return arr
    raise ValueError(
        f"aspect_ratio must be None, scalar, ({dim},), ({n_bodies},), or "
        f"({n_bodies}, {dim}); got {arr.shape}"
    )


def _resolve_material_table(
    mat_table: Any,
    *,
    material_type: str,
    material_kwargs: dict[str, Any] | None,
    matcher_type: str,
    matcher_kwargs: dict[str, Any] | None,
    e_int: float,
) -> Any:
    """Return ``mat_table`` or build one from the material/matcher spec.

    Shared by :func:`build_ga_system` and :func:`build_sphere_system`.
    """
    import jaxdem as jd

    if mat_table is not None:
        return mat_table
    if material_kwargs is None:
        material_kwargs = dict(young=e_int, poisson=0.5, density=1.0)
    material = jd.Material.create(material_type, **dict(material_kwargs))
    matcher = jd.MaterialMatchmaker.create(matcher_type, **dict(matcher_kwargs or {}))
    return jd.MaterialTable.from_materials([material], matcher=matcher)


def _make_collider_kw_resolver(
    collider_type: str, collider_kw: dict[str, Any] | None
) -> Callable[[State], dict[str, Any]]:
    """Build the per-state collider-kw resolver shared by the system builders."""
    user_collider_kw = dict(collider_kw) if collider_kw is not None else None
    collider_key = _normalize_key(collider_type)

    def _resolve(current_state: State) -> dict[str, Any]:
        if user_collider_kw is None:
            if collider_key == "neighborlist":
                return dict(
                    state=current_state,
                    cutoff=float(2.0 * jnp.max(current_state.rad)),
                    skin_fraction=0.05,
                    safety_factor=5.0,
                )
            if collider_key == "celllist":
                return dict(state=current_state)
            return {}
        kw = dict(user_collider_kw)
        if "state" in kw or collider_key in ("neighborlist", "celllist"):
            kw["state"] = current_state  # refresh reference
        return kw

    return _resolve


def build_ga_system(
    particle_radii: Sequence[float] | np.ndarray,
    vertex_counts: Sequence[int] | np.ndarray,
    asperity_radius: float | Sequence[float] | np.ndarray,
    phi: float,
    dim: int,
    *,
    # Body composition
    particle_type: str = "clump",
    core_type: str = "hollow",
    aspect_ratio: Any = None,
    particle_mass: float | Sequence[float] = 1.0,
    n_property_samples: int = 10_000_000,
    mesh_type: str = "thomson",
    mesh_kwargs: dict[str, Any] | None = None,
    asperity_dispersity_type: str = "constant",
    asperity_dispersity_kwargs: dict[str, Any] | None = None,
    # Placement
    domain_type: str = "periodic",
    box_aspect: Sequence[float] | None = None,
    randomize_orientation: bool = True,
    initial_phi_bb: float = 0.3,
    # Compression
    compression_step: float = 1e-3,
    compression_pe_tol: float = 1e-16,
    compression_pe_diff_tol: float = 1e-16,
    max_n_min_steps_per_outer: int = 200_000,
    compression_progress: bool = False,
    fire_dt: float = 1e-2,
    # Final system
    dt: float = 1e-3,
    linear_integrator_type: str = "verlet",
    rotation_integrator_type: str = "verletspiral",
    linear_integrator_kw: dict[str, Any] | None = None,
    rotation_integrator_kw: dict[str, Any] | None = None,
    force_model_type: str = "spring",
    collider_type: str = "neighborlist",
    collider_kw: dict[str, Any] | None = None,
    minimizer: Any = None,
    minimizer_kw: dict[str, Any] | None = None,
    target_fn: Any = None,
    # Material / interaction
    mat_table: Any = None,
    e_int: float = 1.0,
    material_type: str = "elastic",
    material_kwargs: dict[str, Any] | None = None,
    matcher_type: str = "harmonic",
    matcher_kwargs: dict[str, Any] | None = None,
    # DP container (used only when particle_type == "dp")
    dp_em: float | None = 1.0,
    dp_ec: float | None = 1.0,
    dp_eb: float | None = 1.0,
    dp_el: float | None = 1.0,
    dp_gamma: float | None = None,
    dp_tau_s: float | None = None,
    dp_plasticity_type: Literal["edge", "perimeter", "bending", "none", None] = None,
    dp_interior_edges: str | int = "fan",
    dp_is_surface: jax.Array | None = None,
    # Misc
    seed: int | None = None,
) -> tuple[Any, ...]:
    """Catch-all builder: polydisperse GA/DP particles at a target packing fraction.

    Given per-body polydispersity (radii, vertex counts, aspect ratios, etc.),
    builds the state, randomly places each body's bounding sphere at the
    loose bounding-sphere packing fraction ``initial_phi_bb``,
    energy-minimizes the analogue sphere system, transfers the new centroids
    back to the bodies, builds a System (initially with FIRE for
    minimization), quasistatically compresses to the target true-body
    packing fraction ``phi``, and returns the result wrapped in a System
    built with the user-requested integrator, collider, and material.

    For ``particle_type="dp"`` a :class:`DeformableParticleModel` (or a
    plastic variant) is also built and wired into the returned System as
    the ``bonded_force_model``. In that case returns
    ``(state, system, container)``; for clumps/spheres returns
    ``(state, system)``.

    Parameters
    ----------
    particle_radii, vertex_counts, asperity_radius
        Per-body arrays (shape ``(M,)``) or scalars for the last two.
    phi
        Target true-body packing fraction (uses each body's union volume,
        not its bounding sphere).
    dim
        Spatial dimension (2 or 3).
    particle_type
        ``"clump"`` (default) or ``"dp"``. ``nv=1`` clumps are effectively
        spheres.
    core_type
        ``"hollow"`` (default), ``"solid"``, or ``"phantom"``.
    aspect_ratio
        ``None`` (isotropic), scalar, ``(dim,)``, ``(M,)``, or ``(M, dim)``.
    asperity_dispersity_type, asperity_dispersity_kwargs
        Asperity-radius polydispersity. Forwarded as-is to every per-body
        :func:`create_ga_state` call. Theoretical mean of the chosen
        distribution always equals each body's ``asperity_radius``, so
        the bounding/core geometry is unchanged. See
        :func:`create_ga_state` for the supported distributions and
        kwargs.
    domain_type
        ``"periodic"`` or ``"reflect"`` (closed box).
    initial_phi_bb
        Loose *bounding-sphere* packing fraction used for the initial random
        placement. It only needs to be below the jamming density of the
        bounding spheres; the target ``phi`` is reached afterwards by the
        quasistatic compression. Defaults to 0.3.
    compression_step
        Packing-fraction increment for quasistatic compression.
    compression_pe_tol, compression_pe_diff_tol
        Energy tolerances passed to
        :func:`quasistatic_compress_to_packing_fraction`.
    max_n_min_steps_per_outer, compression_progress
        Passed to :func:`quasistatic_compress_to_packing_fraction`.
    fire_dt
        Time step of the internal FIRE system used for compression.
        Defaults to 1e-2.
    dt
        Time step for the returned System.
    linear_integrator_type, rotation_integrator_type
        Integrator types for the final (returned) System.
    linear_integrator_kw, rotation_integrator_kw
        Keyword arguments forwarded to the final System's integrator
        constructors (e.g. thermostat parameters for ``"verlet_rescaling"``).
    force_model_type, collider_type, collider_kw
        Parameters of the final (returned) System.
    mat_table
        Optional pre-built material table. If supplied, material and matcher
        specs are ignored.
    material_type, material_kwargs, matcher_type, matcher_kwargs, e_int
        Material / matcher spec. When ``mat_table`` is None a single
        ``material_type`` is created with ``material_kwargs`` (defaulting
        to ``young=e_int, poisson=0.5, density=1.0``) and paired with the
        given matcher.
    dp_em, dp_ec, dp_eb, dp_el
        DP energy parameters. Unused for ``particle_type="clump"``.
    dp_gamma, dp_tau_s, dp_plasticity_type
        DP plasticity parameters. Unused for ``particle_type="clump"``.
    dp_interior_edges, dp_is_surface
        DP energy / topology parameters. Unused for ``particle_type="clump"``.

    Returns
    -------
    tuple
        ``(state, system)`` for clumps / spheres, or
        ``(state, system, container)`` for DPs.
    """
    import jaxdem as jd

    if particle_type not in ("clump", "dp"):
        raise ValueError(
            f"particle_type must be 'clump' or 'dp'; got {particle_type!r}"
        )

    radii_arr = np.asarray(particle_radii, dtype=float)
    nv_arr = np.asarray(vertex_counts, dtype=int)
    if radii_arr.ndim != 1 or nv_arr.ndim != 1:
        raise ValueError(
            f"particle_radii and vertex_counts must be 1D; got shapes "
            f"{radii_arr.shape} and {nv_arr.shape}"
        )
    M = int(radii_arr.shape[0])
    if nv_arr.shape[0] != M:
        raise ValueError(
            f"particle_radii and vertex_counts must have the same length; got "
            f"{M} and {nv_arr.shape[0]}"
        )

    ar_arr = _broadcast_per_body(asperity_radius, M, "asperity_radius")
    mass_arr = _broadcast_per_body(particle_mass, M, "particle_mass")
    aspect_arr = _normalize_aspect_ratio(aspect_ratio, M, dim)

    if seed is None:
        seed = int(np.random.randint(0, int(1e9)))

    # Group bodies by unique (nv, rad, asp_rad, aspect_tuple, mass). Each
    # group gets one create_ga_state call with N=count (efficient). All
    # bodies share the same mesh_type / mesh_kwargs.
    def _key(i: int) -> tuple[Any, ...]:
        aspect_tup = (
            tuple(float(x) for x in aspect_arr[i]) if aspect_arr is not None else None
        )
        return (
            int(nv_arr[i]),
            float(radii_arr[i]),
            float(ar_arr[i]),
            aspect_tup,
            float(mass_arr[i]),
        )

    keys = [_key(i) for i in range(M)]
    unique_keys: list[tuple[Any, ...]] = []
    seen: set[tuple[Any, ...]] = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)

    # Derive a distinct, deterministic per-group seed from the master seed;
    # reusing the master seed verbatim would give groups with equal nv
    # identical meshes / dispersity draws.
    group_seeds = np.random.SeedSequence(int(seed)).generate_state(len(unique_keys))

    states = []
    for gi, k in enumerate(unique_keys):
        nv, rad, asp_rad, aspect_tup, mass = k
        count = sum(1 for x in keys if x == k)
        aspect_for_call = list(aspect_tup) if aspect_tup is not None else None
        group_state = create_ga_state(
            N=count,
            nv=nv,
            dim=dim,
            particle_radius=rad,
            asperity_radius=asp_rad,
            particle_type=particle_type,
            core_type=core_type,
            aspect_ratio=aspect_for_call,
            particle_mass=mass,
            n_samples=n_property_samples,
            seed=int(group_seeds[gi]),
            mesh_type=mesh_type,
            mesh_kwargs=mesh_kwargs,
            asperity_dispersity_type=asperity_dispersity_type,
            asperity_dispersity_kwargs=asperity_dispersity_kwargs,
        )
        states.append(group_state)

    state = states[0] if len(states) == 1 else State.merge(states[0], states[1:])

    # Initial bounding-sphere placement + random orientation. The starting
    # bounding-sphere packing fraction only needs to be below the jamming
    # density of the bounding spheres so the random placement finds an
    # overlap-free minimum. The target ``phi`` (true-body) is reached
    # afterwards by the quasistatic compression, which handles both
    # compression and decompression directions automatically.
    state, box_size = distribute_bodies(
        state,
        phi=float(initial_phi_bb),
        domain_type=domain_type,
        box_aspect=box_aspect,
        seed=seed,
        randomize_orientation=randomize_orientation,
    )

    # Build the DP container from the (placed, oriented) state.
    container = None
    if particle_type == "dp":
        container = create_dp_container(
            state,
            em=dp_em,
            ec=dp_ec,
            eb=dp_eb,
            el=dp_el,
            gamma=dp_gamma,
            tau_s=dp_tau_s,
            plasticity_type=dp_plasticity_type,
            interior_edges=dp_interior_edges,
            is_surface=dp_is_surface,
        )

    # Material / matcher.
    mat_table = _resolve_material_table(
        mat_table,
        material_type=material_type,
        material_kwargs=material_kwargs,
        matcher_type=matcher_type,
        matcher_kwargs=matcher_kwargs,
        e_int=e_int,
    )

    _resolve_collider_kw = _make_collider_kw_resolver(collider_type, collider_kw)

    # Build a FIRE system for quasistatic compression. It mirrors the
    # returned system's collider / domain / bonded-force setup so that the
    # PE landscape FIRE sees is the same one the user will simulate on.
    # RotationFIRE is used for clumps (where rigid-body rotation matters
    # during minimization) and skipped for DPs (whose node kinematics are
    # translation-only at the rigid level; DP internal relaxation comes
    # from the bonded-force model).
    fire_kw: dict[str, Any] = dict(
        state_shape=state.shape,
        dt=float(fire_dt),
        minimizer=jd.minimizers.fire,
        minimizer_kw={"dt": float(fire_dt)},
        domain_type=domain_type,
        force_model_type="spring",
        collider_type=collider_type,
        collider_kw=_resolve_collider_kw(state),
        mat_table=mat_table,
        domain_kw={"box_size": box_size},
    )
    if container is not None:
        fire_kw["bonded_force_model"] = container
    fire_system = jd.System.create(**fire_kw)

    # Import here to avoid a cross-module import at module-load time.
    from .packing_utils import quasistatic_compress_to_packing_fraction

    state, fire_system, _final_phi, _final_pe = (
        quasistatic_compress_to_packing_fraction(
            state,
            fire_system,
            target_phi=float(phi),
            step=float(compression_step),
            pe_tol=float(compression_pe_tol),
            pe_diff_tol=float(compression_pe_diff_tol),
            max_n_min_steps_per_outer=int(max_n_min_steps_per_outer),
            progress=bool(compression_progress),
        )
    )

    # Build the final system with the user's integrator / collider.
    final_box_size = fire_system.domain.box_size
    final_kw: dict[str, Any] = dict(
        state_shape=state.shape,
        dt=float(dt),
        linear_integrator_type=linear_integrator_type,
        rotation_integrator_type=rotation_integrator_type,
        linear_integrator_kw=linear_integrator_kw,
        rotation_integrator_kw=rotation_integrator_kw,
        domain_type=domain_type,
        force_model_type=force_model_type,
        collider_type=collider_type,
        collider_kw=_resolve_collider_kw(state),
        mat_table=mat_table,
        domain_kw={"box_size": final_box_size},
        minimizer=minimizer,
        minimizer_kw=minimizer_kw,
        target_fn=target_fn,
    )
    if container is not None:
        final_kw["bonded_force_model"] = container
    system = jd.System.create(**final_kw)

    if container is not None:
        return state, system, container
    return state, system


def build_sphere_system(
    particle_radii: Sequence[float] | np.ndarray,
    phi: float,
    dim: int,
    *,
    particle_mass: float | Sequence[float] = 1.0,
    # Placement
    domain_type: str = "periodic",
    box_aspect: Sequence[float] | None = None,
    initial_phi: float = 0.3,
    # Compression
    compression_step: float = 1e-3,
    compression_pe_tol: float = 1e-16,
    compression_pe_diff_tol: float = 1e-16,
    max_n_min_steps_per_outer: int = 200_000,
    compression_progress: bool = False,
    fire_dt: float = 1e-2,
    # Final system
    dt: float = 1e-3,
    linear_integrator_type: str = "verlet",
    rotation_integrator_type: str = "",
    linear_integrator_kw: dict[str, Any] | None = None,
    rotation_integrator_kw: dict[str, Any] | None = None,
    force_model_type: str = "spring",
    collider_type: str = "neighborlist",
    collider_kw: dict[str, Any] | None = None,
    # Material / interaction
    mat_table: Any = None,
    e_int: float = 1.0,
    material_type: str = "elastic",
    material_kwargs: dict[str, Any] | None = None,
    matcher_type: str = "harmonic",
    matcher_kwargs: dict[str, Any] | None = None,
    # Misc
    seed: int | None = None,
    minimizer: Any = None,
    minimizer_kw: dict[str, Any] | None = None,
    target_fn: Any = None,
) -> tuple[State, Any]:
    """Catch-all builder for a polydisperse sphere packing at a target phi.

    Sphere counterpart to :func:`build_ga_system`. Random positions are
    drawn loose (``initial_phi`` — default 0.3 — or the target, whichever is
    smaller) via :func:`random_sphere_configuration`, then quasistatically
    compressed to ``phi`` under the same FIRE/spring setup as
    ``build_ga_system`` (the FIRE time step is ``fire_dt``). Returns
    ``(state, system)`` built with the user-requested integrator, collider,
    and material; ``linear_integrator_kw`` / ``rotation_integrator_kw`` are
    forwarded to the final integrator constructors (e.g. thermostat
    parameters for ``linear_integrator_type="verlet_rescaling"``).

    Parameters mirror :func:`build_ga_system` (minus all the GA/DP-specific
    knobs that don't apply to bare spheres).
    """
    import jaxdem as jd

    from .random_sphere_configuration import random_sphere_configuration

    radii_arr = np.asarray(particle_radii, dtype=float)
    if radii_arr.ndim != 1:
        raise ValueError(f"particle_radii must be 1D; got shape {radii_arr.shape}")
    n_spheres = int(radii_arr.shape[0])
    mass_arr = _broadcast_per_body(particle_mass, n_spheres, "particle_mass")

    if seed is None:
        seed = int(np.random.randint(0, int(1e9)))

    # Initial random placement at a loose phi (or directly at the target
    # if the target is already loose); random_sphere_configuration
    # minimizes so the starting state is overlap-free.
    initial_phi = min(float(initial_phi), float(phi))
    pos_init, box_size_init = random_sphere_configuration(
        particle_radii=radii_arr.tolist(),
        phi=initial_phi,
        dim=int(dim),
        seed=int(seed),
        box_aspect=box_aspect,
        domain_type=domain_type,
    )

    state = create_sphere_state(
        radii=radii_arr,
        dim=int(dim),
        pos=jnp.asarray(pos_init),
        mass=jnp.asarray(mass_arr),
    )

    # Material / matcher.
    mat_table = _resolve_material_table(
        mat_table,
        material_type=material_type,
        material_kwargs=material_kwargs,
        matcher_type=matcher_type,
        matcher_kwargs=matcher_kwargs,
        e_int=e_int,
    )

    _resolve_collider_kw = _make_collider_kw_resolver(collider_type, collider_kw)

    # FIRE system for compression.
    fire_system = jd.System.create(
        state_shape=state.shape,
        dt=float(fire_dt),
        minimizer=jd.minimizers.fire,
        minimizer_kw={"dt": float(fire_dt)},
        domain_type=domain_type,
        force_model_type="spring",
        collider_type=collider_type,
        collider_kw=_resolve_collider_kw(state),
        mat_table=mat_table,
        domain_kw={"box_size": box_size_init},
    )

    from .packing_utils import quasistatic_compress_to_packing_fraction

    state, fire_system, _final_phi, _final_pe = (
        quasistatic_compress_to_packing_fraction(
            state,
            fire_system,
            target_phi=float(phi),
            step=float(compression_step),
            pe_tol=float(compression_pe_tol),
            pe_diff_tol=float(compression_pe_diff_tol),
            max_n_min_steps_per_outer=int(max_n_min_steps_per_outer),
            progress=bool(compression_progress),
        )
    )

    final_box_size = fire_system.domain.box_size
    final_system = jd.System.create(
        state_shape=state.shape,
        dt=float(dt),
        linear_integrator_type=linear_integrator_type,
        rotation_integrator_type=rotation_integrator_type,
        linear_integrator_kw=linear_integrator_kw,
        rotation_integrator_kw=rotation_integrator_kw,
        domain_type=domain_type,
        force_model_type=force_model_type,
        collider_type=collider_type,
        collider_kw=_resolve_collider_kw(state),
        mat_table=mat_table,
        domain_kw={"box_size": final_box_size},
        minimizer=minimizer,
        minimizer_kw=minimizer_kw,
        target_fn=target_fn,
    )

    return state, final_system
