# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Asperity mesh generators for geometric-asperity (GA) particles.

Each ``generate_*_mesh`` function produces ``(nv, dim)`` or ``(N, nv, dim)``
unit-scaled vertex positions — the longest axis has extent 1 — suitable for
consumption by :func:`~jaxdem.utils.particleCreation.create_ga_state`.

Available meshes
----------------
* :func:`generate_thomson_mesh` — generalized Thomson problem on a
  hyper-ellipsoid; iterative Riesz-energy minimization, works in 2D and 3D.
* :func:`generate_icosphere_mesh` — recursive icosahedron subdivision (3D)
  or regular polygon (2D); deterministic; discrete ``nv`` in 3D.
* :func:`generate_fibonacci_sphere_mesh` — golden-angle spiral sphere (3D)
  or evenly-spaced circle (2D); deterministic; any ``nv``.
* :func:`generate_torus_mesh` — quasi-uniform torus surface points (3D).
* :func:`generate_helix_mesh` — right-handed helix (3D) or Archimedean
  spiral (2D).
* :func:`generate_arclength_mesh` — 2D only: equal-arc-length spacing
  along an ellipse / circle perimeter. Closed-form analogue of the
  converged Thomson ground state in 2D (exact in the α→∞ packing limit).
* :func:`generate_faceted_mesh` — polygonal / icosahedral shell. Vertex
  asperities at the corners + face-interior fillers to reach ``nv``.
  Gives genuinely angular, crystalline particles (flat faces, sharp edges)
  as opposed to the smooth-sphere family.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


# --------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------


def _apply_aspect_ratio_renorm(
    verts_np: np.ndarray, aspect_ratio, dim: int
) -> np.ndarray:
    """Stretch by per-axis factors then re-normalize so the longest axis extent is 1.

    Used by non-isotropic base shapes (torus, helix). Unlike the pure
    "normalize axes to max=1" pattern that works for sphere-based meshes,
    non-isotropic bases need a post-hoc renormalization because the base
    extents along each axis aren't all equal.
    """
    if aspect_ratio is None:
        return verts_np
    axes = np.asarray(aspect_ratio, dtype=np.float64)
    if axes.ndim == 0:
        axes = np.full((dim,), float(axes), dtype=np.float64)
    elif axes.shape != (dim,):
        raise ValueError(f"Expected aspect_ratio shape ({dim},), got {axes.shape}.")
    if np.any(axes <= 0):
        raise ValueError("Axes must be strictly positive.")
    out = verts_np * axes
    max_abs = float(np.max(np.abs(out)))
    return out / max_abs if max_abs > 0.0 else out


def _normalize_axes_to_unit(aspect_ratio, dim: int) -> np.ndarray:
    """Return axes ``(dim,)`` scaled so the longest is 1, or ones if None.

    Used by sphere-based meshes (icosphere, fibonacci, arclength) whose
    base extents are already unit so ``pos * axes`` preserves max extent 1.
    """
    if aspect_ratio is None:
        return np.ones((dim,), dtype=np.float64)
    axes = np.asarray(aspect_ratio, dtype=np.float64)
    if axes.ndim == 0:
        axes = np.full((dim,), float(axes), dtype=np.float64)
    elif axes.shape != (dim,):
        raise ValueError(f"Expected aspect_ratio shape ({dim},), got {axes.shape}.")
    if np.any(axes <= 0):
        raise ValueError("Axes must be strictly positive.")
    return axes / np.max(axes)


# --------------------------------------------------------------------
# Thomson (generalized) mesh
# --------------------------------------------------------------------


def _sample_uniform_surface_points(key, nv, axes):
    """
    Sample points uniformly on a hyper-ellipsoid surface.
    """
    inv_axes = 1.0 / axes
    accept_scale = jnp.min(axes)

    accepted_chunks = []
    accepted_count = 0
    current_key = key

    while accepted_count < nv:
        n_candidates = max(32, 2 * (nv - accepted_count))
        current_key, sphere_key, accept_key = jax.random.split(current_key, 3)
        unit_points = jax.random.normal(sphere_key, shape=(n_candidates, axes.shape[0]))
        unit_points /= jnp.linalg.norm(unit_points, axis=-1, keepdims=True)
        weights = jnp.linalg.norm(unit_points * inv_axes, axis=-1)
        accept_prob = accept_scale * weights
        accepted_mask = jax.random.uniform(accept_key, shape=(n_candidates,)) < accept_prob
        accepted = unit_points[accepted_mask]

        accepted_chunks.append(accepted)
        accepted_count += accepted.shape[0]

    return jnp.concatenate(accepted_chunks, axis=0)[:nv] * axes


def random_points_on_hyper_ellipsoid(key, nv, N, dim, aspect_ratio=None, use_uniform_sampling=True):
    """
    Generate nv uniform random points on a dim-dimensional
    unit hyper-ellipsoid surface with a set aspect ratio, repeated N times.
    If using uniform sampling, expensive rejection sampling is used
    to ensure the points are uniform across the surface.
    This gives the exact result for hyper-ellipsoids, but is
    not necessary for hyper-spheres.
    Scaled so that the longest axis is unit-length.
    """
    if aspect_ratio is None:
        axes = jnp.ones((dim,), dtype=jnp.float32)
    else:
        axes = jnp.asarray(aspect_ratio, dtype=jnp.float32)
        if axes.ndim == 0:
            axes = jnp.full((dim,), axes, dtype=jnp.float32)
        elif axes.shape != (dim,):
            raise ValueError(
                f"Expected aspect_ratio to have shape ({dim},), got {axes.shape}."
            )
        if jnp.any(axes <= 0):
            raise ValueError("Hyper-ellipsoid axes must be strictly positive.")
        axes = axes / jnp.max(axes)

    if jnp.all(axes == axes[0]).item() or use_uniform_sampling:
        points = jax.random.normal(key, shape=(N, nv, dim))
        points /= jnp.linalg.norm(points, axis=-1, keepdims=True)
        points *= axes
        return (points[0] if N == 1 else points), axes

    points = jnp.stack(
        [_sample_uniform_surface_points(batch_key, nv, axes) for batch_key in jax.random.split(key, N)]
    )
    return (points[0] if N == 1 else points), axes


def riesz_energy(pos, alpha):
    """Riesz energy kernel.  alpha=1 reduces to the Thomson problem.  alpha=\infty reduces to the packing problem"""
    r_ij = pos[:, None, :] - pos[None, :, :]
    # squared distances (no gradient issue here)
    d_sq = jnp.sum(r_ij**2, axis=-1)
    # fill diagonal with 1.0 BEFORE sqrt, so grad(sqrt(1.0)) = 0.5, not inf
    n = pos.shape[0]
    d_sq = d_sq.at[jnp.diag_indices(n)].set(1.0)
    d_ij = jnp.sqrt(d_sq)
    e_ij = 1.0 / d_ij ** alpha
    # zero out the diagonal so self-interactions don't contribute
    e_ij = e_ij.at[jnp.diag_indices(n)].set(0.0)
    return jnp.sum(jnp.triu(e_ij, k=1))


def project_to_tangent(grad, pos, aspect_ratio):
    """Remove the normal component of the gradient (project onto tangent plane of surface)."""
    normal = pos / aspect_ratio ** 2
    normal = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
    return grad - jnp.sum(grad * normal, axis=-1, keepdims=True) * normal


def retract_to_surface(pos, aspect_ratio):
    """Project point back onto the ellipsoid/ellipse surface."""
    u = pos / aspect_ratio
    u = u / jnp.linalg.norm(u, axis=-1, keepdims=True)
    return u * aspect_ratio


def minimize_on_hyper_ellipsoid(pos, axes, alpha, lr=0.01, steps=1000):
    """Minimize Riesz energy for points constrained to a hyper-ellipsoid surface."""
    axes = jnp.asarray(axes, dtype=pos.dtype)
    energy_grad = jax.grad(riesz_energy)

    if steps == 0:
        return pos, riesz_energy(pos, alpha)

    def step(pos, _):
        g = energy_grad(pos, alpha)
        g_tangent = project_to_tangent(g, pos, axes)
        pos = pos - lr * g_tangent
        pos = retract_to_surface(pos, axes)
        return pos, riesz_energy(pos, alpha)

    pos, _ = jax.lax.scan(step, pos, None, length=steps)
    return pos, riesz_energy(pos, alpha)


def generate_thomson_mesh(
    nv,
    N,
    dim,
    alpha=1.0,
    lr=0.01,
    steps=1000,
    aspect_ratio=None,
    use_uniform_sampling=True,
    batch_size=None,
    seed=None,
):
    """Generate and minimize charges constrained to a hyper-ellipsoid surface."""
    key = jax.random.PRNGKey(np.random.randint(0, 1e9) if seed is None else seed)
    pos, axes = random_points_on_hyper_ellipsoid(
        key,
        nv=nv,
        N=N,
        dim=dim,
        aspect_ratio=aspect_ratio,
        use_uniform_sampling=use_uniform_sampling,
    )
    surface_dim = dim - 1
    scaled_lr = lr / nv ** ((alpha + 2) / surface_dim)
    if N == 1:
        pos, energy = minimize_on_hyper_ellipsoid(
            pos, axes, alpha, lr=scaled_lr, steps=steps
        )
    else:
        if batch_size is not None and batch_size < 1:
            raise ValueError("batch_size must be positive.")

        minimize_fn = lambda x: minimize_on_hyper_ellipsoid(
            x, axes, alpha, lr=scaled_lr, steps=steps
        )
        if batch_size is None:
            pos, energy = jax.vmap(minimize_fn)(pos)
        else:
            pos, energy = jax.lax.map(minimize_fn, pos, batch_size=batch_size)
    if np.any(np.isnan(energy)):
        raise ValueError(f'Minimization failed on {np.mean(np.isnan(energy)) * 100:.2f}% of runs. Try lowering lr!')
    return pos, energy


# --------------------------------------------------------------------
# Icosphere mesh (3D subdivision / 2D regular polygon)
# --------------------------------------------------------------------


def _icosahedron_vertices() -> np.ndarray:
    """Return the 12 vertices of a regular icosahedron on the unit sphere."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array(
        [
            [-1.0, phi, 0.0],
            [1.0, phi, 0.0],
            [-1.0, -phi, 0.0],
            [1.0, -phi, 0.0],
            [0.0, -1.0, phi],
            [0.0, 1.0, phi],
            [0.0, -1.0, -phi],
            [0.0, 1.0, -phi],
            [phi, 0.0, -1.0],
            [phi, 0.0, 1.0],
            [-phi, 0.0, -1.0],
            [-phi, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return verts / np.linalg.norm(verts, axis=-1, keepdims=True)


def _icosahedron_faces() -> np.ndarray:
    """Return the 20 triangular faces as vertex index triples."""
    return np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )


def _subdivide(
    verts: list[np.ndarray], faces: np.ndarray
) -> tuple[list[np.ndarray], np.ndarray]:
    """Loop-style subdivision: split each triangle into 4, project midpoints to unit sphere."""
    midpoint_cache: dict[tuple[int, int], int] = {}

    def _midpoint(a: int, b: int) -> int:
        key = (a, b) if a < b else (b, a)
        cached = midpoint_cache.get(key)
        if cached is not None:
            return cached
        mid = (verts[a] + verts[b]) / 2.0
        mid = mid / np.linalg.norm(mid)
        verts.append(mid)
        idx = len(verts) - 1
        midpoint_cache[key] = idx
        return idx

    new_faces = np.empty((faces.shape[0] * 4, 3), dtype=np.int64)
    for i, (a, b, c) in enumerate(faces):
        ab = _midpoint(int(a), int(b))
        bc = _midpoint(int(b), int(c))
        ca = _midpoint(int(c), int(a))
        new_faces[4 * i + 0] = (a, ab, ca)
        new_faces[4 * i + 1] = (b, bc, ab)
        new_faces[4 * i + 2] = (c, ca, bc)
        new_faces[4 * i + 3] = (ab, bc, ca)
    return verts, new_faces


def _icosphere_level_for_nv(nv: int, max_level: int = 8) -> int:
    """Return the subdivision level whose vertex count equals ``nv``, or -1 if none."""
    for level in range(max_level):
        if 10 * 4 ** level + 2 == nv:
            return level
    return -1


def _valid_icosphere_nvs(max_level: int = 6) -> list[int]:
    return [10 * 4 ** lv + 2 for lv in range(max_level)]


def generate_icosphere_mesh(
    nv: int,
    N: int,
    dim: int,
    aspect_ratio=None,
) -> jax.Array:
    """Generate ``N`` icosphere meshes (3D) or regular polygons (2D).

    In 3D, ``nv`` must be ``10 * 4**level + 2`` for some ``level >= 0``
    (i.e. one of ``{12, 42, 162, 642, 2562, ...}``); each corresponds to a
    subdivision level of the icosahedron. Use :func:`generate_fibonacci_sphere_mesh`
    if you need an arbitrary vertex count on a sphere.

    In 2D, ``nv`` can be any integer and the output is a regular ``nv``-gon
    on the unit circle.

    The mesh is deterministic, so all ``N`` bodies are identical copies;
    per-body random orientation is typically applied downstream by
    :func:`~jaxdem.utils.particleCreation.distribute_bodies`.
    """
    if dim == 3:
        level = _icosphere_level_for_nv(nv)
        if level < 0:
            raise ValueError(
                f"icosphere mesh in 3D requires nv in {_valid_icosphere_nvs()} "
                f"(10*4^level + 2); got nv={nv}. Use mesh_type='fibonacci' for "
                "arbitrary nv."
            )
        verts_list: list[np.ndarray] = list(_icosahedron_vertices())
        faces = _icosahedron_faces()
        for _ in range(level):
            verts_list, faces = _subdivide(verts_list, faces)
        verts_np = np.asarray(verts_list, dtype=np.float64)
    elif dim == 2:
        if nv < 3:
            raise ValueError(
                f"2D icosphere (regular polygon) requires nv >= 3; got nv={nv}."
            )
        theta = 2.0 * np.pi * np.arange(nv, dtype=np.float64) / nv
        verts_np = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    else:
        raise ValueError(f"dim must be 2 or 3; got {dim}.")

    axes = _normalize_axes_to_unit(aspect_ratio, dim)
    verts_np = verts_np * axes

    pos = jnp.asarray(verts_np)
    if N == 1:
        return pos
    return jnp.broadcast_to(pos[None, ...], (N, pos.shape[0], dim))


# --------------------------------------------------------------------
# Fibonacci sphere mesh (3D golden-angle spiral / 2D evenly-spaced circle)
# --------------------------------------------------------------------


def generate_fibonacci_sphere_mesh(
    nv: int,
    N: int,
    dim: int,
    aspect_ratio=None,
) -> jax.Array:
    """Generate ``N`` Fibonacci-sphere meshes (3D) or circles (2D).

    In 3D the points are laid out by a golden-angle spiral (the sunflower
    / Fibonacci lattice): ``z`` is stratified in ``(-1, 1)`` and the
    azimuth advances by the golden angle on each step. The result is a
    near-optimal, deterministic, low-discrepancy covering of the sphere
    for any ``nv >= 1``. It's the "I want uniform points fast" default
    and a drop-in alternative to Thomson that skips the minimization.

    In 2D the output is ``nv`` evenly-spaced points on the unit circle
    (there's no non-trivial 1D analogue of the spiral).

    The mesh is deterministic, so all ``N`` bodies are identical copies;
    per-body random orientation is typically applied downstream by
    :func:`~jaxdem.utils.particleCreation.distribute_bodies`.
    """
    if nv < 1:
        raise ValueError(f"nv must be >= 1; got nv={nv}.")

    if dim == 3:
        i = np.arange(nv, dtype=np.float64)
        golden_angle = np.pi * (3.0 - np.sqrt(5.0))
        z = 1.0 - (2.0 * i + 1.0) / nv
        r_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        theta = golden_angle * i
        verts_np = np.stack([r_xy * np.cos(theta), r_xy * np.sin(theta), z], axis=-1)
    elif dim == 2:
        theta = 2.0 * np.pi * (np.arange(nv, dtype=np.float64) + 0.5) / nv
        verts_np = np.stack([np.cos(theta), np.sin(theta)], axis=-1)
    else:
        raise ValueError(f"dim must be 2 or 3; got {dim}.")

    axes = _normalize_axes_to_unit(aspect_ratio, dim)
    verts_np = verts_np * axes

    pos = jnp.asarray(verts_np)
    if N == 1:
        return pos
    return jnp.broadcast_to(pos[None, ...], (N, pos.shape[0], dim))


# --------------------------------------------------------------------
# Torus mesh (3D only)
# --------------------------------------------------------------------


def generate_torus_mesh(
    nv: int,
    N: int,
    dim: int = 3,
    tube_ratio: float = 0.3,
    aspect_ratio=None,
) -> jax.Array:
    """Generate ``N`` torus surface meshes with ``nv`` quasi-uniform points each.

    The torus is parameterized by two radii: the major radius ``R`` (center
    of the tube → center of the torus) and the minor radius ``r`` (tube
    half-thickness). ``tube_ratio`` sets ``r`` directly under the "longest
    axis has extent 1" convention: ``r = tube_ratio`` and ``R = 1 - tube_ratio``,
    so the torus fits in ``x, y ∈ [-1, 1]`` and ``z ∈ [-r, r]``.

    Points are placed by stratified angular sampling around the major axis
    (``theta``, evenly spaced) paired with a golden-ratio quasi-random phase
    around the tube (``phi``). This gives good 2D coverage of the torus
    surface for any ``nv``, with a slight over-representation of the inner
    rim relative to the outer rim (exact-uniform sampling would require a
    ``(R + r cos(phi))`` area weighting, which we skip for simplicity).

    Useful for non-convex, genus-1 particles — e.g. studies of interlocking
    or linking in packings where non-convexity matters.
    """
    if dim != 3:
        raise ValueError(f"torus mesh is 3D only; got dim={dim}.")
    if not (0.0 < tube_ratio < 1.0):
        raise ValueError(f"tube_ratio must be in (0, 1); got {tube_ratio}.")
    if nv < 1:
        raise ValueError(f"nv must be >= 1; got nv={nv}.")

    r = float(tube_ratio)
    R = 1.0 - r

    i = np.arange(nv, dtype=np.float64)
    theta = 2.0 * np.pi * (i + 0.5) / nv
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    phi = 2.0 * np.pi * ((i * golden) % 1.0)

    ring = R + r * np.cos(phi)
    x = ring * np.cos(theta)
    y = ring * np.sin(theta)
    z = r * np.sin(phi)
    verts_np = np.stack([x, y, z], axis=-1)

    verts_np = _apply_aspect_ratio_renorm(verts_np, aspect_ratio, dim)

    pos = jnp.asarray(verts_np)
    if N == 1:
        return pos
    return jnp.broadcast_to(pos[None, ...], (N, pos.shape[0], dim))


# --------------------------------------------------------------------
# Helix mesh (3D helix / 2D Archimedean spiral)
# --------------------------------------------------------------------


def generate_helix_mesh(
    nv: int,
    N: int,
    dim: int,
    n_turns: float = 3.0,
    helix_radius: float = 0.3,
    aspect_ratio=None,
) -> jax.Array:
    """Generate ``N`` helical meshes (3D) or Archimedean spirals (2D).

    In 3D the points trace a right-handed helix along the ``z`` axis:
    ``nv`` points evenly spaced in the arc parameter, making ``n_turns``
    full turns from ``z = -1`` to ``z = 1`` on a circle of radius
    ``helix_radius``. This gives chiral, rod-like bodies with a controllable
    pitch; good for studies of enantiomeric packing or helical-fiber clumps.

    In 2D the helix degenerates to an Archimedean spiral centered at the
    origin, with ``nv`` points going from near the origin out to the unit
    circle over ``n_turns`` turns.
    """
    if nv < 2:
        raise ValueError(f"helix mesh requires nv >= 2; got nv={nv}.")
    if n_turns <= 0.0:
        raise ValueError(f"n_turns must be positive; got {n_turns}.")

    t = np.linspace(0.0, 1.0, nv, dtype=np.float64)
    theta = 2.0 * np.pi * float(n_turns) * t

    if dim == 3:
        if not (0.0 < helix_radius <= 1.0):
            raise ValueError(f"helix_radius must be in (0, 1]; got {helix_radius}.")
        x = helix_radius * np.cos(theta)
        y = helix_radius * np.sin(theta)
        z = 2.0 * t - 1.0
        verts_np = np.stack([x, y, z], axis=-1)
    elif dim == 2:
        r = t
        verts_np = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=-1)
    else:
        raise ValueError(f"dim must be 2 or 3; got {dim}.")

    verts_np = _apply_aspect_ratio_renorm(verts_np, aspect_ratio, dim)

    pos = jnp.asarray(verts_np)
    if N == 1:
        return pos
    return jnp.broadcast_to(pos[None, ...], (N, pos.shape[0], dim))


# --------------------------------------------------------------------
# Faceted mesh (regular n-gon in 2D / icosahedron in 3D)
# --------------------------------------------------------------------


def _distribute_extras(n_extra: int, n_slots: int) -> np.ndarray:
    """Return a length-``n_slots`` int array of per-slot asperity counts summing to ``n_extra``.

    Divides ``n_extra`` evenly across ``n_slots`` and parks the remainder on
    the first few slots so they each get exactly one extra.
    """
    base = n_extra // n_slots
    rem = n_extra % n_slots
    counts = np.full((n_slots,), base, dtype=np.int64)
    counts[:rem] += 1
    return counts


def _sample_triangle_quasi_uniform(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray, k: int
) -> np.ndarray:
    """Return ``(k, 3)`` quasi-uniform points strictly inside triangle ``(v0, v1, v2)``.

    Uses the standard barycentric remap ``(a, b, c) = (1-sqrt(u1), sqrt(u1)*(1-u2),
    sqrt(u1)*u2)`` of a 2D quasi-random sequence (stratified ``u1`` with
    golden-angle ``u2``), which gives a uniform distribution on the triangle.
    Points are placed strictly in the interior since ``u1 ∈ (0, 1)``, so they
    never coincide with the existing vertex asperities.
    """
    if k <= 0:
        return np.empty((0, v0.shape[0]), dtype=np.float64)
    i = np.arange(k, dtype=np.float64)
    u1 = (i + 0.5) / k
    golden = (np.sqrt(5.0) - 1.0) / 2.0
    u2 = (i * golden) % 1.0
    sqrt_u1 = np.sqrt(u1)
    a = 1.0 - sqrt_u1
    b = sqrt_u1 * (1.0 - u2)
    c = sqrt_u1 * u2
    return a[:, None] * v0[None, :] + b[:, None] * v1[None, :] + c[:, None] * v2[None, :]


def generate_faceted_mesh(
    nv: int,
    N: int,
    dim: int,
    n_facets: int = 6,
    aspect_ratio=None,
) -> jax.Array:
    """Regular n-gon (2D) or icosahedron (3D) with vertex + surface-filler asperities.

    Unlike the smooth-sphere family (thomson / icosphere / fibonacci), this
    mesh keeps the particle genuinely faceted: the shape is a polygon or
    polyhedron with sharp vertices and flat faces, and the asperities sit
    on those flat features rather than being projected to a circumscribed
    sphere.

    **Asperity layout**

    * Vertex asperities are always placed at the corners of the shape
      (``n_facets`` in 2D, 12 for the icosahedron in 3D).
    * The remaining ``nv - n_vertices`` asperities are distributed uniformly
      across the surface primitives — edges in 2D, triangular face interiors
      in 3D — to achieve an (approximately) uniform surface density. If
      ``nv - n_vertices`` is not divisible by the number of primitives, the
      first few get one extra asperity so the total exactly matches ``nv``.

    In 2D, edge-interior asperities are evenly spaced along each edge
    (excluding the endpoints, which are already vertex asperities).

    In 3D, face-interior asperities are quasi-uniformly sampled inside each
    triangular face via a barycentric-coordinate remap. They stay on the
    flat face — not projected to the circumscribing sphere — so the
    particle's faceted character is preserved (asperities on faces sit
    closer to the center than vertex asperities).

    Parameters
    ----------
    nv : int
        Total number of asperities. Must be ``>= n_facets`` in 2D and
        ``>= 12`` in 3D. Larger ``nv`` → higher surface density.
    N : int
        Number of bodies (all identical copies; per-body orientation is
        handled downstream).
    dim : int
        Spatial dimension (2 or 3).
    n_facets : int
        2D only: number of polygon sides / vertices. Must be ``>= 3``.
        Ignored in 3D (the icosahedron has 12 vertices / 20 faces fixed).
    aspect_ratio : None, scalar, or (dim,) array-like
        Axis stretch in the usual "normalize to max=1" convention.

    Returns
    -------
    jax.Array
        Shape ``(nv, dim)`` if ``N == 1`` else ``(N, nv, dim)``. Vertex
        asperities appear first in the array, followed by edge/face
        fillers grouped by primitive.
    """
    if dim == 2:
        if n_facets < 3:
            raise ValueError(f"n_facets must be >= 3; got {n_facets}.")
        if nv < n_facets:
            raise ValueError(
                f"nv must be >= n_facets (={n_facets}) for a 2D faceted mesh; got nv={nv}."
            )
        angles = 2.0 * np.pi * np.arange(n_facets, dtype=np.float64) / n_facets
        vertices = np.stack([np.cos(angles), np.sin(angles)], axis=-1)

        extras_per_edge = _distribute_extras(nv - n_facets, n_facets)
        points = [vertices]
        for i in range(n_facets):
            k_i = int(extras_per_edge[i])
            if k_i > 0:
                v_start = vertices[i]
                v_end = vertices[(i + 1) % n_facets]
                t = (np.arange(k_i, dtype=np.float64) + 1.0) / (k_i + 1)
                edge_points = v_start[None, :] + t[:, None] * (v_end - v_start)[None, :]
                points.append(edge_points)
        verts_np = np.concatenate(points, axis=0)
    elif dim == 3:
        n_vertex = 12
        n_faces = 20
        if nv < n_vertex:
            raise ValueError(
                f"nv must be >= 12 for a 3D faceted (icosahedron) mesh; got nv={nv}."
            )
        verts = _icosahedron_vertices()
        faces = _icosahedron_faces()
        extras_per_face = _distribute_extras(nv - n_vertex, n_faces)

        points = [verts]
        for i in range(n_faces):
            k_i = int(extras_per_face[i])
            if k_i > 0:
                a, b, c = faces[i]
                face_pts = _sample_triangle_quasi_uniform(
                    verts[a], verts[b], verts[c], k_i
                )
                points.append(face_pts)
        verts_np = np.concatenate(points, axis=0)
    else:
        raise ValueError(f"dim must be 2 or 3; got {dim}.")

    axes = _normalize_axes_to_unit(aspect_ratio, dim)
    verts_np = verts_np * axes

    pos = jnp.asarray(verts_np)
    if N == 1:
        return pos
    return jnp.broadcast_to(pos[None, ...], (N, pos.shape[0], dim))


# --------------------------------------------------------------------
# Uniform arc-length mesh (2D only)
# --------------------------------------------------------------------


def generate_arclength_mesh(
    nv: int,
    N: int,
    dim: int = 2,
    aspect_ratio=None,
    n_fine: int | None = None,
) -> jax.Array:
    """2D mesh with equal arc-length spacing along the ellipse / circle perimeter.

    This is the closed-form analogue of the converged (``n_steps → ∞``) Thomson
    ground state in 2D. On a circle the two coincide exactly — both give the
    regular ``nv``-gon. On an ellipse, "uniform neighbor distance" (= equal
    arc-length spacing) is exactly the ``α → ∞`` packing-problem limit of
    Riesz-energy minimization and a very close approximation to the ``α = 1``
    Thomson ground state. Use this when you'd otherwise run Thomson with a huge
    ``n_steps`` budget on a 2D particle — it arrives at essentially the same
    answer in one numerical integration, deterministically.

    The algorithm inverts the arc-length parameterization of the ellipse
    ``(a cos t, b sin t)`` numerically: trapezoidal-rule the arclength
    integrand on a fine grid, then linearly interpolate the angles
    corresponding to ``nv`` equally spaced target arc lengths. Accuracy is
    controlled by ``n_fine`` (auto-sized to ``max(10_000, 200 * nv)`` by default).

    Parameters
    ----------
    nv : int
        Number of surface vertices. Must be ``>= 3``.
    N : int
        Number of bodies (all identical copies; per-body orientation is
        handled downstream).
    dim : int
        Must be 2. In 3D "uniform neighbor distance" is generically the
        multi-point Thomson problem — use :func:`generate_thomson_mesh`.
    aspect_ratio : None, scalar, or (2,) array-like
        Ellipse semi-axes in the usual "normalize to max=1" convention.
        ``None`` gives a circle.
    n_fine : int, optional
        Grid resolution for the arc-length integral. ``None`` auto-sizes.

    Returns
    -------
    jax.Array
        Shape ``(nv, 2)`` if ``N == 1`` else ``(N, nv, 2)``.
    """
    if dim != 2:
        raise ValueError(
            f"arclength mesh is 2D only; got dim={dim}. "
            "Use mesh_type='thomson' in 3D for the equivalent ground state."
        )
    if nv < 3:
        raise ValueError(f"arclength mesh requires nv >= 3; got nv={nv}.")

    axes = _normalize_axes_to_unit(aspect_ratio, dim)
    a = float(axes[0])
    b = float(axes[1])

    # Circle shortcut: arc-length-uniform is exactly equal angular spacing.
    if np.isclose(a, b):
        theta = 2.0 * np.pi * (np.arange(nv, dtype=np.float64) + 0.5) / nv
        verts_np = np.stack([a * np.cos(theta), b * np.sin(theta)], axis=-1)
    else:
        if n_fine is None:
            n_fine = max(10_000, 200 * nv)
        t_fine = np.linspace(0.0, 2.0 * np.pi, n_fine + 1)
        # Ellipse arc-length integrand: ds/dt = sqrt((a sin t)^2 + (b cos t)^2).
        ds_dt = np.sqrt((a * np.sin(t_fine)) ** 2 + (b * np.cos(t_fine)) ** 2)
        dt = t_fine[1] - t_fine[0]
        # Cumulative arc length at each t_fine via trapezoidal rule.
        s_cumulative = np.concatenate(
            [[0.0], np.cumsum(0.5 * (ds_dt[:-1] + ds_dt[1:]) * dt)]
        )
        L_total = float(s_cumulative[-1])
        s_targets = L_total * (np.arange(nv, dtype=np.float64) + 0.5) / nv
        t_targets = np.interp(s_targets, s_cumulative, t_fine)
        verts_np = np.stack(
            [a * np.cos(t_targets), b * np.sin(t_targets)], axis=-1
        )

    pos = jnp.asarray(verts_np)
    if N == 1:
        return pos
    return jnp.broadcast_to(pos[None, ...], (N, pos.shape[0], dim))
