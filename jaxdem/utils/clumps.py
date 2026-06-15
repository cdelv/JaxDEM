# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from typing import TYPE_CHECKING, Any
from functools import partial

from dataclasses import replace
from .quaternion import Quaternion
from .linalg import norm2

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..materials import MaterialTable


@jax.jit(static_argnames=("n", "dim"), inline=True)
def _generate_golden_lattice(n: int, dim: int = 2) -> jax.Array:
    phi = 1.32471795724475 if dim == 2 else 1.22074408460576
    exponents = 1.0 + jax.lax.iota(int, size=dim)
    alphas = 1.0 / jnp.power(phi, exponents)
    ids = 1.0 + jax.lax.iota(int, size=n)
    return (0.5 + jnp.outer(ids, alphas)) % 1.0


def _compute_uniform_union_properties(
    pos: jax.Array,
    rad: jax.Array,
    clump_mass: jax.Array,
    n_samples: int = 50_000,
    sample_batch_size: int = 4096,
    clump_batch_size: int = 32,
) -> tuple[jax.Array, ...]:
    """Compute rigid-body properties of clumps defined as unions of overlapping
    spheres (3D) or disks (2D), under the assumption of **uniform mass density
    throughout the clump volume**.

    Each clump is described by ``nv`` sphere-vertices (centers ``pos`` and
    radii ``rad``); the clump itself is the *union* of those spheres.  The
    total clump mass is fixed (``clump_mass``) and assumed to be distributed
    uniformly over the union volume, so the mass density is simply
    ``clump_mass / volume``.  Overlaps between sphere-vertices are handled
    correctly: the union volume (and therefore the density) never
    double-counts overlapped regions.

    For each clump the function returns:

    * ``volume``  -- union volume, estimated by Monte Carlo integration over
      the clump's bounding box.
    * ``com``     -- center of mass, in the input/world frame.
    * ``inertia`` -- principal moments of inertia (diagonal entries of the
      inertia tensor in the body frame): length-3 in 3D, length-1 in 2D.
    * ``q``       -- unit quaternion ``[w, x, y, z]`` rotating from the body
      frame (principal axes) to the world frame.
    * ``pos_p``   -- each sphere-vertex center in the body frame, i.e.
      ``pos - com`` rotated by ``q^{-1}``.  These are what downstream rigid-
      body integrators actually consume: COM-relative, axis-aligned positions
      that stay constant under rigid motion.

    These outputs are the full set of properties a rigid-body simulator needs
    for each clump: mass is given, volume / density come from here, COM and
    inertia set up the equations of motion, and ``q`` + ``pos_p`` let the
    integrator place each sphere-vertex at any pose during a simulation.

    Accuracy scales with ``n_samples``; ``n_samples = 10_000_000`` gives
    noise-free results on typical clumps, ``50_000`` (the default) is useful
    for quick sanity checks.

    Input shapes:
        pos:        (N, nv, dim) or (nv, dim) for a single clump
        rad:        (N, nv), (nv,), or scalar
        clump_mass: (N,) or scalar

    Returns:
        volume:  (N,)
        com:     (N, dim)
        inertia: (N, 3) in 3D or (N, 1) in 2D
        q:       (N, 4)
        pos_p:   (N, nv, dim)

    The computation is split into two phases:

    * **Phase 1** (Monte Carlo accumulation) runs a per-clump kernel that
      scans over sample batches of ``sample_batch_size`` and is vmapped
      across ``clump_batch_size`` clumps per outer ``jax.lax.scan`` step.
      Only per-clump aggregates (``count``, ``sum_pos``, ``sum_r_sq``,
      ``sum_outer``, ``box_vol``) are produced; the ``n_samples`` axis is
      fully reduced away.  When ``N <= clump_batch_size`` the outer scan and
      any clump-axis padding are skipped in favour of a single vmap.
    * **Phase 2** vmaps the O(1)-per-clump finalization (volume / COM /
      inertia / quaternion) over all N clumps.  Small, fast kernel.

    Both batch sizes should be powers of two.  The defaults
    ``sample_batch_size=4096`` and ``clump_batch_size=32`` are a good
    starting point on modern GPUs; they are exposed here so callers can
    tune memory footprint and per-kernel occupancy for their hardware.
    """
    pos = jnp.asarray(pos, dtype=float)
    single_clump = pos.ndim == 2
    if single_clump:
        pos = pos[None, ...]
    if pos.ndim != 3:
        raise ValueError("pos must have shape (N, nv, dim) or (nv, dim).")

    n_clumps, nv, dim = pos.shape

    rad = jnp.asarray(rad, dtype=pos.dtype)
    if rad.ndim == 0:
        rad = jnp.full((n_clumps, nv), rad, dtype=pos.dtype)
    elif rad.ndim == 1:
        if rad.shape != (nv,):
            raise ValueError(f"Expected rad to have shape ({nv},), got {rad.shape}.")
        rad = jnp.broadcast_to(rad[None, :], (n_clumps, nv))
    elif rad.ndim == 2:
        if single_clump and rad.shape == (nv, 1):
            rad = rad.reshape(1, nv)
        elif rad.shape != (n_clumps, nv):
            raise ValueError(
                f"Expected rad to have shape ({n_clumps}, {nv}), got {rad.shape}."
            )
    else:
        raise ValueError("rad must be scalar, (nv,), or (N, nv).")

    clump_mass = jnp.asarray(clump_mass, dtype=pos.dtype)
    if clump_mass.ndim == 0:
        clump_mass = jnp.full((n_clumps,), clump_mass, dtype=pos.dtype)
    elif clump_mass.shape != (n_clumps,):
        raise ValueError(
            f"Expected clump_mass to have shape ({n_clumps},), got {clump_mass.shape}."
        )

    if bool(jnp.any(rad <= 0.0)):
        raise ValueError("All radii must be strictly positive.")
    if bool(jnp.any(clump_mass <= 0.0)):
        raise ValueError("All clump masses must be strictly positive.")
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    if sample_batch_size <= 0:
        raise ValueError("sample_batch_size must be positive.")
    if clump_batch_size <= 0:
        raise ValueError("clump_batch_size must be positive.")

    # single spheres are treated analytically
    if nv == 1:
        sphere_pos = pos[:, 0, :]  # (n_clumps, dim)
        sphere_rad = rad[:, 0]  # (n_clumps,)
        if dim == 3:
            volume = (4.0 / 3.0) * jnp.pi * sphere_rad**3
            inertia_scalar = 0.4 * clump_mass * sphere_rad**2  # 2/5 m r^2
            inertia = jnp.broadcast_to(inertia_scalar[:, None], (n_clumps, 3))
        else:
            volume = jnp.pi * sphere_rad**2
            inertia_scalar = 0.5 * clump_mass * sphere_rad**2
            inertia = inertia_scalar[:, None]
        com = sphere_pos
        q = jnp.tile(jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=pos.dtype), (n_clumps, 1))
        pos_p = jnp.zeros_like(pos)
        if single_clump:
            return volume[0], com[0], inertia[0], q[0], pos_p[0]
        return volume, com, inertia, q, pos_p

    volume, _mass, com, inertia, q, pos_p = _union_properties_kernel(
        pos,
        rad,
        jnp.ones_like(rad),
        clump_mass,
        n_samples,
        sample_batch_size,
        clump_batch_size,
    )

    if single_clump:
        return volume[0], com[0], inertia[0], q[0], pos_p[0]
    return volume, com, inertia, q, pos_p


@partial(
    jax.jit,
    static_argnames=(
        "n_samples",
        "sample_batch_size",
        "clump_batch_size",
    ),
)
def _union_properties_kernel(
    pos: jax.Array,
    rad: jax.Array,
    vertex_density: jax.Array,
    clump_mass: jax.Array | None,
    n_samples: int,
    sample_batch_size: int,
    clump_batch_size: int,
) -> tuple[jax.Array, ...]:
    """Shared Monte Carlo kernel for sphere/disk-union rigid-body properties.

    Each sample point is weighted by the *maximum* density of the
    sphere-vertices containing it (``vertex_density``, shape ``(N, nv)``),
    so overlapping regions are never double-counted. If ``clump_mass`` is
    ``None`` the total mass is the Monte Carlo mass integral
    ``sum(rho) * sample_volume``; otherwise the given mass is distributed
    proportionally to the sampled density field (for uniform
    ``vertex_density`` this is exactly the uniform-density assumption of
    :func:`_compute_uniform_union_properties`).

    Returns ``(volume, mass, com, inertia, q, pos_p)`` with one leading
    clump axis.
    """
    n_clumps, nv, dim = pos.shape
    n_batches = max(1, (n_samples + sample_batch_size - 1) // sample_batch_size)
    effective_samples = n_batches * sample_batch_size
    points_u = _generate_golden_lattice(effective_samples, dim=dim).reshape(
        n_batches, sample_batch_size, dim
    )

    # ---------- Phase 1: per-clump Monte Carlo accumulation ----------
    def phase1_single(
        positions: jax.Array, radii: jax.Array, densities: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        bb_min = jnp.min(positions - radii[:, None], axis=0)
        bb_max = jnp.max(positions + radii[:, None], axis=0)
        box_range = bb_max - bb_min
        box_vol = jnp.prod(box_range)
        radii_sq = jnp.square(radii)

        def accumulate(
            carry: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
            pts_u: jax.Array,
        ) -> tuple[tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], None]:
            count, sum_rho, sum_pos, sum_r_sq, sum_outer = carry
            pts = bb_min + pts_u * box_range
            diff = pts[:, None, :] - positions[None, :, :]
            inside_v = norm2(diff) < radii_sq[None, :]
            inside = jnp.any(inside_v, axis=-1).astype(positions.dtype)
            # Per-sample density: max density of the containing vertices,
            # so overlapping regions are never double-counted.
            rho = jnp.max(inside_v * densities[None, :], axis=-1)
            count = count + jnp.sum(inside)
            sum_rho = sum_rho + jnp.sum(rho)
            sum_pos = sum_pos + jnp.sum(pts * rho[:, None], axis=0)
            sum_r_sq = sum_r_sq + jnp.sum(rho * norm2(pts))
            sum_outer = sum_outer + jnp.einsum("n,ni,nj->ij", rho, pts, pts)
            return (count, sum_rho, sum_pos, sum_r_sq, sum_outer), None

        init = (
            jnp.zeros((), dtype=positions.dtype),
            jnp.zeros((), dtype=positions.dtype),
            jnp.zeros((dim,), dtype=positions.dtype),
            jnp.zeros((), dtype=positions.dtype),
            jnp.zeros((dim, dim), dtype=positions.dtype),
        )
        (count, sum_rho, sum_pos, sum_r_sq, sum_outer), _ = jax.lax.scan(
            accumulate, init, points_u
        )
        return count, sum_rho, sum_pos, sum_r_sq, sum_outer, box_vol

    phase1_vmapped = jax.vmap(phase1_single)

    if n_clumps <= clump_batch_size:
        # Fast path: skip outer scan + padding entirely.
        count, sum_rho, sum_pos, sum_r_sq, sum_outer, box_vol = phase1_vmapped(
            pos, rad, vertex_density
        )
    else:
        # Pad to a multiple of clump_batch_size by duplicating the first clump,
        # reshape into (n_chunks, clump_batch_size, ...), scan over chunks, then
        # slice the pad off the outputs.
        pad = (-n_clumps) % clump_batch_size
        if pad:
            pos_pad = jnp.broadcast_to(pos[:1], (pad, nv, dim))
            rad_pad = jnp.broadcast_to(rad[:1], (pad, nv))
            den_pad = jnp.broadcast_to(vertex_density[:1], (pad, nv))
            pos_padded = jnp.concatenate([pos, pos_pad], axis=0)
            rad_padded = jnp.concatenate([rad, rad_pad], axis=0)
            den_padded = jnp.concatenate([vertex_density, den_pad], axis=0)
        else:
            pos_padded = pos
            rad_padded = rad
            den_padded = vertex_density

        n_padded = n_clumps + pad
        n_chunks = n_padded // clump_batch_size

        pos_c = pos_padded.reshape(n_chunks, clump_batch_size, nv, dim)
        rad_c = rad_padded.reshape(n_chunks, clump_batch_size, nv)
        den_c = den_padded.reshape(n_chunks, clump_batch_size, nv)

        def phase1_scan_body(
            carry: Any, clump_chunk: tuple[jax.Array, jax.Array, jax.Array]
        ) -> tuple[
            Any,
            tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
        ]:
            positions, radii, densities = clump_chunk
            return carry, phase1_vmapped(positions, radii, densities)

        _, (count, sum_rho, sum_pos, sum_r_sq, sum_outer, box_vol) = jax.lax.scan(
            phase1_scan_body, None, (pos_c, rad_c, den_c)
        )
        count = count.reshape(n_padded)[:n_clumps]
        sum_rho = sum_rho.reshape(n_padded)[:n_clumps]
        sum_pos = sum_pos.reshape(n_padded, dim)[:n_clumps]
        sum_r_sq = sum_r_sq.reshape(n_padded)[:n_clumps]
        sum_outer = sum_outer.reshape(n_padded, dim, dim)[:n_clumps]
        box_vol = box_vol.reshape(n_padded)[:n_clumps]

    # ---------- Phase 2: per-clump finalization (no n_samples dim) ----------
    sample_volume = box_vol / effective_samples  # (n_clumps,)
    mass_mc = sum_rho * sample_volume  # Monte Carlo mass integral
    total_mass = mass_mc if clump_mass is None else clump_mass

    def phase2_single(
        count: jax.Array,
        sum_rho: jax.Array,
        sum_pos: jax.Array,
        sum_r_sq: jax.Array,
        sum_outer: jax.Array,
        sample_volume: jax.Array,
        mass_mc: jax.Array,
        total_mass: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        volume = count * sample_volume
        com = sum_pos / sum_rho
        # Rescale the density-weighted Monte Carlo moments so they integrate
        # to the requested total mass (scale == density for uniform unions).
        scale = total_mass / mass_mc

        if dim == 3:
            eye = jnp.eye(3, dtype=pos.dtype)
            inertia_origin = (sum_r_sq * eye - sum_outer) * sample_volume
            inertia = scale * (
                inertia_origin - mass_mc * (jnp.sum(com**2) * eye - jnp.outer(com, com))
            )
            inertia = 0.5 * (inertia + inertia.T)
            eigvals, eigvecs = jnp.linalg.eigh(inertia)
            sign = jnp.sign(jnp.linalg.det(eigvecs))
            eigvecs = eigvecs.at[:, -1].set(eigvecs[:, -1] * sign)
            rot = Rotation.from_matrix(eigvecs)
            q_xyzw = rot.as_quat()
            q = jnp.concatenate([q_xyzw[3:4], q_xyzw[:3]])
            return volume, com, eigvals, q

        # 2D: principal angle of the COM-frame second-moment matrix.
        cov = sum_outer * sample_volume - mass_mc * jnp.outer(com, com)
        theta = jnp.arctan2(2.0 * cov[1, 0], cov[0, 0] - cov[1, 1]) / 2.0
        half_theta = 0.5 * theta
        q = jnp.array(
            [jnp.cos(half_theta), 0.0, 0.0, jnp.sin(half_theta)],
            dtype=pos.dtype,
        )
        inertia = scale * (sum_r_sq * sample_volume - mass_mc * jnp.sum(com**2))
        return volume, com, inertia.reshape(1), q

    volume, com, inertia, q = jax.vmap(phase2_single)(
        count, sum_rho, sum_pos, sum_r_sq, sum_outer, sample_volume, mass_mc, total_mass
    )
    quat = Quaternion(q[:, None, 0:1], q[:, None, 1:])
    pos_p = Quaternion.rotate_back(quat, pos - com[:, None, :])

    return volume, total_mass, com, inertia, q, pos_p


def compute_clump_properties(
    state: State,
    mat_table: MaterialTable,
    n_samples: int = 50_000,
    sample_batch_size: int = 4096,
    clump_batch_size: int = 32,
) -> State:
    """Compute mass / COM / inertia / orientation for every multi-sphere clump.

    Each clump (group of spheres sharing ``state.clump_id``) is treated as
    the union of its spheres with per-vertex material density
    ``mat_table.density[state.mat_id]`` (overlaps take the maximum density,
    never double-counted). Properties are obtained by Monte Carlo
    integration via the shared batched kernel
    :func:`_union_properties_kernel`; single-sphere clumps keep their
    existing analytic state values.

    This function performs host-side grouping of spheres into clumps, so it
    cannot be wrapped in ``jax.jit`` itself; the heavy Monte Carlo kernel it
    calls is jitted.
    """
    dim = state.dim
    pos = state.pos
    n_groups = int(jnp.max(state.clump_id)) + 1
    counts = jnp.bincount(state.clump_id, length=n_groups)
    is_clump = counts[state.clump_id] > 1

    multi_ids = jnp.where(counts > 1)[0]
    if multi_ids.shape[0] == 0:
        return state

    # Build a (n_groups, nv_max) table of sphere indices per clump. Padding
    # slots repeat the clump's first sphere: a duplicated sphere does not
    # change the union, so the kernel result is unaffected.
    nv_max = int(jnp.max(counts))
    order = jnp.argsort(state.clump_id)
    sorted_ids = state.clump_id[order]
    starts = jnp.concatenate(
        [jnp.zeros((1,), dtype=counts.dtype), jnp.cumsum(counts[:-1])]
    )
    slot = jnp.arange(state.N) - starts[sorted_ids]
    idx_table = jnp.zeros((n_groups, nv_max), dtype=int).at[sorted_ids, slot].set(order)
    valid_slot = jnp.arange(nv_max)[None, :] < counts[:, None]
    idx_table = jnp.where(valid_slot, idx_table, idx_table[:, :1])

    sel = idx_table[multi_ids]  # (n_multi, nv_max)
    volume_g, mass_g, com_g, inertia_g, q_g, _pos_p_g = _union_properties_kernel(
        pos[sel],
        state.rad[sel],
        mat_table.density[state.mat_id][sel],
        None,
        n_samples,
        sample_batch_size,
        clump_batch_size,
    )

    inertia_width = inertia_g.shape[-1]
    tm = jnp.zeros((n_groups,), dtype=pos.dtype).at[multi_ids].set(mass_g)
    cm = jnp.zeros((n_groups, dim), dtype=pos.dtype).at[multi_ids].set(com_g)
    it = (
        jnp.zeros((n_groups, inertia_width), dtype=pos.dtype)
        .at[multi_ids]
        .set(inertia_g)
    )
    qt = jnp.zeros((n_groups, 4), dtype=pos.dtype).at[multi_ids].set(q_g)

    new_mass = jnp.where(is_clump, tm[state.clump_id], state.mass)
    new_com = jnp.where(is_clump[:, None], cm[state.clump_id], state.pos_c)
    new_inertia = jnp.where(is_clump[:, None], it[state.clump_id], state.inertia)

    new_q_arr = jnp.where(
        is_clump[:, None],
        qt[state.clump_id],
        jnp.concatenate([state.q.w, state.q.xyz], axis=-1),
    )

    new_q = Quaternion(new_q_arr[..., 0:1], new_q_arr[..., 1:])
    new_pos_p = new_q.rotate_back(new_q, pos - new_com)
    state = replace(
        state,
        mass=new_mass,
        pos_c=new_com,
        inertia=new_inertia,
        q=new_q,
        pos_p=new_pos_p,
    )

    return state
