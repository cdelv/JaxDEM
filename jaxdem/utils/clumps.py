# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation

from typing import TYPE_CHECKING
from functools import partial

from . import Quaternion
from .linalg import norm2

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..materials import MaterialTable


@partial(jax.jit, static_argnames=("n", "dim"), inline=True)
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

    n_batches = max(1, (n_samples + sample_batch_size - 1) // sample_batch_size)
    effective_samples = n_batches * sample_batch_size
    points_u = _generate_golden_lattice(effective_samples, dim=dim).reshape(
        n_batches, sample_batch_size, dim
    )

    # ---------- Phase 1: per-clump Monte Carlo accumulation ----------
    def phase1_single(positions: jax.Array, radii: jax.Array):
        bb_min = jnp.min(positions - radii[:, None], axis=0)
        bb_max = jnp.max(positions + radii[:, None], axis=0)
        box_range = bb_max - bb_min
        box_vol = jnp.prod(box_range)
        radii_sq = jnp.square(radii)

        def accumulate(carry, pts_u):
            count, sum_pos, sum_r_sq, sum_outer = carry
            pts = bb_min + pts_u * box_range
            diff = pts[:, None, :] - positions[None, :, :]
            inside = jnp.any(norm2(diff) < radii_sq[None, :], axis=-1).astype(
                positions.dtype
            )
            count = count + jnp.sum(inside)
            sum_pos = sum_pos + jnp.sum(pts * inside[:, None], axis=0)
            sum_r_sq = sum_r_sq + jnp.sum(inside * norm2(pts))
            sum_outer = sum_outer + jnp.einsum("n,ni,nj->ij", inside, pts, pts)
            return (count, sum_pos, sum_r_sq, sum_outer), None

        init = (
            jnp.zeros((), dtype=positions.dtype),
            jnp.zeros((dim,), dtype=positions.dtype),
            jnp.zeros((), dtype=positions.dtype),
            jnp.zeros((dim, dim), dtype=positions.dtype),
        )
        (count, sum_pos, sum_r_sq, sum_outer), _ = jax.lax.scan(
            accumulate, init, points_u
        )
        return count, sum_pos, sum_r_sq, sum_outer, box_vol

    phase1_vmapped = jax.vmap(phase1_single)

    if n_clumps <= clump_batch_size:
        # Fast path: skip outer scan + padding entirely.
        count, sum_pos, sum_r_sq, sum_outer, box_vol = phase1_vmapped(pos, rad)
    else:
        # Pad to a multiple of clump_batch_size by duplicating the first clump,
        # reshape into (n_chunks, clump_batch_size, ...), scan over chunks, then
        # slice the pad off the outputs.
        pad = (-n_clumps) % clump_batch_size
        if pad:
            pos_pad = jnp.broadcast_to(pos[:1], (pad, nv, dim))
            rad_pad = jnp.broadcast_to(rad[:1], (pad, nv))
            pos_padded = jnp.concatenate([pos, pos_pad], axis=0)
            rad_padded = jnp.concatenate([rad, rad_pad], axis=0)
        else:
            pos_padded = pos
            rad_padded = rad

        n_padded = n_clumps + pad
        n_chunks = n_padded // clump_batch_size

        pos_c = pos_padded.reshape(n_chunks, clump_batch_size, nv, dim)
        rad_c = rad_padded.reshape(n_chunks, clump_batch_size, nv)

        def phase1_scan_body(carry, clump_chunk):
            positions, radii = clump_chunk
            return carry, phase1_vmapped(positions, radii)

        _, (count, sum_pos, sum_r_sq, sum_outer, box_vol) = jax.lax.scan(
            phase1_scan_body, None, (pos_c, rad_c)
        )
        count = count.reshape(n_padded)[:n_clumps]
        sum_pos = sum_pos.reshape(n_padded, dim)[:n_clumps]
        sum_r_sq = sum_r_sq.reshape(n_padded)[:n_clumps]
        sum_outer = sum_outer.reshape(n_padded, dim, dim)[:n_clumps]
        box_vol = box_vol.reshape(n_padded)[:n_clumps]

    # ---------- Phase 2: per-clump finalization (no n_samples dim) ----------
    def phase2_single(count, sum_pos, sum_r_sq, sum_outer, box_vol, total_mass):
        sample_volume = box_vol / effective_samples
        volume = count * sample_volume
        com = sum_pos * sample_volume / volume
        density = total_mass / volume
        cov = sum_outer * sample_volume - volume * jnp.outer(com, com)

        if dim == 3:
            inertia_origin = (
                sum_r_sq * jnp.eye(3, dtype=pos.dtype) - sum_outer
            ) * sample_volume
            inertia = density * (
                inertia_origin
                - volume
                * (jnp.sum(com**2) * jnp.eye(3, dtype=pos.dtype) - jnp.outer(com, com))
            )
            inertia = 0.5 * (inertia + inertia.T)
            eigvals, eigvecs = jnp.linalg.eigh(inertia)
            sign = jnp.sign(jnp.linalg.det(eigvecs))
            eigvecs = eigvecs.at[:, -1].set(eigvecs[:, -1] * sign)
            rot = Rotation.from_matrix(eigvecs)
            q_xyzw = rot.as_quat()
            q = jnp.concatenate([q_xyzw[3:4], q_xyzw[:3]])
            return volume, com, eigvals, q

        theta = jnp.arctan2(cov[1, 0], cov[0, 0] - cov[1, 1]) / 2.0
        q = jnp.array(
            [jnp.cos(theta), 0.0, 0.0, jnp.sin(theta)],
            dtype=pos.dtype,
        )
        inertia_origin = sum_r_sq * sample_volume
        inertia = density * (inertia_origin - volume * jnp.sum(com**2))
        return volume, com, inertia.reshape(1), q

    volume, com, inertia, q = jax.vmap(phase2_single)(
        count, sum_pos, sum_r_sq, sum_outer, box_vol, clump_mass
    )
    quat = Quaternion(q[:, None, 0:1], q[:, None, 1:])
    pos_p = Quaternion.rotate_back(quat, pos - com[:, None, :])

    if single_clump:
        return volume[0], com[0], inertia[0], q[0], pos_p[0]
    return volume, com, inertia, q, pos_p


@partial(jax.jit, static_argnames=("n_samples",))
def compute_clump_properties(
    state: State, mat_table: MaterialTable, n_samples: int = 50_000
) -> State:
    dim = state.dim
    clump_ids = jnp.arange(state.N)
    counts = jnp.bincount(state.clump_id, length=state.N)
    points_u = _generate_golden_lattice(n_samples, dim=state.dim)
    pos = state.pos

    def solve_monte_carlo(c_id: jax.Array) -> tuple[jax.Array, ...]:
        is_in_clump = state.clump_id == c_id

        # --- Bounding Box & Points ---
        inf = jnp.inf
        local_min = pos - state.rad[:, None]
        local_max = pos + state.rad[:, None]

        clump_min_b = jnp.min(jnp.where(is_in_clump[:, None], local_min, inf), axis=0)
        clump_max_b = jnp.max(jnp.where(is_in_clump[:, None], local_max, -inf), axis=0)

        box_vol = jnp.prod(clump_max_b - clump_min_b)
        points = clump_min_b + points_u * (clump_max_b - clump_min_b)

        # --- Filter Logic ---
        eff_rad = jnp.where(is_in_clump, state.rad, 0.0)
        eff_densities = jnp.where(is_in_clump, mat_table.density[state.mat_id], 0.0)

        diff = points[:, None, :] - pos[None, :, :]
        dists_sq = norm2(diff)
        inside_mask = dists_sq < jnp.square(eff_rad[None, :])

        densities_per_point = jnp.where(inside_mask, eff_densities[None, :], 0.0)
        rho = jnp.max(densities_per_point, axis=-1)

        # --- Mass & COM ---
        vol_per_sample = box_vol / n_samples
        total_mass = jnp.sum(rho) * vol_per_sample

        rho_r = points * rho[:, None]
        com = jnp.sum(rho_r, axis=0) * vol_per_sample / total_mass

        # --- Inertia & Orientation ---
        r_prime = points - com
        r_sq = norm2(r_prime)

        if dim == 3:
            term1 = jnp.sum(
                rho[:, None, None] * r_sq[:, None, None] * jnp.eye(3)[None, :, :],
                axis=0,
            )
            term2 = jnp.einsum("n,ni,nj->ij", rho, r_prime, r_prime)
            i_tensor = (term1 - term2) * vol_per_sample

            i_tensor = 0.5 * (i_tensor + i_tensor.T)
            eigvals, eigvecs = jnp.linalg.eigh(i_tensor)

            rot = Rotation.from_matrix(eigvecs)
            q_xyzw = rot.as_quat()
            q_update = jnp.concatenate([q_xyzw[3:4], q_xyzw[:3]])

            return total_mass, com, eigvals, q_update

        # 2D Case: Use Covariance Matrix to determine orientation
        cov = jnp.einsum("n,ni,nj->ij", rho, r_prime, r_prime) * vol_per_sample
        _eigvals_cov, eigvecs = jnp.linalg.eigh(cov)

        # Convert 2D rotation matrix (eigvecs) to angle theta
        # Column 0 is the new X-axis
        theta = jnp.arctan2(eigvecs[1, 0], eigvecs[0, 0])

        # Convert angle to Quaternion (rotation around Z)
        half_theta = theta / 2.0
        q_update = jnp.array([jnp.cos(half_theta), 0.0, 0.0, jnp.sin(half_theta)])

        # Scalar polar moment of inertia
        i_scalar = jnp.sum(rho * r_sq) * vol_per_sample
        i_res = i_scalar.reshape(1)

        return total_mass, com, i_res, q_update

    tm, cm, it, qt = jax.vmap(solve_monte_carlo)(clump_ids)
    is_clump = counts[state.clump_id] > 1

    new_mass = jnp.where(is_clump, tm[state.clump_id], state.mass)
    new_com = jnp.where(is_clump[:, None], cm[state.clump_id], state.pos_c)
    new_inertia = jnp.where(is_clump[:, None], it[state.clump_id], state.inertia)

    new_q_arr = jnp.where(
        is_clump[:, None],
        qt[state.clump_id],
        jnp.concatenate([state.q.w, state.q.xyz], axis=-1),
    )

    state.mass = new_mass
    state.pos_c = new_com
    state.inertia = new_inertia
    state.q = Quaternion(new_q_arr[..., 0:1], new_q_arr[..., 1:])
    state.pos_p = state.q.rotate_back(state.q, pos - state.pos_c)

    return state
