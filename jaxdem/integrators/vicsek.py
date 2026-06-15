# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Vicsek-style integrators (extrinsic and intrinsic noise)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import jax
import jax.numpy as jnp

from ..colliders.neighbor_list import NeighborList as NeighborListCollider
from ..utils.linalg import cross, dot, norm2, unit
from . import LinearIntegrator, free_mask

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


def _vicsek_alignment(
    state: State,
    system: System,
    neighbor_radius: jax.Array,
    max_neighbors: int,
) -> tuple[State, System, jax.Array, jax.Array]:
    """Shared Vicsek pre-processing: neighbor query and clump-wise averages.

    Builds the neighbor list, averages neighbor velocities (including self),
    and reduces both the alignment vector and the accumulated force to a single
    clump-wise value broadcast to all clump members.

    Returns
    -------
    Tuple[State, System, jax.Array, jax.Array]
        The (possibly reordered) state and system, the clump-broadcast force,
        and the clump-broadcast average neighbor velocity.
    """
    # Neighbor list query. Some colliders may sort the returned state.
    state, system, nl, _overflow = system.collider.create_neighbor_list(
        state, system, neighbor_radius, max_neighbors
    )

    pos = state.pos
    vel = state.vel

    # Gather neighbor velocities (with padding-safe indexing).
    safe_nl = jnp.maximum(nl, 0)
    nbr_vel = vel[safe_nl]  # (N, K, dim)
    valid = nl != -1

    # If the collider neighbor list radius includes skin, filter by true Vicsek radius.
    # This is a compile-time branch (resolved when tracing) based on collider type.
    if isinstance(system.collider, NeighborListCollider):
        dr = system.domain._displacement(pos[:, None, :], pos[safe_nl], system)
        dist_sq = norm2(dr)
        r2 = neighbor_radius**2
        valid = valid & (dist_sq <= r2)

    # Average neighbor velocity including self.
    sum_v = jnp.sum(nbr_vel * valid[..., None], axis=1) + vel
    count = jnp.sum(valid, axis=1) + 1
    avg_v = sum_v / count[..., None]

    # Clump-wise average of avg_v (so each clump has a single alignment vector).
    counts = jnp.bincount(state.clump_id, length=state.N)
    counts_safe = jnp.where(counts > 0, counts, 1).astype(avg_v.dtype)
    avg_v_clump = (
        jax.ops.segment_sum(avg_v, state.clump_id, num_segments=state.N)
        / counts_safe[..., None]
    )
    avg_v = avg_v_clump[state.clump_id]

    # Clump-wise force (state.force is expected to already be clump-broadcasted,
    # but compute robustly anyway).
    force_clump = (
        jax.ops.segment_sum(state.force, state.clump_id, num_segments=state.N)
        / counts_safe[..., None]
    )
    force = force_clump[state.clump_id]

    return state, system, force, avg_v


@jax.jit(inline=True)
def _apply_desired_velocity(
    state: State, system: System, v_des: jax.Array
) -> tuple[State, System]:
    """Shared Vicsek tail: set free particles' velocities and advance positions."""
    state.vel = v_des * free_mask(state)
    state.pos_c = state.pos_c + system.dt * state.vel
    return state, system


@LinearIntegrator.register("vicsek_extrinsic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class VicsekExtrinsic(LinearIntegrator):
    """Vicsek-model integrator with **extrinsic** (vectorial) noise.

    This integrator implements a Vicsek-like update rule by directly setting the
    translational velocity magnitude to ``v0`` each step, based on the direction
    of a vector that combines:

    - the current accumulated force vector (from colliders + force functions),
    - the average neighbor velocity direction (including self),
    - an additive random unit vector scaled by ``eta`` (extrinsic noise).

    Notes
    -----
    - Noise is generated **per clump** (one sample per rigid body) and then
      broadcast to all clump members so clumps move coherently.
    - Neighbor lists may be cached (e.g., NeighborList collider) or may sort the
      state (e.g., some cell-list builders). This integrator uses the returned
      state from ``create_neighbor_list`` for consistency.

    """

    neighbor_radius: jax.Array
    eta: jax.Array
    v0: jax.Array
    max_neighbors: int = jax.tree.static()

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="VicsekExtrinsic.step_after_force")
    def step_after_force(state: State, system: System) -> tuple[State, System]:
        vicsek = cast(VicsekExtrinsic, system.linear_integrator)

        state, system, force, avg_v = _vicsek_alignment(
            state, system, vicsek.neighbor_radius, vicsek.max_neighbors
        )

        # Random unit vector per clump (extrinsic/vectorial noise).
        system.key, noise_key = jax.random.split(system.key)
        if state.dim == 2:
            angles = jax.random.uniform(
                noise_key, shape=(state.N,), minval=0.0, maxval=2.0 * jnp.pi
            )
            unit_clump = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        else:
            # Isotropic direction in R^3 by normalizing a Gaussian vector.
            raw = jax.random.normal(
                noise_key, shape=(state.N, 3), dtype=state.vel.dtype
            )
            unit_clump = unit(raw)
        noise_dir = unit_clump[state.clump_id]

        f = force + avg_v + vicsek.eta * noise_dir
        v_des = vicsek.v0 * unit(f)

        return _apply_desired_velocity(state, system, v_des)


@LinearIntegrator.register("vicsek_intrinsic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class VicsekIntrinsic(LinearIntegrator):
    """Vicsek-model integrator with **intrinsic** noise.

    This variant perturbs the *direction* of the desired motion by applying a
    random rotation to the normalized base direction (rather than adding a random
    vector in force space as in the extrinsic / vectorial-noise variant).

    The base direction is computed from:
    - the current accumulated force vector (from colliders + force functions),
    - the average neighbor velocity direction (including self),
    then noise is applied per clump and broadcast to all clump members so clumps
    move coherently.
    """

    neighbor_radius: jax.Array
    eta: jax.Array
    v0: jax.Array
    max_neighbors: int = jax.tree.static()

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="VicsekIntrinsic.step_after_force")
    def step_after_force(state: State, system: System) -> tuple[State, System]:
        vicsek = cast(VicsekIntrinsic, system.linear_integrator)

        state, system, force, avg_v = _vicsek_alignment(
            state, system, vicsek.neighbor_radius, vicsek.max_neighbors
        )

        base_dir = unit(force + avg_v)

        # Intrinsic noise: randomly rotate the base direction.
        # Angle perturbation in [-pi, pi] scaled by eta, generated per clump.
        system.key, noise_key = jax.random.split(system.key)
        dtheta_clump = (2.0 * jax.random.uniform(noise_key, shape=(state.N,)) - 1.0) * (
            jnp.pi * vicsek.eta
        )
        dtheta = dtheta_clump[state.clump_id]
        if state.dim == 2:
            c = jnp.cos(dtheta)
            s = jnp.sin(dtheta)
            # Rotate base_dir by dtheta: [x', y'] = [c -s; s c] [x, y]
            rot = jnp.stack([c, -s, s, c], axis=-1).reshape((-1, 2, 2))
            dir_clump = jnp.einsum("nij,nj->ni", rot, base_dir)
        else:
            # 3D: sample random rotation axis (uniform on sphere), then rotate by angle.
            system.key, axis_key = jax.random.split(system.key)
            raw_axis_clump = jax.random.normal(
                axis_key, shape=(state.N, 3), dtype=state.vel.dtype
            )
            axis = unit(raw_axis_clump[state.clump_id])

            # Rodrigues' rotation formula: v' = v c + (k x v) s + k (k·v) (1-c)
            c = jnp.cos(dtheta)[..., None]
            s = jnp.sin(dtheta)[..., None]
            k = axis
            v = base_dir
            k_cross_v = cross(k, v)
            k_dot_v = dot(k, v)[..., None]
            dir_clump = v * c + k_cross_v * s + k * k_dot_v * (1.0 - c)

        v_des = vicsek.v0 * dir_clump

        return _apply_desired_velocity(state, system, v_des)


__all__ = ["VicsekExtrinsic", "VicsekIntrinsic"]
