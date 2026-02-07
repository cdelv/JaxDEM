# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""Vicsek-style integrators (extrinsic and intrinsic noise)."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Tuple, cast

from . import LinearIntegrator
from ..colliders.neighbor_list import NeighborList as NeighborListCollider

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@LinearIntegrator.register("vicsek_extrinsic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class VicsekExtrinsic(LinearIntegrator):
    """
    Vicsek-model integrator with **extrinsic** (vectorial) noise.

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
    max_neighbors: int = field(metadata={"static": True})

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"), inline=True)
    @partial(jax.named_call, name="VicsekExtrinsic.step_after_force")
    def step_after_force(state: "State", system: "System") -> Tuple["State", "System"]:
        vicsek = cast(VicsekExtrinsic, system.linear_integrator)

        # Neighbor list query. Some colliders may sort the returned state.
        state, system, nl, _overflow = system.collider.create_neighbor_list(
            state, system, vicsek.neighbor_radius, vicsek.max_neighbors
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
            dr = system.domain.displacement(pos[:, None, :], pos[safe_nl], system)
            dist_sq = jnp.sum(dr * dr, axis=-1)
            r2 = vicsek.neighbor_radius**2
            valid = valid & (dist_sq <= r2)

        # Average neighbor velocity including self.
        sum_v = jnp.sum(nbr_vel * valid[..., None], axis=1) + vel
        count = jnp.sum(valid, axis=1) + 1
        avg_v = sum_v / count[..., None]

        # Clump-wise average of avg_v (so each clump has a single alignment vector).
        counts = jnp.bincount(state.clump_ID, length=state.N)
        counts_safe = jnp.where(counts > 0, counts, 1).astype(avg_v.dtype)
        avg_v_clump = (
            jax.ops.segment_sum(avg_v, state.clump_ID, num_segments=state.N)
            / counts_safe[..., None]
        )
        avg_v = avg_v_clump[state.clump_ID]

        # Clump-wise force (state.force is expected to already be clump-broadcasted,
        # but compute robustly anyway).
        force = state.force
        force_clump = (
            jax.ops.segment_sum(force, state.clump_ID, num_segments=state.N)
            / counts_safe[..., None]
        )
        force = force_clump[state.clump_ID]

        # Random unit vector per clump (extrinsic/vectorial noise).
        system.key, noise_key = jax.random.split(system.key)
        if state.dim == 2:
            angles = jax.random.uniform(
                noise_key, shape=(state.N,), minval=0.0, maxval=2.0 * jnp.pi
            )
            unit_clump = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
        else:
            # Isotropic direction in R^3 by normalizing a Gaussian vector.
            raw = jax.random.normal(noise_key, shape=(state.N, 3), dtype=vel.dtype)
            nrm = jnp.linalg.norm(raw, axis=-1, keepdims=True)
            nrm = jnp.where(nrm == 0.0, 1.0, nrm)
            unit_clump = raw / nrm
        unit = unit_clump[state.clump_ID]

        f = force + avg_v + vicsek.eta * unit
        norm = jnp.linalg.norm(f, axis=-1, keepdims=True)
        norm = jnp.where(norm == 0.0, 1.0, norm)
        v_des = vicsek.v0 * (f / norm)

        mask_free = (1 - state.fixed)[..., None]
        state.vel = v_des * mask_free
        state.pos_c = state.pos_c + system.dt * state.vel
        return state, system


@LinearIntegrator.register("vicsek_intrinsic")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class VicsekIntrinsic(LinearIntegrator):
    """
    Vicsek-model integrator with **intrinsic** noise.

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
    max_neighbors: int = field(metadata={"static": True})

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"), inline=True)
    @partial(jax.named_call, name="VicsekIntrinsic.step_after_force")
    def step_after_force(state: "State", system: "System") -> Tuple["State", "System"]:
        vicsek = cast(VicsekIntrinsic, system.linear_integrator)

        # Neighbor list query. Some colliders may sort the returned state.
        state, system, nl, _overflow = system.collider.create_neighbor_list(
            state, system, vicsek.neighbor_radius, vicsek.max_neighbors
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
            dr = system.domain.displacement(pos[:, None, :], pos[safe_nl], system)
            dist_sq = jnp.sum(dr * dr, axis=-1)
            r2 = vicsek.neighbor_radius**2
            valid = valid & (dist_sq <= r2)

        # Average neighbor velocity including self.
        sum_v = jnp.sum(nbr_vel * valid[..., None], axis=1) + vel
        count = jnp.sum(valid, axis=1) + 1
        avg_v = sum_v / count[..., None]

        # Clump-wise average of avg_v (so each clump has a single alignment vector).
        counts = jnp.bincount(state.clump_ID, length=state.N)
        counts_safe = jnp.where(counts > 0, counts, 1).astype(avg_v.dtype)
        avg_v_clump = (
            jax.ops.segment_sum(avg_v, state.clump_ID, num_segments=state.N)
            / counts_safe[..., None]
        )
        avg_v = avg_v_clump[state.clump_ID]

        # Clump-wise force (state.force is expected to already be clump-broadcasted,
        # but compute robustly anyway).
        force = state.force
        force_clump = (
            jax.ops.segment_sum(force, state.clump_ID, num_segments=state.N)
            / counts_safe[..., None]
        )
        force = force_clump[state.clump_ID]

        base = force + avg_v
        base_norm = jnp.linalg.norm(base, axis=-1, keepdims=True)
        base_norm = jnp.where(base_norm == 0.0, 1.0, base_norm)
        base_dir = base / base_norm

        # Intrinsic noise: randomly rotate the base direction.
        system.key, noise_key = jax.random.split(system.key)
        if state.dim == 2:
            # Angle perturbation in [-pi, pi] scaled by eta.
            dtheta = (2.0 * jax.random.uniform(noise_key, shape=(state.N,)) - 1.0) * (
                jnp.pi * vicsek.eta
            )
            c = jnp.cos(dtheta)
            s = jnp.sin(dtheta)
            # Rotate base_dir by dtheta: [x', y'] = [c -s; s c] [x, y]
            rot = jnp.stack([c, -s, s, c], axis=-1).reshape((-1, 2, 2))
            dir_clump = jnp.einsum("nij,nj->ni", rot, base_dir)
        else:
            # 3D: sample random rotation axis (uniform on sphere), then rotate by angle.
            # Angle perturbation in [-pi, pi] scaled by eta.
            dtheta = (2.0 * jax.random.uniform(noise_key, shape=(state.N,)) - 1.0) * (
                jnp.pi * vicsek.eta
            )
            system.key, axis_key = jax.random.split(system.key)
            raw_axis = jax.random.normal(axis_key, shape=(state.N, 3), dtype=vel.dtype)
            axis_nrm = jnp.linalg.norm(raw_axis, axis=-1, keepdims=True)
            axis_nrm = jnp.where(axis_nrm == 0.0, 1.0, axis_nrm)
            axis = raw_axis / axis_nrm

            # Rodrigues' rotation formula: v' = v c + (k x v) s + k (k·v) (1-c)
            c = jnp.cos(dtheta)[..., None]
            s = jnp.sin(dtheta)[..., None]
            k = axis
            v = base_dir
            k_cross_v = jnp.cross(k, v)
            k_dot_v = jnp.sum(k * v, axis=-1, keepdims=True)
            dir_clump = v * c + k_cross_v * s + k * k_dot_v * (1.0 - c)

        # One noise sample per clump; broadcast to all clump members.
        dir_clump = dir_clump[state.clump_ID]

        v_des = vicsek.v0 * dir_clump
        mask_free = (1 - state.fixed)[..., None]
        state.vel = v_des * mask_free
        state.pos_c = state.pos_c + system.dt * state.vel
        return state, system


__all__ = ["VicsekExtrinsic", "VicsekIntrinsic"]
