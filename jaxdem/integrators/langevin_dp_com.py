# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Langevin thermostat acting on deformable-particle COM velocities."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

from . import LinearIntegrator

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@partial(jax.named_call, name="LangevinDPCOM._as_particle_vector")
def _as_particle_vector(field: jax.Array, state: State) -> jax.Array:
    """Broadcast scalar or per-particle thermostat fields to ``state.vel.shape``."""
    arr = jnp.asarray(field, dtype=state.vel.dtype)
    if arr.ndim == 0:
        return jnp.broadcast_to(arr, state.vel.shape)
    if arr.shape == state.mass.shape:
        return jnp.broadcast_to(arr[..., None], state.vel.shape)
    return jnp.broadcast_to(arr, state.vel.shape)


@LinearIntegrator.register("langevin_dp_com")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class LangevinDPCOM(LinearIntegrator):
    r"""Langevin thermostat that acts on deformable-particle COM velocities.

    Vertices sharing the same ``state.bond_id`` are treated as belonging to the
    same deformable particle. The deterministic force kick remains per vertex,
    but the stochastic Ornstein-Uhlenbeck update is applied only to the
    mass-weighted center-of-mass velocity of each deformable particle. Internal
    relative velocities are preserved by construction.

    Bonds containing any fixed vertex are excluded from the thermostat step.
    """

    gamma: jax.Array
    k_B: jax.Array
    temperature: jax.Array

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="LangevinDPCOM.step_before_force")
    def step_before_force(state: State, system: System) -> tuple[State, System]:
        langevin = cast(LangevinDPCOM, system.linear_integrator)
        dt = system.dt
        mask = (1 - state.fixed)[..., None]

        accel = state.force / state.mass[..., None]
        state.vel += (dt / 2) * accel * mask
        state.pos_c += (dt / 2) * state.vel

        bond_id = state.bond_id
        num_segments = state.N
        vel_dtype = state.vel.dtype

        bond_counts = jax.ops.segment_sum(
            jnp.ones((state.N,), dtype=vel_dtype),
            bond_id,
            num_segments=num_segments,
        )
        bond_fixed_counts = jax.ops.segment_sum(
            state.fixed.astype(vel_dtype),
            bond_id,
            num_segments=num_segments,
        )
        bond_free = (bond_counts > 0) & (bond_fixed_counts == 0)

        bond_mass = jax.ops.segment_sum(
            state.mass,
            bond_id,
            num_segments=num_segments,
        )
        bond_momentum = jax.ops.segment_sum(
            state.mass[..., None] * state.vel,
            bond_id,
            num_segments=num_segments,
        )
        bond_vel = bond_momentum / jnp.maximum(bond_mass[..., None], 1.0)
        relative_vel = state.vel - bond_vel[bond_id]

        gamma = _as_particle_vector(langevin.gamma, state)
        kT = _as_particle_vector(langevin.k_B * langevin.temperature, state)
        gamma_bond = jax.ops.segment_sum(gamma, bond_id, num_segments=num_segments)
        gamma_bond /= jnp.maximum(bond_counts[..., None], 1.0)
        kT_bond = jax.ops.segment_sum(kT, bond_id, num_segments=num_segments)
        kT_bond /= jnp.maximum(bond_counts[..., None], 1.0)

        c1 = jnp.exp(-gamma_bond * dt)
        c2 = jnp.sqrt(
            kT_bond
            / jnp.maximum(bond_mass[..., None], 1.0)
            * (1.0 - jnp.exp(-2.0 * gamma_bond * dt))
        )
        system.key, noise_key = jax.random.split(system.key)
        noise = jax.random.normal(noise_key, shape=bond_vel.shape, dtype=vel_dtype)
        thermostatted_bond_vel = c1 * bond_vel + c2 * noise
        new_bond_vel = jnp.where(bond_free[..., None], thermostatted_bond_vel, bond_vel)

        state.vel = relative_vel + new_bond_vel[bond_id]
        state.pos_c += (dt / 2) * state.vel
        return state, system

    @staticmethod
    @partial(jax.jit, inline=True)
    @partial(jax.named_call, name="LangevinDPCOM.step_after_force")
    def step_after_force(state: State, system: System) -> tuple[State, System]:
        mask = (1 - state.fixed)[..., None]
        accel = state.force / state.mass[..., None]
        state.vel += (system.dt / 2) * accel * mask
        return state, system


__all__ = ["LangevinDPCOM"]
