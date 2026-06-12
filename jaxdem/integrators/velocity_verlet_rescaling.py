# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Velocity Verlet integrator with periodic velocity-rescaling thermostat."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

from . import LinearIntegrator, free_mask
from .velocity_verlet import VelocityVerlet
from ..utils.thermal import (
    compute_translational_kinetic_energy_per_particle,
    compute_rotational_kinetic_energy_per_particle,
    count_dynamic_dofs,
)

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@LinearIntegrator.register("verlet_rescaling")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class VelocityVerletRescaling(VelocityVerlet):
    r"""Velocity Verlet with periodic velocity-rescaling thermostat.

    Every ``rescale_every`` steps, the translational (and optionally rotational)
    velocities are uniformly rescaled so that the instantaneous kinetic
    temperature matches ``temperature``.  Between rescalings the dynamics are
    purely Newtonian (standard Velocity Verlet, inherited from
    :class:`VelocityVerlet`).

    The rescaling is applied at the end of ``step_after_force``, after the
    second Verlet half-kick, so that the terminal velocities on rescaling
    steps are exactly at the target temperature.

    Fixed particles are excluded from the thermostat statistics (kinetic
    energy sums and drift mean) and their prescribed velocities are never
    modified.

    Parameters
    ----------
    k_B : jax.Array
        Boltzmann constant (set to 1.0 for reduced units).
    temperature : jax.Array
        Target temperature :math:`T`.
    rescale_every : jax.Array
        Rescale velocities every this many steps (scalar integer).
    can_rotate : jax.Array
        Whether to include rotational DOF in the thermostat (0 or 1).
        When 1, angular velocities are rescaled together with linear
        velocities and rotational KE counts toward the temperature.
    subtract_drift : jax.Array
        Whether to remove centre-of-mass drift before rescaling (0 or 1).
        When 1, the COM velocity is subtracted on rescaling steps before
        measuring temperature. Mainly relevant for small systems.
    """

    k_B: jax.Array
    temperature: jax.Array
    rescale_every: jax.Array
    can_rotate: jax.Array
    subtract_drift: jax.Array

    @staticmethod
    @jax.jit(inline=True)
    @partial(jax.named_call, name="VelocityVerletRescaling.step_after_force")
    def step_after_force(state: State, system: System) -> tuple[State, System]:
        integrator = cast(VelocityVerletRescaling, system.linear_integrator)
        can_rot = integrator.can_rotate

        # --- Standard Verlet half-kick (reuse VelocityVerlet) ---
        state, system = VelocityVerlet.step_after_force(state, system)

        # --- Conditional velocity rescaling ---
        should_rescale = (system.step_count % integrator.rescale_every) == 0

        free = free_mask(state)  # (..., N, 1) bool
        free_f = free.astype(state.vel.dtype)
        n_free = jnp.maximum(jnp.sum(free_f, axis=(-2, -1)), 1.0)

        drift = (
            jnp.sum(state.vel * free_f, axis=-2, keepdims=True)
            / n_free[..., None, None]
        )
        drift = drift * (should_rescale * integrator.subtract_drift)
        state.vel = jnp.where(free, state.vel - drift, state.vel)

        ke_trans = jnp.sum(
            compute_translational_kinetic_energy_per_particle(state) * free_f[..., 0],
            axis=-1,
        )
        ke_rot = jnp.sum(
            compute_rotational_kinetic_energy_per_particle(state) * free_f[..., 0],
            axis=-1,
        )
        ke_total = ke_trans + ke_rot * can_rot

        _, n_dof_v, n_dof_w = count_dynamic_dofs(
            state, can_rotate=True, subtract_drift=False
        )
        dim = state.vel.shape[-1]
        n_dof = (n_dof_v - dim * integrator.subtract_drift) + n_dof_w * can_rot

        T_current = 2.0 * ke_total / (integrator.k_B * jnp.maximum(n_dof, 1.0))
        scale = jnp.sqrt(integrator.temperature / jnp.maximum(T_current, 1e-30))
        scale = jnp.where(should_rescale, scale, 1.0)[..., None, None]

        state.vel = jnp.where(free, state.vel * scale, state.vel)
        ang_scale = jnp.where(can_rot, scale, 1.0)
        state.ang_vel = jnp.where(free, state.ang_vel * ang_scale, state.ang_vel)

        return state, system


__all__ = ["VelocityVerletRescaling"]
