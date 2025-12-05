# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""FIRE energy minimizer.

Reference: https://doi.org/10.1103/PhysRevLett.97.170201
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp

from dataclasses import dataclass, replace
from functools import partial
from typing import TYPE_CHECKING, Tuple

from . import LinearMinimizer
from ..integrators import LinearIntegrator

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@LinearMinimizer.register("linearfire")
@LinearIntegrator.register("linearfire")  # also register as linear integrator
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class LinearFIRE(LinearMinimizer):
    """FIRE energy minimizer for linear DOFs.

    Notes
    -----
    - Adaptive FIRE state (``dt``, ``alpha``, counters, etc.) lives on this
      integrator dataclass and is updated *functionally* via :class:`System`.
    - No FIRE-specific fields are stored on :class:`System` or :class:`State`.
    """

    # Hyperparameters (as JAX arrays)
    alpha_init: jax.Array  # initial mixing factor
    f_inc: jax.Array  # dt increase factor
    f_dec: jax.Array  # dt decrease factor
    f_alpha: jax.Array  # mixing factor decrease factor
    N_min: jax.Array  # minimum number of downhill steps before increasing dt
    N_bad_max: jax.Array  # maximum number of uphill steps before stopping
    dt_max_scale: jax.Array  # maximum dt scale relative to System.dt
    dt_min_scale: jax.Array  # minimum dt scale relative to System.dt

    # Adaptive state (updated every step, also as JAX arrays)
    dt: jax.Array
    dt_min: jax.Array
    dt_max: jax.Array
    alpha: jax.Array
    N_good: jax.Array
    N_bad: jax.Array

    @classmethod
    def Create(
        cls,
        alpha_init: float = 0.1,
        f_inc: float = 1.1,
        f_dec: float = 0.5,
        f_alpha: float = 0.99,
        N_min: int = 5,
        N_bad_max: int = 10,
        dt_max_scale: float = 10.0,
        dt_min_scale: float = 1e-3,
    ) -> "LinearFIRE":
        """Create a LinearFIRE minimizer with JAX array parameters.
        
        Parameters
        ----------
        alpha_init : float, optional
            Initial mixing factor. Default is 0.1.
        f_inc : float, optional
            Time step increase factor. Default is 1.1.
        f_dec : float, optional
            Time step decrease factor. Default is 0.5.
        f_alpha : float, optional
            Mixing factor decrease factor. Default is 0.99.
        N_min : int, optional
            Minimum number of downhill steps before increasing dt. Default is 5.
        N_bad_max : int, optional
            Maximum number of uphill steps before stopping. Default is 10.
        dt_max_scale : float, optional
            Maximum dt scale relative to System.dt. Default is 10.0.
        dt_min_scale : float, optional
            Minimum dt scale relative to System.dt. Default is 1e-3.
        
        Returns
        -------
        LinearFIRE
            A new minimizer instance with JAX array parameters.
        """
        return cls(
            alpha_init=jnp.array(alpha_init),
            f_inc=jnp.array(f_inc),
            f_dec=jnp.array(f_dec),
            f_alpha=jnp.array(f_alpha),
            N_min=jnp.array(N_min),
            N_bad_max=jnp.array(N_bad_max),
            dt_max_scale=jnp.array(dt_max_scale),
            dt_min_scale=jnp.array(dt_min_scale),
            # Initialize adaptive state to zero arrays
            dt=jnp.array(0.0),
            dt_min=jnp.array(0.0),
            dt_max=jnp.array(0.0),
            alpha=jnp.array(0.0),
            N_good=jnp.array(0),
            N_bad=jnp.array(0),
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="LinearFIRE.step_before_force")
    def step_before_force(state: "State", system: "System") -> Tuple["State", "System"]:
        """FIRE update and first half of the velocity-Verlet-like step."""
        fire = system.linear_integrator

        dt = fire.dt
        dt_min = fire.dt_min
        dt_max = fire.dt_max
        alpha = fire.alpha
        N_good = fire.N_good
        N_bad = fire.N_bad

        alpha_init = fire.alpha_init
        f_inc = fire.f_inc
        f_dec = fire.f_dec
        f_alpha = fire.f_alpha
        N_min = fire.N_min
        N_bad_max = fire.N_bad_max

        mask_free = (1 - state.fixed)

        # FIRE power: dot(f, v)
        power = jnp.sum(state.force * state.vel)

        def _active_branch(carry):
            dt, alpha, N_good, N_bad = carry

            def downhill(_):
                # Moving downhill: increase inertia and maybe dt
                N_good2 = N_good + 1
                N_bad2 = jnp.zeros_like(N_bad)

                dt2 = jnp.where(
                    N_good2 > N_min, jnp.minimum(dt * f_inc, dt_max), dt
                )
                alpha2 = jnp.where(N_good2 > N_min, alpha * f_alpha, alpha)

                dt_reverse2 = jnp.array(0.0, dtype=dt.dtype)
                velocity_scale2 = mask_free
                return dt2, alpha2, N_good2, N_bad2, dt_reverse2, velocity_scale2

            def uphill(_):
                # Moving uphill: reduce inertia, shrink dt, possibly stop
                N_good2 = jnp.zeros_like(N_good)
                N_bad2 = N_bad + 1

                dt_candidate = jnp.maximum(dt * f_dec, dt_min)
                alpha2 = alpha_init
                dt_reverse2 = -dt_candidate
                velocity_scale2 = jnp.zeros_like(mask_free)

                # If uphill for too many steps, stop by setting dt = 0
                done = N_bad2 > N_bad_max
                dt2 = jnp.where(done, 0.0, dt_candidate)
                return dt2, alpha2, N_good2, N_bad2, dt_reverse2, velocity_scale2

            return jax.lax.cond(
                power > 0.0, downhill, uphill, operand=None
            )

        def _inactive_branch(carry):
            # dt == 0: system is stopped; zero velocities and alpha, keep counters
            dt, alpha, N_good, N_bad = carry
            dt_reverse2 = jnp.array(0.0, dtype=dt.dtype)
            velocity_scale2 = jnp.zeros_like(mask_free)
            alpha2 = jnp.zeros_like(alpha)
            return dt, alpha2, N_good, N_bad, dt_reverse2, velocity_scale2

        dt, alpha, N_good, N_bad, dt_reverse, velocity_scale = jax.lax.cond(
            dt != 0.0,
            _active_branch,
            _inactive_branch,
            operand=(dt, alpha, N_good, N_bad),
        )

        # Apply reverse half-step and velocity scaling
        state.pos += state.vel * mask_free[..., None] * dt_reverse / 2.0
        state.vel *= velocity_scale[..., None]

        # Velocity Verlet: first half-kick
        accel = state.force / state.mass[..., None]
        state.vel += accel * mask_free[..., None] * dt / 2.0

        # Mix velocities and forces (FIRE projection)
        vel_norm = jnp.sqrt(jnp.sum(state.vel ** 2, axis=-1))
        force_norm = jnp.sqrt(jnp.sum(state.force ** 2, axis=-1))
        mix_mask = (force_norm > 1e-16) * mask_free
        mixing_ratio = vel_norm / (force_norm + 1e-16) * alpha * mix_mask
        state.vel = (
            state.vel * (1.0 - alpha) * mask_free[..., None]
            + state.force * mixing_ratio[..., None]
        )

        # Re-apply velocity scaling if we stopped motion
        state.vel *= velocity_scale[..., None]
        state.pos += state.vel * mask_free[..., None] * dt / 2.0

        # Write back updated FIRE state into the System integrator
        new_fire = replace(
            fire,
            dt=dt,
            dt_min=dt_min,
            dt_max=dt_max,
            alpha=alpha,
            N_good=N_good,
            N_bad=N_bad,
        )
        system = dataclasses.replace(system, linear_integrator=new_fire)

        return state, system

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="LinearFIRE.step_after_force")
    def step_after_force(state: "State", system: "System") -> Tuple["State", "System"]:
        """Second half of the velocity-Verlet-like step using adaptive dt."""
        fire = system.linear_integrator
        dt = fire.dt
        mask_free = (1 - state.fixed)[..., None]

        accel = state.force / state.mass[..., None]
        state.vel += accel * mask_free * dt / 2.0
        return state, system

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="LinearFIRE.initialize")
    def initialize(state: "State", system: "System") -> Tuple["State", "System"]:
        """Initialize FIRE state from the System and current forces."""
        fire = system.linear_integrator

        # Zero initial velocities and compute forces once
        state.vel *= 0.0
        state, system = system.collider.compute_force(state, system)

        # Calculate the initial parameters
        dt0 = system.dt
        fire = replace(
            fire,
            dt=dt0,
            dt_min=dt0 * fire.dt_min_scale,
            dt_max=dt0 * fire.dt_max_scale,
            alpha=fire.alpha_init,
            N_good=0,
            N_bad=0,
        )
        system = dataclasses.replace(system, linear_integrator=fire)
        return state, system
