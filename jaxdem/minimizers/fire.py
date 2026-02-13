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
from typing import TYPE_CHECKING, Any, Callable, Tuple, cast

from . import LinearMinimizer, RotationMinimizer
from ..integrators import LinearIntegrator, RotationIntegrator
from ..integrators.velocity_verlet_spiral import omega_dot
from ..utils.quaternion import Quaternion

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System


@partial(jax.jit, inline=True)
@partial(jax.named_call, name="fire._control_update")
def _fire_control_update(
    *,
    dt: jax.Array,
    alpha: jax.Array,
    N_good: jax.Array,
    N_bad: jax.Array,
    dt_min: jax.Array,
    dt_max: jax.Array,
    alpha_init: jax.Array,
    f_inc: jax.Array,
    f_dec: jax.Array,
    f_alpha: jax.Array,
    N_min: jax.Array,
    N_bad_max: jax.Array,
    mask_free: jax.Array,
    power: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Shared FIRE control-law update.

    Returns (dt, alpha, N_good, N_bad, dt_reverse, velocity_scale).

    Notes
    -----
    - This is shared between linear and rotational FIRE so both can use identical
      dt/alpha/counter update logic.
    - `velocity_scale` is per-particle and is used to zero (or keep) velocities.
    """

    def _active_branch(
        carry: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        dt, alpha, N_good, N_bad = carry

        def downhill(
            _: None,
        ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            N_good2 = N_good + 1
            N_bad2 = jnp.zeros_like(N_bad)

            dt2 = jnp.where(N_good2 > N_min, jnp.minimum(dt * f_inc, dt_max), dt)
            alpha2 = jnp.where(N_good2 > N_min, alpha * f_alpha, alpha)

            dt_reverse2 = jnp.array(0.0, dtype=dt.dtype)
            velocity_scale2 = mask_free
            return dt2, alpha2, N_good2, N_bad2, dt_reverse2, velocity_scale2

        def uphill(
            _: None,
        ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            N_good2 = jnp.zeros_like(N_good)
            N_bad2 = N_bad + 1

            dt_candidate = jnp.maximum(dt * f_dec, dt_min)
            alpha2 = alpha_init
            dt_reverse2 = -dt_candidate
            velocity_scale2 = jnp.zeros_like(mask_free)

            done = N_bad2 > N_bad_max
            dt2 = jnp.where(done, 0.0, dt_candidate)
            return dt2, alpha2, N_good2, N_bad2, dt_reverse2, velocity_scale2

        return jax.lax.cond(power > 0.0, downhill, uphill, operand=None)

    def _inactive_branch(
        carry: Tuple[jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
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

    return dt, alpha, N_good, N_bad, dt_reverse, velocity_scale


@LinearMinimizer.register("linearfire")
@LinearIntegrator.register("linearfire")
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
    # FIRE coupling flags (JAX bool scalars; used via lax.cond/where inside jit)
    attempt_couple: jax.Array
    coupled: jax.Array
    is_master: jax.Array
    # Shared per-step outputs (stored so a coupled partner can consume them)
    dt_reverse: jax.Array
    velocity_scale: jax.Array

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
        attempt_couple: bool = True,
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
            # Coupling defaults (set during initialize if a partner exists)
            attempt_couple=jnp.array(attempt_couple),
            coupled=jnp.array(False),
            is_master=jnp.array(True),
            dt_reverse=jnp.array(0.0),
            velocity_scale=jnp.array(0.0),
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="LinearFIRE.step_before_force")
    def step_before_force(state: "State", system: "System") -> Tuple["State", "System"]:
        """FIRE update and first half of the velocity-Verlet-like step."""
        fire = cast(LinearFIRE, system.linear_integrator)

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

        mask_free = 1 - state.fixed

        # FIRE power
        power_lin = jnp.sum(state.force * state.vel)
        power_global = power_lin + jnp.sum(state.torque * state.angVel)
        coupled_master = jnp.logical_and(fire.coupled, fire.is_master)
        power = jnp.where(coupled_master, power_global, power_lin)

        dt, alpha, N_good, N_bad, dt_reverse, velocity_scale = _fire_control_update(
            dt=dt,
            alpha=alpha,
            N_good=N_good,
            N_bad=N_bad,
            dt_min=dt_min,
            dt_max=dt_max,
            alpha_init=alpha_init,
            f_inc=f_inc,
            f_dec=f_dec,
            f_alpha=f_alpha,
            N_min=N_min,
            N_bad_max=N_bad_max,
            mask_free=mask_free,
            power=power,
        )

        # Apply reverse half-step and velocity scaling
        state.pos_c += state.vel * mask_free[..., None] * dt_reverse / 2.0
        state.vel *= velocity_scale[..., None]

        # Velocity Verlet: first half-kick
        accel = state.force / state.mass[..., None]
        state.vel += accel * mask_free[..., None] * dt / 2.0

        # Mix velocities and forces (FIRE projection)
        vel_norm = jnp.sqrt(jnp.sum(state.vel**2, axis=-1))
        force_norm = jnp.sqrt(jnp.sum(state.force**2, axis=-1))
        mix_mask = (force_norm > 1e-16) * mask_free
        mixing_ratio = vel_norm / (force_norm + 1e-16) * alpha * mix_mask
        state.vel = (
            state.vel * (1.0 - alpha) * mask_free[..., None]
            + state.force * mixing_ratio[..., None]
        )

        # Re-apply velocity scaling if we stopped motion
        state.vel *= velocity_scale[..., None]
        state.pos_c += state.vel * mask_free[..., None] * dt / 2.0

        # Write back updated FIRE state into the System integrator.
        # If coupled, also synchronize the partner integrator's control state so it can
        # consume dt/alpha/counters/dt_reverse/velocity_scale in its step_before_force.
        new_fire = replace(
            fire,
            dt=dt,
            dt_min=dt_min,
            dt_max=dt_max,
            alpha=alpha,
            N_good=N_good,
            N_bad=N_bad,
            dt_reverse=dt_reverse,
            velocity_scale=velocity_scale,
        )
        system = dataclasses.replace(system, linear_integrator=new_fire)
        if isinstance(system.rotation_integrator, RotationFIRE):
            rot_fire = system.rotation_integrator
            do_sync = jnp.logical_and(coupled_master, rot_fire.coupled)

            def _sync(sys: "System") -> "System":
                return dataclasses.replace(
                    sys,
                    rotation_integrator=replace(
                        rot_fire,
                        dt=dt,
                        dt_min=dt_min,
                        dt_max=dt_max,
                        alpha=alpha,
                        N_good=N_good,
                        N_bad=N_bad,
                        dt_reverse=dt_reverse,
                        velocity_scale=velocity_scale,
                    ),
                )

            system = jax.lax.cond(do_sync, _sync, lambda s: s, system)

        return state, system

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="LinearFIRE.step_after_force")
    def step_after_force(state: "State", system: "System") -> Tuple["State", "System"]:
        """Second half of the velocity-Verlet-like step using adaptive dt."""
        fire = cast(LinearFIRE, system.linear_integrator)
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
        fire = cast(LinearFIRE, system.linear_integrator)

        # Zero initial velocities and compute forces once
        state.vel *= 0.0
        state, system = system.collider.compute_force(state, system)
        state, system = system.force_manager.apply(state, system)

        # Calculate the initial parameters
        dt0 = system.dt
        mask_free = 1 - state.fixed
        fire = replace(
            fire,
            dt=dt0,
            dt_min=dt0 * fire.dt_min_scale,
            dt_max=dt0 * fire.dt_max_scale,
            alpha=fire.alpha_init,
            N_good=jnp.array(0),
            N_bad=jnp.array(0),
            dt_reverse=jnp.array(0.0, dtype=dt0.dtype),
            velocity_scale=mask_free,
        )

        # Attempt to couple to rotational FIRE if present.
        if isinstance(system.rotation_integrator, RotationFIRE):
            rot_fire0 = system.rotation_integrator
            do_couple = jnp.logical_and(fire.attempt_couple, rot_fire0.attempt_couple)

            def _couple(_: None) -> Tuple["LinearFIRE", "RotationFIRE"]:
                fire2 = replace(
                    fire, coupled=jnp.array(True), is_master=jnp.array(True)
                )
                rot_fire2 = replace(
                    rot_fire0,
                    # Hyperparams (sync to master)
                    alpha_init=fire2.alpha_init,
                    f_inc=fire2.f_inc,
                    f_dec=fire2.f_dec,
                    f_alpha=fire2.f_alpha,
                    N_min=fire2.N_min,
                    N_bad_max=fire2.N_bad_max,
                    dt_max_scale=fire2.dt_max_scale,
                    dt_min_scale=fire2.dt_min_scale,
                    # Adaptive state
                    dt=fire2.dt,
                    dt_min=fire2.dt_min,
                    dt_max=fire2.dt_max,
                    alpha=fire2.alpha,
                    N_good=fire2.N_good,
                    N_bad=fire2.N_bad,
                    # Coupling flags / shared outputs
                    coupled=jnp.array(True),
                    is_master=jnp.array(False),
                    dt_reverse=fire2.dt_reverse,
                    velocity_scale=mask_free,
                )
                return fire2, rot_fire2

            def _no_couple(_: None) -> Tuple["LinearFIRE", "RotationFIRE"]:
                # Keep the same output PyTree types/shapes as the coupled branch.
                rot_fire2 = replace(
                    rot_fire0,
                    dt_reverse=jnp.array(0.0, dtype=dt0.dtype),
                    velocity_scale=mask_free,
                )
                return (
                    replace(fire, coupled=jnp.array(False), is_master=jnp.array(True)),
                    rot_fire2,
                )

            fire2, rot_fire2 = jax.lax.cond(
                do_couple, _couple, _no_couple, operand=None
            )
            system = dataclasses.replace(
                system, linear_integrator=fire2, rotation_integrator=rot_fire2
            )
        else:
            fire2 = replace(fire, coupled=jnp.array(False), is_master=jnp.array(True))
            system = dataclasses.replace(system, linear_integrator=fire2)

        return state, system


@RotationMinimizer.register("rotationfire")
@RotationIntegrator.register("rotationfire")
@jax.tree_util.register_dataclass
@dataclass(slots=True)
class RotationFIRE(RotationMinimizer):
    """FIRE energy minimizer for rotation DOFs.

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
    # FIRE coupling flags (JAX bool scalars; used via lax.cond/where inside jit)
    attempt_couple: jax.Array
    coupled: jax.Array
    is_master: jax.Array
    # Shared per-step outputs (stored so a coupled partner can consume them)
    dt_reverse: jax.Array
    velocity_scale: jax.Array

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
        attempt_couple: bool = True,
    ) -> "RotationFIRE":
        """Create a RotationFIRE minimizer with JAX array parameters.

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
        RotationFIRE
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
            # Coupling defaults (set during initialize if a partner exists)
            attempt_couple=jnp.array(attempt_couple),
            coupled=jnp.array(False),
            is_master=jnp.array(False),
            dt_reverse=jnp.array(0.0),
            velocity_scale=jnp.array(0.0),
        )

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="RotationFIRE.step_before_force")
    def step_before_force(state: "State", system: "System") -> Tuple["State", "System"]:
        """FIRE update and first half of the velocity-Verlet-like step."""
        fire = cast(RotationFIRE, system.rotation_integrator)

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

        mask_free = 1 - state.fixed

        # pad to 3d if in 2d
        if state.dim == 2:
            angVel_lab_3d = jnp.pad(state.angVel, ((0, 0), (2, 0)), constant_values=0.0)
            torque_lab_3d = jnp.pad(state.torque, ((0, 0), (2, 0)), constant_values=0.0)
        else:  # state.dim == 3
            angVel_lab_3d = state.angVel
            torque_lab_3d = state.torque

        # rotate angular velocities and torques to body frame
        angVel = state.q.rotate_back(state.q, angVel_lab_3d)
        torque = state.q.rotate_back(state.q, torque_lab_3d)

        follower = jnp.logical_and(fire.coupled, jnp.logical_not(fire.is_master))

        def _consume(
            _: None,
        ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            return dt, alpha, N_good, N_bad, fire.dt_reverse, fire.velocity_scale

        def _update(
            _: None,
        ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
            power = jnp.sum(torque * angVel)
            return _fire_control_update(
                dt=dt,
                alpha=alpha,
                N_good=N_good,
                N_bad=N_bad,
                dt_min=dt_min,
                dt_max=dt_max,
                alpha_init=alpha_init,
                f_inc=f_inc,
                f_dec=f_dec,
                f_alpha=f_alpha,
                N_min=N_min,
                N_bad_max=N_bad_max,
                mask_free=mask_free,
                power=power,
            )

        dt, alpha, N_good, N_bad, dt_reverse, velocity_scale = jax.lax.cond(
            follower, _consume, _update, operand=None
        )

        # Apply reverse half-step
        w_norm2 = jnp.sum(angVel * angVel, axis=-1, keepdims=True)
        w_norm = jnp.sqrt(w_norm2)
        theta1 = dt_reverse * w_norm / 2
        w_norm = jnp.where(w_norm == 0, 1.0, w_norm)
        state.q @= Quaternion(
            jnp.cos(theta1),
            jnp.sin(theta1) * angVel / w_norm,
        )
        # normalize quarternion
        state.q = state.q.unit(state.q)

        # Scale velocities
        angVel *= velocity_scale[..., None]

        # Velocity Verlet: first half-kick
        dt_2 = dt / 2
        inv_inertia = 1 / state.inertia
        k1 = dt_2 * omega_dot(angVel, torque, state.inertia, inv_inertia)
        k2 = dt_2 * omega_dot(angVel + k1, torque, state.inertia, inv_inertia)
        k3 = dt_2 * omega_dot(
            angVel + 0.25 * (k1 + k2), torque, state.inertia, inv_inertia
        )
        angVel += (1 - state.fixed)[..., None] * (k1 + k2 + 4.0 * k3) / 6.0

        # Mix angular velocities and torques (FIRE projection)
        ang_vel_norm = jnp.sqrt(jnp.sum(angVel * angVel, axis=-1))
        torque_norm = jnp.sqrt(jnp.sum(torque * torque, axis=-1))
        mix_mask = (torque_norm > 1e-16) * mask_free
        mixing_ratio = ang_vel_norm / (torque_norm + 1e-16) * alpha * mix_mask
        angVel = (
            angVel * (1.0 - alpha) * (1 - state.fixed)[..., None]
            + torque * mixing_ratio[..., None]
        )

        # Re-apply velocity scaling if we stopped motion
        angVel *= velocity_scale[..., None]

        # Apply final half-step
        w_norm2 = jnp.sum(angVel * angVel, axis=-1, keepdims=True)
        w_norm = jnp.sqrt(w_norm2)
        theta1 = dt * w_norm / 2
        w_norm = jnp.where(w_norm == 0, 1.0, w_norm)
        state.q @= Quaternion(
            jnp.cos(theta1),
            jnp.sin(theta1) * angVel / w_norm,
        )
        # normalize quarternion
        state.q = state.q.unit(state.q)

        # rotate angular velocity back to lab frame and save it in the state
        if state.dim == 2:
            state.angVel = angVel[..., -1:]
        else:
            state.angVel = state.q.rotate(state.q, angVel)

        # Write back updated FIRE state into the System integrator
        new_fire = replace(
            fire,
            dt=dt,
            dt_min=dt_min,
            dt_max=dt_max,
            alpha=alpha,
            N_good=N_good,
            N_bad=N_bad,
            dt_reverse=dt_reverse,
            velocity_scale=velocity_scale,
        )
        system = dataclasses.replace(system, rotation_integrator=new_fire)

        return state, system

    @staticmethod
    @partial(jax.jit, donate_argnames=("state", "system"))
    @partial(jax.named_call, name="RotationFIRE.step_after_force")
    def step_after_force(state: "State", system: "System") -> Tuple["State", "System"]:
        """Second half of the velocity-Verlet-like step using adaptive dt."""
        fire = cast(RotationFIRE, system.rotation_integrator)
        dt = fire.dt

        # pad to 3d if needed
        if state.dim == 2:
            angVel_lab_3d = jnp.pad(state.angVel, ((0, 0), (2, 0)), constant_values=0.0)
            torque_lab_3d = jnp.pad(state.torque, ((0, 0), (2, 0)), constant_values=0.0)
        else:
            angVel_lab_3d = state.angVel
            torque_lab_3d = state.torque

        # rotate angular velocities and torques to body frame
        angVel = state.q.rotate_back(state.q, angVel_lab_3d)
        torque = state.q.rotate_back(state.q, torque_lab_3d)

        # update angular velocities
        dt_2 = dt / 2
        inv_inertia = 1 / state.inertia
        k1 = dt_2 * omega_dot(angVel, torque, state.inertia, inv_inertia)
        k2 = dt_2 * omega_dot(angVel + k1, torque, state.inertia, inv_inertia)
        k3 = dt_2 * omega_dot(
            angVel + (k1 + k2) / 4, torque, state.inertia, inv_inertia
        )
        angVel += (1 - state.fixed)[..., None] * (k1 + k2 + 4.0 * k3) / 6.0

        # rotate angular velocities back to lab frame and save in state
        if state.dim == 2:
            state.angVel = angVel[..., -1:]
        else:
            state.angVel = state.q.rotate(state.q, angVel)  # to lab

        return state, system

    @staticmethod
    @jax.jit
    @partial(jax.named_call, name="RotationFIRE.initialize")
    def initialize(state: "State", system: "System") -> Tuple["State", "System"]:
        """Initialize FIRE state from the System and current forces."""
        fire = cast(RotationFIRE, system.rotation_integrator)

        # Zero initial velocities and compute forces once
        state.angVel *= 0.0
        state, system = system.collider.compute_force(state, system)
        state, system = system.force_manager.apply(state, system)

        # Calculate the initial parameters
        dt0 = system.dt
        mask_free = 1 - state.fixed
        fire = replace(
            fire,
            dt=dt0,
            dt_min=dt0 * fire.dt_min_scale,
            dt_max=dt0 * fire.dt_max_scale,
            alpha=fire.alpha_init,
            N_good=jnp.array(0),
            N_bad=jnp.array(0),
            dt_reverse=jnp.array(0.0, dtype=dt0.dtype),
            velocity_scale=mask_free,
        )

        # Attempt to couple to linear FIRE if present.
        if isinstance(system.linear_integrator, LinearFIRE):
            lin_fire0 = system.linear_integrator
            do_couple = jnp.logical_and(fire.attempt_couple, lin_fire0.attempt_couple)

            def _couple(_: None) -> Tuple["RotationFIRE", "LinearFIRE"]:
                # Ensure the linear integrator is marked as master/coupled.
                lin_fire2 = replace(
                    lin_fire0, coupled=jnp.array(True), is_master=jnp.array(True)
                )
                fire2 = replace(
                    fire,
                    # Hyperparams + adaptive state from master
                    alpha_init=lin_fire2.alpha_init,
                    f_inc=lin_fire2.f_inc,
                    f_dec=lin_fire2.f_dec,
                    f_alpha=lin_fire2.f_alpha,
                    N_min=lin_fire2.N_min,
                    N_bad_max=lin_fire2.N_bad_max,
                    dt_max_scale=lin_fire2.dt_max_scale,
                    dt_min_scale=lin_fire2.dt_min_scale,
                    dt=lin_fire2.dt,
                    dt_min=lin_fire2.dt_min,
                    dt_max=lin_fire2.dt_max,
                    alpha=lin_fire2.alpha,
                    N_good=lin_fire2.N_good,
                    N_bad=lin_fire2.N_bad,
                    dt_reverse=lin_fire2.dt_reverse,
                    velocity_scale=mask_free,
                    coupled=jnp.array(True),
                    is_master=jnp.array(False),
                )
                return fire2, lin_fire2

            def _no_couple(_: None) -> Tuple["RotationFIRE", "LinearFIRE"]:
                # Keep the same output PyTree types/shapes as the coupled branch.
                lin_fire2 = replace(
                    lin_fire0,
                    dt_reverse=jnp.array(0.0, dtype=dt0.dtype),
                    velocity_scale=mask_free,
                )
                return (
                    replace(fire, coupled=jnp.array(False), is_master=jnp.array(False)),
                    lin_fire2,
                )

            fire2, lin_fire2 = jax.lax.cond(
                do_couple, _couple, _no_couple, operand=None
            )
            system = dataclasses.replace(
                system, rotation_integrator=fire2, linear_integrator=lin_fire2
            )
        else:
            fire2 = replace(fire, coupled=jnp.array(False), is_master=jnp.array(False))
            system = dataclasses.replace(system, rotation_integrator=fire2)

        return state, system
