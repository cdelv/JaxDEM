# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Optax custom optimizers for energy minimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple

import optax  # type: ignore[import-untyped]

from ..utils.linalg import unit_and_norm
from ..utils.quaternion import Quaternion

if TYPE_CHECKING:
    from ..state import State

@jax.jit
def _quaternion_to_rotvec(q: Quaternion) -> jax.Array:
    """Map a unit quaternion to its axis-angle rotation vector."""
    q_u = q.unit(q)
    sign = jnp.where(q_u.w < 0.0, -1.0, 1.0)
    w = q_u.w * sign
    xyz = q_u.xyz * sign
    axis, sin_half = unit_and_norm(xyz)
    angle = 2.0 * jnp.arctan2(sin_half, w)
    return axis * angle


@jax.jit
def _rotvec_to_quaternion(rotvec: jax.Array) -> Quaternion:
    """Map an axis-angle rotation vector to a unit quaternion."""
    axis, angle = unit_and_norm(rotvec)
    half_angle = 0.5 * angle
    return Quaternion(jnp.cos(half_angle), axis * jnp.sin(half_angle))


@jax.jit
def _state_to_params(state: State) -> jax.Array:
    """Pack state center of mass positions and orientations to optimization parameter array."""
    rotvec = _quaternion_to_rotvec(state.q)
    if state.dim == 2:
        return jnp.concatenate([state.pos_c, rotvec[..., 2:3]], axis=-1)
    else:
        return jnp.concatenate([state.pos_c, rotvec], axis=-1)


@jax.jit
def _params_to_state(state: State, params: jax.Array) -> State:
    """Unpack optimization parameter array back to state positions and orientations."""
    if state.dim == 2:
        pos_c = params[..., 0:2]
        rotvec = jnp.concatenate(
            [jnp.zeros_like(pos_c), params[..., 2:3]],
            axis=-1,
        )
    else:
        pos_c = params[..., 0:3]
        rotvec = params[..., 3:6]
    q = _rotvec_to_quaternion(rotvec)
    return replace(state, pos_c=pos_c, q=q.unit(q))


class CustomGradientTransformation(optax.GradientTransformationExtraArgs):  # type: ignore[misc]
    _constructor: Any
    type_name: str
    kw: dict[str, Any]

    def __new__(
        cls,
        init_fn: Any,
        update_fn: Any,
        _constructor: Any,
        kw: dict[str, Any],
        type_name: str = "",
    ) -> CustomGradientTransformation:
        obj = super().__new__(cls, init_fn, update_fn)
        obj._constructor = _constructor
        obj.type_name = type_name
        obj.kw = kw
        return obj


class FIREState(NamedTuple):
    vel: jax.Array
    dt: jax.Array
    alpha: jax.Array
    N_good: jax.Array
    N_bad: jax.Array


def fire(
    dt: float,
    alpha_init: float = 0.1,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    f_alpha: float = 0.99,
    N_min: int = 5,
    N_bad_max: int = 10,
    dt_max_scale: float = 10.0,
    dt_min_scale: float = 1e-3,
) -> Any:
    """Fast Inertial Relaxation Engine (FIRE) custom optax optimizer."""
    try:
        import optax
    except ImportError:
        raise ImportError(
            "optax must be installed to use optax optimizers. Install it via 'pip install optax'"
        )

    def init(params: jax.Array) -> FIREState:
        return FIREState(
            vel=jnp.zeros_like(params),
            dt=jnp.array(dt),
            alpha=jnp.array(alpha_init),
            N_good=jnp.array(0),
            N_bad=jnp.array(0),
        )

    def update(
        updates: jax.Array,
        state: FIREState,
        params: jax.Array | None = None,
        **kwargs: Any,
    ) -> tuple[jax.Array, FIREState]:
        F = -updates
        v_old = state.vel + F * state.dt / 2.0
        power = jnp.sum(F * v_old)

        dt_cand_inc = jnp.minimum(state.dt * f_inc, dt * dt_max_scale)
        dt_cand_dec = jnp.maximum(state.dt * f_dec, dt * dt_min_scale)

        def uphill(_: Any) -> tuple[Any, ...]:
            return (
                dt_cand_dec,
                jnp.array(alpha_init),
                jnp.array(0),
                state.N_bad + 1,
                -dt_cand_dec,
                0.0,
            )

        def downhill(_: Any) -> tuple[Any, ...]:
            N_good_new = state.N_good + 1
            dt_new = jnp.where(N_good_new > N_min, dt_cand_inc, state.dt)
            alpha_new = jnp.where(
                N_good_new > N_min, state.alpha * f_alpha, state.alpha
            )
            return (
                dt_new,
                alpha_new,
                N_good_new,
                jnp.array(0),
                0.0,
                1.0,
            )

        new_dt, new_alpha, new_N_good, new_N_bad, dt_reverse, velocity_scale = (
            jax.lax.cond(power > 0.0, downhill, uphill, operand=None)
        )

        v_temp = v_old * velocity_scale
        v_half = v_temp + F * new_dt / 2.0

        v_half_norm = jnp.sqrt(jnp.sum(v_half**2, axis=-1, keepdims=True))
        F_norm = jnp.sqrt(jnp.sum(F**2, axis=-1, keepdims=True))
        mixing_ratio = jnp.where(F_norm > 1e-16, v_half_norm / F_norm * new_alpha, 0.0)
        v_half = v_half * (1.0 - new_alpha) + F * mixing_ratio
        v_half = v_half * velocity_scale

        updates_to_apply = v_old * dt_reverse / 2.0 + v_half * new_dt / 2.0

        new_state = FIREState(
            vel=v_half,
            dt=new_dt,
            alpha=new_alpha,
            N_good=new_N_good,
            N_bad=new_N_bad,
        )
        return updates_to_apply, new_state

    kw = {
        "dt": dt,
        "alpha_init": alpha_init,
        "f_inc": f_inc,
        "f_dec": f_dec,
        "f_alpha": f_alpha,
        "N_min": N_min,
        "N_bad_max": N_bad_max,
        "dt_max_scale": dt_max_scale,
        "dt_min_scale": dt_min_scale,
    }
    return CustomGradientTransformation(init, update, fire, kw, type_name="fire")


class DampedNewtonianState(NamedTuple):
    vel: jax.Array
    dt: jax.Array


def damped_newtonian(
    dt: float,
    gamma: float = 0.5,
) -> Any:
    """Damped Newtonian dynamics custom optax optimizer."""
    try:
        import optax
    except ImportError:
        raise ImportError(
            "optax must be installed to use optax optimizers. Install it via 'pip install optax'"
        )

    def init(params: jax.Array) -> DampedNewtonianState:
        return DampedNewtonianState(
            vel=jnp.zeros_like(params),
            dt=jnp.array(dt),
        )

    def update(
        updates: jax.Array,
        state: DampedNewtonianState,
        params: jax.Array | None = None,
        **kwargs: Any,
    ) -> tuple[jax.Array, DampedNewtonianState]:
        F = -updates
        v_half_prev = state.vel

        v_k = (v_half_prev + F * state.dt / 2.0) / (1.0 + gamma * state.dt / 2.0)

        v_half = v_k * (1.0 - gamma * state.dt / 2.0) + F * state.dt / 2.0

        updates_to_apply = v_half * state.dt

        new_state = DampedNewtonianState(vel=v_half, dt=state.dt)
        return updates_to_apply, new_state

    kw = {
        "dt": dt,
        "gamma": gamma,
    }
    return CustomGradientTransformation(init, update, damped_newtonian, kw, type_name="damped_newtonian")
