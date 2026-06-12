# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Optax custom optimizers for energy minimization."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp
import optax  # type: ignore[import-untyped]

from ..utils.linalg import norm2
from ..utils.quaternion import Quaternion

if TYPE_CHECKING:
    from ..state import State


@jax.jit
def _quaternion_to_rotvec(q: Quaternion) -> jax.Array:
    r"""Map a unit quaternion to its axis-angle rotation vector ensuring correct gradients.

    Parameters
    ----------
    q : Quaternion
        The unit quaternion to convert.

    Returns
    -------
    jax.Array
        The 3D axis-angle rotation vector :math:`\vec{\theta} = \theta \hat{u}`.
    """
    q_u = q.unit(q)
    sign = jnp.where(q_u.w < 0.0, -1.0, 1.0)
    w = q_u.w * sign
    xyz = q_u.xyz * sign

    # 1. Compute norm safely, bypassing unit_and_norm
    n2 = norm2(xyz)[..., None]
    safe_n2 = jnp.where(n2 == 0.0, 1.0, n2)
    s = jnp.sqrt(safe_n2)

    # 2. Evaluate the singularity safely.
    # At v=0, this term approaches 2.0.
    factor = jnp.where(n2 == 0.0, 2.0, 2.0 * jnp.arctan2(s, w) / s)

    # 3. Multiply by the un-normalized vector
    return xyz * factor


@jax.jit
def _rotvec_to_quaternion(rotvec: jax.Array) -> Quaternion:
    r"""Map an axis-angle rotation vector to a unit quaternion.

    Parameters
    ----------
    rotvec : jax.Array
        The 3D axis-angle rotation vector :math:`\vec{\theta} = \theta \hat{u}`.

    Returns
    -------
    Quaternion
        The corresponding unit quaternion.
    """
    return Quaternion.from_rotvec(rotvec)


@jax.jit
def _state_to_params(state: State) -> jax.Array:
    """Pack state center of mass positions and orientations to optimization parameter array.

    Parameters
    ----------
    state : State
        The current simulation state.

    Returns
    -------
    jax.Array
        A packed array of shape `(N, 3)` in 2D or `(N, 6)` in 3D representing
        positions and rotation vectors.
    """
    rotvec = _quaternion_to_rotvec(state.q)
    if state.dim == 2:
        return jnp.concatenate([state.pos_c, rotvec[..., 2:3]], axis=-1)
    else:
        return jnp.concatenate([state.pos_c, rotvec], axis=-1)


@jax.jit
def _params_to_state(state: State, params: jax.Array) -> State:
    """Unpack optimization parameter array back to state positions and orientations.

    Parameters
    ----------
    state : State
        The reference state from which to copy fields.
    params : jax.Array
        The packed parameter array of shape `(N, 3)` in 2D or `(N, 6)` in 3D.

    Returns
    -------
    State
        The updated simulation state.
    """
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
    """Custom optax gradient transformation wrapper for DEM energy minimization.

    This class extends `optax.GradientTransformationExtraArgs` to support
    serialization and custom equality/hashing for user-defined minimization routines.
    """

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

    @property
    def metadata(self) -> dict[str, Any]:
        from jaxdem.utils import encode_callable

        return {
            "constructor": encode_callable(self._constructor),
            "kw": self.kw,
        }

    def __copy__(self) -> CustomGradientTransformation:
        # Immutable bundle of pure functions: safe to share.
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> CustomGradientTransformation:
        # NamedTuple's default reduce protocol only passes the tuple fields to
        # __new__, losing `_constructor`/`kw` and crashing deepcopy of any
        # System holding a minimizer. The object is an immutable bundle of
        # pure functions, so sharing it is the correct deep copy.
        return self

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CustomGradientTransformation):
            return False
        return self.type_name == other.type_name and self.kw == other.kw

    def __hash__(self) -> int:
        kw_items = tuple(sorted((k, str(v)) for k, v in self.kw.items()))
        return hash((self.type_name, kw_items))


class FIREState(NamedTuple):
    """Internal state for the Fast Inertial Relaxation Engine (FIRE) optimizer.

    Attributes
    ----------
    vel : jax.Array
        The current velocity parameters of shape `(N, d)`.
    dt : jax.Array
        The current step size.
    alpha : jax.Array
        The current mixing parameter.
    N_good : jax.Array
        Number of consecutive steps with positive power ($P > 0$).
    N_bad : jax.Array
        Number of consecutive steps with negative power ($P \le 0$).
    """

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
    r"""Fast Inertial Relaxation Engine (FIRE) custom optax optimizer.

    The FIRE algorithm accelerates or decelerates dynamics depending on the power
    computed between the force and the velocity. It is a widely used algorithm
    for energy minimization of granular particles.

    Mathematical Formulation
    ------------------------
    At each step:

    1. Update the velocities and positions:

       .. math::
           v_{old} &= v(t) + F(t) \cdot \frac{dt}{2} \\
           P &= F(t) \cdot v_{old}

    2. Update the algorithm parameters depending on the power :math:`P`:

       - **Downhill Step (:math:`P > 0`):**

         .. math::
             N_{good} &\to N_{good} + 1 \\
             N_{bad} &\to 0 \\
             dt &\to \begin{cases} \min(dt \cdot f_{inc}, dt_{max}) & \text{if } N_{good} > N_{min} \\ dt & \text{otherwise} \end{cases} \\
             \alpha &\to \begin{cases} \alpha \cdot f_{\alpha} & \text{if } N_{good} > N_{min} \\ \alpha & \text{otherwise} \end{cases}

       - **Uphill Step (:math:`P \le 0`):**

         .. math::
             N_{good} &\to 0 \\
             N_{bad} &\to N_{bad} + 1 \\
             dt &\to \max(dt \cdot f_{dec}, dt_{min}) \\
             \alpha &\to \alpha_{init} \\
             v_{old} &\to 0

    3. Perform velocity mixing:

       .. math::
           v_{half} &= v_{old} \cdot (1 - \alpha) + \hat{F}(t) \cdot |v_{old}| \cdot \alpha \\
           v(t + dt) &= v_{half} + F(t) \cdot \frac{dt}{2}

    Parameters
    ----------
    dt : float
        The base time step.
    alpha_init : float, default 0.1
        The initial mixing coefficient.
    f_inc : float, default 1.1
        The factor by which the time step increases on downhill steps.
    f_dec : float, default 0.5
        The factor by which the time step decreases on uphill steps.
    f_alpha : float, default 0.99
        The decay factor for the mixing coefficient.
    N_min : int, default 5
        The number of consecutive downhill steps required to increase the time step.
    N_bad_max : int, default 10
        The maximum number of uphill steps before performing resets.
    dt_max_scale : float, default 10.0
        The maximum time step scale limit: :math:`dt_{max} = dt \cdot dt_{max\_scale}`.
    dt_min_scale : float, default 1e-3
        The minimum time step scale limit: :math:`dt_{min} = dt \cdot dt_{min\_scale}`.

    Returns
    -------
    CustomGradientTransformation
        An optax gradient transformation for the FIRE algorithm.

    Reference
    ---------
    Bitzek et al., Structural Relaxation Made Simple, Phys. Rev. Lett. 97, 170201 (2006)
    """

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
            N_bad_new = state.N_bad + 1
            # After N_bad_max consecutive uphill steps, dt has been cut so many
            # times the dynamics stall: reset dt to the base time step and
            # restart the uphill counter.
            exceeded = N_bad_new > N_bad_max
            dt_new = jnp.where(
                exceeded, jnp.asarray(dt, dtype=state.dt.dtype), dt_cand_dec
            )
            N_bad_new = jnp.where(exceeded, 0, N_bad_new)
            return (
                dt_new,
                jnp.array(alpha_init),
                jnp.array(0),
                N_bad_new,
                -dt_new,
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

        v_half_norm = optax.safe_norm(v_half, min_norm=1e-16, axis=-1, keepdims=True)
        F_norm = optax.safe_norm(F, min_norm=1e-16, axis=-1, keepdims=True)
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
    """Internal state for the damped Newtonian dynamics optimizer.

    Attributes
    ----------
    vel : jax.Array
        The current velocity parameters of shape `(N, d)`.
    dt : jax.Array
        The current step size.
    """

    vel: jax.Array
    dt: jax.Array


def damped_newtonian(
    dt: float,
    gamma: float = 0.5,
) -> Any:
    r"""Damped Newtonian dynamics custom optax optimizer.

    This optimizer implements a velocity-verlet-like scheme with a linear velocity damping term
    to drive the system toward energy minimization.

    Mathematical Formulation
    ------------------------
    At each step :math:`k`, the parameters are advanced using:

    .. math::
        v_{k} &= \frac{v_{half} + F(t) \cdot \frac{dt}{2}}{1 + \gamma \cdot \frac{dt}{2}} \\
        v(t+dt) &= v_{k} \cdot \left(1 - \gamma \cdot \frac{dt}{2}\right) + F(t) \cdot \frac{dt}{2} \\
        x(t+dt) &= x(t) + v(t+dt) \cdot dt

    Parameters
    ----------
    dt : float
        The time step.
    gamma : float, default 0.5
        The damping coefficient.

    Returns
    -------
    CustomGradientTransformation
        An optax gradient transformation for the damped Newtonian algorithm.
    """

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
    return CustomGradientTransformation(
        init, update, damped_newtonian, kw, type_name="damped_newtonian"
    )
