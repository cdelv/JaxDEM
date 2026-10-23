# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Custom optax optimizers for energy minimization."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax  # type: ignore[import-untyped]


def _freeze_metadata(value: Any) -> Any:
    """Recursively detach mutable containers used as static metadata."""
    if isinstance(value, Mapping):
        return MappingProxyType({k: _freeze_metadata(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze_metadata(v) for v in value)
    return value


def _metadata_key(value: Any) -> Any:
    """Return a stable hash key without invoking array-valued equality."""
    if isinstance(value, Mapping):
        return tuple(sorted((k, _metadata_key(v)) for k, v in value.items()))
    if isinstance(value, tuple):
        return tuple(_metadata_key(v) for v in value)
    if isinstance(value, frozenset):
        return frozenset(_metadata_key(v) for v in value)
    try:
        hash(value)
    except TypeError:
        return (type(value), id(value))
    return value


def _thaw_metadata(value: Any) -> Any:
    """Copy frozen metadata into JSON-compatible mutable containers."""
    if isinstance(value, Mapping):
        return {k: _thaw_metadata(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_thaw_metadata(v) for v in value)
    if isinstance(value, frozenset):
        return [_thaw_metadata(v) for v in value]
    return value


class CustomGradientTransformation(optax.GradientTransformationExtraArgs):  # type: ignore[misc]
    """Custom optax gradient transformation wrapper for DEM energy minimization.

    This class extends `optax.GradientTransformationExtraArgs` to support
    serialization and custom equality and hashing for minimization routines.
    ``requires_value=False`` opts gradient-only optimizers into physical
    force evaluation without computing energy at every iteration. Nested
    wrappers inherit this capability from the supplied update function unless
    explicitly overridden; plain Optax transformations require values by default.

    Custom updates follow the Optax extra-argument interface and accept
    unused keyword arguments. Factories can wrap plain transformations with
    ``optax.with_extra_args_support`` before passing their functions here.
    """

    _constructor: Any
    type_name: str
    kw: Mapping[str, Any]
    _kw_key: Any
    requires_value: bool

    def __new__(
        cls,
        init_fn: Any,
        update_fn: Any,
        _constructor: Any,
        kw: dict[str, Any],
        type_name: str = "",
        *,
        requires_value: bool | None = None,
    ) -> CustomGradientTransformation:
        if requires_value is None:
            requires_value = bool(getattr(update_fn, "requires_value", True))

        def update(updates: Any, state: Any, params: Any = None, **kwargs: Any) -> Any:
            return update_fn(updates, state, params, **kwargs)

        setattr(update, "requires_value", requires_value)
        obj = super().__new__(cls, init_fn, update)
        object.__setattr__(obj, "_constructor", _constructor)
        object.__setattr__(obj, "type_name", type_name)
        object.__setattr__(obj, "requires_value", requires_value)
        # Static JAX arguments must not change their equality/hash after they
        # have been used as compilation-cache keys.
        object.__setattr__(obj, "kw", _freeze_metadata(kw))
        object.__setattr__(obj, "_kw_key", _metadata_key(obj.kw))
        return obj

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("CustomGradientTransformation is immutable")

    @property
    def metadata(self) -> dict[str, Any]:
        from jaxdem.utils import encode_callable

        return {
            "constructor": encode_callable(self._constructor),
            "kw": _thaw_metadata(self.kw),
        }

    def __copy__(self) -> CustomGradientTransformation:
        # Immutable bundle of pure functions: safe to share.
        return self

    def __deepcopy__(
        self, memo: dict[int, Any] | None = None
    ) -> CustomGradientTransformation:
        # NamedTuple's default reduce protocol only passes the tuple fields to
        # the constructor, but we added type_name and kw. So deepcopy fails by
        # default. The object is an immutable bundle of
        # pure functions, so sharing it is the correct deep copy.
        return self

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, CustomGradientTransformation):
            return False
        return (
            self._constructor is other._constructor
            and self._kw_key == other._kw_key
            and self.requires_value == other.requires_value
        )

    def __hash__(self) -> int:
        return hash((id(self._constructor), self._kw_key, self.requires_value))

    def __ne__(self, other: Any) -> bool:
        # tuple.__ne__ compares only init/update, bypassing our static metadata.
        return not self == other


class FIREState(NamedTuple):
    r"""Internal state for the Fast Inertial Relaxation Engine (FIRE) optimizer.

    Attributes
    ----------
    vel : jax.Array
        The current velocity parameters (same PyTree structure as params).
    dt : jax.Array
        The current step size.
    alpha : jax.Array
        The current mixing parameter.
    N_good : jax.Array
        Number of consecutive steps with positive power ($P > 0$).
    N_bad : jax.Array
        Number of consecutive steps with nonpositive power ($P \le 0$).
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

    The FIRE algorithm accelerates or decelerates the dynamics based on the power
    of the force and the velocity. It minimizes the energy of granular particles.

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
             \alpha &\to \alpha_{init}

    3. Kick, mix velocities, and update positions using the adapted step size
       :math:`h`. Let :math:`s=1` for positive power and zero otherwise:

       .. math::
           w &= s v_{old} + F h / 2 \\
           v_{new} &= s\left[(1-\alpha)w + \alpha\frac{\|w\|}{\|F\|}F\right] \\
           \Delta x &= \frac{1}{2}\left[v_{old} h_{reverse} + v_{new} h\right]

       Here :math:`h_{reverse}=0` downhill and :math:`-h` uphill, so an
       uphill step backtracks and stores zero velocity. Mixing norms are
       computed along each leaf's last axis (absolute values for scalars);
       a zero force contributes zero to the directional mixing term. Power
       is summed across all leaves. After more than ``N_bad_max`` consecutive
       uphill steps, the step size returns to its base value and the uphill
       counter resets.

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
        The maximum number of uphill steps before a reset.
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

    def init(params: Any) -> FIREState:
        return FIREState(
            vel=jax.tree.map(jnp.zeros_like, params),
            dt=jnp.array(dt),
            alpha=jnp.array(alpha_init),
            N_good=jnp.array(0),
            N_bad=jnp.array(0),
        )

    def update(
        updates: Any,
        state: FIREState,
        params: Any | None = None,
        **kwargs: Any,
    ) -> tuple[Any, FIREState]:
        F = jax.tree.map(lambda u: -u, updates)
        v_old = jax.tree.map(lambda v, f: v + f * state.dt / 2.0, state.vel, F)
        power = sum(
            jnp.sum(f * v) for f, v in zip(jax.tree.leaves(F), jax.tree.leaves(v_old))
        )

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

        v_temp = jax.tree.map(lambda v: v * velocity_scale, v_old)
        v_half = jax.tree.map(lambda vt, f: vt + f * new_dt / 2.0, v_temp, F)

        def normalize(x: jax.Array) -> jax.Array:
            return optax.safe_norm(
                x, min_norm=0.0, axis=-1 if x.ndim else None, keepdims=True
            )

        v_half_norm = jax.tree.map(normalize, v_half)
        F_norm = jax.tree.map(normalize, F)
        mixing_ratio = jax.tree.map(
            lambda fn, vn: vn / jnp.where(fn > 0.0, fn, 1.0) * new_alpha,
            F_norm,
            v_half_norm,
        )
        v_half = jax.tree.map(
            lambda v, f, m: (v * (1.0 - new_alpha) + f * m) * velocity_scale,
            v_half,
            F,
            mixing_ratio,
        )

        updates_to_apply = jax.tree.map(
            lambda vo, vh: vo * dt_reverse / 2.0 + vh * new_dt / 2.0, v_old, v_half
        )

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
    return CustomGradientTransformation(
        init, update, fire, kw, type_name="fire", requires_value=False
    )


class DampedNewtonianState(NamedTuple):
    """Internal state for the damped Newtonian dynamics optimizer.

    Attributes
    ----------
    vel : Any
        The current velocity parameters (same PyTree structure as params).
    dt : jax.Array
        The current step size.
    """

    vel: Any
    dt: jax.Array


def damped_newtonian(
    dt: float,
    gamma: float = 0.5,
) -> Any:
    r"""Damped Newtonian dynamics custom optax optimizer.

    This optimizer advances the parameters with a velocity-verlet-like scheme and
    a linear velocity damping term to minimize the energy of the system.

    Mathematical Formulation
    ------------------------
    At each step :math:`k`, the optimizer advances the parameters with:

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

    def init(params: Any) -> DampedNewtonianState:
        return DampedNewtonianState(
            vel=jax.tree.map(jnp.zeros_like, params),
            dt=jnp.array(dt),
        )

    def update(
        updates: Any,
        state: DampedNewtonianState,
        params: Any | None = None,
        **kwargs: Any,
    ) -> tuple[Any, DampedNewtonianState]:
        F = jax.tree.map(lambda u: -u, updates)
        v_half_prev = state.vel

        def update_vel(vh: jax.Array, f: jax.Array) -> jax.Array:
            vk = (vh + f * state.dt / 2.0) / (1.0 + gamma * state.dt / 2.0)
            return vk * (1.0 - gamma * state.dt / 2.0) + f * state.dt / 2.0

        v_half = jax.tree.map(update_vel, v_half_prev, F)
        updates_to_apply = jax.tree.map(lambda v: v * state.dt, v_half)

        new_state = DampedNewtonianState(vel=v_half, dt=state.dt)
        return updates_to_apply, new_state

    kw = {
        "dt": dt,
        "gamma": gamma,
    }
    return CustomGradientTransformation(
        init,
        update,
        damped_newtonian,
        kw,
        type_name="damped_newtonian",
        requires_value=False,
    )


class ConjugateGradientState(NamedTuple):
    r"""Internal state for the nonlinear conjugate gradient optimizer.

    Attributes
    ----------
    grad_prev : Any
        Gradient from the previous step, :math:`g_{k-1}` (PyTree like params).
    dir_prev : Any
        Previous search direction, :math:`d_{k-1}` (PyTree like params).
    count : jax.Array
        Iteration counter (forces a steepest-descent first step).
    """

    grad_prev: Any
    dir_prev: Any
    # NamedTuple inherits tuple.count, but this field intentionally stores the
    # optimizer iteration as an array for JAX transformations.
    count: jax.Array  # type: ignore[assignment]


def _scale_by_conjugate_gradient() -> Any:
    r"""Optax transform producing a Polak--Ribiere(+) conjugate-gradient direction.

    Maps the incoming gradient :math:`g_k` to the search direction

    .. math::
        d_k = -g_k + \beta_k \, d_{k-1}, \qquad
        \beta_k = \max\!\left(0,\, \frac{g_k \cdot (g_k - g_{k-1})}{g_{k-1}\cdot g_{k-1}}\right),

    i.e. Polak--Ribiere with the :math:`\beta \ge 0` restart (PR+). The first
    step, and any step whose direction is not downhill, fall back to steepest
    descent (Powell restart). Scale the direction with
    :func:`optax.scale_by_zoom_linesearch`.
    """

    def _dot(a: Any, b: Any) -> jax.Array:
        return sum(
            (jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))),
            start=jnp.array(0.0),
        )

    def init(params: Any) -> ConjugateGradientState:
        zeros = jax.tree.map(jnp.zeros_like, params)
        return ConjugateGradientState(
            grad_prev=zeros, dir_prev=zeros, count=jnp.zeros([], dtype=int)
        )

    def update(
        updates: Any,
        state: ConjugateGradientState,
        params: Any | None = None,
        **kwargs: Any,
    ) -> tuple[Any, ConjugateGradientState]:
        g = updates
        y = jax.tree.map(lambda gk, gp: gk - gp, g, state.grad_prev)
        denom = _dot(state.grad_prev, state.grad_prev)
        beta = jnp.where(denom > 0.0, _dot(g, y) / denom, 0.0)
        # PR+ restart (and steepest descent on the first step).
        beta = jnp.where(state.count == 0, 0.0, jnp.maximum(beta, 0.0))

        direction = jax.tree.map(lambda gk, dp: -gk + beta * dp, g, state.dir_prev)
        # Powell restart: if the direction is not downhill, reset to -g.
        not_descent = _dot(g, direction) >= 0.0
        direction = jax.tree.map(
            lambda gk, d: jnp.where(not_descent, -gk, d), g, direction
        )

        new_state = ConjugateGradientState(
            grad_prev=g, dir_prev=direction, count=state.count + 1
        )
        return direction, new_state

    return optax.GradientTransformationExtraArgs(init, update)


def conjugate_gradient(max_linesearch_steps: int = 20) -> Any:
    r"""Nonlinear conjugate gradient (Polak--Ribiere+) custom optax optimizer.

    The optimizer builds the search direction with a Polak--Ribiere(+) update
    (:func:`_scale_by_conjugate_gradient`) and chooses the step length with a
    strong-Wolfe zoom line search (:func:`optax.scale_by_zoom_linesearch`). For
    energy minimization, this optimizer uses little memory and often converges
    in fewer steps than :func:`fire <jaxdem.minimizers.optimizers.fire>`.

    .. note::
        A line search evaluates the energy several times per outer step, so
        :func:`~jaxdem.minimizers.routines.minimize` performs more than one force/energy
        evaluation per iteration (unlike FIRE). The line search requires a
        scalar objective, which ``minimize`` supplies automatically.

    Parameters
    ----------
    max_linesearch_steps : int, default 20
        Maximum number of zoom line-search iterations per step.

    Returns
    -------
    CustomGradientTransformation
        An optax gradient transformation for nonlinear conjugate gradient.

    Reference
    ---------
    Nocedal & Wright, Numerical Optimization, 2nd ed., Ch. 5 (Algorithm 5.4, PR+).
    """
    base = optax.chain(
        _scale_by_conjugate_gradient(),
        optax.scale_by_zoom_linesearch(max_linesearch_steps=max_linesearch_steps),
    )
    kw = {"max_linesearch_steps": max_linesearch_steps}
    return CustomGradientTransformation(
        base.init, base.update, conjugate_gradient, kw, type_name="conjugate_gradient"
    )
