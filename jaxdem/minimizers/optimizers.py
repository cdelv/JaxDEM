# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Custom optax optimizers for energy minimization."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Any, NamedTuple

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
    momentum : Any
        The half-step momentum (same PyTree structure as params). It equals
        the velocity when every degree of freedom has unit mass.
    dt : jax.Array
        The current step size.
    alpha : jax.Array
        The current mixing parameter.
    N_good : jax.Array
        Number of consecutive steps with positive power (:math:`P > 0`).
    """

    momentum: Any
    dt: jax.Array
    alpha: jax.Array
    N_good: jax.Array


def _tree_dot(a: Any, b: Any) -> jax.Array:
    """Inner product summed over every leaf of two matching PyTrees."""
    return sum(
        (jnp.vdot(x, y) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b))),
        start=jnp.array(0.0),
    )


def fire(
    dt: float,
    alpha_init: float = 0.1,
    f_inc: float = 1.1,
    f_dec: float = 0.5,
    f_alpha: float = 0.99,
    N_min: int = 5,
    dt_max_scale: float = 10.0,
    dt_min_scale: float = 1e-3,
) -> Any:
    r"""Fast Inertial Relaxation Engine (FIRE) custom optax optimizer.

    FIRE runs velocity-Verlet dynamics, steers the velocity toward the force,
    and adapts the time step and the steering from the power of the force
    and the velocity.

    Every degree of freedom moves with its physical mass. The optimizer reads
    the inverse mass operator :math:`M^{-1}` from the extra update argument
    ``inverse_mass``, a function that maps a force-like PyTree to the matching
    velocity. :func:`~jaxdem.minimizers.minimize` supplies body masses for
    translations and the orientation-dependent inverse inertia tensor for
    rotations. Without ``inverse_mass``, every degree of freedom has unit mass.

    Mathematical Formulation
    ------------------------
    The optimizer stores the half-step momentum :math:`p`. At each step, with
    the force :math:`F = -\nabla U` and the velocity :math:`v = M^{-1} p`:

    1. Complete the previous velocity-Verlet step and compute the power:

       .. math::
           p &\leftarrow p + F \cdot \frac{\Delta t}{2} \\
           P &= F \cdot M^{-1} p

       The power sums over every degree of freedom.

    2. Update the algorithm parameters depending on the power :math:`P`:

       - **Downhill Step (:math:`P > 0`):**

         .. math::
             N_{good} &\to N_{good} + 1 \\
             \Delta t &\to \begin{cases} \min(\Delta t \cdot f_{inc}, \Delta t_{max}) & \text{if } N_{good} > N_{min} \\ \Delta t & \text{otherwise} \end{cases} \\
             \alpha &\to \begin{cases} \alpha \cdot f_{\alpha} & \text{if } N_{good} > N_{min} \\ \alpha & \text{otherwise} \end{cases}

       - **Uphill Step (:math:`P \le 0`):**

         .. math::
             N_{good} &\to 0 \\
             \Delta t &\to \max(\Delta t \cdot f_{dec}, \Delta t_{min}) \\
             \alpha &\to \alpha_{init} \\
             x &\to x - M^{-1} p \cdot \frac{\Delta t}{2} \\
             p &\to 0

         The step backtracks half a step with the reduced time step (FIRE
         2.0) and does not move forward. The next step starts from rest at the
         backtracked configuration with a freshly evaluated force.

    3. After a downhill step, start the next velocity-Verlet step, mix the
       momentum, and move:

       .. math::
           p &\leftarrow p + F \cdot \frac{\Delta t}{2} \\
           p &\leftarrow (1 - \alpha)\, p
               + \alpha \, \frac{\lVert v \rVert_M}{\lVert F \rVert_{M^{-1}}} \, F \\
           x &\leftarrow x + M^{-1} p \cdot \Delta t

       where :math:`\lVert v \rVert_M^2 = v \cdot p` and
       :math:`\lVert F \rVert_{M^{-1}}^2 = F \cdot M^{-1} F` sum over every
       degree of freedom. This is the standard global FIRE mixing,
       :math:`v \leftarrow (1 - \alpha)\, v + \alpha \lVert v \rVert \hat{F}`,
       in the mass-weighted coordinates :math:`M^{1/2} x`. Translations and
       rotations share one norm with units of energy. For equal masses, it
       reduces to the standard form.

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

    Guénolé et al., Assessment and optimization of the fast inertial relaxation
    engine (FIRE) for energy minimization in atomistic simulations and its
    implementation in LAMMPS, Comput. Mater. Sci. 175, 109584 (2020)
    """

    def init(params: Any) -> FIREState:
        return FIREState(
            momentum=jax.tree.map(jnp.zeros_like, params),
            dt=jnp.array(dt),
            alpha=jnp.array(alpha_init),
            N_good=jnp.array(0),
        )

    def update(
        updates: Any,
        state: FIREState,
        params: Any | None = None,
        *,
        inverse_mass: Callable[[Any], Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, FIREState]:
        if inverse_mass is None:
            inverse_mass = lambda tree: tree  # noqa: E731
        F = jax.tree.map(lambda u: -u, updates)
        p_old = jax.tree.map(lambda p, f: p + f * state.dt / 2.0, state.momentum, F)
        # The inverse mass is linear, so two applications give every velocity.
        v_old = inverse_mass(p_old)
        accel = inverse_mass(F)
        downhill = _tree_dot(F, v_old) > 0.0

        N_good = jnp.where(downhill, state.N_good + 1, 0)
        accelerate = N_good > N_min
        new_dt = jnp.where(
            downhill,
            jnp.where(
                accelerate, jnp.minimum(state.dt * f_inc, dt * dt_max_scale), state.dt
            ),
            jnp.maximum(state.dt * f_dec, dt * dt_min_scale),
        )
        new_alpha = jnp.where(
            downhill,
            jnp.where(accelerate, state.alpha * f_alpha, state.alpha),
            alpha_init,
        )

        p_half = jax.tree.map(
            lambda p, f: jnp.where(downhill, p, 0.0) + f * new_dt / 2.0, p_old, F
        )
        v_half = jax.tree.map(
            lambda v, a: jnp.where(downhill, v, 0.0) + a * new_dt / 2.0, v_old, accel
        )
        speed2 = jnp.maximum(_tree_dot(v_half, p_half), 0.0)
        force2 = jnp.maximum(_tree_dot(F, accel), 0.0)
        mixing = new_alpha * jnp.where(
            force2 > 0.0, jnp.sqrt(speed2 / jnp.where(force2 > 0.0, force2, 1.0)), 0.0
        )

        def mix(x: jax.Array, f: jax.Array) -> jax.Array:
            return jnp.where(downhill, x * (1.0 - new_alpha) + f * mixing, 0.0)

        p_half = jax.tree.map(mix, p_half, F)
        v_half = jax.tree.map(mix, v_half, accel)
        updates_to_apply = jax.tree.map(
            lambda vo, vh: jnp.where(downhill, vh * new_dt, -vo * new_dt / 2.0),
            v_old,
            v_half,
        )

        new_state = FIREState(
            momentum=p_half,
            dt=new_dt,
            alpha=new_alpha,
            N_good=N_good,
        )
        return updates_to_apply, new_state

    kw = {
        "dt": dt,
        "alpha_init": alpha_init,
        "f_inc": f_inc,
        "f_dec": f_dec,
        "f_alpha": f_alpha,
        "N_min": N_min,
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
        denom = _tree_dot(state.grad_prev, state.grad_prev)
        beta = jnp.where(denom > 0.0, _tree_dot(g, y) / denom, 0.0)
        # PR+ restart (and steepest descent on the first step).
        beta = jnp.where(state.count == 0, 0.0, jnp.maximum(beta, 0.0))

        direction = jax.tree.map(lambda gk, dp: -gk + beta * dp, g, state.dir_prev)
        # Powell restart: if the direction is not downhill, reset to -g.
        not_descent = _tree_dot(g, direction) >= 0.0
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
