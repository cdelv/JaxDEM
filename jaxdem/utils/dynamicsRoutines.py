# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Jit-compiled routines for controlling temperature and density via basic rescaling.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from dataclasses import replace
from functools import partial
from typing import TYPE_CHECKING, Callable, Optional, Tuple

from .thermal import compute_temperature, scale_to_temperature, set_temperature
from .packingUtils import compute_packing_fraction, scale_to_packing_fraction

if TYPE_CHECKING:  # pragma: no cover
    from ..state import State
    from ..system import System

# Schedule signature:
#   k: 1..K (rescale event index)
#   K: total number of rescale events over the protocol
#   start: initial value
#   target: final value
# returns: setpoint to apply at event k
ScheduleFn = Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]


def _linear_schedule(k: jax.Array, K: jax.Array, start: jax.Array, target: jax.Array) -> jax.Array:
    Kf = jnp.maximum(K.astype(float), 1.0)
    alpha = k.astype(float) / Kf
    return start + alpha * (target - start)


def _resolve_target(
    start: jax.Array,
    *,
    target: Optional[float],
    delta: Optional[float],
) -> Tuple[bool, jax.Array]:
    """Returns (enabled, final_target). If neither target nor delta is provided, control is disabled."""
    if target is None and delta is None:
        return False, start
    if target is not None and delta is not None:
        # Python-side error at trace time (intentional; ambiguous input)
        raise ValueError("Provide either target=... or delta=..., not both.")
    if target is not None:
        return True, jnp.asarray(target, dtype=float)
    return True, start + jnp.asarray(delta, dtype=float)  # delta is not None here


def _zero_velocities(state: "State", can_rotate: bool) -> "State":
    # Use replace to avoid in-place mutation surprises.
    return replace(
        state,
        vel=jnp.zeros_like(state.vel),
        angVel=jnp.zeros_like(state.angVel) * can_rotate,
    )


def _maybe_init_temperature_if_zero(
    state: "State",
    *,
    enabled: bool,
    start_setpoint: jax.Array,
    can_rotate: bool,
    subtract_drift: bool,
    k_B: float,
    seed: int,
) -> "State":
    """Guard: if initial T == 0 and we intend to control temperature, initialize velocities to start_setpoint."""
    if not enabled:
        return state

    T0 = compute_temperature(state, can_rotate, subtract_drift, k_B)

    def init_nonzero(_):
        # If requested start_setpoint is 0, we can just zero velocities deterministically.
        return jax.lax.cond(
            start_setpoint <= 0.0,
            lambda __: _zero_velocities(state, can_rotate),
            lambda __: set_temperature(
                state,
                start_setpoint,
                can_rotate,
                subtract_drift,
                seed=seed,  # IMPORTANT: must not be None inside jit
                k_B=k_B,
            ),
            operand=None,
        )

    return jax.lax.cond(T0 <= 0.0, init_nonzero, lambda _: state, operand=None)


def _controlled_steps_chunk(
    state: "State",
    system: "System",
    *,
    n: int,
    unroll: int,
    # protocol context (to keep schedules consistent across chunking / rollout)
    step0: jax.Array,          # step_count at protocol start
    total_n: int,              # total integration steps in the whole protocol (not just this chunk)
    rescale_every: int,
    # temperature control config
    temp_enabled: bool,
    T_start: jax.Array,
    T_target: jax.Array,
    temperature_schedule: Optional[ScheduleFn],
    can_rotate: bool,
    subtract_drift: bool,
    k_B: float,
    # density control config
    dens_enabled: bool,
    pf_start: jax.Array,
    pf_target: jax.Array,
    density_schedule: Optional[ScheduleFn],
    pf_min: float,
) -> Tuple["State", "System"]:
    """Runs n integration steps with optional rescaling hooks; jittable and chunkable."""
    f = rescale_every
    if f <= 0:
        # No rescaling at all; just delegate to the fast path.
        return system._steps(state, system, n, unroll=unroll)

    schedule_T = _linear_schedule if temperature_schedule is None else temperature_schedule
    schedule_pf = _linear_schedule if density_schedule is None else density_schedule

    # total number of rescale events over the *entire* protocol
    K = ((step0 + total_n) // f) - (step0 // f)

    @partial(jax.named_call, name="dynamicsRoutines._controlled_steps_chunk.body")
    def body(carry, _):
        st, sys = carry

        # --- identical to System._steps body (with the hook inserted later) ---
        sys = replace(sys, time=sys.time + sys.dt, step_count=sys.step_count + 1)

        st, sys = sys.domain.apply(st, sys)
        st, sys = sys.linear_integrator.step_before_force(st, sys)
        st, sys = sys.rotation_integrator.step_before_force(st, sys)

        st, sys = sys.force_manager.apply(st, sys)
        st, sys = sys.collider.compute_force(st, sys)

        st, sys = sys.linear_integrator.step_after_force(st, sys)
        st, sys = sys.rotation_integrator.step_after_force(st, sys)
        # ---------------------------------------------------------------

        do_rescale = (sys.step_count % f) == 0

        def apply_rescale(carry2):
            st2, sys2 = carry2

            # rescale-event index (1..K) at the current step
            k = (sys2.step_count // f) - (step0 // f)

            # --- temperature rescaling ---
            def do_temp(_):
                T_set = schedule_T(k, K, T_start, T_target)
                T_set = jnp.maximum(T_set, 0.0)

                # Guard: if target is 0, avoid 0/0 in scale_to_temperature; just zero velocities.
                return jax.lax.cond(
                    T_set <= 0.0,
                    lambda __: _zero_velocities(st2, can_rotate),
                    lambda __: scale_to_temperature(st2, T_set, can_rotate, subtract_drift, k_B=k_B),
                    operand=None,
                )

            st3 = jax.lax.cond(temp_enabled, do_temp, lambda _: st2, operand=None)

            # --- density rescaling ---
            def do_dens(_):
                pf_set = schedule_pf(k, K, pf_start, pf_target)

                # Guard: if pf_set <= 0, warn and clamp to a tiny positive value to avoid NaNs.
                def warn_and_clamp(__):
                    jax.debug.print(
                        "Warning: requested packing fraction <= 0 (pf_set={pf}). Clamping to {pf_min}.",
                        pf=pf_set,
                        pf_min=pf_min,
                    )
                    return jnp.asarray(pf_min, dtype=float)

                pf_set2 = jax.lax.cond(pf_set <= 0.0, warn_and_clamp, lambda __: pf_set, operand=None)
                return scale_to_packing_fraction(st3, sys2, pf_set2)

            st4, sys4 = jax.lax.cond(dens_enabled, do_dens, lambda _: (st3, sys2), operand=None)
            return st4, sys4

        st, sys = jax.lax.cond(do_rescale, apply_rescale, lambda x: x, operand=(st, sys))
        return (st, sys), None

    (state, system), _ = jax.lax.scan(body, (state, system), xs=None, length=n, unroll=unroll)
    return state, system


@partial(
    jax.jit,
    static_argnames=(
        "n",
        "unroll",
        "rescale_every",
        "temperature_schedule",
        "density_schedule",
        "can_rotate",
        "subtract_drift",
        "pf_min",
        "init_temp_seed",
    ),
    donate_argnames=("state", "system"),
)
def control_nvt_density(
    state: "State",
    system: "System",
    *,
    n: int,
    rescale_every: int,
    # temperature control: choose one of (temperature_target, temperature_delta) or neither
    temperature_target: Optional[float] = None,
    temperature_delta: Optional[float] = None,
    # density control: choose one of (packing_fraction_target, packing_fraction_delta) or neither
    packing_fraction_target: Optional[float] = None,
    packing_fraction_delta: Optional[float] = None,
    # dynamics/thermo params
    can_rotate: bool = True,
    subtract_drift: bool = True,
    k_B: float = 1.0,
    # schedule overrides (must be JIT-static callables)
    temperature_schedule: Optional[ScheduleFn] = None,
    density_schedule: Optional[ScheduleFn] = None,
    # safety controls
    pf_min: float = 1e-12,
    init_temp_seed: int = 0,
    unroll: int = 2,
) -> Tuple["State", "System"]:
    """
    Runs a protocol for n integration steps, applying (optional) NVT rescaling and/or density rescaling
    whenever system.step_count is divisible by rescale_every.

    Notes
    - rescale_every is in *integration steps* (System.step_count units).
    - Provide either target or delta for each controlled quantity (or neither to disable).
    - temperature_schedule / density_schedule must be JIT-static (passed as static_argnames).
    """
    step0 = system.step_count
    total_n = n

    # --- determine starts ---
    T0 = compute_temperature(state, can_rotate, subtract_drift, k_B)
    pf0 = compute_packing_fraction(state, system)

    temp_enabled, T_target = _resolve_target(T0, target=temperature_target, delta=temperature_delta)
    dens_enabled, pf_target = _resolve_target(pf0, target=packing_fraction_target, delta=packing_fraction_delta)

    # For temperature, if T0==0 and control is enabled, initialize to the "start setpoint".
    # Default start setpoint is just T0 (if nonzero), otherwise use the final target (common desired behavior).
    T_start = jax.lax.cond(T0 > 0.0, lambda _: T0, lambda _: T_target, operand=None)
    state = _maybe_init_temperature_if_zero(
        state,
        enabled=temp_enabled,
        start_setpoint=T_start,
        can_rotate=can_rotate,
        subtract_drift=subtract_drift,
        k_B=k_B,
        seed=init_temp_seed,
    )

    # Recompute starts after possible initialization (important if T0 was 0).
    T_start = compute_temperature(state, can_rotate, subtract_drift, k_B)
    pf_start = compute_packing_fraction(state, system)

    state, system = _controlled_steps_chunk(
        state,
        system,
        n=n,
        unroll=unroll,
        step0=step0,
        total_n=total_n,
        rescale_every=rescale_every,
        temp_enabled=temp_enabled,
        T_start=T_start,
        T_target=T_target,
        temperature_schedule=temperature_schedule,
        can_rotate=can_rotate,
        subtract_drift=subtract_drift,
        k_B=k_B,
        dens_enabled=dens_enabled,
        pf_start=pf_start,
        pf_target=pf_target,
        density_schedule=density_schedule,
        pf_min=pf_min,
    )
    return state, system


@partial(
    jax.jit,
    static_argnames=(
        "n",
        "stride",
        "unroll",
        "rescale_every",
        "temperature_schedule",
        "density_schedule",
        "can_rotate",
        "subtract_drift",
        "pf_min",
        "init_temp_seed",
    ),
    donate_argnames=("state", "system"),
)
def control_nvt_density_rollout(
    state: "State",
    system: "System",
    *,
    n: int,                 # number of saved frames
    stride: int = 1,        # integration steps between frames (like System.trajectory_rollout)
    rescale_every: int = 1,
    temperature_target: Optional[float] = None,
    temperature_delta: Optional[float] = None,
    packing_fraction_target: Optional[float] = None,
    packing_fraction_delta: Optional[float] = None,
    can_rotate: bool = True,
    subtract_drift: bool = True,
    k_B: float = 1.0,
    temperature_schedule: Optional[ScheduleFn] = None,
    density_schedule: Optional[ScheduleFn] = None,
    pf_min: float = 1e-12,
    init_temp_seed: int = 0,
    unroll: int = 2,
) -> Tuple["State", "System", Tuple["State", "System"]]:
    """Rollout variant (like System.trajectory_rollout), with globally-consistent schedules across the whole rollout."""
    step0 = system.step_count
    total_n = n * stride

    T0 = compute_temperature(state, can_rotate, subtract_drift, k_B)
    pf0 = compute_packing_fraction(state, system)

    temp_enabled, T_target = _resolve_target(T0, target=temperature_target, delta=temperature_delta)
    dens_enabled, pf_target = _resolve_target(pf0, target=packing_fraction_target, delta=packing_fraction_delta)

    T_start = jax.lax.cond(T0 > 0.0, lambda _: T0, lambda _: T_target, operand=None)
    state = _maybe_init_temperature_if_zero(
        state,
        enabled=temp_enabled,
        start_setpoint=T_start,
        can_rotate=can_rotate,
        subtract_drift=subtract_drift,
        k_B=k_B,
        seed=init_temp_seed,
    )

    T_start = compute_temperature(state, can_rotate, subtract_drift, k_B)
    pf_start = compute_packing_fraction(state, system)

    def frame_body(carry, _):
        st, sys = carry
        st, sys = _controlled_steps_chunk(
            st,
            sys,
            n=stride,
            unroll=unroll,
            step0=step0,
            total_n=total_n,
            rescale_every=rescale_every,
            temp_enabled=temp_enabled,
            T_start=T_start,
            T_target=T_target,
            temperature_schedule=temperature_schedule,
            can_rotate=can_rotate,
            subtract_drift=subtract_drift,
            k_B=k_B,
            dens_enabled=dens_enabled,
            pf_start=pf_start,
            pf_target=pf_target,
            density_schedule=density_schedule,
            pf_min=pf_min,
        )
        return (st, sys), (st, sys)

    if state.batch_size > 1:
        frame_body = jax.vmap(frame_body, in_axes=(0, None))

    (state, system), traj = jax.lax.scan(frame_body, (state, system), xs=None, length=n)
    return state, system, traj