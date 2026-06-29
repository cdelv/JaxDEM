# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Jamming routines.
https://doi.org/10.1103/PhysRevE.68.011306.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from functools import partial

from typing import TYPE_CHECKING, Any, NamedTuple

from .contacts import compute_contact_pressure
from .packing_utils import (
    _host_body_grouping,
    _scale_to_packing_fraction_grouped,
    compute_packing_fraction,
    compute_particle_volume,
)

if TYPE_CHECKING:
    from ..state import State
    from ..system import System


class JamResult(NamedTuple):
    """Result of :func:`bisection_jam`.

    Behaves like the historical 6-tuple (same field order), but the named
    fields make the intent explicit at the call site, e.g.
    ``result.jammed_state`` instead of ``result[2]``.
    """

    unjammed_state: "State"
    """Last *unjammed* state visited by the bisection."""
    unjammed_system: "System"
    """System matching :attr:`unjammed_state`."""
    jammed_state: "State"
    """The jammed state (usually what you want)."""
    jammed_system: "System"
    """System matching :attr:`jammed_state`."""
    packing_fraction: jax.Array
    """Packing fraction of the jammed state."""
    potential_energy: jax.Array
    """Per-particle potential energy of the jammed state."""


@partial(
    jax.jit, static_argnames=["n_minimization_steps", "n_jamming_steps", "verbose"]
)
def bisection_jam(
    state: State,
    system: System,
    n_minimization_steps: int = 1000000,
    pe_tol: float = 1e-16,
    pe_diff_tol: float = 1e-16,
    n_jamming_steps: int = 10000,
    packing_fraction_tolerance: float = 1e-10,
    packing_fraction_increment: float = 1e-3,
    verbose: bool = True,
) -> JamResult:
    """Find the nearest jammed state for a given state and system.
    Uses bisection search with state reversion.

    Parameters
    ----------
    state : State
        The state to jam.
    system : System
        The system to jam.
    n_minimization_steps : int, optional
        The number of steps to take in the minimization.  Should be large.  Typically 1e6.
    pe_tol : float, optional
        The tolerance for the potential energy.  Should be very small.  Typically 1e-16.
    pe_diff_tol : float, optional
        The tolerance for the difference in potential energy across subsequent steps.  Should be very small.  Typically 1e-16.
    n_jamming_steps : int, optional
        The number of steps in the jamming loop.  Typically 1e4.
    packing_fraction_tolerance : float, optional
        The tolerance for the packing fraction to determine convergence.  Typically 1e-10
    packing_fraction_increment : float, optional
        The initial increment for the packing fraction.  Typically 1e-3.  Larger increments make it faster in the unjammed region, but makes minimization of the earliest detected jammed states take much longer.
    verbose : bool, optional
        If ``True`` (default), print per-iteration progress via
        ``jax.debug.print``. Set to ``False`` to silence the prints and avoid
        the per-iteration host callbacks they incur.

    Returns
    -------
    JamResult
        A named tuple ``(unjammed_state, unjammed_system, jammed_state,
        jammed_system, packing_fraction, potential_energy)``; unpacking it
        like the historical 6-tuple keeps working.

    """
    # cannot proceed if the initial state is jammed
    state, system, n_steps, final_pe = system.minimize(
        state,
        system,
        max_steps=n_minimization_steps,
        pe_tol=pe_tol,
        pe_diff_tol=pe_diff_tol,
    )
    is_initially_jammed = final_pe > pe_tol

    def print_warning() -> None:
        jax.debug.print(
            "Warning: Initial state is already jammed (PE={pe} > tol={tol}). Skipping.",
            pe=final_pe,
            tol=pe_tol,
        )
        return

    if verbose:
        jax.lax.cond(is_initially_jammed, print_warning, lambda: None)
        jax.debug.print("Initial minimization took {n_steps} steps.", n_steps=n_steps)
    initial_packing_fraction = compute_packing_fraction(state, system)

    # Body grouping depends only on the (static) bond/clump topology: pay the
    # host callback once here instead of once per bisection iteration (which
    # would force a host round-trip per loop step and break async dispatch).
    group_id = jax.pure_callback(
        _host_body_grouping,
        jax.ShapeDtypeStruct((state.N,), int),  # type: ignore[no-untyped-call]
        state.clump_id,
        state.bond_id,
        vmap_method="sequential",
    )

    init_carry = (
        0,  # iteration
        is_initially_jammed,  # is_jammed
        state,
        system,  # current state/system
        state,
        system,  # last unjammed state/system
        initial_packing_fraction,  # current packing fraction
        initial_packing_fraction,  # low packing fraction
        -1.0,  # high packing fraction (initially set to -1.0)
        final_pe,  # final potential energy
    )

    def cond_fun(carry: tuple[Any, ...]) -> jax.Array:
        i, is_jammed, _, _, _, _, _, _, _, _ = carry
        return (i < n_jamming_steps) & (~is_jammed)

    def body_fun(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        i, _, state, system, last_state, last_system, pf, pf_low, pf_high, _ = carry

        # minimize the state
        state, system, n_steps, final_pe = system.minimize(
            state,
            system,
            max_steps=n_minimization_steps,
            pe_tol=pe_tol,
            pe_diff_tol=pe_diff_tol,
        )

        is_jammed = final_pe > pe_tol

        def jammed_branch(_: None) -> tuple[Any, ...]:
            new_pf_high = pf
            new_pf = (new_pf_high + pf_low) / 2.0
            return (
                new_pf,
                pf_low,
                new_pf_high,
                last_state,
                last_system,
                last_state,
                last_system,
            )

        def unjammed_branch(
            _: None,
        ) -> tuple[
            Any, ...
        ]:  # if unjammed, save current as last unjammed, increment or bisect
            new_last_state = state
            new_last_system = system
            new_pf_low = pf

            def bisect() -> (
                jax.Array
            ):  # if a jammed state is known, perform a bisection search
                return (pf_high + new_pf_low) / 2.0

            def increment() -> (
                jax.Array
            ):  # if no jammed state is known, increment the packing fraction
                return new_pf_low + packing_fraction_increment

            new_pf = jax.lax.cond(pf_high > 0, bisect, increment)
            return (
                new_pf,
                new_pf_low,
                pf_high,
                state,
                system,
                new_last_state,
                new_last_system,
            )

        (
            new_pf,
            new_pf_low,
            new_pf_high,
            new_state,
            new_system,
            new_last_state,
            new_last_system,
        ) = jax.lax.cond(is_jammed, jammed_branch, unjammed_branch, operand=None)

        # check if the packing fraction is converged and print
        ratio = new_pf_high / new_pf_low
        is_jammed = (jnp.abs(ratio - 1.0) < packing_fraction_tolerance) & (
            new_pf_high > 0
        )
        if verbose:
            jax.debug.print(
                "Step: {i} -  phi={pf}, PE={pe} after {n_steps} steps",
                i=i + 1,
                pf=pf,
                pe=final_pe,
                n_steps=n_steps,
            )

        next_state, next_system = _scale_to_packing_fraction_grouped(
            new_state, new_system, new_pf, group_id
        )

        return (
            i + 1,
            is_jammed,
            next_state,
            next_system,
            new_last_state,
            new_last_system,
            new_pf,
            new_pf_low,
            new_pf_high,
            final_pe,
        )

    final_carry = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    _, _, _, _, last_state, last_system, final_pf, _, pf_high, final_pe = final_carry
    last_jammed_pf = jnp.where(pf_high > 0, pf_high, final_pf)
    last_jammed_state, last_jammed_system = _scale_to_packing_fraction_grouped(
        last_state, last_system, last_jammed_pf, group_id
    )
    last_jammed_state, last_jammed_system, _, final_pe = last_jammed_system.minimize(
        last_jammed_state,
        last_jammed_system,
        max_steps=n_minimization_steps,
        pe_tol=pe_tol,
        pe_diff_tol=pe_diff_tol,
    )
    return JamResult(
        unjammed_state=last_state,
        unjammed_system=last_system,
        jammed_state=last_jammed_state,
        jammed_system=last_jammed_system,
        packing_fraction=final_pf,
        potential_energy=final_pe,
    )


def pressure_bisection_jam(
    state: State,
    system: System,
    *,
    n_minimization_steps: int = 1_000_000,
    pe_tol: float = 1e-16,
    pe_diff_tol: float = 1e-16,
    pressure_threshold: float = 1e-7,
    pressure_band_factor: float = 1.01,
    growth_rate: float = 1.001,
    fine_growth_rate: float = 1.000001,
    length_ratio_tolerance: float = 1e-14,
    n_jamming_steps: int = 10_000,
    pressure_cutoff: float | None = None,
    pressure_max_neighbors: int | None = None,
    verbose: bool = True,
) -> JamResult:
    r"""Find the nearest jammed state via a *pressure-band* bisection search.

    This is a JaxDEM port of the classic single-system C++ ``Disk::Jam``
    routine. Where :func:`bisection_jam` works in packing-fraction space and
    classifies a state as jammed/unjammed with a single potential-energy
    threshold, this routine mirrors the C++ algorithm faithfully:

    * The control variable is the **characteristic box length**
      ``L = prod(box_size) ** (1 / dim)``. Compression decreases ``L`` and the
      bisection is performed *linearly in* ``L`` (not in packing fraction).
    * The jamming criterion is a **pressure band** ``[P_lo, P_hi]`` with
      ``P_lo = pressure_threshold`` and ``P_hi = pressure_band_factor * P_lo``.
      A configuration is *unjammed* if ``P < P_lo``, *over-compressed* if
      ``P > P_hi``, and *accepted* (a successful jammed packing) if ``P`` lands
      inside the band.
    * Two phases are used, exactly as in the original: a **coarse** phase that
      multiplicatively compresses by ``growth_rate`` until the first
      over-compression brackets the jamming point, followed by a **fine** phase
      (``fine_growth_rate``) that bisects until either the pressure lands in the
      band or the bracket collapses to ``|L_hi / L_lo - 1| < length_ratio_tolerance``.

    Unlike :func:`bisection_jam`, this routine relies on host-side control flow
    and on :func:`~jaxdem.utils.contacts.compute_contact_pressure` (which is not
    ``jit``-safe), so it runs on a **single system** and is neither ``jit``-ed
    nor ``vmap``-able. Loop over systems in Python (or use :func:`bisection_jam`)
    if you need many packings.

    .. note::
        The default ``pressure_threshold`` (``1e-7``) comes from the original
        C++ code's unit system. Pressure scales with the contact stiffness, so
        you will typically need to tune ``pressure_threshold`` to your own
        units to obtain a meaningfully marginal packing.

    Parameters
    ----------
    state, system
        The (single) state/system to jam. Assumed to start *unjammed*; if the
        initial minimized pressure already exceeds ``P_hi`` the routine warns
        and returns the input unchanged.
    n_minimization_steps : int, optional
        Maximum FIRE iterations per minimization. Typically ``1e6``.
    pe_tol, pe_diff_tol : float, optional
        Minimizer convergence tolerances.
    pressure_threshold : float, optional
        Lower edge ``P_lo`` of the target pressure band.
    pressure_band_factor : float, optional
        ``P_hi = pressure_band_factor * P_lo`` (``> 1``). Default ``1.01``.
    growth_rate : float, optional
        Coarse multiplicative compression rate (``> 1``). Each unjammed step
        shrinks the box as ``L /= growth_rate``. Default ``1.001``.
    fine_growth_rate : float, optional
        Compression rate used in the refinement phase. Default ``1.000001``.
    length_ratio_tolerance : float, optional
        Convergence tolerance on ``|L_hi / L_lo - 1|``. Default ``1e-14``.
    n_jamming_steps : int, optional
        Hard cap on the total number of outer (minimize + classify) iterations
        across both phases. Default ``1e4``.
    pressure_cutoff, pressure_max_neighbors : optional
        Forwarded to :func:`~jaxdem.utils.contacts.compute_contact_pressure`.
    verbose : bool, optional
        If ``True`` (default), print per-iteration progress.

    Returns
    -------
    JamResult
        ``(unjammed_state, unjammed_system, jammed_state, jammed_system,
        packing_fraction, potential_energy)`` for the jammed packing, matching
        :func:`bisection_jam`.
    """
    p_lo = float(pressure_threshold)
    p_hi = float(pressure_band_factor) * p_lo
    dim = int(state.dim)

    # Total particle volume is fixed during jamming (radii do not change), so
    # the box length maps to a packing fraction via phi = V / L**dim.
    volume = float(compute_particle_volume(state))

    def length_of(system: System) -> float:
        return float(jnp.prod(system.domain.box_size)) ** (1.0 / dim)

    def packing_fraction_for_length(length: float) -> float:
        return volume / (length**dim)

    # Body grouping depends only on the (static) topology; compute it once.
    group_id = jnp.asarray(_host_body_grouping(state.clump_id, state.bond_id))

    def pressure_of(state: State, system: System) -> tuple[State, System, float]:
        state, system, pressure = compute_contact_pressure(
            state, system, pressure_cutoff, pressure_max_neighbors
        )
        return state, system, float(pressure)

    # Initial relaxation and over-compression guard.
    state, system, _, final_pe = system.minimize(
        state,
        system,
        max_steps=n_minimization_steps,
        pe_tol=pe_tol,
        pe_diff_tol=pe_diff_tol,
    )
    state, system, pressure = pressure_of(state, system)
    if pressure > p_hi:
        if verbose:
            print(
                f"Warning: Initial state is already over-compressed "
                f"(P={pressure:.5e} > P_hi={p_hi:.5e}). Skipping."
            )
        return JamResult(
            unjammed_state=state,
            unjammed_system=system,
            jammed_state=state,
            jammed_system=system,
            packing_fraction=compute_packing_fraction(state, system),
            potential_energy=jnp.asarray(final_pe),
        )

    length = length_of(system)
    # Bracket bounds: L_hi is the largest *unjammed* box seen, L_lo the
    # smallest *over-compressed* box seen (mirrors L_h / L_l in the C++ code).
    # A bound is "unknown" while it is negative.
    length_hi = -1.0
    length_lo = -1.0

    # Last fully relaxed *unjammed* configuration; every new box is produced by
    # affinely rescaling this reference (== the C++ ``x_old`` reversion).
    last_state, last_system = state, system

    iteration = 0
    success = False
    final_pe = float(final_pe)

    def do_step(
        state: State,
        system: System,
        last_state: State,
        last_system: System,
        length: float,
        length_hi: float,
        length_lo: float,
        rate: float,
        break_on_over: bool,
        check_convergence: bool,
    ) -> tuple[State, System, State, System, float, float, float, float, str]:
        state, system, _, pe = system.minimize(
            state,
            system,
            max_steps=n_minimization_steps,
            pe_tol=pe_tol,
            pe_diff_tol=pe_diff_tol,
        )
        state, system, pressure = pressure_of(state, system)
        pe = float(pe)

        if verbose:
            print(
                f"Step {iteration}: L={length:.8e}, "
                f"phi={packing_fraction_for_length(length):.8e}, "
                f"P={pressure:.5e}, PE={pe:.5e}"
            )

        status = "continue"
        if pressure < p_lo:  # unjammed -> compress further
            last_state, last_system = state, system
            length_hi = length
            if length_lo > 0.0:  # bracket known: bisect, then resume growth
                length = 0.5 * (length_hi + length_lo)
                length_lo = -1.0
            else:
                length /= rate
        elif pressure > p_hi:  # over-compressed -> record bound and bisect
            length_lo = length
            length = 0.5 * (length_hi + length_lo)
            if break_on_over:
                status = "break"
        else:  # pressure inside the band -> success
            status = "success"

        if (
            check_convergence
            and length_hi > 0.0
            and length_lo > 0.0
            and abs(length_hi / length_lo - 1.0) < length_ratio_tolerance
        ):
            status = "converged"

        # Produce the next trial box by rescaling the last unjammed reference,
        # unless we have already accepted a packing.
        if status in ("continue", "break"):
            state, system = _scale_to_packing_fraction_grouped(
                last_state,
                last_system,
                packing_fraction_for_length(length),
                group_id,
            )

        return (
            state,
            system,
            last_state,
            last_system,
            length,
            length_hi,
            length_lo,
            pe,
            status,
        )

    # Phase 1: coarse compression until the first over-compression brackets it.
    status = "continue"
    while iteration < n_jamming_steps and status == "continue":
        (
            state,
            system,
            last_state,
            last_system,
            length,
            length_hi,
            length_lo,
            final_pe,
            status,
        ) = do_step(
            state,
            system,
            last_state,
            last_system,
            length,
            length_hi,
            length_lo,
            growth_rate,
            break_on_over=True,
            check_convergence=False,
        )
        iteration += 1
    success = status == "success"

    # Phase 2: fine bisection to the pressure band or the length tolerance.
    if not success:
        status = "continue"
        while iteration < n_jamming_steps and status == "continue":
            (
                state,
                system,
                last_state,
                last_system,
                length,
                length_hi,
                length_lo,
                final_pe,
                status,
            ) = do_step(
                state,
                system,
                last_state,
                last_system,
                length,
                length_hi,
                length_lo,
                fine_growth_rate,
                break_on_over=False,
                check_convergence=True,
            )
            iteration += 1
        success = status == "success"

    if verbose and not success:
        print(
            "Warning: pressure band not reached; returning the marginally "
            "jammed bracket bound."
        )

    # Recover the jammed packing: rescale the last unjammed configuration to the
    # jammed box length and relax it once more. On success ``length`` already
    # holds the in-band box; otherwise fall back to the over-compressed bound.
    jammed_length = length if success else (length_lo if length_lo > 0.0 else length)
    jammed_state, jammed_system = _scale_to_packing_fraction_grouped(
        last_state, last_system, packing_fraction_for_length(jammed_length), group_id
    )
    jammed_state, jammed_system, _, final_pe = jammed_system.minimize(
        jammed_state,
        jammed_system,
        max_steps=n_minimization_steps,
        pe_tol=pe_tol,
        pe_diff_tol=pe_diff_tol,
    )

    return JamResult(
        unjammed_state=last_state,
        unjammed_system=last_system,
        jammed_state=jammed_state,
        jammed_system=jammed_system,
        packing_fraction=compute_packing_fraction(jammed_state, jammed_system),
        potential_energy=jnp.asarray(final_pe),
    )


@partial(
    jax.jit, static_argnames=["n_minimization_steps", "n_jamming_steps", "verbose"]
)
def pe_band_jam(
    state: State,
    system: System,
    n_minimization_steps: int = 1_000_000,
    pe_tol: float = 1e-16,
    pe_diff_tol: float = 1e-16,
    pe_band_factor: float = 2.0,
    packing_fraction_increment: float = 1e-3,
    n_jamming_steps: int = 10_000,
    verbose: bool = True,
) -> JamResult:
    r"""Find a jammed state via an adaptive, halving packing-fraction step.

    This is a third jamming strategy that, like :func:`bisection_jam`, works in
    packing-fraction space and uses the per-particle potential energy as its
    criterion -- but instead of a single jammed/unjammed threshold it targets a
    **potential-energy band** ``[pe_tol, pe_band_factor * pe_tol]`` with an
    *adaptive step size*:

    * Start from ``packing_fraction_increment`` (typically ``1e-3``).
    * If ``PE/N < pe_tol`` the configuration is under-compressed -> **compress**
      (increase the packing fraction by the current increment, shrinking the
      box).
    * If ``PE/N > pe_band_factor * pe_tol`` it is over-compressed -> **expand**
      (decrease the packing fraction).
    * Otherwise ``PE/N`` is inside the band -> **exit** (success).

    Every time the search *reverses direction* (compress -> expand or
    expand -> compress) the increment is **halved**, so the step adaptively
    refines once it brackets the band -- a self-bracketing bisection that needs
    no separately tracked bracket bounds.

    Unlike :func:`bisection_jam` and :func:`pressure_bisection_jam`, this routine
    **does not revert** to the last sub-threshold configuration: each new box is
    produced by affinely rescaling the *current* (just-minimized) state. The
    minimizer already returns ``PE/N`` directly, so this routine -- like
    :func:`bisection_jam` -- is fully ``jit``/``vmap`` compatible.

    Parameters
    ----------
    state, system
        The state/system to jam.
    n_minimization_steps : int, optional
        Maximum FIRE iterations per minimization. Typically ``1e6``.
    pe_tol, pe_diff_tol : float, optional
        Minimizer convergence tolerances. ``pe_tol`` also sets the lower edge of
        the target PE band.
    pe_band_factor : float, optional
        The PE band is ``[pe_tol, pe_band_factor * pe_tol]`` (``> 1``).
        Default ``2.0`` (i.e. the upper edge is ``2 * pe_tol``).
    packing_fraction_increment : float, optional
        Initial packing-fraction step. Default ``1e-3``.
    n_jamming_steps : int, optional
        Hard cap on the number of (minimize + classify) iterations.
        Default ``1e4``.
    verbose : bool, optional
        If ``True`` (default), print per-iteration progress via
        ``jax.debug.print``.

    Returns
    -------
    JamResult
        ``(unjammed_state, unjammed_system, jammed_state, jammed_system,
        packing_fraction, potential_energy)``. ``unjammed_state`` is the most
        recent configuration seen with ``PE/N < pe_tol`` (defaulting to the
        input if none was seen); ``jammed_state`` is the final in-band packing.
    """
    pe_lo = pe_tol
    pe_hi = pe_band_factor * pe_tol

    initial_packing_fraction = compute_packing_fraction(state, system)

    # Body grouping depends only on the (static) topology; compute it once.
    group_id = jax.pure_callback(
        _host_body_grouping,
        jax.ShapeDtypeStruct((state.N,), int),  # type: ignore[no-untyped-call]
        state.clump_id,
        state.bond_id,
        vmap_method="sequential",
    )

    init_carry = (
        0,  # iteration
        jnp.asarray(False),  # done
        state,
        system,  # current state/system
        state,
        system,  # last sub-threshold ("unjammed") state/system
        initial_packing_fraction,  # current packing fraction
        jnp.asarray(packing_fraction_increment, float),  # current increment
        jnp.asarray(0, int),  # previous step direction in {-1, 0, +1}
        jnp.asarray(jnp.inf),  # final PE/N
    )

    def cond_fun(carry: tuple[Any, ...]) -> jax.Array:
        i, done, *_ = carry
        return (i < n_jamming_steps) & (~done)

    def body_fun(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        (
            i,
            _,
            state,
            system,
            last_state,
            last_system,
            pf,
            increment,
            prev_dir,
            _,
        ) = carry

        state, system, n_steps, pe = system.minimize(
            state,
            system,
            max_steps=n_minimization_steps,
            pe_tol=pe_tol,
            pe_diff_tol=pe_diff_tol,
        )

        below = pe < pe_lo  # under-compressed -> compress
        above = pe > pe_hi  # over-compressed -> expand
        done = ~(below | above)  # inside the band -> success

        direction = jnp.where(below, 1, jnp.where(above, -1, 0))

        # Halve the increment whenever the search reverses direction.
        switched = (prev_dir != 0) & (direction != 0) & (direction != prev_dir)
        new_increment = jnp.where(switched, 0.5 * increment, increment)

        new_pf = pf + direction.astype(pf.dtype) * new_increment

        # Track the most recent below-band configuration purely for the return
        # value; the search itself never reverts to it.
        new_last_state, new_last_system = jax.lax.cond(
            below,
            lambda: (state, system),
            lambda: (last_state, last_system),
        )

        new_prev_dir = jnp.where(done, prev_dir, direction)

        # Rescale the *current* (non-reverted) state to the new box. On success
        # leave the accepted in-band state untouched.
        next_state, next_system = jax.lax.cond(
            done,
            lambda: (state, system),
            lambda: _scale_to_packing_fraction_grouped(
                state, system, new_pf, group_id
            ),
        )
        carry_pf = jnp.where(done, pf, new_pf)

        if verbose:
            jax.debug.print(
                "Step: {i} - phi={pf}, increment={inc}, PE/N={pe} after {n} steps",
                i=i + 1,
                pf=pf,
                inc=new_increment,
                pe=pe,
                n=n_steps,
            )

        return (
            i + 1,
            done,
            next_state,
            next_system,
            new_last_state,
            new_last_system,
            carry_pf,
            new_increment,
            new_prev_dir,
            pe,
        )

    final_carry = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    (
        _,
        _,
        final_state,
        final_system,
        last_state,
        last_system,
        _,
        _,
        _,
        _,
    ) = final_carry

    # Ensure the returned packing is relaxed; this is a no-op on a successful
    # in-band exit but matters if the loop hit ``max_jamming_steps`` (where the
    # final state was rescaled but not yet minimized).
    final_state, final_system, _, final_pe = final_system.minimize(
        final_state,
        final_system,
        max_steps=n_minimization_steps,
        pe_tol=pe_tol,
        pe_diff_tol=pe_diff_tol,
    )

    return JamResult(
        unjammed_state=last_state,
        unjammed_system=last_system,
        jammed_state=final_state,
        jammed_system=final_system,
        packing_fraction=compute_packing_fraction(final_state, final_system),
        potential_energy=jnp.asarray(final_pe),
    )
