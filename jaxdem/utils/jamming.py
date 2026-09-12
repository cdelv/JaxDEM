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

from ..minimizers.routines import MAX_STEPS, MinimizeInfo
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

    @property
    def converged(self):
        """Whether the search returned validated jammed scalars."""
        return jnp.isfinite(self.packing_fraction) & jnp.isfinite(self.potential_energy)


class JammingInfo(NamedTuple):
    """Search diagnostics: status 0=success, 1=initially jammed,
    2=unresolved minimization, 3=outer iteration limit.
    """

    converged: jax.Array
    status: jax.Array
    steps: jax.Array
    minimization: MinimizeInfo


def _bisect_jamming(
    state,
    system,
    payload,
    relax,
    scale,
    *,
    pe_tol,
    n_jamming_steps,
    packing_fraction_tolerance,
    packing_fraction_increment,
    verbose,
):
    """Shared device loop, including opaque contact history in each bound.

    ``relax(state, system, payload)`` returns state, system, payload, steps,
    energy per sphere, MinimizeInfo. Both bounds require mechanical convergence
    before energy classifies the equilibrated state. Return the cached high
    bound without reconstruction.
    """
    phi = compute_packing_fraction(state, system)
    empty = MinimizeInfo(
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(jnp.inf),
        jnp.asarray(jnp.inf),
        jnp.asarray(MAX_STEPS),
    )
    trial = (state, system, payload)
    # status -1 means running. Unknown bounds have phi=-1.
    carry = (
        0,
        jnp.asarray(-1),
        trial,
        trial,
        trial,
        phi,
        jnp.asarray(-1.0),
        jnp.asarray(-1.0),
        jnp.asarray(jnp.nan),
        empty,
        empty,
    )

    def condition(c):
        return (c[0] < n_jamming_steps) & (c[1] == -1)

    def step(c):
        it, _, trial, low, high, phi, lo, hi, high_pe, high_info, _ = c
        st, sy, history, n, pe, info = relax(*trial)
        evaluated = (st, sy, history)
        valid = info.converged & jnp.isfinite(pe) & (pe >= 0.0)
        below = valid & (pe <= pe_tol)
        above = valid & (pe > pe_tol)
        initial_high = above & (lo < 0)
        low = jax.lax.cond(below, lambda: evaluated, lambda: low)
        high = jax.lax.cond(above, lambda: evaluated, lambda: high)
        lo = jnp.where(below, phi, lo)
        hi = jnp.where(above, phi, hi)
        high_pe = jnp.where(above, pe, high_pe)
        high_info = jax.lax.cond(above, lambda: info, lambda: high_info)
        bracketed = (lo > 0) & (hi > 0)
        done = valid & bracketed & ((hi / lo - 1.0) <= packing_fraction_tolerance)
        status = jnp.where(
            ~valid, 2, jnp.where(initial_high, 1, jnp.where(done, 0, -1))
        )
        next_phi = jnp.where(hi > 0, 0.5 * (lo + hi), lo + packing_fraction_increment)
        trial = jax.lax.cond(
            status == -1,
            lambda: (*scale(low[0], low[1], next_phi), low[2]),
            lambda: evaluated,
        )
        if verbose:
            jax.debug.print(
                "Step {i}: phi={phi}, PE/N={pe}, steps={n}, balanced={ok}",
                i=it + 1,
                phi=phi,
                pe=pe,
                n=n,
                ok=info.converged,
            )
        return (
            it + 1,
            status,
            trial,
            low,
            high,
            next_phi,
            lo,
            hi,
            high_pe,
            high_info,
            info,
        )

    c = jax.lax.while_loop(condition, step, carry)
    it, status, trial, low, high, _, lo, hi, high_pe, high_info, last_info = c
    status = jnp.where(status == -1, 3, status)
    success = status == 0
    info = JammingInfo(
        success, status, it, jax.lax.cond(success, lambda: high_info, lambda: last_info)
    )
    # Invalid scalars prevent exhausted/unresolved searches masquerading as a jam.
    return (
        low,
        high,
        jnp.where(success, hi, jnp.nan),
        jnp.where(success, high_pe, jnp.nan),
        info,
    )


@partial(
    jax.jit,
    static_argnames=[
        "n_minimization_steps",
        "n_jamming_steps",
        "verbose",
        "return_info",
    ],
)
def bisection_jam(
    state: State,
    system: System,
    n_minimization_steps: int = 1000000,
    pe_tol: float = 1e-16,
    n_jamming_steps: int = 10000,
    packing_fraction_tolerance: float = 1e-10,
    packing_fraction_increment: float = 1e-3,
    verbose: bool = True,
    *,
    force_tol: float = 1e-12,
    torque_tol: float | None = None,
    return_info: bool = False,
):
    """Bracket repulsive jamming using mechanically validated high states.

    Every trial must meet BOTH force/torque tolerances before its energy can
    classify it and change either bracket bound. The returned
    jammed state is the stored equilibrated high state, including its collider
    history; no final reconstruction/minimization can change its classification.

    Energy normalization remains per constituent sphere. Tolerances are in
    force and torque units (``torque_tol=None`` inherits ``force_tol``).
    Returns the historical six-field :class:`JamResult`. On failed relaxation,
    an initially jammed input, or outer-step exhaustion, packing_fraction and
    potential_energy are NaN. Check ``result.converged``. ``return_info=True``
    returns ``(result, JammingInfo)`` with the specific failure status.
    This routine remains jit/vmap compatible; no per-step host checks are used.
    """
    if system.target_fn is not None:
        raise ValueError("bisection_jam requires a repulsive physical energy objective")
    group_id = jax.pure_callback(
        _host_body_grouping,
        jax.ShapeDtypeStruct((state.N,), int),
        state.clump_id,
        state.bond_id,
        vmap_method="sequential",
    )

    def relax(st, sy, history):
        st, sy, n, pe, info = sy.minimize(
            st,
            sy,
            max_steps=n_minimization_steps,
            force_tol=force_tol,
            torque_tol=torque_tol,
            return_info=True,
        )
        return st, sy, history, n, pe, info

    def scale(st, sy, phi):
        return _scale_to_packing_fraction_grouped(st, sy, phi, group_id)

    low, high, phi, pe, info = _bisect_jamming(
        state,
        system,
        (),
        relax,
        scale,
        pe_tol=pe_tol,
        n_jamming_steps=n_jamming_steps,
        packing_fraction_tolerance=packing_fraction_tolerance,
        packing_fraction_increment=packing_fraction_increment,
        verbose=verbose,
    )
    result = JamResult(low[0], low[1], high[0], high[1], phi, pe)
    return (result, info) if return_info else result


def pressure_bisection_jam(
    state: State,
    system: System,
    *,
    n_minimization_steps: int = 1_000_000,
    pressure_threshold: float = 1e-7,
    pressure_band_factor: float = 1.01,
    growth_rate: float = 1.001,
    fine_growth_rate: float = 1.000001,
    length_ratio_tolerance: float = 1e-14,
    n_jamming_steps: int = 10_000,
    pressure_cutoff: float | None = None,
    pressure_max_neighbors: int | None = None,
    verbose: bool = True,
    force_tol: float = 1e-12,
    torque_tol: float | None = None,
) -> JamResult:
    r"""Find the nearest jammed state via a *pressure-band* bisection search.

    This is a JaxDEM port of the classic single-system C++ ``Disk::Jam``
    routine. Where :func:`bisection_jam` works in packing-fraction space and
    classifies a state as jammed or unjammed with a single potential-energy
    threshold, this routine follows the C++ algorithm:

    * The control variable is the **characteristic box length**
      ``L = prod(box_size) ** (1 / dim)``. Compression decreases ``L`` and the
      bisection is performed *linearly in* ``L`` (not in packing fraction).
    * The jamming criterion is a **pressure band** ``[P_lo, P_hi]`` with
      ``P_lo = pressure_threshold`` and ``P_hi = pressure_band_factor * P_lo``.
      A configuration is *unjammed* if ``P < P_lo``, *over-compressed* if
      ``P > P_hi``, and *accepted* (a successful jammed packing) if ``P`` lands
      inside the band.
    * The routine runs two phases, as in the original. A **coarse** phase
      compresses multiplicatively by ``growth_rate`` until the first
      over-compression brackets the jamming point. A **fine** phase
      (``fine_growth_rate``) then bisects until the pressure lands in the band
      or the bracket collapses to
      ``|L_hi / L_lo - 1| < length_ratio_tolerance``.

    Unlike :func:`bisection_jam`, this routine relies on host-side control
    flow and on :func:`~jaxdem.utils.contacts.compute_contact_pressure`,
    which is not ``jit``-safe. It runs on a **single system** and is neither
    ``jit``-ed nor ``vmap``-able. Loop over systems in Python (or use
    :func:`bisection_jam`) if you need many packings.

    .. note::
        The default ``pressure_threshold`` (``1e-7``) comes from the original
        C++ code's unit system. Pressure scales with the contact stiffness, so
        you will typically need to tune ``pressure_threshold`` to your own
        units to get a meaningfully marginal packing.

    Parameters
    ----------
    state, system
        The (single) state/system to jam. The state must start *unjammed*.
        If the initial minimized pressure already exceeds ``P_hi``, the
        routine raises ValueError.
    n_minimization_steps : int, optional
        Maximum FIRE iterations per minimization. Typically ``1e6``.
    force_tol, torque_tol : float, optional
        Maximum free-body force and torque norms for mechanical convergence.
        Torque tolerance defaults to the numerical force tolerance.
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

    def relax_checked(st, sy, **kwargs):
        st, sy, steps, pe, info = sy.minimize(
            st,
            sy,
            force_tol=force_tol,
            torque_tol=torque_tol,
            return_info=True,
            **kwargs,
        )
        if not bool(info.converged):
            raise RuntimeError(
                f"Pressure jamming minimization failed (status={int(info.status)}, "
                f"force={float(info.force_max)}, torque={float(info.torque_max)})"
            )
        return st, sy, steps, pe

    if system.target_fn is not None:
        raise ValueError("pressure_bisection_jam requires a physical energy objective")
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
    state, system, _, final_pe = relax_checked(
        state,
        system,
        max_steps=n_minimization_steps,
    )
    state, system, pressure = pressure_of(state, system)
    if pressure > p_hi:
        raise ValueError("Initial state is already above the requested pressure band")

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
        state, system, _, pe = relax_checked(
            state,
            system,
            max_steps=n_minimization_steps,
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
            "jammed bracket bound, which must still pass final validation."
        )

    # Recover the jammed packing: rescale the last unjammed configuration to the
    # jammed box length and relax it once more. On success ``length`` already
    # holds the in-band box; otherwise fall back to the over-compressed bound.
    jammed_length = length if success else (length_lo if length_lo > 0.0 else length)
    jammed_state, jammed_system = _scale_to_packing_fraction_grouped(
        last_state, last_system, packing_fraction_for_length(jammed_length), group_id
    )
    jammed_state, jammed_system, _, final_pe = relax_checked(
        jammed_state,
        jammed_system,
        max_steps=n_minimization_steps,
    )

    jammed_state, jammed_system, final_pressure = pressure_of(
        jammed_state, jammed_system
    )
    if not (p_lo <= final_pressure <= p_hi):
        raise RuntimeError("Pressure jamming did not reach the requested pressure band")

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
    pe_band_factor: float = 2.0,
    packing_fraction_increment: float = 1e-3,
    n_jamming_steps: int = 10_000,
    verbose: bool = True,
    *,
    force_tol: float = 1e-12,
    torque_tol: float | None = None,
) -> JamResult:
    r"""Find a jammed state via an adaptive, halving packing-fraction step.

    Like :func:`bisection_jam`, this strategy works in packing-fraction space
    and uses the per-particle potential energy as its criterion. Instead of a
    single jammed/unjammed threshold, it targets a **potential-energy band**
    ``[pe_tol, pe_band_factor * pe_tol]`` with an *adaptive step size*:

    * Start from ``packing_fraction_increment`` (typically ``1e-3``).
    * If ``PE/N < pe_tol`` the configuration is under-compressed -> **compress**
      (increase the packing fraction by the current increment, shrinking the
      box).
    * If ``PE/N > pe_band_factor * pe_tol`` it is over-compressed -> **expand**
      (decrease the packing fraction).
    * An in-band state is accepted only after force AND torque convergence.

    Every time the search *reverses direction* (compress -> expand or
    expand -> compress) it **halves** the increment. The step refines
    itself once it brackets the band -- a self-bracketing bisection that
    needs no separately tracked bracket bounds.

    Unlike :func:`bisection_jam` and :func:`pressure_bisection_jam`, this routine
    **does not revert** to the last sub-threshold configuration: it produces
    each new box by affinely rescaling the *current* (just-minimized) state. The
    minimizer already returns ``PE/N`` directly, so this routine -- like
    :func:`bisection_jam` -- is fully ``jit``/``vmap`` compatible.

    Parameters
    ----------
    state, system
        The state/system to jam.
    n_minimization_steps : int, optional
        Maximum FIRE iterations per minimization. Typically ``1e6``.
    pe_tol : float, optional
        Lower edge of the energy band, applied after mechanical relaxation.
    force_tol, torque_tol : float, optional
        Both maximum free-body norms must pass before each trial is classified
        for compression, expansion, or acceptance. Torque tolerance defaults
        to the numerical force tolerance.
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
        recent configuration seen with ``PE/N < pe_tol`` (it defaults to the
        input if none was seen). ``jammed_state`` is the final in-band
        packing. Failed or exhausted searches return NaN phi/energy; check
        result.converged before accepting or saving the packing.
    """
    if system.target_fn is not None:
        raise ValueError("pe_band_jam requires a repulsive physical energy objective")
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

    empty_info = MinimizeInfo(
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(jnp.inf),
        jnp.asarray(jnp.inf),
        jnp.asarray(MAX_STEPS),
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
        empty_info,
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
            _,
        ) = carry

        state, system, n_steps, pe, info = system.minimize(
            state,
            system,
            max_steps=n_minimization_steps,
            force_tol=force_tol,
            torque_tol=torque_tol,
            return_info=True,
        )

        valid = info.converged & jnp.isfinite(pe) & (pe >= 0.0)
        below = valid & (pe < pe_lo)  # under-compressed -> compress
        above = valid & (pe > pe_hi)  # over-compressed -> expand
        accepted = valid & (pe >= pe_lo) & (pe <= pe_hi)
        done = accepted | ~valid  # failures stop, but do not certify a packing

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
            lambda: _scale_to_packing_fraction_grouped(state, system, new_pf, group_id),
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
            info,
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
        final_pe,
        info,
    ) = final_carry
    success = info.converged & (final_pe >= pe_lo) & (final_pe <= pe_hi)
    return JamResult(
        unjammed_state=last_state,
        unjammed_system=last_system,
        jammed_state=final_state,
        jammed_system=final_system,
        packing_fraction=jnp.where(
            success, compute_packing_fraction(final_state, final_system), jnp.nan
        ),
        potential_energy=jnp.where(success, final_pe, jnp.nan),
    )
