# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Search for equilibrated packings using energy or pressure criteria.

The drivers change the box size and minimize each trial configuration before
classifying it. They return particle states, packing fraction, energy per
constituent sphere, and optional convergence diagnostics.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from enum import IntEnum
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple

import jax
import jax.numpy as jnp

from ..minimizers.routines import COLLIDER_OVERFLOW, MAX_STEPS, NONFINITE, MinimizeInfo
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


class JamReason(IntEnum):
    """Why a jamming search stopped; bracket exhaustion is not acceptance."""

    MAX_STEPS = 0
    TARGET_REACHED = 1
    BRACKET_CONVERGED = 2
    INITIAL_OVERCOMPRESSED = 3
    MINIMIZATION_FAILED = 4
    NONFINITE = 5
    SEARCH_OVERFLOW = 6
    INVALID_MEASUREMENT = 7
    BRACKET_EXHAUSTED = 8


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class JamResult:
    """Particle states and measured scalars from a jamming search.

    Attributes
    ----------
    unjammed_state : State
        Most recent equilibrated state below the driver's energy or pressure
        threshold. Defaults to the input state if no such trial was found.
    unjammed_system : System
        System associated with ``unjammed_state``.
    jammed_state : State
        Accepted state on success. On failure, contains a state retained by
        the driver for inspection; see its Returns section.
    jammed_system : System
        System associated with ``jammed_state``, including its collider state.
    packing_fraction : jax.Array
        Particle volume divided by box volume for the accepted state.
        NaN if the search failed.
    potential_energy : jax.Array
        Potential energy divided by the number of constituent spheres for
        the accepted state. NaN if the search failed.

    Notes
    -----
    Iteration yields six values in the order shown above. ``reason`` and
    ``minimizer_reason`` identify the termination conditions; ``info``
    holds the residuals, trial count and maximum inner iteration count.
    ``steps`` counts trial minimizations, not cumulative inner iterations.
    Check ``converged`` before using its contents as a jammed packing.
    """

    unjammed_state: State
    unjammed_system: System
    jammed_state: State
    jammed_system: System
    packing_fraction: jax.Array
    potential_energy: jax.Array
    reason: jax.Array
    info: JammingInfo

    @property
    def converged(self) -> jax.Array:
        return self.info.converged

    @property
    def minimizer_reason(self) -> jax.Array:
        return self.info.minimization.reason

    @property
    def steps(self) -> jax.Array:
        """Number of attempted trial minimizations, including failed attempts."""
        return self.info.steps

    @property
    def max_minimization_steps(self) -> jax.Array:
        return self.info.max_minimization_steps

    def __iter__(self) -> Iterator[Any]:
        yield self.unjammed_state
        yield self.unjammed_system
        yield self.jammed_state
        yield self.jammed_system
        yield self.packing_fraction
        yield self.potential_energy

    def __len__(self) -> int:
        return 6

    def __getitem__(self, index: int | slice) -> Any:
        return tuple(self)[index]


class JammingInfo(NamedTuple):
    """Convergence diagnostics for a jamming search.

    Attributes
    ----------
    converged : jax.Array
        Whether the driver met its acceptance criterion.
    status : jax.Array
        Integer search status:

        * 0: accepted a mechanically equilibrated packing.
        * 1: the initial trial exceeded the energy threshold in
          ``bisection_jam`` or the upper pressure bound in
          ``pressure_bisection_jam``.
        * 2: minimization failed, or the trial energy or pressure was
          nonfinite or negative.
        * 3: reached the maximum number of jamming steps.
        * 4: refinement stopped outside the target band because the next
          trial coordinate was numerically unchanged.
    steps : jax.Array
        Number of attempted minimizations, including the initial trial and
        any failed attempt.
    minimization : MinimizeInfo
        Force and torque residuals, finiteness, convergence flag, and
        minimizer status for the accepted state on success or the last
        attempted trial on failure. Before any trial, convergence and
        finiteness are false, residuals are infinite, and the minimizer
        status is ``MAX_STEPS``.
    max_minimization_steps : jax.Array
        Largest number of minimizer steps taken by any trial, including
        failed attempts. Zero if no minimization was attempted.
    """

    converged: jax.Array
    status: jax.Array
    steps: jax.Array
    minimization: MinimizeInfo
    max_minimization_steps: jax.Array


def _jam_reason(
    info: JammingInfo,
    success_reason: JamReason,
    *,
    measurement_finite: bool | jax.Array = True,
) -> jax.Array:
    """Map search and minimizer diagnostics to a jamming termination code."""
    return jnp.select(
        [
            info.status == 0,
            info.status == 1,
            info.status == 3,
            info.status == 4,
            info.minimization.status == COLLIDER_OVERFLOW,
            (info.minimization.status == NONFINITE)
            | (info.minimization.converged & ~jnp.asarray(measurement_finite)),
            info.minimization.converged,
        ],
        [
            int(success_reason),
            int(JamReason.INITIAL_OVERCOMPRESSED),
            int(JamReason.MAX_STEPS),
            int(JamReason.BRACKET_EXHAUSTED),
            int(JamReason.SEARCH_OVERFLOW),
            int(JamReason.NONFINITE),
            int(JamReason.INVALID_MEASUREMENT),
        ],
        default=int(JamReason.MINIMIZATION_FAILED),
    )


def _bisect_jamming(
    state: State,
    system: System,
    relax: Callable[..., tuple[Any, ...]],
    scale: Callable[..., tuple[State, System]],
    *,
    pe_tol: float,
    pe_band_factor: float,
    n_jamming_steps: int,
    packing_fraction_tolerance: float,
    packing_fraction_increment: float,
    verbose: bool,
) -> tuple[tuple[Any, ...], tuple[Any, ...], jax.Array, jax.Array, JammingInfo]:
    """Bisect an energy threshold while retaining each equilibrated bound.

    Each trial is relaxed and classified by energy only if its minimization
    converged and its energy is finite and nonnegative. Energies at or below
    ``pe_tol`` update the lower bound on packing fraction; higher energies
    update the upper bound. Before an upper bound exists, increase packing fraction
    by ``packing_fraction_increment``. Once both bounds exist, use their
    midpoint. Each trial is scaled from the stored state at the lower bound.

    Parameters
    ----------
    state, system
        Initial particle state and simulation system.
    relax : callable
        ``relax(state, system)`` returns
        ``(state, system, steps, energy, MinimizeInfo)``. Energy
        must be normalized per constituent sphere.
    scale : callable
        ``scale(state, system, packing_fraction)`` returns the state and
        system rescaled to the requested packing fraction.
    pe_tol : float
        Energy threshold separating the lower and upper bounds.
    pe_band_factor : float
        Maximum accepted upper-bound energy as a multiple of ``pe_tol``.
    n_jamming_steps : int
        Maximum number of calls to ``relax``.
    packing_fraction_tolerance : float
        Accept the upper bound when ``phi_hi / phi_lo - 1`` is at most
        this value and both bounds have been established.
    packing_fraction_increment : float
        Additive step in packing fraction before establishing an upper bound.
    verbose : bool
        Print the packing fraction, energy, and minimizer steps for each trial.

    Returns
    -------
    low, high : tuple
        Stored ``(state, system)`` for each bound. An unset bound
        contains the input values.
    packing_fraction, potential_energy : jax.Array
        Packing fraction and energy of the accepted upper bound, or NaN
        if the search failed.
    info : JammingInfo
        Search status, minimizer diagnostics, and iteration counts.

    Notes
    -----
    The search uses ``jax.lax.while_loop``. The callbacks must support
    JAX tracing.
    """
    phi = compute_packing_fraction(state, system)
    empty = MinimizeInfo(
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(jnp.inf),
        jnp.asarray(jnp.inf),
        jnp.asarray(MAX_STEPS),
    )
    trial = (state, system)
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
        jnp.asarray(0, dtype=int),
    )

    def condition(c: tuple[Any, ...]) -> jax.Array:
        return (c[0] < n_jamming_steps) & (c[1] == -1)

    def step(c: tuple[Any, ...]) -> tuple[Any, ...]:
        (
            it,
            _,
            trial,
            low,
            high,
            phi,
            lo,
            hi,
            high_pe,
            high_info,
            _,
            max_minimization_steps,
        ) = c
        st, sy, n, pe, info = relax(*trial)
        max_minimization_steps = jnp.maximum(max_minimization_steps, n)
        evaluated = (st, sy)
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
        phi_converged = (hi / lo - 1.0) <= packing_fraction_tolerance
        energy_converged = high_pe <= pe_band_factor * pe_tol
        done = valid & bracketed & phi_converged & energy_converged
        next_phi = jnp.where(hi > 0, 0.5 * (lo + hi), lo + packing_fraction_increment)
        exhausted = bracketed & ((next_phi == lo) | (next_phi == hi))
        status = jnp.where(
            ~valid,
            2,
            jnp.where(
                initial_high,
                1,
                jnp.where(done, 0, jnp.where(exhausted, 4, -1)),
            ),
        )
        trial = jax.lax.cond(
            status == -1,
            lambda: scale(low[0], low[1], next_phi),
            lambda: evaluated,
        )
        if verbose:
            jax.debug.print(
                "Step {i}: phi={phi}, PE/N={pe}, steps={n}",
                i=it + 1,
                phi=phi,
                pe=pe,
                n=n,
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
            max_minimization_steps,
        )

    c = jax.lax.while_loop(condition, step, carry)
    (
        it,
        status,
        _trial,
        low,
        high,
        _,
        _lo,
        hi,
        high_pe,
        high_info,
        last_info,
        max_minimization_steps,
    ) = c
    status = jnp.where(status == -1, 3, status)
    success = status == 0
    info = JammingInfo(
        success,
        status,
        jnp.asarray(it),
        jax.lax.cond(success, lambda: high_info, lambda: last_info),
        max_minimization_steps,
    )
    # Invalid scalars prevent exhausted/unresolved searches masquerading as a jam.
    return (
        low,
        high,
        jnp.where(success, hi, jnp.nan),
        jnp.where(success, high_pe, jnp.nan),
        info,
    )


_MINIMIZER_KEYS = ("max_steps", "force_tol", "torque_tol", "advance_history")


def _minimizer_kwargs(minimizer_kwargs: Mapping[str, Any] | None) -> dict[str, Any]:
    """Merge user arguments for ``System.minimize`` with the jamming defaults."""
    provided = {} if minimizer_kwargs is None else dict(minimizer_kwargs)
    unknown = set(provided).difference(_MINIMIZER_KEYS)
    if unknown:
        names = ", ".join(sorted(unknown))
        allowed = ", ".join(_MINIMIZER_KEYS)
        raise TypeError(
            f"unknown minimizer_kwargs {names}; expected a subset of {allowed}"
        )
    resolved = {
        "max_steps": 100_000_000,
        "force_tol": 1e-14,
        "torque_tol": 1e-14,
        "advance_history": True,
    }
    resolved.update(provided)
    return resolved


def bisection_jam(
    state: State,
    system: System,
    pe_tol: float = 1e-16,
    n_jamming_steps: int = 10000,
    packing_fraction_tolerance: float = 1e-10,
    packing_fraction_increment: float = 1e-3,
    verbose: bool = True,
    *,
    pe_band_factor: float = 2.0,
    minimizer_kwargs: Mapping[str, Any] | None = None,
    return_info: bool = False,
) -> JamResult | tuple[JamResult, JammingInfo]:
    """Locate an energy threshold by bisecting packing fraction.

    Minimize each trial with the system's configured minimizer. A trial can
    update a bound only after its maximum net force and torque norms satisfy
    their tolerances and its energy is finite and nonnegative. Write
    ``e = E / N``, where ``N`` is the number of constituent spheres. Trials
    with ``e <= pe_tol`` set the lower bound; trials with ``e > pe_tol`` set
    the upper bound. The accepted upper bound must additionally satisfy
    ``e <= pe_band_factor * pe_tol``.

    Until an upper bound is found, increase packing fraction by
    ``packing_fraction_increment``. Then choose each trial packing fraction
    as the midpoint of the two bounds. Rescale the stored state at the lower
    bound to create each trial, preserving body sizes and box aspect ratio.
    Accept the stored state at the upper bound when
    ``phi_hi / phi_lo - 1 <= packing_fraction_tolerance`` and its energy is
    at most ``pe_band_factor * pe_tol``.

    Parameters
    ----------
    state : State
        Initial particle configuration. Its minimized energy must be at or
        below ``pe_tol`` to establish the first lower bound.
    system : System
        Domain, interactions, collider, and configured minimizer. The search
        requires a repulsive physical energy and ``system.target_fn=None``.
    pe_tol : float, optional
        Threshold for energy per constituent sphere, applied after
        mechanical equilibration. This classifies an equilibrated trial; it
        is not a minimizer tolerance.
    n_jamming_steps : int, optional
        Maximum number of trial minimizations, including the initial trial.
    packing_fraction_tolerance : float, optional
        Maximum relative separation of the two bounds on packing fraction
        required for acceptance.
    packing_fraction_increment : float, optional
        Additive step in packing fraction used before finding an upper bound.
    verbose : bool, optional
        Print packing fraction, energy, and minimizer steps for each trial.
    pe_band_factor : float, optional
        Multiplier greater than one defining the maximum accepted upper-bound
        energy. Defaults to 2.
    minimizer_kwargs : mapping, optional
        Arguments forwarded to :meth:`System.minimize <jaxdem.system.System.minimize>` for every trial.
        Recognized keys are ``max_steps``, ``force_tol``, ``torque_tol``, and
        ``advance_history``. Omitted keys use ``max_steps=100_000_000``,
        ``force_tol=1e-14``, ``torque_tol=1e-14``, and
        ``advance_history=True``. ``torque_tol=None`` uses the numerical
        value of ``force_tol``. Every value must be a concrete Python object.
        Pair history advances on accepted steps unless ``advance_history``
        is false.
    return_info : bool, optional
        Return ``(result, info)`` when true; otherwise return ``result``.

    Returns
    -------
    result : JamResult
        The stored lower and upper states and their systems, with the upper
        bound's packing fraction and energy on success. State and collider
        history are retained together. On failure, the stored bounds are
        returned for inspection, with NaN packing fraction and energy.
        An unset bound contains the input state and system.
    info : JammingInfo, optional
        Returned when ``return_info=True``. Includes search status, force
        and torque diagnostics, trial count, and maximum minimizer steps.

    Raises
    ------
    ValueError
        If ``system.target_fn`` is set or ``pe_band_factor`` is not finite
        and greater than one.

    Notes
    -----
    Failed minimization, an initial energy above the threshold, exhaustion of
    the packing-fraction bracket before the energy band is reached, or
    exhaustion of the jamming step budget ends the search with
    ``result.converged=False``. The search supports ``jax.jit`` and
    ``jax.vmap``.
    """
    if not math.isfinite(pe_band_factor) or pe_band_factor <= 1.0:
        raise ValueError("pe_band_factor must be finite and greater than one")
    options = _minimizer_kwargs(minimizer_kwargs)
    return _bisection_jam(
        state,
        system,
        pe_tol,
        pe_band_factor,
        n_jamming_steps,
        packing_fraction_tolerance,
        packing_fraction_increment,
        verbose,
        options["max_steps"],
        options["force_tol"],
        options["torque_tol"],
        options["advance_history"],
        return_info,
    )


@partial(
    jax.jit,
    static_argnames=[
        "n_jamming_steps",
        "verbose",
        "return_info",
        "max_steps",
        "force_tol",
        "torque_tol",
        "advance_history",
    ],
)
def _bisection_jam(
    state: State,
    system: System,
    pe_tol: float,
    pe_band_factor: float,
    n_jamming_steps: int,
    packing_fraction_tolerance: float,
    packing_fraction_increment: float,
    verbose: bool,
    max_steps: int,
    force_tol: float,
    torque_tol: float | None,
    advance_history: bool,
    return_info: bool,
) -> JamResult | tuple[JamResult, JammingInfo]:
    if system.target_fn is not None:
        raise ValueError("bisection_jam requires a repulsive physical energy objective")
    group_id = jax.pure_callback(
        _host_body_grouping,
        jax.ShapeDtypeStruct((state.N,), jnp.int32),  # type: ignore[no-untyped-call]
        state.clump_id,
        state.bond_id,
        vmap_method="sequential",
    )

    def relax(st: State, sy: System) -> tuple[Any, ...]:
        result = sy.minimize(
            st,
            sy,
            max_steps=max_steps,
            force_tol=force_tol,
            torque_tol=torque_tol,
            advance_history=advance_history,
        )
        st, sy, n, pe = result
        return st, sy, n, pe, result.info

    def scale(st: State, sy: System, phi: float) -> tuple[State, System]:
        return _scale_to_packing_fraction_grouped(st, sy, phi, group_id)

    low, high, phi, pe, info = _bisect_jamming(
        state,
        system,
        relax,
        scale,
        pe_tol=pe_tol,
        pe_band_factor=pe_band_factor,
        n_jamming_steps=n_jamming_steps,
        packing_fraction_tolerance=packing_fraction_tolerance,
        packing_fraction_increment=packing_fraction_increment,
        verbose=verbose,
    )
    result = JamResult(
        low[0],
        low[1],
        high[0],
        high[1],
        phi,
        pe,
        _jam_reason(info, JamReason.BRACKET_CONVERGED),
        info,
    )
    return (result, info) if return_info else result


bisection_jam.clear_cache = _bisection_jam.clear_cache  # type: ignore[attr-defined]


def pressure_bisection_jam(
    state: State,
    system: System,
    *,
    pressure_threshold: float = 1e-7,
    pressure_band_factor: float = 1.01,
    growth_rate: float = 1.001,
    fine_growth_rate: float = 1.000001,
    length_ratio_tolerance: float = 1e-14,
    n_jamming_steps: int = 10_000,
    verbose: bool = True,
    minimizer_kwargs: Mapping[str, Any] | None = None,
    return_info: bool = False,
) -> JamResult | tuple[JamResult, JammingInfo]:
    """Find an equilibrated packing within a prescribed contact pressure band.

    The target is ``P_lo <= P <= P_hi``, where
    ``P_lo = pressure_threshold`` and
    ``P_hi = pressure_band_factor * pressure_threshold``. Minimize each trial,
    require force and torque convergence, then compute its contact pressure.
    Accept the first trial inside the band and return that state and system.

    The search variable is box length ``L = prod(box_size) ** (1 / dim)``.
    Before finding a trial above the pressure band, compress by dividing
    ``L`` by ``growth_rate``. A trial above the band records a shorter box
    length and sets the next trial to the midpoint between it and the stored
    length below the band. Subsequent compression uses ``fine_growth_rate``
    whenever no length above the band is stored. A trial below the band uses
    any stored length above the band to choose a midpoint, then clears that
    bound.

    Every trial is produced by rescaling the most recent equilibrated state
    below the band. Rescaling preserves body sizes and box aspect ratio.
    Stop without acceptance if refinement exhausts the length bracket or
    the proposed box length is numerically unchanged.

    Parameters
    ----------
    state : State
        Initial particle configuration. Its minimized pressure must not
        exceed the upper edge of the target band.
    system : System
        Domain, interactions, collider, and configured minimizer. The search
        requires a repulsive physical energy and ``system.target_fn=None``.
    pressure_threshold : float, optional
        Positive lower edge of the pressure band, in the system's pressure
        units.
    pressure_band_factor : float, optional
        Multiplier greater than one defining the upper edge of the band.
    growth_rate : float, optional
        Compression factor for box length, greater than one. Used before
        the first trial above the pressure band.
    fine_growth_rate : float, optional
        Compression factor for box length, greater than one. Used after
        the first trial above the band when no upper pressure bound is stored.
    length_ratio_tolerance : float, optional
        Stop refinement outside the band when
        ``abs(L_below / L_above - 1)`` is smaller than this value.
    n_jamming_steps : int, optional
        Maximum number of trial minimizations, including the initial trial.
    verbose : bool, optional
        Print box length, packing fraction, pressure, energy, and minimizer
        steps for each trial with a valid pressure measurement.
    minimizer_kwargs : mapping, optional
        Arguments forwarded to :meth:`System.minimize <jaxdem.system.System.minimize>` for every trial.
        Recognized keys are ``max_steps``, ``force_tol``, ``torque_tol``, and
        ``advance_history``. Omitted keys use ``max_steps=100_000_000``,
        ``force_tol=1e-14``, ``torque_tol=1e-14``, and
        ``advance_history=True``. ``torque_tol=None`` uses the numerical
        value of ``force_tol``. Pair history advances on accepted steps
        unless ``advance_history`` is false.
    return_info : bool, optional
        Return ``(result, info)`` when true; otherwise return ``result``.

    Returns
    -------
    result : JamResult
        The most recent state below the band and the accepted state, each
        with its system. The unjammed fields contain the input if no trial
        below the band was found. On failure, the jammed fields contain the last
        evaluated state and system, or the input if no trial was attempted;
        packing fraction and energy are NaN. Reported energy is per
        constituent sphere.
    info : JammingInfo, optional
        Returned when ``return_info=True``. Includes search status, force
        and torque diagnostics, trial count, and maximum minimizer steps.

    Raises
    ------
    ValueError
        If ``system.target_fn`` is set or contact pressure analysis reports
        a neighbor list overflow.

    Notes
    -----
    Failed minimization, invalid energy or pressure, an initial state above
    the band, or exhaustion of the jamming step budget or pressure bracket
    ends the search with ``result.converged=False``. Energies and pressures
    used for classification must be finite and nonnegative.

    The search uses Python control flow and contact analysis on one system.
    It cannot be transformed with ``jax.jit`` or ``jax.vmap``.
    """

    if system.target_fn is not None:
        raise ValueError("pressure_bisection_jam requires a physical energy objective")
    options = _minimizer_kwargs(minimizer_kwargs)
    p_lo = float(pressure_threshold)
    p_hi = float(pressure_band_factor) * p_lo
    dim = int(state.dim)
    volume = float(compute_particle_volume(state))
    group_id = jnp.asarray(_host_body_grouping(state.clump_id, state.bond_id))

    # L_hi is the below-band box length; L_lo is the above-band length.
    length_hi = -1.0
    length_lo = -1.0
    last_state, last_system = state, system
    fine = False
    iteration = 0
    max_minimization_steps = 0
    status = 3
    pe = jnp.asarray(jnp.nan)
    pressure = float("nan")
    min_info = MinimizeInfo(
        jnp.asarray(False),
        jnp.asarray(False),
        jnp.asarray(jnp.inf),
        jnp.asarray(jnp.inf),
        jnp.asarray(MAX_STEPS),
    )

    while iteration < n_jamming_steps:
        relaxation = system.minimize(state, system, **options)
        state, system, n_steps, pe = relaxation
        min_info = relaxation.info
        iteration += 1
        max_minimization_steps = max(max_minimization_steps, int(n_steps))
        if (
            not bool(min_info.converged)
            or not math.isfinite(float(pe))
            or float(pe) < 0
        ):
            status = 2
            break

        state, system, pressure_value = compute_contact_pressure(state, system)
        pressure = float(pressure_value)
        if not math.isfinite(pressure) or pressure < 0:
            status = 2
            break
        length = float(jnp.prod(system.domain.box_size)) ** (1.0 / dim)
        if verbose:
            print(
                f"Step {iteration}: L={length:.8e}, phi={volume / length**dim:.8e}, "
                f"P={pressure:.5e}, PE/N={float(pe):.5e}, steps={int(n_steps)}"
            )

        if p_lo <= pressure <= p_hi:
            status = 0
            break
        if pressure < p_lo:
            last_state, last_system = state, system
            length_hi = length
            if length_lo > 0:
                next_length = 0.5 * (length_hi + length_lo)
                length_lo = -1.0
            else:
                next_length = length / (fine_growth_rate if fine else growth_rate)
        else:
            if length_hi < 0:
                status = 1
                break
            length_lo = length
            next_length = 0.5 * (length_hi + length_lo)

        if (
            fine
            and length_hi > 0
            and length_lo > 0
            and abs(length_hi / length_lo - 1.0) < length_ratio_tolerance
        ) or next_length == length:
            status = 4
            break
        fine = fine or pressure > p_hi
        if iteration == n_jamming_steps:
            break
        # Each pressure trial starts from the latest equilibrated low state.
        state, system = _scale_to_packing_fraction_grouped(
            last_state, last_system, volume / next_length**dim, group_id
        )

    success = status == 0
    info = JammingInfo(
        jnp.asarray(success),
        jnp.asarray(status),
        jnp.asarray(iteration),
        min_info,
        jnp.asarray(max_minimization_steps),
    )
    result = JamResult(
        unjammed_state=last_state,
        unjammed_system=last_system,
        jammed_state=state,
        jammed_system=system,
        packing_fraction=(
            compute_packing_fraction(state, system) if success else jnp.asarray(jnp.nan)
        ),
        potential_energy=(jnp.asarray(pe) if success else jnp.asarray(jnp.nan)),
        reason=_jam_reason(
            info,
            JamReason.TARGET_REACHED,
            measurement_finite=math.isfinite(float(pe)) and math.isfinite(pressure),
        ),
        info=info,
    )
    return (result, info) if return_info else result


def pe_band_jam(
    state: State,
    system: System,
    pe_tol: float = 1e-16,
    pe_band_factor: float = 2.0,
    packing_fraction_increment: float = 1e-3,
    n_jamming_steps: int = 10_000,
    verbose: bool = True,
    *,
    minimizer_kwargs: Mapping[str, Any] | None = None,
    return_info: bool = False,
) -> JamResult | tuple[JamResult, JammingInfo]:
    """Find an equilibrated packing within a prescribed energy band.

    The target is ``pe_tol <= E / N <= pe_band_factor * pe_tol``, where
    ``N`` is the number of constituent spheres. Minimize each trial and
    require force and torque convergence before classifying its energy.

    Begin with a step in packing fraction of ``packing_fraction_increment``.
    Increase packing fraction when the energy is below the band and decrease
    it when the energy is above the band. Halve the step before applying it
    whenever the direction changes between compression and expansion. Each
    trial is produced by rescaling the current equilibrated state, preserving
    body sizes and box aspect ratio. Accept the first trial inside the band.
    An initial state above the band starts the search with expansion.

    Parameters
    ----------
    state : State
        Initial particle configuration.
    system : System
        Domain, interactions, collider, and configured minimizer. The search
        requires a repulsive physical energy and ``system.target_fn=None``.
    pe_tol : float, optional
        Lower edge of the band for energy per constituent sphere, applied after
        mechanical equilibration. This classifies an equilibrated trial; it
        is not a minimizer tolerance.
    pe_band_factor : float, optional
        Multiplier greater than one defining the upper edge of the band.
    packing_fraction_increment : float, optional
        Initial magnitude of the additive step in packing fraction.
    n_jamming_steps : int, optional
        Maximum number of trial minimizations, including the initial trial.
    verbose : bool, optional
        Print packing fraction, step size, energy, and minimizer steps for
        each trial.
    minimizer_kwargs : mapping, optional
        Arguments forwarded to :meth:`System.minimize <jaxdem.system.System.minimize>` for every trial.
        Recognized keys are ``max_steps``, ``force_tol``, ``torque_tol``, and
        ``advance_history``. Omitted keys use ``max_steps=100_000_000``,
        ``force_tol=1e-14``, ``torque_tol=1e-14``, and
        ``advance_history=True``. ``torque_tol=None`` uses the numerical
        value of ``force_tol``. Every value must be a concrete Python object.
        Pair history advances on accepted steps unless ``advance_history``
        is false.
    return_info : bool, optional
        Return ``(result, info)`` when true; otherwise return ``result``.

    Returns
    -------
    result : JamResult
        The most recent state below the band and the accepted state, each
        with its system. The unjammed fields contain the input if no trial
        below the band was found. On failure, the jammed fields contain the last
        evaluated state and system, or the input if no trial was attempted;
        packing fraction and energy are NaN. Reported energy is per
        constituent sphere.
    info : JammingInfo, optional
        Returned when ``return_info=True``. Includes search status, force
        and torque diagnostics, trial count, and maximum minimizer steps.

    Raises
    ------
    ValueError
        If ``system.target_fn`` is set.

    Notes
    -----
    Failed minimization, nonfinite or negative energy, or exhaustion of the
    jamming step budget ends the search with ``result.converged=False``.
    The search supports ``jax.jit`` and ``jax.vmap``.
    """
    options = _minimizer_kwargs(minimizer_kwargs)
    return _pe_band_jam(
        state,
        system,
        pe_tol,
        pe_band_factor,
        packing_fraction_increment,
        n_jamming_steps,
        verbose,
        options["max_steps"],
        options["force_tol"],
        options["torque_tol"],
        options["advance_history"],
        return_info,
    )


@partial(
    jax.jit,
    static_argnames=[
        "n_jamming_steps",
        "verbose",
        "return_info",
        "max_steps",
        "force_tol",
        "torque_tol",
        "advance_history",
    ],
)
def _pe_band_jam(
    state: State,
    system: System,
    pe_tol: float,
    pe_band_factor: float,
    packing_fraction_increment: float,
    n_jamming_steps: int,
    verbose: bool,
    max_steps: int,
    force_tol: float,
    torque_tol: float | None,
    advance_history: bool,
    return_info: bool,
) -> JamResult | tuple[JamResult, JammingInfo]:
    if system.target_fn is not None:
        raise ValueError("pe_band_jam requires a repulsive physical energy objective")
    pe_lo = pe_tol
    pe_hi = pe_band_factor * pe_tol

    initial_packing_fraction = compute_packing_fraction(state, system)

    # Body grouping depends only on the (static) topology; compute it once.
    group_id = jax.pure_callback(
        _host_body_grouping,
        jax.ShapeDtypeStruct((state.N,), jnp.int32),  # type: ignore[no-untyped-call]
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
        jnp.asarray(-1),  # running
        state,
        system,  # current state/system
        state,
        system,  # last sub-threshold ("unjammed") state/system
        initial_packing_fraction,  # current packing fraction
        jnp.asarray(packing_fraction_increment, float),  # current increment
        jnp.asarray(0, int),  # previous step direction in {-1, 0, +1}
        jnp.asarray(jnp.inf),  # final PE/N
        empty_info,
        jnp.asarray(0, dtype=int),  # maximum inner step count
    )

    def cond_fun(carry: tuple[Any, ...]) -> jax.Array:
        i, status, *_ = carry
        return (i < n_jamming_steps) & (status == -1)

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
            max_minimization_steps,
        ) = carry

        relaxation = system.minimize(
            state,
            system,
            max_steps=max_steps,
            force_tol=force_tol,
            torque_tol=torque_tol,
            advance_history=advance_history,
        )
        state, system, n_steps, pe = relaxation
        info = relaxation.info
        max_minimization_steps = jnp.maximum(max_minimization_steps, n_steps)

        valid = info.converged & jnp.isfinite(pe) & (pe >= 0.0)
        below = valid & (pe < pe_lo)  # under-compressed -> compress
        above = valid & (pe > pe_hi)  # over-compressed -> expand
        accepted = valid & (pe >= pe_lo) & (pe <= pe_hi)
        status = jnp.where(~valid, 2, jnp.where(accepted, 0, -1))
        done = (status != -1) | (i + 1 == n_jamming_steps)

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

        # Preserve the evaluated state on every exit so diagnostics match it.
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
            status,
            next_state,
            next_system,
            new_last_state,
            new_last_system,
            carry_pf,
            new_increment,
            new_prev_dir,
            pe,
            info,
            max_minimization_steps,
        )

    final_carry = jax.lax.while_loop(cond_fun, body_fun, init_carry)
    (
        iterations,
        status,
        final_state,
        final_system,
        last_state,
        last_system,
        _,
        _,
        _,
        final_pe,
        min_info,
        max_minimization_steps,
    ) = final_carry
    status = jnp.where(status == -1, 3, status)
    success = status == 0
    info = JammingInfo(
        success, status, jnp.asarray(iterations), min_info, max_minimization_steps
    )
    result = JamResult(
        unjammed_state=last_state,
        unjammed_system=last_system,
        jammed_state=final_state,
        jammed_system=final_system,
        packing_fraction=jnp.where(
            success, compute_packing_fraction(final_state, final_system), jnp.nan
        ),
        potential_energy=jnp.where(success, final_pe, jnp.nan),
        reason=_jam_reason(
            info, JamReason.TARGET_REACHED, measurement_finite=jnp.isfinite(final_pe)
        ),
        info=info,
    )
    return (result, info) if return_info else result


pe_band_jam.clear_cache = _pe_band_jam.clear_cache  # type: ignore[attr-defined]


def rollback_pe_band_jam(
    state: State,
    system: System,
    pe_tol: float = 1e-16,
    pe_band_factor: float = 2.0,
    packing_fraction_increment: float = 1e-3,
    increment_growth_factor: float = 1.0,
    n_jamming_steps: int = 10_000,
    verbose: bool = True,
    *,
    minimizer_kwargs: Mapping[str, Any] | None = None,
    return_info: bool = False,
) -> JamResult | tuple[JamResult, JammingInfo]:
    """Compress from the last sub-threshold state with rollback on overshoot.

    The target is ``pe_tol < E / N <= pe_band_factor * pe_tol``. Every trial
    starts from the most recently accepted state with ``E / N <= pe_tol``.
    A sub-threshold trial is committed as the new lower state, preserving its
    updated contact history. A trial above the band is discarded, the packing
    fraction increment is halved, and the next trial is regenerated from the
    unchanged lower state. A trial inside the band is accepted immediately.

    After a sub-threshold trial, multiply the current increment by
    ``increment_growth_factor`` up to the original
    ``packing_fraction_increment``. Regrowth is disabled by default; set
    ``increment_growth_factor`` greater than 1 to enable it.

    Parameters
    ----------
    state : State
        Initial particle configuration. Its minimized energy must be at or
        below ``pe_tol`` to establish the first lower state.
    system : System
        Domain, interactions, collider, and configured minimizer. The search
        requires a repulsive physical energy and ``system.target_fn=None``.
    pe_tol : float, optional
        Lower edge of the accepted energy-per-sphere band.
    pe_band_factor : float, optional
        Multiplier greater than one defining the upper edge of the band.
    packing_fraction_increment : float, optional
        Initial and maximum additive compression step in packing fraction.
    increment_growth_factor : float, optional
        Factor at least one applied to the step after each committed
        sub-threshold trial, capped by ``packing_fraction_increment``.
        Defaults to 1, which disables regrowth.
    n_jamming_steps : int, optional
        Maximum number of trial minimizations, including the initial trial.
    verbose : bool, optional
        Print packing fraction, step size, energy, and minimizer steps for
        each trial.
    minimizer_kwargs : mapping, optional
        Arguments forwarded to :meth:`System.minimize <jaxdem.system.System.minimize>` for every trial.
        Recognized keys are ``max_steps``, ``force_tol``, ``torque_tol``, and
        ``advance_history``. Omitted keys use ``max_steps=100_000_000``,
        ``force_tol=1e-14``, ``torque_tol=1e-14``, and
        ``advance_history=True``.
    return_info : bool, optional
        Return ``(result, info)`` when true; otherwise return ``result``.

    Returns
    -------
    result : JamResult
        The final committed sub-threshold state and accepted in-band state,
        each with its system. On failure, the jammed fields contain the last
        evaluated trial, while packing fraction and energy are NaN.
    info : JammingInfo, optional
        Returned when ``return_info=True``.

    Raises
    ------
    ValueError
        If an option is outside its documented range.

    Notes
    -----
    This is a one-sided continuation rather than a density bisection. It never
    continues from an above-band state. Failed minimization, an initially
    overcompressed state, numerical step exhaustion, or exhaustion of the
    outer step budget returns ``result.converged=False``. The search supports
    ``jax.jit`` and ``jax.vmap``.
    """
    if not math.isfinite(pe_band_factor) or pe_band_factor <= 1.0:
        raise ValueError("pe_band_factor must be finite and greater than one")
    if (
        not math.isfinite(packing_fraction_increment)
        or packing_fraction_increment <= 0.0
    ):
        raise ValueError("packing_fraction_increment must be finite and positive")
    if not math.isfinite(increment_growth_factor) or increment_growth_factor < 1.0:
        raise ValueError("increment_growth_factor must be finite and at least one")
    options = _minimizer_kwargs(minimizer_kwargs)
    return _rollback_pe_band_jam(
        state,
        system,
        pe_tol,
        pe_band_factor,
        packing_fraction_increment,
        increment_growth_factor,
        n_jamming_steps,
        verbose,
        options["max_steps"],
        options["force_tol"],
        options["torque_tol"],
        options["advance_history"],
        return_info,
    )


@partial(
    jax.jit,
    static_argnames=[
        "n_jamming_steps",
        "verbose",
        "return_info",
        "max_steps",
        "force_tol",
        "torque_tol",
        "advance_history",
    ],
)
def _rollback_pe_band_jam(
    state: State,
    system: System,
    pe_tol: float,
    pe_band_factor: float,
    packing_fraction_increment: float,
    increment_growth_factor: float,
    n_jamming_steps: int,
    verbose: bool,
    max_steps: int,
    force_tol: float,
    torque_tol: float | None,
    advance_history: bool,
    return_info: bool,
) -> JamResult | tuple[JamResult, JammingInfo]:
    if system.target_fn is not None:
        raise ValueError(
            "rollback_pe_band_jam requires a repulsive physical energy objective"
        )
    pe_hi = pe_band_factor * pe_tol
    initial_packing_fraction = compute_packing_fraction(state, system)
    initial_increment = jnp.asarray(packing_fraction_increment, float)

    group_id = jax.pure_callback(
        _host_body_grouping,
        jax.ShapeDtypeStruct((state.N,), jnp.int32),  # type: ignore[no-untyped-call]
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
        0,
        jnp.asarray(-1),  # running
        state,
        system,  # current trial
        state,
        system,  # last committed sub-threshold state
        initial_packing_fraction,  # current trial packing fraction
        initial_packing_fraction,  # lower-state packing fraction
        initial_increment,
        jnp.asarray(jnp.inf),  # last evaluated PE/N
        empty_info,
        jnp.asarray(0, dtype=int),
    )

    def condition(carry: tuple[Any, ...]) -> jax.Array:
        iteration, status, *_ = carry
        return (iteration < n_jamming_steps) & (status == -1)

    def step(carry: tuple[Any, ...]) -> tuple[Any, ...]:
        (
            iteration,
            _,
            trial_state,
            trial_system,
            low_state,
            low_system,
            trial_phi,
            low_phi,
            increment,
            _,
            _,
            max_minimization_steps,
        ) = carry

        relaxation = trial_system.minimize(
            trial_state,
            trial_system,
            max_steps=max_steps,
            force_tol=force_tol,
            torque_tol=torque_tol,
            advance_history=advance_history,
        )
        evaluated_state, evaluated_system, n_steps, pe = relaxation
        min_info = relaxation.info
        max_minimization_steps = jnp.maximum(max_minimization_steps, n_steps)

        valid = min_info.converged & jnp.isfinite(pe) & (pe >= 0.0)
        below = valid & (pe <= pe_tol)
        accepted = valid & (pe > pe_tol) & (pe <= pe_hi)
        initial_overcompressed = valid & (iteration == 0) & ~below
        status = jnp.where(
            ~valid,
            2,
            jnp.where(
                initial_overcompressed,
                1,
                jnp.where(accepted, 0, -1),
            ),
        )

        new_low_state, new_low_system = jax.lax.cond(
            below,
            lambda: (evaluated_state, evaluated_system),
            lambda: (low_state, low_system),
        )
        new_low_phi = jnp.where(below, trial_phi, low_phi)
        grown_increment = jnp.minimum(
            initial_increment, increment * increment_growth_factor
        )
        new_increment = jnp.where(
            pe > pe_hi,
            0.5 * increment,
            jnp.where(below, grown_increment, increment),
        )
        next_phi = new_low_phi + new_increment
        exhausted = valid & (status == -1) & (next_phi == new_low_phi)
        status = jnp.where(exhausted, 4, status)
        done = (status != -1) | (iteration + 1 == n_jamming_steps)

        next_state, next_system = jax.lax.cond(
            done,
            lambda: (evaluated_state, evaluated_system),
            lambda: _scale_to_packing_fraction_grouped(
                new_low_state, new_low_system, next_phi, group_id
            ),
        )
        carry_phi = jnp.where(done, trial_phi, next_phi)

        if verbose:
            jax.debug.print(
                "Step {i}: phi={phi}, increment={inc}, PE/N={pe}, steps={n}",
                i=iteration + 1,
                phi=trial_phi,
                inc=increment,
                pe=pe,
                n=n_steps,
            )

        return (
            iteration + 1,
            status,
            next_state,
            next_system,
            new_low_state,
            new_low_system,
            carry_phi,
            new_low_phi,
            new_increment,
            pe,
            min_info,
            max_minimization_steps,
        )

    final_carry = jax.lax.while_loop(condition, step, init_carry)
    (
        iterations,
        status,
        final_state,
        final_system,
        low_state,
        low_system,
        _,
        _,
        _,
        final_pe,
        min_info,
        max_minimization_steps,
    ) = final_carry
    status = jnp.where(status == -1, 3, status)
    success = status == 0
    info = JammingInfo(
        success,
        status,
        jnp.asarray(iterations),
        min_info,
        max_minimization_steps,
    )
    result = JamResult(
        unjammed_state=low_state,
        unjammed_system=low_system,
        jammed_state=final_state,
        jammed_system=final_system,
        packing_fraction=jnp.where(
            success, compute_packing_fraction(final_state, final_system), jnp.nan
        ),
        potential_energy=jnp.where(success, final_pe, jnp.nan),
        reason=_jam_reason(
            info,
            JamReason.TARGET_REACHED,
            measurement_finite=jnp.isfinite(final_pe),
        ),
        info=info,
    )
    return (result, info) if return_info else result


rollback_pe_band_jam.clear_cache = _rollback_pe_band_jam.clear_cache  # type: ignore[attr-defined]
