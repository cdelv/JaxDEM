# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Jamming drivers: equilibration, brackets, failure diagnostics, and vmap."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.minimizers import TerminationReason
from jaxdem.minimizers.routines import (
    COLLIDER_OVERFLOW,
    MAX_STEPS,
    NONFINITE,
    MinimizationResult,
    MinimizeInfo,
    minimize,
)
from jaxdem.utils import jamming
from jaxdem.utils.jamming import (
    JamReason,
    bisection_jam,
    pe_band_jam,
    pressure_bisection_jam,
    rollback_pe_band_jam,
)

BAND_DRIVERS = (pe_band_jam, pressure_bisection_jam)


def _lattice(dim=2, spacing=1.1):
    pos = np.indices((3,) * dim).reshape(dim, -1).T * spacing
    state = jd.State.create(pos=jnp.asarray(pos), rad=jnp.full(len(pos), 0.5))
    material = jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)
    system = jd.System.create(
        state=state,
        collider_type="naive",
        dt=0.01,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(dim, 3.0 * spacing)},
        mat_table=jd.MaterialTable.from_materials([material]),
        minimizer=jd.minimizers.fire,
    )
    return state, system


def _pair(dim=2, overlap=1e-7):
    pos = jnp.zeros((2, dim)).at[1, 0].set(1.0 - overlap)
    st = jd.State.create(pos=pos, rad=jnp.full(2, 0.5), mass=jnp.ones(2))
    mat = jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)
    sy = jd.System.create(
        st.shape,
        collider_type="naive",
        dt=0.01,
        minimizer=jd.minimizers.fire,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(dim, 10.0)},
        mat_table=jd.MaterialTable.from_materials([mat]),
    )
    return st, sy


def _nonfinite_system():
    state = jd.State.create(
        pos=jnp.asarray([[jnp.nan, 0.0], [2.0, 0.0]]),
        rad=jnp.asarray([0.2, 0.2]),
    )
    return state, jd.System.create(state=state, collider_type="naive")


def _band_kwargs(driver):
    return (
        {"pe_tol": 1.0, "pe_band_factor": 2.0}
        if driver is pe_band_jam
        else {"pressure_threshold": 1.0, "pressure_band_factor": 2.0}
    )


@pytest.fixture
def patched_minimize(monkeypatch):
    pe_band_jam.clear_cache()

    def install(energies, counts, statuses=None):
        energies = jnp.asarray(energies, dtype=float)
        counts = jnp.asarray(counts, dtype=int)
        statuses = (
            jnp.zeros(len(counts), dtype=int)
            if statuses is None
            else jnp.asarray(statuses)
        )

        def fake_minimize(state, system, **kwargs):
            index = state.vel[0, 0].astype(int)
            status = statuses[index]
            marker = jnp.full_like(state.vel, index + 1)
            info = MinimizeInfo(
                status == 0, status != 3, jnp.asarray(0.0), jnp.asarray(0.0), status
            )
            return MinimizationResult(
                replace(state, vel=marker),
                system,
                counts[index],
                energies[index],
                info.reason,
                info,
            )

        monkeypatch.setattr(jd.System, "minimize", staticmethod(fake_minimize))
        monkeypatch.setattr(
            jamming,
            "compute_contact_pressure",
            lambda st, sy: (st, sy, energies[st.vel[0, 0].astype(int) - 1]),
        )

    yield install
    pe_band_jam.clear_cache()


def test_bisection_returns_the_validated_high_state():
    st, sy = _lattice()
    result, info = bisection_jam(
        st,
        sy,
        minimizer_kwargs={"max_steps": 2},
        n_jamming_steps=60,
        pe_tol=1e-8,
        packing_fraction_increment=0.03,
        packing_fraction_tolerance=1e-6,
        verbose=False,
        return_info=True,
    )
    assert result.converged and info.converged and info.minimization.converged
    assert 0 <= int(info.max_minimization_steps) <= 2
    assert 1e-8 < result.potential_energy <= 2e-8
    actual = jd.utils.compute_packing_fraction(
        result.jammed_state, result.jammed_system
    )
    np.testing.assert_allclose(result.packing_fraction, actual, rtol=1e-12)
    again = minimize(result.jammed_state, result.jammed_system, max_steps=0)
    assert again.info.converged
    np.testing.assert_allclose(again[3], result.potential_energy, rtol=1e-10)
    assert len(result) == 6
    low = minimize(result.unjammed_state, result.unjammed_system, max_steps=0)
    assert low.info.converged and 0.0 <= low[3] <= 1e-8


@pytest.mark.parametrize("driver", BAND_DRIVERS)
def test_three_dimensional_band_drivers_return_balanced_in_band_states(driver):
    state, system = _lattice(dim=3)
    kwargs = (
        {"pe_tol": 1e-6, "pe_band_factor": 2.0, "packing_fraction_increment": 0.03}
        if driver is pe_band_jam
        else {
            "pressure_threshold": 1e-4,
            "pressure_band_factor": 1.2,
            "growth_rate": 1.02,
            "fine_growth_rate": 1.002,
        }
    )
    result, info = driver(
        state,
        system,
        minimizer_kwargs={"max_steps": 2},
        n_jamming_steps=80,
        verbose=False,
        return_info=True,
        **kwargs,
    )
    assert result.converged and info.converged and info.minimization.converged
    assert int(info.status) == 0 and 0 < int(info.steps) <= 80
    assert 0 <= int(info.max_minimization_steps) <= 2
    checked = minimize(result.jammed_state, result.jammed_system, max_steps=0)
    assert checked.info.converged
    np.testing.assert_allclose(
        checked.info.force_max, info.minimization.force_max, atol=1e-14
    )
    np.testing.assert_allclose(
        checked.info.torque_max, info.minimization.torque_max, atol=1e-14
    )
    np.testing.assert_allclose(result.potential_energy, checked[3], rtol=1e-10)
    if driver is pe_band_jam:
        assert 1e-6 <= result.potential_energy <= 2e-6
    else:
        _, _, pressure = jamming.compute_contact_pressure(
            result.jammed_state, result.jammed_system
        )
        assert 1e-4 <= pressure <= 1.2e-4


@pytest.mark.parametrize("outer_limit", [False, True])
def test_bisection_failure_cannot_look_jammed(outer_limit):
    st, sy = _pair(overlap=-0.1 if outer_limit else 0.1)
    result, info = bisection_jam(
        st,
        sy,
        minimizer_kwargs={"max_steps": 0},
        n_jamming_steps=1,
        verbose=False,
        return_info=True,
    )
    assert not result.converged and not info.converged
    assert jnp.isnan(result.packing_fraction) and jnp.isnan(result.potential_energy)
    assert int(info.status) == (3 if outer_limit else 2)
    assert int(info.max_minimization_steps) == 0


@pytest.mark.parametrize("dim,driver", [(2, "bisection"), (3, "energy")])
def test_unbalanced_small_energy_cannot_establish_lower_bound(dim, driver):
    st, sy = _pair(dim)
    kw = {
        "minimizer_kwargs": {"max_steps": 3},
        "n_jamming_steps": 4,
        "pe_tol": 1e-14,
        "verbose": False,
    }
    if driver == "bisection":
        result, info = bisection_jam(st, sy, return_info=True, **kw)
        assert not info.converged and int(info.status) == 2
        assert int(info.steps) == 1 and int(info.minimization.status) == MAX_STEPS
        assert int(info.max_minimization_steps) == 3
        assert not info.minimization.converged
    else:
        result = pe_band_jam(st, sy, **kw)
    assert not result.converged
    assert jnp.isnan(result.packing_fraction) and jnp.isnan(result.potential_energy)
    np.testing.assert_array_equal(result.unjammed_state.pos_c, st.pos_c)
    np.testing.assert_array_equal(
        result.unjammed_system.domain.box_size, sy.domain.box_size
    )


def test_jamming_propagates_overflow_failure():
    st = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]]), rad=jnp.full(3, 0.5)
    )
    mat = jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)
    sy = jd.System.create(
        state=st,
        dt=0.01,
        minimizer=jd.minimizers.fire,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(2, 10.0)},
        force_model_type="spring",
        collider_type="neighborlist",
        collider_kw={
            "cutoff": 1.0,
            "skin": 0.05,
            "max_neighbors": 1,
            "secondary_collider_type": "naive",
        },
        mat_table=jd.MaterialTable.from_materials([mat]),
    )
    result, info = bisection_jam(
        st,
        sy,
        minimizer_kwargs={"max_steps": 10},
        n_jamming_steps=2,
        verbose=False,
        return_info=True,
    )
    assert not result.converged and not info.converged
    assert int(info.minimization.status) == COLLIDER_OVERFLOW
    assert int(info.max_minimization_steps) == 0


@pytest.mark.parametrize(
    "driver",
    [bisection_jam, pe_band_jam, pressure_bisection_jam, rollback_pe_band_jam],
)
def test_jamming_reports_nonfinite_minimization_without_retries(driver):
    state, system = _nonfinite_system()
    result = driver(
        state,
        system,
        minimizer_kwargs={"max_steps": 2},
        n_jamming_steps=3,
        verbose=False,
    )
    assert result.reason == JamReason.NONFINITE
    assert result.minimizer_reason == TerminationReason.NONFINITE
    assert int(result.steps) == 1
    assert int(result.max_minimization_steps) == 0
    assert len(result) == 6
    assert tuple(result)[2] is result.jammed_state


@pytest.mark.parametrize("driver", BAND_DRIVERS)
def test_outer_limit_preserves_last_evaluated_state(driver):
    state, system = _lattice()
    result, info = driver(
        state,
        system,
        n_jamming_steps=1,
        minimizer_kwargs={"max_steps": 0},
        verbose=False,
        return_info=True,
    )
    assert isinstance(result, jamming.JamResult) and isinstance(
        info, jamming.JammingInfo
    )
    assert not result.converged and not info.converged
    assert np.isnan(result.packing_fraction) and np.isnan(result.potential_energy)
    assert int(info.status) == 3
    assert int(info.steps) == 1 and int(info.max_minimization_steps) == 0
    assert bool(info.minimization.converged)
    np.testing.assert_array_equal(result.jammed_state.pos_c, state.pos_c)
    np.testing.assert_array_equal(
        result.jammed_system.domain.box_size, system.domain.box_size
    )
    zero_step = driver(state, system, n_jamming_steps=0, verbose=False)
    assert isinstance(zero_step, jamming.JamResult) and len(zero_step) == 6
    assert zero_step.reason == JamReason.MAX_STEPS


@pytest.mark.parametrize("driver", BAND_DRIVERS)
def test_maximum_includes_earlier_trials_and_returns_accepted_snapshot(
    driver, patched_minimize
):
    patched_minimize([0.0, 0.0, 1.5], [2, 9, 1])
    state, system = _lattice()
    result, info = driver(
        state,
        system,
        n_jamming_steps=3,
        minimizer_kwargs={"max_steps": 10},
        verbose=False,
        return_info=True,
        **_band_kwargs(driver),
    )
    assert result.converged and info.converged and int(info.status) == 0
    assert int(info.steps) == 3 and int(info.max_minimization_steps) == 9
    assert float(result.potential_energy) == 1.5
    np.testing.assert_array_equal(result.jammed_state.vel, np.full_like(state.vel, 3))
    np.testing.assert_array_equal(result.unjammed_state.vel, np.full_like(state.vel, 2))


@pytest.mark.parametrize("driver", BAND_DRIVERS)
def test_failed_trial_reports_nested_status_and_maximum(driver, patched_minimize):
    patched_minimize([0.0, 1.5], [2, 10], [0, 2])
    state, system = _lattice()
    result, info = driver(
        state,
        system,
        n_jamming_steps=4,
        minimizer_kwargs={"max_steps": 10},
        verbose=False,
        return_info=True,
        **_band_kwargs(driver),
    )
    assert not result.converged and not info.converged
    assert int(info.status) == 2 and int(info.minimization.status) == 2
    assert result.reason == JamReason.MINIMIZATION_FAILED
    assert int(info.steps) == 2 and int(info.max_minimization_steps) == 10
    np.testing.assert_array_equal(result.unjammed_state.vel, np.full_like(state.vel, 1))
    np.testing.assert_array_equal(result.jammed_state.vel, np.full_like(state.vel, 2))


def test_pressure_initially_above_band_is_a_reported_failure():
    # Compressed 3x3 periodic lattice is force-balanced with P = 1/9 >> 1.2e-4.
    state, system = _lattice(dim=2, spacing=0.9)
    result, info = pressure_bisection_jam(
        state,
        system,
        pressure_threshold=1e-4,
        pressure_band_factor=1.2,
        minimizer_kwargs={"max_steps": 1},
        return_info=True,
        verbose=False,
    )
    assert not result.converged and int(info.status) == 1
    assert int(info.steps) == 1 and info.minimization.converged


def test_pressure_initial_relaxation_failure_does_not_claim_invalid_pressure(
    patched_minimize,
):
    patched_minimize([0.0], [5], [2])
    state, system = _lattice()
    result = pressure_bisection_jam(state, system, verbose=False)
    assert result.reason == JamReason.MINIMIZATION_FAILED
    assert result.max_minimization_steps == 5


def test_pressure_bracket_exhaustion_is_a_reported_failure(patched_minimize):
    patched_minimize([0.0, 3.0, 3.0], [1, 6, 2])
    state, system = _lattice()
    result, info = pressure_bisection_jam(
        state,
        system,
        return_info=True,
        verbose=False,
        length_ratio_tolerance=0.1,
        **_band_kwargs(pressure_bisection_jam),
    )
    assert not result.converged and int(info.status) == 4
    assert int(info.steps) == 3 and int(info.max_minimization_steps) == 6
    assert info.minimization.converged


def test_energy_band_bracket_exhaustion_is_a_reported_failure(monkeypatch):
    bisection_jam.clear_cache()

    def discontinuous_minimize(state, system, **_kwargs):
        phi = jamming.compute_packing_fraction(state, system)
        energy = jnp.where(phi < 0.7, 0.0, 3.0)
        info = MinimizeInfo(
            jnp.asarray(True),
            jnp.asarray(True),
            jnp.asarray(0.0),
            jnp.asarray(0.0),
            jnp.asarray(0),
        )
        return MinimizationResult(state, system, 1, energy, info.reason, info)

    monkeypatch.setattr(jd.System, "minimize", staticmethod(discontinuous_minimize))
    state, system = _lattice()
    try:
        result, info = bisection_jam(
            state,
            system,
            pe_tol=1.0,
            pe_band_factor=2.0,
            packing_fraction_increment=0.03,
            packing_fraction_tolerance=0.0,
            n_jamming_steps=100,
            return_info=True,
            verbose=False,
        )
    finally:
        bisection_jam.clear_cache()

    assert not result.converged and int(info.status) == 4
    assert result.reason == JamReason.BRACKET_EXHAUSTED
    assert int(info.steps) < 100
    assert np.isnan(result.packing_fraction) and np.isnan(result.potential_energy)
    low_phi = jamming.compute_packing_fraction(
        result.unjammed_state, result.unjammed_system
    )
    high_phi = jamming.compute_packing_fraction(
        result.jammed_state, result.jammed_system
    )
    np.testing.assert_allclose(low_phi, high_phi, rtol=1e-14)


@pytest.mark.parametrize(
    ("growth_options", "expected_steps"),
    [({}, 5), ({"increment_growth_factor": 2.0}, 6)],
)
def test_rollback_halves_rejected_steps_with_optional_regrowth(
    monkeypatch, growth_options, expected_steps
):
    state = jd.State.create(
        pos=jnp.asarray([[0.0, 0.0], [2.0, 0.0]]), rad=jnp.asarray([0.2, 0.2])
    )
    system = jd.System.create(state=state, collider_type="naive")
    phi_initial = jd.utils.compute_packing_fraction(state, system)
    original = type(system).minimize

    def staged_minimize(state, system, **kwargs):
        result = original(state, system, **kwargs)
        delta_phi = jd.utils.compute_packing_fraction(state, system) - phi_initial
        energy = jnp.where(
            delta_phi <= 0.100001,
            0.5,
            jnp.where(delta_phi <= 0.170001, 1.5, 3.0),
        )
        return replace(
            result,
            energy=energy,
            reason=jnp.asarray(TerminationReason.FORCE_TORQUE_TOLERANCE),
            info=result.info._replace(
                converged=jnp.asarray(True),
                finite=jnp.asarray(True),
                status=jnp.asarray(0),
            ),
        )

    rollback_pe_band_jam.clear_cache()
    monkeypatch.setattr(type(system), "minimize", staticmethod(staged_minimize))
    try:
        result = rollback_pe_band_jam(
            state,
            system,
            pe_tol=1.0,
            pe_band_factor=2.0,
            packing_fraction_increment=0.2,
            n_jamming_steps=10,
            verbose=False,
            minimizer_kwargs={"max_steps": 1},
            **growth_options,
        )
    finally:
        rollback_pe_band_jam.clear_cache()

    assert result.converged
    assert result.steps == expected_steps
    assert result.potential_energy == pytest.approx(1.5)
    assert jd.utils.compute_packing_fraction(
        result.unjammed_state, result.unjammed_system
    ) == pytest.approx(float(phi_initial) + 0.1)
    assert jd.utils.compute_packing_fraction(
        result.jammed_state, result.jammed_system
    ) == pytest.approx(float(phi_initial) + 0.15)


def test_pressure_nan_is_not_accepted_as_target(monkeypatch):
    state = jd.State.create(
        pos=jnp.asarray([[0.0, 0.0], [2.0, 0.0]]), rad=jnp.asarray([0.2, 0.2])
    )
    system = jd.System.create(state=state, collider_type="naive")
    monkeypatch.setattr(
        "jaxdem.utils.jamming.compute_contact_pressure",
        lambda st, sy: (st, sy, jnp.asarray(jnp.nan)),
    )
    result = pressure_bisection_jam(
        state,
        system,
        n_jamming_steps=2,
        verbose=False,
        minimizer_kwargs={"max_steps": 1},
    )
    assert result.reason == JamReason.NONFINITE


def test_accepted_pressure_trial_is_returned_without_extra_relaxation(monkeypatch):
    state = jd.State.create(
        pos=jnp.asarray([[0.0, 0.0], [2.0, 0.0]]), rad=jnp.asarray([0.2, 0.2])
    )
    system = jd.System.create(state=state, collider_type="naive")
    original = type(system).minimize
    calls = 0

    def staged_minimize(state, system, **kwargs):
        nonlocal calls
        calls += 1
        result = original(state, system, **kwargs)
        if calls > 1:
            return replace(
                result,
                energy=jnp.asarray(jnp.nan),
                reason=jnp.asarray(TerminationReason.NONFINITE),
                info=result.info._replace(
                    converged=jnp.asarray(False),
                    finite=jnp.asarray(False),
                    status=jnp.asarray(NONFINITE),
                ),
            )
        return result

    monkeypatch.setattr(type(system), "minimize", staticmethod(staged_minimize))
    monkeypatch.setattr(
        "jaxdem.utils.jamming.compute_contact_pressure",
        lambda st, sy: (st, sy, jnp.asarray(1.005e-7)),
    )
    result = pressure_bisection_jam(
        state,
        system,
        n_jamming_steps=2,
        verbose=False,
        minimizer_kwargs={"max_steps": 1},
    )
    assert calls == 1
    assert result.reason == JamReason.TARGET_REACHED
    assert result.converged


def test_pe_band_info_survives_jit_vmap():
    pos = jnp.array([[0.0, 0.0], [1.1, 0.0]])
    state = jd.State.create(pos=pos, rad=jnp.full(2, 0.5))
    _, template = _lattice()
    system = jd.System.create(
        state=state,
        collider_type="naive",
        dt=0.01,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(2, 10.0)},
        mat_table=template.mat_table,
        minimizer=jd.minimizers.fire,
    )

    def run(separation):
        trial = replace(state, pos_c=state.pos_c.at[1, 0].set(separation))
        return pe_band_jam(
            trial,
            system,
            minimizer_kwargs={"max_steps": 0},
            n_jamming_steps=1,
            verbose=False,
            return_info=True,
        )

    result, info = jax.jit(jax.vmap(run))(jnp.array([1.1, 0.9]))
    np.testing.assert_array_equal(info.status, [3, 2])
    np.testing.assert_array_equal(info.steps, [1, 1])
    np.testing.assert_array_equal(info.max_minimization_steps, [0, 0])
    np.testing.assert_array_equal(info.minimization.converged, [True, False])
    np.testing.assert_array_equal(result.converged, [False, False])


def test_jamming_rejects_invalid_options():
    state, system = _nonfinite_system()
    with pytest.raises(TypeError, match="n_minimization_steps"):
        bisection_jam(
            state,
            system,
            verbose=False,
            minimizer_kwargs={"n_minimization_steps": 3},
        )
    for factor in (1.0, 0.5, np.inf):
        with pytest.raises(ValueError, match="pe_band_factor"):
            bisection_jam(state, system, pe_band_factor=factor, verbose=False)
    for kwargs in (
        {"pe_band_factor": 1.0},
        {"packing_fraction_increment": 0.0},
        {"increment_growth_factor": 0.5},
    ):
        with pytest.raises(ValueError):
            rollback_pe_band_jam(state, system, verbose=False, **kwargs)
