"""The band drivers report the same diagnostics and failures as bisection."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.minimizers.routines import MinimizationResult, MinimizeInfo, minimize
from jaxdem.utils import jamming

DRIVERS = (jamming.pe_band_jam, jamming.pressure_bisection_jam)


def lattice(dim=2):
    pos = np.indices((3,) * dim).reshape(dim, -1).T * 1.1
    state = jd.State.create(pos=jnp.asarray(pos), rad=jnp.full(len(pos), 0.5))
    material = jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)
    system = jd.System.create(
        state=state,
        collider_type="naive",
        dt=0.01,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(dim, 3.3)},
        mat_table=jd.MaterialTable.from_materials([material]),
        minimizer=jd.minimizers.fire,
    )
    return state, system


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("outer_steps", [0, 1])
def test_outer_limit_preserves_last_evaluated_state(driver, outer_steps):
    state, system = lattice()
    result, info = driver(
        state,
        system,
        n_jamming_steps=outer_steps,
        n_minimization_steps=0,
        verbose=False,
        return_info=True,
    )
    assert isinstance(result, jamming.JamResult) and isinstance(
        info, jamming.JammingInfo
    )
    assert not result.converged and not info.converged
    assert np.isnan(result.packing_fraction) and np.isnan(result.potential_energy)
    assert int(info.status) == 3
    assert int(info.steps) == outer_steps and int(info.max_minimization_steps) == 0
    assert bool(info.minimization.converged) == (outer_steps > 0)
    np.testing.assert_array_equal(result.jammed_state.pos_c, state.pos_c)
    np.testing.assert_array_equal(
        result.jammed_system.domain.box_size, system.domain.box_size
    )
    default = driver(state, system, n_jamming_steps=0, verbose=False)
    assert isinstance(default, jamming.JamResult) and len(default) == 6


@pytest.mark.parametrize("driver", DRIVERS)
def test_three_dimensional_result_is_equilibrated(driver):
    state, system = lattice(dim=3)
    kwargs = (
        {"pe_tol": 1e-6, "pe_band_factor": 2.0, "packing_fraction_increment": 0.03}
        if driver is jamming.pe_band_jam
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
        n_minimization_steps=2,
        n_jamming_steps=80,
        verbose=False,
        return_info=True,
        **kwargs,
    )
    assert result.converged and info.converged and info.minimization.converged
    checked = minimize(result.jammed_state, result.jammed_system, max_steps=0)
    assert checked.info.converged
    np.testing.assert_allclose(result.potential_energy, checked[3], rtol=1e-10)
    if driver is jamming.pe_band_jam:
        assert 1e-6 <= result.potential_energy <= 2e-6
    else:
        _, _, pressure = jamming.compute_contact_pressure(
            result.jammed_state, result.jammed_system
        )
        assert 1e-4 <= pressure <= 1.2e-4


@pytest.fixture
def patched_minimize(monkeypatch):
    # Stubs exercise search bookkeeping without depending on FIRE iteration counts.
    jamming.pe_band_jam.clear_cache()

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
    jamming.pe_band_jam.clear_cache()


def band_kwargs(driver):
    return (
        {"pe_tol": 1.0, "pe_band_factor": 2.0}
        if driver is jamming.pe_band_jam
        else {"pressure_threshold": 1.0, "pressure_band_factor": 2.0}
    )


@pytest.mark.parametrize("driver", DRIVERS)
def test_maximum_includes_earlier_trials_and_returns_accepted_snapshot(
    driver, patched_minimize
):
    patched_minimize([0.0, 0.0, 1.5], [2, 9, 1])
    state, system = lattice()
    result, info = driver(
        state,
        system,
        n_jamming_steps=3,
        n_minimization_steps=10,
        verbose=False,
        return_info=True,
        **band_kwargs(driver),
    )
    assert result.converged and info.converged and int(info.status) == 0
    assert int(info.steps) == 3 and int(info.max_minimization_steps) == 9
    assert float(result.potential_energy) == 1.5
    np.testing.assert_array_equal(result.jammed_state.vel, np.full_like(state.vel, 3))
    np.testing.assert_array_equal(result.unjammed_state.vel, np.full_like(state.vel, 2))


@pytest.mark.parametrize("driver", DRIVERS)
@pytest.mark.parametrize("min_status", [2, 3, 4])
def test_failed_trial_reports_nested_status_and_maximum(
    driver, min_status, patched_minimize
):
    patched_minimize([0.0, 1.5], [2, 10], [0, min_status])
    state, system = lattice()
    result, info = driver(
        state,
        system,
        n_jamming_steps=4,
        n_minimization_steps=10,
        verbose=False,
        return_info=True,
        **band_kwargs(driver),
    )
    assert not result.converged and not info.converged
    assert int(info.status) == 2 and int(info.minimization.status) == min_status
    assert (
        result.reason
        == {
            2: jamming.JamReason.MINIMIZATION_FAILED,
            3: jamming.JamReason.NONFINITE,
            4: jamming.JamReason.SEARCH_OVERFLOW,
        }[min_status]
    )
    assert int(info.steps) == 2 and int(info.max_minimization_steps) == 10
    np.testing.assert_array_equal(result.unjammed_state.vel, np.full_like(state.vel, 1))
    np.testing.assert_array_equal(result.jammed_state.vel, np.full_like(state.vel, 2))


def test_pressure_initially_above_band_is_a_reported_failure(patched_minimize):
    patched_minimize([3.0], [7])
    state, system = lattice()
    result, info = jamming.pressure_bisection_jam(
        state,
        system,
        return_info=True,
        verbose=False,
        **band_kwargs(DRIVERS[1]),
    )
    assert not result.converged and int(info.status) == 1
    assert int(info.steps) == 1 and int(info.max_minimization_steps) == 7
    assert info.minimization.converged


def test_pressure_initial_relaxation_failure_does_not_claim_invalid_pressure(
    patched_minimize,
):
    patched_minimize([0.0], [5], [2])
    state, system = lattice()
    result = jamming.pressure_bisection_jam(state, system, verbose=False)
    assert result.reason == jamming.JamReason.MINIMIZATION_FAILED
    assert result.max_minimization_steps == 5


def test_pressure_bracket_exhaustion_is_a_reported_failure(patched_minimize):
    patched_minimize([0.0, 3.0, 3.0], [1, 6, 2])
    state, system = lattice()
    result, info = jamming.pressure_bisection_jam(
        state,
        system,
        return_info=True,
        verbose=False,
        length_ratio_tolerance=0.1,
        **band_kwargs(DRIVERS[1]),
    )
    assert not result.converged and int(info.status) == 4
    assert int(info.steps) == 3 and int(info.max_minimization_steps) == 6
    assert info.minimization.converged


def test_nonfinite_pressure_cannot_be_accepted(monkeypatch, patched_minimize):
    patched_minimize([1.5], [4])
    monkeypatch.setattr(
        jamming, "compute_contact_pressure", lambda st, sy: (st, sy, jnp.nan)
    )
    state, system = lattice()
    result, info = jamming.pressure_bisection_jam(
        state, system, return_info=True, verbose=False
    )
    assert not result.converged and int(info.status) == 2
    assert int(info.steps) == 1 and int(info.max_minimization_steps) == 4


def test_pe_band_info_survives_jit_vmap():
    pos = jnp.array([[0.0, 0.0], [1.1, 0.0]])
    state = jd.State.create(pos=pos, rad=jnp.full(2, 0.5))
    _, template = lattice()
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
        return jamming.pe_band_jam(
            trial,
            system,
            n_minimization_steps=0,
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
