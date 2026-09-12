"""Mechanical convergence must survive low/constant energy and rigid rotations."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
import jaxdem as jd
from jaxdem.minimizers.routines import (
    CONVERGED,
    MAX_STEPS,
    NONFINITE,
    convergence_info,
    minimize,
)
from jaxdem.utils.jamming import bisection_jam, pe_band_jam, pressure_bisection_jam


def pair(dim=2, overlap=1e-7):
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


@pytest.mark.parametrize("dim", [2, 3])
def test_small_energy_does_not_certify_force_balance(dim):
    st, sy = pair(dim)
    out = sy.minimize(st, sy, max_steps=3, return_info=True)
    assert int(out[2]) == 3
    assert not out[-1].converged
    assert 0.0 <= out[3] <= 1e-14
    assert int(out[-1].status) == MAX_STEPS


def test_energy_plateau_is_not_convergence():
    st, sy = pair(overlap=0.1)
    sy = replace(sy, minimizer=optax.sgd(0.0))
    out = minimize(st, sy, max_steps=4, return_info=True)
    assert int(out[2]) == 4
    np.testing.assert_allclose(out[3], minimize(st, sy, max_steps=0)[3])
    assert not out[-1].converged


@pytest.mark.parametrize("dim", [2, 3])
def test_balanced_force_does_not_hide_a_clump_torque(dim):
    centers = np.zeros((4, dim))
    centers[2, :2] = [0.5, -0.9]
    centers[3, :2] = [-0.5, 0.9]
    st, sy = pair(dim)
    st = jd.State.create(
        pos=jnp.asarray(centers),
        rad=jnp.full(4, 0.5),
        clump_id=jnp.array([0, 0, 1, 2]),
        fixed=jnp.array([False, False, True, True]),
    )
    arms = jnp.zeros((4, dim)).at[0, 0].set(0.5).at[1, 0].set(-0.5)
    st.pos_p = arms
    sy = jd.System.create(
        st.shape,
        collider_type="naive",
        dt=0.01,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(dim, 10.0)},
        mat_table=sy.mat_table,
        minimizer=jd.minimizers.fire,
    )
    out = minimize(
        st, sy, max_steps=0, force_tol=1e-10, torque_tol=1e-10, return_info=True
    )
    assert out[-1].force_max < 1e-12
    assert out[-1].torque_max > 0.09
    assert not out[-1].converged


def test_fixed_reactions_and_contact_free_states():
    st, sy = pair(overlap=0.1)
    st.fixed = jnp.ones(2, dtype=bool)
    out = minimize(st, sy, max_steps=0, return_info=True)
    assert out[-1].converged and out[-1].force_max == 0
    st, sy = pair(overlap=-0.1)
    out = minimize(st, sy, return_info=True)
    assert int(out[2]) == 0 and int(out[-1].status) == CONVERGED
    assert len(minimize(st, sy)) == 4


def test_nonfinite_is_a_failure():
    st, sy = pair()
    st.pos_c = st.pos_c.at[0, 0].set(jnp.nan)
    out = minimize(st, sy, max_steps=2, return_info=True)
    assert not out[-1].converged and int(out[-1].status) == NONFINITE


def test_norms_are_per_body_and_rotation_invariant():
    force = jnp.tile(jnp.array([[0.8, 0.8]]), (17, 1))
    torque = jnp.zeros((17, 1))
    info = convergence_info(
        force,
        torque,
        jnp.zeros(17, dtype=bool),
        1.0,
        1.0,
    )
    assert not info.converged
    assert float(info.force_max) == pytest.approx(np.sqrt(1.28))


def test_jit_vmap_preserves_per_system_convergence():
    st, sy = pair()

    def run(dx):
        trial = replace(st, pos_c=st.pos_c.at[1, 0].set(1.0 + dx))
        return minimize(trial, sy, max_steps=0, return_info=True)[-1].converged

    np.testing.assert_array_equal(
        jax.jit(jax.vmap(run))(jnp.array([0.1, -0.1])), [True, False]
    )


def test_custom_signed_objective_uses_its_gradient():
    st, sy = pair(overlap=-0.1)
    sy = replace(sy, target_fn=lambda s, y: jnp.sum((s.pos_c - 2.0) ** 2) - 100.0)
    out = minimize(st, sy, max_steps=1, return_info=True)
    assert int(out[2]) == 1 and not out[-1].converged
    assert out[3] < 0.0


def lattice():
    st, sy = pair()
    xy = (
        np.stack(
            np.meshgrid(np.arange(3), np.arange(3), indexing="ij"), axis=-1
        ).reshape(-1, 2)
        * 1.1
    )
    st = jd.State.create(pos=jnp.asarray(xy), rad=jnp.full(9, 0.5))
    sy = jd.System.create(
        st.shape,
        collider_type="naive",
        dt=0.01,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(2, 3.3)},
        mat_table=sy.mat_table,
        minimizer=jd.minimizers.fire,
    )
    return st, sy


def test_bisection_returns_the_validated_high_state():
    st, sy = lattice()
    result, info = bisection_jam(
        st,
        sy,
        n_minimization_steps=2,
        n_jamming_steps=60,
        pe_tol=1e-8,
        packing_fraction_increment=0.03,
        packing_fraction_tolerance=1e-6,
        verbose=False,
        return_info=True,
    )
    assert result.converged and info.converged and info.minimization.converged
    assert result.potential_energy > 1e-8
    actual = jd.utils.compute_packing_fraction(
        result.jammed_state, result.jammed_system
    )
    np.testing.assert_allclose(result.packing_fraction, actual, rtol=1e-12)
    again = minimize(
        result.jammed_state, result.jammed_system, max_steps=0, return_info=True
    )
    assert again[-1].converged
    np.testing.assert_allclose(again[3], result.potential_energy, rtol=1e-10)
    assert len(result) == 6
    low = minimize(
        result.unjammed_state, result.unjammed_system, max_steps=0, return_info=True
    )
    assert low[-1].converged and 0.0 <= low[3] <= 1e-8


@pytest.mark.parametrize("outer_limit", [False, True])
def test_bisection_failure_cannot_look_jammed(outer_limit):
    st, sy = pair(overlap=-0.1 if outer_limit else 0.1)
    result, info = bisection_jam(
        st,
        sy,
        n_minimization_steps=0,
        n_jamming_steps=1,
        verbose=False,
        return_info=True,
    )
    assert not result.converged and not info.converged
    assert jnp.isnan(result.packing_fraction) and jnp.isnan(result.potential_energy)
    assert int(info.status) == (3 if outer_limit else 2)


def test_other_jamming_drivers_reject_unresolved_relaxation():
    st, sy = pair(overlap=0.1)
    result = pe_band_jam(
        st, sy, n_minimization_steps=0, n_jamming_steps=1, verbose=False
    )
    assert not result.converged
    with pytest.raises(RuntimeError, match="minimization failed"):
        pressure_bisection_jam(
            st, sy, n_minimization_steps=0, n_jamming_steps=1, verbose=False
        )


@pytest.mark.parametrize("driver", ["energy", "pressure"])
def test_other_jamming_drivers_return_balanced_in_band_states(driver):
    st, sy = lattice()
    if driver == "energy":
        result = pe_band_jam(
            st,
            sy,
            pe_tol=1e-6,
            pe_band_factor=2.0,
            packing_fraction_increment=0.03,
            n_minimization_steps=2,
            n_jamming_steps=60,
            verbose=False,
        )
        assert 1e-6 <= result.potential_energy <= 2e-6
    else:
        result = pressure_bisection_jam(
            st,
            sy,
            pressure_threshold=1e-4,
            pressure_band_factor=1.2,
            growth_rate=1.02,
            fine_growth_rate=1.002,
            n_minimization_steps=2,
            n_jamming_steps=60,
            verbose=False,
        )
        from jaxdem.utils.contacts import compute_contact_pressure

        _, _, pressure = compute_contact_pressure(
            result.jammed_state, result.jammed_system
        )
        assert 1e-4 <= pressure <= 1.2e-4
    assert result.converged
    checked = minimize(
        result.jammed_state, result.jammed_system, max_steps=0, return_info=True
    )
    assert checked[-1].converged
    np.testing.assert_allclose(checked[3], result.potential_energy, rtol=1e-10)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("driver", ["bisection", "energy"])
def test_unbalanced_small_energy_cannot_establish_lower_bound(dim, driver):
    st, sy = pair(dim)
    kw = dict(n_minimization_steps=3, n_jamming_steps=4, pe_tol=1e-14, verbose=False)
    if driver == "bisection":
        result, info = bisection_jam(st, sy, return_info=True, **kw)
        assert not info.converged and int(info.status) == 2
        assert int(info.steps) == 1 and int(info.minimization.status) == MAX_STEPS
        assert not info.minimization.converged
    else:
        result = pe_band_jam(st, sy, **kw)
    assert not result.converged
    assert jnp.isnan(result.packing_fraction) and jnp.isnan(result.potential_energy)
    np.testing.assert_array_equal(result.unjammed_state.pos_c, st.pos_c)
    np.testing.assert_array_equal(
        result.unjammed_system.domain.box_size, sy.domain.box_size
    )


@pytest.mark.parametrize(
    "optimizer", [jd.minimizers.fire, jd.minimizers.damped_newtonian]
)
@pytest.mark.parametrize("max_steps", [0, 5])
def test_physical_relaxation_evaluates_energy_once(monkeypatch, optimizer, max_steps):
    import jaxdem.minimizers.routines as routines

    st, sy = pair(overlap=0.1)
    sy = replace(sy, minimizer=optimizer(dt=0.01))
    original_energy = routines.compute_potential_energy
    evaluations = []

    def counted_energy(state, system):
        pe = original_energy(state, system)
        jax.debug.callback(lambda e: evaluations.append(float(e)), pe, ordered=True)
        return pe

    routines.minimize.clear_cache()
    monkeypatch.setattr(routines, "compute_potential_energy", counted_energy)
    try:
        result = routines.minimize(
            st, sy, max_steps=max_steps, force_tol=0.0, torque_tol=0.0, return_info=True
        )
        jax.block_until_ready(result)
        jax.effects_barrier()
        assert int(result[2]) == max_steps
        assert len(evaluations) == 1
        expected = original_energy(result[0], result[1]) / st.N
        np.testing.assert_allclose(result[3], expected, rtol=1e-12)
    finally:
        routines.minimize.clear_cache()


def test_conjugate_gradient_keeps_objective_evaluations():
    st, sy = pair(overlap=0.1)
    sy = replace(sy, minimizer=jd.minimizers.conjugate_gradient())
    initial_energy = minimize(st, sy, max_steps=0)[3]
    result = minimize(st, sy, max_steps=8, return_info=True)
    assert result[-1].finite
    assert 0.0 <= result[3] < initial_energy
    np.testing.assert_allclose(
        result[3], jd.utils.compute_potential_energy(result[0], result[1]) / st.N
    )


def test_force_only_path_rejects_nonfinite_final_energy(monkeypatch):
    import jaxdem.minimizers.routines as routines

    st, sy = pair(overlap=-0.1)
    routines.minimize.clear_cache()
    monkeypatch.setattr(
        routines, "compute_potential_energy", lambda st, sy: jnp.array(jnp.nan)
    )
    try:
        out = routines.minimize(st, sy, max_steps=0, return_info=True)
        assert not out[-1].converged and not out[-1].finite
        assert int(out[-1].status) == NONFINITE
    finally:
        routines.minimize.clear_cache()


@pytest.mark.parametrize("dim", [2, 3])
def test_sphere_relaxation_helper_uses_force_tolerance(dim):
    from jaxdem.utils.random_sphere_configuration import minimize_sphere_configuration

    pos = np.zeros((2, dim))
    pos[1, 0] = 0.9
    relaxed, box = minimize_sphere_configuration(
        [0.5, 0.5], pos, phi=0.05, dim=dim, force_tol=1e-10
    )
    delta = np.asarray(relaxed[1] - relaxed[0])
    delta = delta - np.asarray(box) * np.round(delta / np.asarray(box))
    assert np.linalg.norm(delta) >= 1.0 - 1e-8


def test_quasistatic_compression_forwards_mechanical_tolerances():
    from jaxdem.utils.packing_utils import quasistatic_compress_to_packing_fraction

    st, sy = lattice()
    target = float(jd.utils.compute_packing_fraction(st, sy)) + 0.01
    st, sy, phi, pe = quasistatic_compress_to_packing_fraction(
        st,
        sy,
        target,
        step=0.005,
        force_tol=1e-11,
        torque_tol=1e-11,
        max_n_min_steps_per_outer=2,
        max_n_outer_steps=5,
    )
    assert float(phi) == pytest.approx(target)
    checked = minimize(
        st, sy, max_steps=0, force_tol=1e-11, torque_tol=1e-11, return_info=True
    )
    assert checked[-1].converged
    np.testing.assert_allclose(pe, checked[3])
