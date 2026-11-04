"""Pair virial kernels and constant-pressure minimization."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.minimizers.routines import minimize
from jaxdem.utils.contacts import compute_contact_pressure

COLLIDERS = ["naive", "CellList", "MultiCellList", "NeighborList"]


def packing(dim, collider="naive", clumps=False, n=24, phi=0.9, seed=0, **system_kw):
    rng = np.random.default_rng(seed)
    volume = n * (np.pi / 4 if dim == 2 else np.pi / 6)
    length = (volume / phi) ** (1 / dim)
    centers = rng.uniform(0.0, length, size=(n, dim))
    if clumps:
        st = jd.State.create(dim=dim)
        for center in centers[: n // 2]:
            body = np.zeros((2, dim))
            body[1, 0] = 0.6
            st = jd.State.add_clump(
                st, pos=jnp.asarray(body + center), rad=jnp.full(2, 0.5)
            )
    else:
        st = jd.State.create(pos=jnp.asarray(centers), rad=jnp.full(n, 0.5))
    mat = jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)
    kw = dict(
        state=st,
        collider_type=collider,
        dt=0.01,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(dim, length), "anchor": jnp.full(dim, 0.25)},
        mat_table=jd.MaterialTable.from_materials([mat]),
        minimizer=jd.minimizers.fire,
    )
    if collider == "NeighborList":
        kw["collider_kw"] = {"cutoff": 1.2, "skin": 0.3}
    kw.update(system_kw)
    return st, jd.System.create(**kw)


def dilated_energy(st, sy, log_scale):
    factor = jnp.exp(log_scale)
    anchor = sy.domain.anchor
    box = sy.domain.box_size * factor
    st = replace(st, pos_c=anchor + (st.pos_c - anchor) * factor)
    sy = replace(sy, domain=replace(sy.domain, box_size=box, inv_box_size=1.0 / box))
    return sy.collider.compute_potential_energy(st, sy)[2]


def virial_pressure(st, sy):
    _, _, virial = sy.collider.compute_force(
        st, sy, advance_history=False, quasistatic=True, virial=True
    )
    return float(virial) / (st.dim * float(jnp.prod(sy.domain.box_size)))


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("collider", COLLIDERS)
@pytest.mark.parametrize("clumps", [False, True])
def test_virial_is_minus_the_dilation_derivative(dim, collider, clumps):
    st, sy = packing(dim, collider, clumps)
    state, _, virial = sy.collider.compute_force(
        st, sy, advance_history=False, quasistatic=True, virial=True
    )
    plain_state, _ = sy.collider.compute_force(
        st, sy, advance_history=False, quasistatic=True
    )
    np.testing.assert_array_equal(state.force, plain_state.force)
    h = 1e-5
    finite_difference = -(dilated_energy(st, sy, h) - dilated_energy(st, sy, -h)) / (
        2 * h
    )
    assert float(virial) > 0.0
    np.testing.assert_allclose(float(virial), float(finite_difference), rtol=1e-7)
    if not clumps:
        _, _, pressure = compute_contact_pressure(st, sy)
        volume = float(jnp.prod(sy.domain.box_size))
        np.testing.assert_allclose(float(virial), float(pressure) * dim * volume)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("collider", ["naive", "NeighborList"])
def test_minimize_reaches_target_pressure(dim, collider):
    st, sy = packing(dim, collider)
    target = 1e-3
    result = minimize(
        st, sy, max_steps=100_000, force_tol=1e-9, pressure=target, pressure_tol=1e-9
    )
    assert result.info.converged
    assert float(result.info.force_max) <= 1e-9
    _, _, pressure = compute_contact_pressure(result.state, result.system)
    np.testing.assert_allclose(float(pressure), target, atol=1e-9)
    np.testing.assert_allclose(float(result.info.pressure), target, atol=1e-9)
    assert not np.allclose(result.system.domain.box_size, sy.domain.box_size)
    np.testing.assert_allclose(
        result.system.domain.inv_box_size, 1.0 / result.system.domain.box_size
    )
    # Reported energy is the potential energy per sphere, not the enthalpy.
    _, _, energy = result.system.collider.compute_potential_energy(
        result.state, result.system
    )
    np.testing.assert_allclose(float(result.energy), float(energy) / st.N)


def test_clumps_relax_to_the_body_centre_virial_pressure():
    st, sy = packing(2, clumps=True, phi=0.8)
    target = 1e-3
    result = minimize(
        st, sy, max_steps=100_000, force_tol=1e-9, pressure=target, pressure_tol=1e-9
    )
    assert result.info.converged
    assert float(result.info.torque_max) <= 1e-9
    np.testing.assert_allclose(
        virial_pressure(result.state, result.system), target, atol=1e-9
    )
    np.testing.assert_allclose(float(result.info.pressure), target, atol=1e-9)


def test_value_based_optimizer_uses_the_same_enthalpy():
    st, sy = packing(2, minimizer=jd.minimizers.conjugate_gradient)
    target = 1e-3
    result = minimize(
        st, sy, max_steps=20_000, force_tol=1e-9, pressure=target, pressure_tol=1e-9
    )
    assert result.info.converged
    np.testing.assert_allclose(
        virial_pressure(result.state, result.system), target, atol=1e-9
    )


def test_history_advances_with_the_box():
    st, sy = packing(2, n=16, phi=0.85)
    mat = jd.Material.create(
        "elasticfrict", density=1.0, young=1.0, poisson=0.0, e=1.0, mu=0.5, mu_r=0.0
    )
    sy = jd.System.create(
        state=st,
        collider_type="naive",
        dt=0.01,
        domain_type="periodic",
        domain_kw={"box_size": sy.domain.box_size},
        mat_table=jd.MaterialTable.from_materials([mat]),
        minimizer=jd.minimizers.fire,
        force_model_type="cundallstrack",
    )
    result = minimize(
        st,
        sy,
        max_steps=100_000,
        force_tol=1e-9,
        pressure=1e-3,
        pressure_tol=1e-9,
        advance_history=True,
    )
    assert result.info.converged
    assert bool(jnp.any(result.system.collider.history != 0.0))
    np.testing.assert_allclose(
        virial_pressure(result.state, result.system), 1e-3, atol=1e-9
    )


def test_vmap_over_target_pressures():
    st, sy = packing(2, n=16)

    def final_pressure(target):
        result = minimize(
            st, sy, max_steps=50_000, force_tol=1e-8, pressure=target, pressure_tol=1e-8
        )
        return result.info.pressure, result.info.converged

    targets = jnp.array([1e-4, 1e-3])
    pressures, converged = jax.jit(jax.vmap(final_pressure))(targets)
    assert bool(jnp.all(converged))
    np.testing.assert_allclose(np.asarray(pressures), np.asarray(targets), atol=1e-8)


def test_fixed_box_path_is_untouched():
    st, sy = packing(2, n=8)
    result = minimize(st, sy, max_steps=5)
    assert bool(jnp.isnan(result.info.pressure))
    np.testing.assert_array_equal(result.system.domain.box_size, sy.domain.box_size)
    assert len(sy.collider.compute_force(st, sy)) == 2


def test_pressure_rejects_unsupported_objectives():
    st, sy = packing(2, n=8)
    with pytest.raises(ValueError, match="physical energy"):
        minimize(
            st, replace(sy, target_fn=lambda s, _: jnp.sum(s.pos_c**2)), pressure=1.0
        )

    def pull(pos, state, system):
        return -pos, jnp.zeros_like(state.torque)

    def pull_energy(pos, state, system):
        return 0.5 * jnp.sum(pos**2, axis=-1)

    _, managed = packing(
        2, n=8, force_manager_kw={"force_functions": ((pull, pull_energy),)}
    )
    with pytest.raises(ValueError, match="pair forces only"):
        minimize(st, managed, pressure=1.0)
