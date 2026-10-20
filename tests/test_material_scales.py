"""Material mixing and contact limits must preserve their physical scales."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.forces.cundall_strack import _pair_coefficients, geometric_mean_friction
from jaxdem.material_matchmakers.harmonic import HarmonicMaterialMatchmaker


@pytest.mark.parametrize("x64", [False, True])
def test_material_means_avoid_dimensionful_products(x64):
    with jax.enable_x64(x64):
        a = jnp.array([0.0, 1e-30, 1.0, 1e30])
        b = 3 * a
        actual = HarmonicMaterialMatchmaker.get_effective_property(a, b)
        np.testing.assert_allclose(actual[1:] / a[1:], 1.5, rtol=2e-6)
        assert actual[0] == 0
        one_zero = jnp.array([0.0, 3.0])
        gradient = jax.grad(
            lambda pair: HarmonicMaterialMatchmaker.get_effective_property(*pair)
        )(one_zero)
        np.testing.assert_allclose(gradient, [2.0, 0.0], rtol=5e-6)
        gradients = jax.grad(
            lambda x, y: jnp.sum(
                HarmonicMaterialMatchmaker.get_effective_property(x, y)
            ),
            argnums=(0, 1),
        )(a, b)
        np.testing.assert_allclose(gradients[0][1:], 18 / 16, rtol=5e-6)
        np.testing.assert_allclose(gradients[1][1:], 2 / 16, rtol=5e-6)
        assert all(np.isfinite(g).all() for g in gradients)
        np.testing.assert_allclose(
            geometric_mean_friction(a, b)[1:] / a[1:], np.sqrt(3), rtol=2e-6
        )


@pytest.mark.parametrize("x64", [False, True])
def test_elastic_contact_coefficients_preserve_modulus_and_mass_units(x64):
    with jax.enable_x64(x64):
        state = jd.State.create(
            pos=jnp.array([[0.0, 0.0], [0.8, 0.0]]),
            rad=jnp.array([0.5, 0.75]),
            mass=jnp.array([2.0, 3.0]),
            mat_id=jnp.arange(2),
        )
        materials = [
            jd.Material.create(
                "elasticfrict", density=1, young=young, poisson=nu, mu=0.4, e=0.7
            )
            for young, nu in [(100, 0.25), (200, 0.3)]
        ]
        system = jd.System.create(
            state=state,
            force_model_type="cundallstrack",
            mat_table=jd.MaterialTable.from_materials(materials),
            collider_type="naive",
        )
        # Independent series compliances, reduced mass, and oscillator damping.
        kn = 2 / (1 / 50 + 1 / 150)
        kt = 1 / (1.25 / 50 + 1.3 / 150)
        reduced_mass = 1 / (1 / 2 + 1 / 3)
        beta = -np.log(0.7) / np.sqrt(np.pi**2 + np.log(0.7) ** 2)
        expected = [
            kn,
            kt,
            2 * beta * np.sqrt(kn * reduced_mass),
            2 * beta * np.sqrt(kt * reduced_mass),
        ]
        for scale in [1e-25, 1.0, 1e25]:
            scaled_state = replace(state, mass=state.mass * scale)
            scaled_system = replace(
                system,
                mat_table=replace(
                    system.mat_table,
                    props={
                        **system.mat_table.props,
                        "young": system.mat_table.young * scale,
                    },
                ),
            )
            coefficients = jnp.stack(
                _pair_coefficients(0, 1, scaled_state, scaled_system)[:4]
            )
            np.testing.assert_allclose(coefficients / scale, expected, rtol=3e-6)


@pytest.mark.parametrize("scale", [1e-32, 1.0, 1e32])
@pytest.mark.parametrize("displacement", [0.1, 3.0])
def test_coulomb_force_and_return_mapping_respect_stiffness_units(scale, displacement):
    # The extreme force scales require float64 norms, independently of input units.
    with jax.enable_x64(True):
        state = jd.State.create(
            pos=jnp.array([[0.0, 0.0], [0.8, 0.0]]), rad=jnp.full(2, 0.5)
        )
        material = jd.Material.create(
            "cundallstrackparams",
            density=1.0,
            k_n=100 * scale,
            k_t=10 * scale,
            b_n=0.0,
            b_t=0.0,
            mu=1.0,
            mu_r=0.0,
        )
        system = jd.System.create(
            state=state,
            force_model=jd.forces.CundallStrackForce(parameterization="coefficients"),
            mat_table=jd.MaterialTable.from_materials([material]),
            collider_type="neighborlist",
            collider_kw={"cutoff": 1.0, "max_neighbors": 2},
        )
        history = jnp.array([0.0, displacement, -1.0, 0.0])
        force, _, advanced = system.force_model.force(
            0, 1, state.pos, state, system, history
        )
        expected_tangent = min(10 * displacement, 20)
        np.testing.assert_allclose(force / scale, [-20, -expected_tangent], rtol=1e-12)
        np.testing.assert_allclose(advanced[:2], [0, min(displacement, 2)], rtol=1e-12)


@pytest.mark.parametrize("scale", [1e-18, 1.0])
def test_fire_mixes_small_nonzero_forces_without_an_absolute_cutoff(scale):
    optimizer = jd.fire(dt=1.0)
    state = optimizer.init(jnp.zeros((1, 2)))
    state = state._replace(vel=jnp.array([[0.0, scale]]))
    update, advanced = optimizer.update(jnp.array([[-scale, 0.0]]), state)
    expected = np.array([[0.9 + 0.1 * np.sqrt(2), 0.9]])
    np.testing.assert_allclose(advanced.vel / scale, expected, rtol=2e-6)
    np.testing.assert_allclose(update / scale, expected / 2, rtol=2e-6)
    zero, _ = optimizer.update(jnp.zeros((1, 2)), optimizer.init(jnp.zeros((1, 2))))
    np.testing.assert_array_equal(zero, 0)
