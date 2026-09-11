# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
from dataclasses import replace
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

import jaxdem as jd
from jaxdem.minimizers import conjugate_gradient


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class CustomDefaultSpring(jd.forces.SpringForce):
    pass


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class CustomOptOutSpring(jd.forces.SpringForce):
    @property
    def supports_analytical_energy_gradient(self) -> bool:
        return False


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class ExplodingInitializer(jd.integrators.VelocityVerlet):
    @staticmethod
    def initialize(state, system):
        raise AssertionError("minimization must not initialize integrators")


def test_conjugate_gradient_uses_supported_analytic_vjp_with_cell_list():
    """Line-search differentiation must not reverse through cell traversal."""
    state = jd.State.create(pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]), rad=jnp.ones(2))
    system = jd.System.create(
        state=state,
        collider_type="CellList",
        minimizer=conjugate_gradient,
        minimizer_kw={"max_linesearch_steps": 2},
    )
    final_state, _, steps, energy = system.minimize(
        state,
        system,
        max_steps=1,
        pe_tol=-1.0,
        pe_diff_tol=-1.0,
        force_tol=-1.0,
    )
    assert int(steps) == 1
    assert bool(jnp.all(jnp.isfinite(final_state.pos)))
    assert bool(jnp.isfinite(energy))


def test_minimizer_does_not_invoke_integrator_initialization():
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
        rad=jnp.ones(2),
        vel=jnp.array([[0.3, -0.2], [0.1, 0.4]]),
    )
    system = jd.System.create(
        state=state,
        collider_type="naive",
        minimizer=jd.fire,
        minimizer_kw={"dt": 1.0e-4},
    )
    system = replace(system, linear_integrator=ExplodingInitializer())
    final_state, _, steps, energy = system.minimize(state, system, max_steps=0)
    assert int(steps) == 0
    assert bool(jnp.all(jnp.isfinite(final_state.pos)))
    assert bool(jnp.all(final_state.vel == state.vel))
    assert bool(jnp.isfinite(energy))


def test_custom_force_controls_default_minimizer_capability():
    state = jd.State.create(pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]), rad=jnp.ones(2))
    base = jd.System.create(
        state=state,
        collider_type="naive",
        minimizer=jd.fire,
        minimizer_kw={"dt": 1.0e-4},
    )

    opted_out = replace(base, force_model=CustomOptOutSpring())
    with pytest.raises(ValueError, match="analytical energy gradient"):
        opted_out.minimize(state, opted_out, max_steps=0)

    default_capability = replace(base, force_model=CustomDefaultSpring())
    assert default_capability.force_model.supports_analytical_energy_gradient
    _, _, steps, energy = default_capability.minimize(
        state, default_capability, max_steps=0
    )
    assert int(steps) == 0
    assert bool(jnp.isfinite(energy))


def test_custom_force_target_fallback_does_not_require_capability():
    state = jd.State.create(pos=jnp.array([[0.5, 0.0]]), rad=jnp.ones(1))
    system = jd.System.create(
        state=state,
        collider_type="naive",
        minimizer=jd.fire,
        minimizer_kw={"dt": 1.0e-4},
        target_fn=lambda trial, _: jnp.sum(trial.pos**2),
    )
    system = replace(system, force_model=CustomOptOutSpring())
    _, _, steps, energy = system.minimize(state, system, max_steps=0)
    assert int(steps) == 0
    assert bool(jnp.isfinite(energy))


def test_composites_aggregate_analytical_gradient_capability():
    supported = jd.forces.LawCombiner(
        laws=(CustomDefaultSpring(), jd.forces.HertzianForce())
    )
    unsupported = jd.forces.LawCombiner(
        laws=(CustomDefaultSpring(), CustomOptOutSpring())
    )
    assert supported.supports_analytical_energy_gradient
    assert not unsupported.supports_analytical_energy_gradient

    router = jd.forces.ForceRouter.from_dict(2, {(0, 1): unsupported})
    assert not router.supports_analytical_energy_gradient


def constant_push(pos, state, system):
    return jnp.ones_like(pos), jnp.zeros_like(state.torque)


def test_default_minimizer_rejects_force_without_matching_energy():
    state = jd.State.create(pos=jnp.array([[0.0, 0.0]]), rad=jnp.array([0.5]))
    system = jd.System.create(
        state=state,
        collider_type="naive",
        minimizer=jd.fire,
        minimizer_kw={"dt": 1.0e-3},
        force_manager_kw={"force_functions": (constant_push,)},
    )
    with pytest.raises(ValueError, match="matching energy function"):
        system.minimize(state, system, max_steps=1)


@pytest.mark.parametrize("collider_type", ["CellList", "MultiCellList"])
def test_free_domain_minimizer_prepares_trial_search_bounds(collider_type):
    state = jd.State.create(
        pos=jnp.array([[100.0, 0.0], [101.5, 0.0]]), rad=jnp.ones(2)
    )
    system = jd.System.create(
        state=state,
        collider_type=collider_type,
        minimizer=jd.fire,
        minimizer_kw={"dt": 1.0e-4},
    )
    final_state, _, steps, _ = system.minimize(state, system, max_steps=1)
    assert int(steps) == 1
    assert bool(jnp.all(jnp.isfinite(final_state.pos)))


def test_fixed_particles_do_not_block_force_tolerance_convergence():
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [1.5, 0.0]]),
        rad=jnp.ones(2),
        fixed=jnp.ones(2, dtype=bool),
    )
    system = jd.System.create(
        state=state,
        collider_type="naive",
        minimizer=jd.fire,
        minimizer_kw={"dt": 1.0e-4},
    )
    final_state, _, steps, _ = system.minimize(
        state, system, max_steps=5, pe_tol=-1.0, pe_diff_tol=-1.0, force_tol=0.0
    )
    assert int(steps) == 0
    assert bool(jnp.all(final_state.pos == state.pos))


def test_default_energy_gradient_excludes_damping_and_buffered_loads():
    import jax
    import numpy as np
    from jaxdem.minimizers.routines import _objective_energy, _state_to_delta_params

    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.8, 0.0]]),
        rad=jnp.full(2, 0.5),
        vel=jnp.array([[2.0, 1.0], [0.0, 0.0]]),
    )
    table = jd.MaterialTable.from_materials(
        [
            jd.Material.create(
                "elasticfrict",
                density=1.0,
                young=100.0,
                poisson=0.3,
                e=0.5,
                mu=0.5,
                mu_r=0.0,
            )
        ]
    )
    system = jd.System.create(
        state=state, force_model_type="cundallstrack", mat_table=table
    )
    system = system.force_manager.add_force(
        state, system, jnp.ones_like(state.pos) * 100.0
    )
    params = _state_to_delta_params(state)
    (value, (_, returned)), gradient = jax.value_and_grad(
        _objective_energy, has_aux=True
    )(params, state, system)
    np.testing.assert_allclose(value, 1.0, rtol=1e-5)
    np.testing.assert_allclose(
        gradient["pos_c"], [[10.0, 0.0], [-10.0, 0.0]], rtol=1e-5, atol=1e-5
    )
    np.testing.assert_array_equal(
        returned.force_manager.external_force, system.force_manager.external_force
    )
