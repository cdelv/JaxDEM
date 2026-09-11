# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
from dataclasses import replace
from dataclasses import dataclass
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import optax

import jaxdem as jd
from jaxdem.minimizers import conjugate_gradient
from jaxdem.minimizers.optimizers import CustomGradientTransformation
from jaxdem.minimizers.routines import TerminationReason


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


def _named_sgd_factory(rate):
    def same_name():
        return optax.sgd(rate)

    return same_name


def test_custom_optimizer_static_identity_includes_factory_closure():
    first_factory = _named_sgd_factory(0.1)
    second_factory = _named_sgd_factory(0.2)
    first_opt = first_factory()
    second_opt = second_factory()
    first = CustomGradientTransformation(
        first_opt.init, first_opt.update, first_factory, {}, type_name="same_name"
    )
    second = CustomGradientTransformation(
        second_opt.init, second_opt.update, second_factory, {}, type_name="same_name"
    )

    assert first != second
    assert hash(first) != hash(second)
    with pytest.raises(TypeError):
        first.kw["learning_rate"] = 1.0
    nested_opt = first_factory()
    nested = CustomGradientTransformation(
        nested_opt.init,
        nested_opt.update,
        jd.fire,
        {"options": {"rates": [0.1, 0.2]}},
    )
    with pytest.raises(TypeError):
        nested.kw["options"]["rates"] = ()
    exported = nested.metadata
    json.dumps(exported)
    exported["kw"]["options"]["rates"] = (9.0,)
    assert nested.kw["options"]["rates"] == (0.1, 0.2)
    for name, value in (
        ("kw", {}),
        ("_constructor", second_factory),
        ("_kw_key", ()),
        ("type_name", "changed"),
    ):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(first, name, value)

    @jax.jit(static_argnames="optimizer")
    def update(gradient, *, optimizer):
        params = jnp.array(0.0)
        updates, _ = optimizer.update(gradient, optimizer.init(params), params=params)
        return updates

    assert update(jnp.array(1.0), optimizer=first) == pytest.approx(-0.1)
    assert update(jnp.array(1.0), optimizer=second) == pytest.approx(-0.2)


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


def test_minimizer_reports_termination_reason_without_breaking_unpacking():
    state = jd.State.create(pos=jnp.array([[0.0, 0.0]]), rad=jnp.array([0.5]))
    system = jd.System.create(state=state, collider_type="naive")

    result = system.minimize(
        state,
        system,
        max_steps=3,
        pe_tol=-1.0,
        pe_diff_tol=-1.0,
        force_tol=0.0,
    )
    _, _, steps, _ = result
    assert len(result) == 4
    assert result[2] is steps
    assert int(steps) == 0
    assert TerminationReason(int(result.reason)) is TerminationReason.FORCE_TOLERANCE

    exhausted = system.minimize(
        state,
        system,
        max_steps=0,
        pe_tol=-1.0,
        pe_diff_tol=-1.0,
        force_tol=-1.0,
    )
    assert TerminationReason(int(exhausted.reason)) is TerminationReason.MAX_STEPS


def test_minimizer_stops_immediately_on_nonfinite_objective():
    state = jd.State.create(pos=jnp.zeros((1, 2)))
    system = jd.System.create(
        state=state, target_fn=lambda trial, _: jnp.sum(trial.pos) * jnp.nan
    )
    result = system.minimize(
        state,
        system,
        max_steps=3,
        pe_tol=-1.0,
        pe_diff_tol=-1.0,
        force_tol=-1.0,
    )
    assert int(result.steps) == 0
    assert TerminationReason(int(result.reason)) is TerminationReason.NONFINITE


def test_minimizer_reports_existing_search_overflow_without_stepping():
    state = jd.State.create(pos=jnp.zeros((1, 2)))
    system = jd.System.create(state=state)
    system = replace(system, search_overflow=jnp.asarray(True))
    result = system.minimize(
        state,
        system,
        max_steps=3,
        pe_tol=-1.0,
        pe_diff_tol=-1.0,
        force_tol=-1.0,
    )
    assert int(result.steps) == 0
    assert TerminationReason(int(result.reason)) is TerminationReason.SEARCH_OVERFLOW


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
        state=state,
        force_model_type="cundallstrack",
        mat_table=table,
        collider_type="NeighborList",
        collider_kw={"state": state, "cutoff": 1.0, "max_neighbors": 1},
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


def test_logical_body_topology_uses_representative_total_properties():
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.0, 0.0], [3.0, 0.0]]),
        clump_id=jnp.array([0, 0, 1]),
        mass=jnp.array([4.0, 4.0, 2.0]),
        volume=jnp.array([5.0, 5.0, 1.0]),
        inertia=jnp.array([[6.0], [6.0], [2.0]]),
    )
    records = state.body_mass_properties(validate=True)
    np.testing.assert_array_equal(records["valid"], [True, True, False])
    np.testing.assert_array_equal(records["representative"], [0, 2, 0])
    np.testing.assert_array_equal(records["member_count"], [2, 1, 0])
    np.testing.assert_allclose(records["mass"], [4.0, 2.0, 0.0])
    np.testing.assert_allclose(records["volume"], [5.0, 1.0, 0.0])
    np.testing.assert_allclose(records["inertia"], [[6.0], [2.0], [0.0]])


def test_logical_body_topology_supports_heterogeneous_batches():
    state = jd.State.create(
        pos=jnp.zeros((2, 3, 2)),
        clump_id=jnp.array([[0, 0, 1], [0, 1, 2]]),
        fixed=jnp.array([[True, True, False], [False, True, False]]),
    )
    topology = state.body_topology()
    np.testing.assert_array_equal(
        topology.valid, [[True, True, False], [True, True, True]]
    )
    np.testing.assert_array_equal(topology.member_count, [[2, 1, 0], [1, 1, 1]])
    np.testing.assert_array_equal(
        topology.fixed, [[True, False, False], [False, True, False]]
    )


def test_logical_body_topology_accepts_empty_states():
    state = jd.State.create(pos=jnp.zeros((0, 2)))
    topology = state.body_topology()
    assert topology.valid.shape == (0,)
    assert topology.representative.shape == (0,)
    assert state.body_mass_properties()["mass"].shape == (0,)


def test_minimizer_parameters_have_one_effective_coordinate_per_clump():
    from jaxdem.minimizers.routines import (
        _delta_params_to_state,
        _state_to_delta_params,
    )

    state = jd.State.create(
        pos=jnp.array([[0.0, -0.5], [0.0, 0.5], [3.0, 0.0]]),
        clump_id=jnp.array([0, 0, 1]),
    )
    topology = state.body_topology()
    params = _state_to_delta_params(state, topology)
    assert params["pos_c"].shape == state.pos_c.shape
    np.testing.assert_array_equal(params["pos_c"][2], [0.0, 0.0])

    moved = dict(params)
    moved["pos_c"] = params["pos_c"].at[0].set(jnp.array([2.0, 4.0]))
    moved["pos_c"] = moved["pos_c"].at[1].set(jnp.array([-1.0, 7.0]))
    trial = _delta_params_to_state(state, moved)
    np.testing.assert_allclose(trial.pos_c[:2], [[2.0, 4.0], [2.0, 4.0]])
    np.testing.assert_allclose(trial.pos_c[2], [-1.0, 7.0])


def test_target_gradient_reduces_member_motion_to_one_body_coordinate():
    from jaxdem.minimizers.routines import (
        _delta_params_to_state,
        _state_to_delta_params,
    )

    state = jd.State.create(
        pos=jnp.array([[0.0, -0.5], [0.0, 0.5], [3.0, 0.0]]),
        clump_id=jnp.array([0, 0, 1]),
    )
    params = _state_to_delta_params(state, state.body_topology())

    def objective(p):
        trial = _delta_params_to_state(state, p)
        return jnp.sum(trial.pos_c[..., 0])

    grad = jax.grad(objective)(params)["pos_c"]
    np.testing.assert_array_equal(grad[:, 0], [2.0, 1.0, 0.0])


def test_species_capacity_is_a_recursive_force_model_capability():
    unrestricted = CustomDefaultSpring()
    inner = jd.ForceRouter.from_dict(2, {(0, 0): unrestricted})
    outer = jd.ForceRouter.from_dict(3, {(0, 0): jd.LawCombiner(laws=(inner,))})

    assert unrestricted.species_capacity is None
    assert inner.species_capacity == 2
    assert outer.species_capacity == 2
    assert jd.LawCombiner(laws=(unrestricted, outer)).species_capacity == 2
