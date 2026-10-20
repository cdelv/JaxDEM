# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Every built-in component retains its full configuration in both storage formats."""

from dataclasses import replace
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.domains import Domain
from jaxdem.integrators import LinearIntegrator, RotationIntegrator
from jaxdem.material_matchmakers import MaterialMatchmaker
from jaxdem.utils import h5
from tests.test_checkpoints import _assert_leaves_equal

ARCHITECTURES = [
    "ActorCritic",
    "SharedActorCritic",
    "LSTMActorCritic",
    "MinGRUActorCritic",
]
ACTION_SPACES = ["Free", "Box", "MaxNorm", "Blockwise", "discrete"]


def test_model_serialization_matrix_covers_builtin_models_and_action_spaces():
    from jaxdem.rl.models import Model
    from jaxdem.rl.action_spaces import ActionSpace

    for factory, names in [(Model, ARCHITECTURES), (ActionSpace, ACTION_SPACES[:-1])]:
        expected = {
            cls
            for cls in factory._registry.values()
            if cls.__module__.startswith("jaxdem.")
        }
        assert {factory._registry[name.lower()] for name in names} == expected


# One component changes at a time. Aliases share a serializer; distinct classes do not.
COMPONENTS = {
    "force_model": (
        jd.ForceModel,
        [
            "spring",
            "hertz",
            "wca",
            "wcashifted",
            "lennardjones",
            "cundallstrack",
            "lawcombiner",
            "forcerouter",
            "spherefacetspring",
            "facetfacetspring",
        ],
    ),
    "domain": (Domain, ["free", "periodic", "leesedwards", "reflect", "reflectsphere"]),
    "collider": (
        jd.Collider,
        ["", "naive", "celllist", "multicelllist", "neighborlist"],
    ),
    "linear_integrator": (
        LinearIntegrator,
        ["", "euler", "verlet", "langevin", "verletrescaling"],
    ),
    "rotation_integrator": (RotationIntegrator, ["", "spiral", "verletspiral"]),
    "bonded_force_model": (
        jd.BondedForceModel,
        [
            "deformableparticlemodel",
            "plasticdeformableparticlemodel",
            "plasticperimeterdeformableparticlemodel",
            "plasticbendingdeformableparticlemodel",
        ],
    ),
    "material_matchmaker": (MaterialMatchmaker, ["harmonic", "linear"]),
}
CASES = [(family, name) for family, (_, names) in COMPONENTS.items() for name in names]


def configured_system(family, name):
    if family == "bonded_force_model":
        from tests.test_plastic_reference_state import _plastic_system

        kinds = {
            "plasticdeformableparticlemodel": "edge",
            "plasticperimeterdeformableparticlemodel": "perimeter",
            "plasticbendingdeformableparticlemodel": "bending",
        }
        state, system, _ = _plastic_system(kinds.get(name, "edge"))
        if name == "deformableparticlemodel":
            from jaxdem.utils.particle_creation import create_dp_container

            system = replace(
                system, bonded_force_model=create_dp_container(state, el=1.0, eb=1.0)
            )
        return jd.System.initialize(state, system)
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0, 0.0], [0.8, 0.1, 0.0]]),
        rad=jnp.array([0.4, 0.6]),
        mass=jnp.array([2.0, 3.0]),
        vel=jnp.array([[0.1, 0.2, 0.0], [-0.2, 0.1, 0.3]]),
        ang_vel=jnp.array([[0.1, 0.2, 0.3], [-0.2, 0.3, 0.1]]),
    )
    law = jd.forces.SpringForce()
    material = jd.Material.create("elastic", young=2.0, density=1.0, poisson=0.25)
    options = dict(
        collider_type="NeighborList",
        collider_kw={
            "cutoff": 1.0,
            "skin": 0.1,
            "max_neighbors": 8,
            "secondary_collider_type": "CellList",
        },
    )
    if family == "force_model":
        if name in {"wca", "wcashifted", "lennardjones"}:
            material = jd.Material.create("lj", epsilon=0.7, density=1.0)
        if name == "cundallstrack":
            material = jd.Material.create(
                "elasticfrict", young=2.0, density=1.0, poisson=0.25, mu=0.6, e=0.8
            )
        if name == "lawcombiner":
            law = jd.LawCombiner(
                laws=(jd.forces.SpringForce(), jd.forces.HertzianForce())
            )
        elif name == "forcerouter":
            law = jd.ForceRouter.from_dict(
                2,
                {
                    (0, 0): jd.LawCombiner(
                        laws=(jd.forces.SpringForce(), jd.forces.HertzianForce())
                    )
                },
            )
        else:
            law = jd.ForceModel.create(name)
        if name in {"spherefacetspring", "facetfacetspring"}:
            vertices = jnp.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
            state = jd.State.add_facet(
                jd.State.create(dim=3), vertices, thickness=0.1, mass=2.0
            )
            if name == "spherefacetspring":
                state = jd.State.merge(
                    state,
                    jd.State.create(
                        pos=jnp.array([[0.5, 0.5, 0.2]]), rad=jnp.array([0.2])
                    ),
                )
            else:
                state = jd.State.add_facet(
                    state,
                    vertices + jnp.array([0.1, 0.1, 0.15]),
                    thickness=0.1,
                    mass=3.0,
                    rigid=False,
                )
    elif family == "domain":
        options.update(domain_type=name)
        if name != "free":
            options["domain_kw"] = {
                "box_size": jnp.array([4.0, 5.0, 6.0]),
                "anchor": jnp.array([-2.0, -2.5, -3.0]),
            }
        if name == "leesedwards":
            options["domain_kw"].update(
                alpha=2,
                beta=0,
                gamma=0.3,
                gamma_dot=0.1,
                box_rate=jnp.array([0.01, 0.0, 0.02]),
            )
    elif family == "collider":
        options["collider_type"] = name
        if name != "neighborlist":
            options["collider_kw"] = {}
    elif family != "material_matchmaker":
        options[family + "_type"] = name
        if name == "langevin":
            options[family + "_kw"] = dict(
                gamma=jnp.array(0.2), k_B=jnp.array(1.0), temperature=jnp.array(0.7)
            )
        if name == "verletrescaling":
            options[family + "_kw"] = dict(
                temperature=0.7, rescale_every=3, can_rotate=True
            )
    materials = [material]
    matcher = None
    if family == "material_matchmaker":
        materials = [jd.Material.create(name, **kw) for name, kw in MATERIALS.items()]
        matcher = MaterialMatchmaker.create(name)
    system = jd.System.create(
        state=state,
        force_model=law,
        dt=0.001,
        seed=13,
        mat_table=jd.MaterialTable.from_materials(materials, matcher=matcher),
        **options,
    )
    return jd.System.initialize(state, system)


MATERIALS = {
    "elastic": dict(young=2.0, poisson=0.25, density=1.0),
    "elasticfrict": dict(young=3.0, poisson=0.3, density=2.0, mu=0.6, e=0.8, mu_r=0.1),
    "cundallstrackparams": dict(
        k_n=2.0, k_t=1.0, b_n=0.1, b_t=0.2, mu=0.4, mu_r=0.2, density=3.0
    ),
    "lj": dict(epsilon=0.7, density=4.0),
}


def test_serialized_material_table_includes_every_builtin_material_class():
    expected = {
        cls
        for cls in jd.Material._registry.values()
        if cls.__module__.startswith("jaxdem.")
    }
    assert {jd.Material._registry[name] for name in MATERIALS} == expected


@pytest.mark.parametrize("family", COMPONENTS)
def test_serialization_table_covers_all_builtin_classes(family):
    factory, names = COMPONENTS[family]
    expected = {
        cls
        for cls in factory._registry.values()
        if cls.__module__.startswith("jaxdem.")
    }
    assert {factory._registry[name] for name in names} == expected


@pytest.mark.parametrize("storage", ["hdf5", "orbax"])
@pytest.mark.parametrize(
    "family,name", CASES, ids=[f"{family}-{name or 'noop'}" for family, name in CASES]
)
def test_component_configuration_and_continuation(tmp_path, storage, family, name):
    state, system = configured_system(family, name)
    assert_round_trip_and_continue(tmp_path, storage, state, system)


def assert_round_trip_and_continue(tmp_path, storage, state, system):
    state, system = jd.System.step(state, system)
    if storage == "hdf5":
        path = str(tmp_path / "state.h5")
        h5.save((state, system), path)
        restored = h5.load(path)
    else:
        with jd.CheckpointWriter(tmp_path / "checkpoint") as writer:
            writer.save(state, system)
        with jd.CheckpointLoader(tmp_path / "checkpoint") as loader:
            restored = loader.load()
    _assert_leaves_equal(state, restored[0], "stored state")
    _assert_leaves_equal(system, restored[1], "stored system")
    expected = jd.System.step(state, system, n=3)
    actual = jd.System.step(*restored, n=3)
    _assert_leaves_equal(
        expected[0], actual[0], "continued state", rtol=2e-6, atol=1e-7
    )
    _assert_leaves_equal(
        expected[1], actual[1], "continued system", rtol=2e-6, atol=1e-7
    )
    return restored


@pytest.mark.parametrize("storage", ["hdf5", "orbax"])
@pytest.mark.parametrize("optimizer", ["fire", "damped_newtonian", "adam"])
def test_minimizer_and_importable_callbacks_keep_configuration(
    tmp_path, storage, optimizer
):
    import optax
    from tests.custom_forces import advance_shear_before_step, mark_after_step
    from tests.test_checkpoints import dummy_target_fn

    state, system = configured_system("domain", "leesedwards")
    factory = optax.adam if optimizer == "adam" else getattr(jd, optimizer)
    kw = {"learning_rate": 0.03} if optimizer == "adam" else {"dt": 0.02}
    system = jd.System.create(
        state=state,
        dt=0.001,
        domain=system.domain,
        minimizer=factory,
        minimizer_kw=kw,
        target_fn=dummy_target_fn,
        user_pre_step_actions=advance_shear_before_step,
        user_post_step_actions=mark_after_step,
    )
    restored = assert_round_trip_and_continue(tmp_path, storage, state, system)
    # Exercise executable reconstruction, beyond comparing the saved metadata.
    expected = factory(**kw)
    actual = restored[1].minimizer
    params = jnp.array([[0.7, -0.2], [0.4, 0.3]])
    expected_state, actual_state = expected.init(params), actual.init(params)
    for step in range(2):
        gradient = 0.4 * params + (step + 1) * jnp.array([[0.1, -0.3], [0.2, 0.4]])
        reference, expected_state = expected.update(gradient, expected_state, params)
        update, actual_state = actual.update(gradient, actual_state, params)
        _assert_leaves_equal(reference, update, "restored optimizer update")
        _assert_leaves_equal(expected_state, actual_state, "restored optimizer state")
        params = optax.apply_updates(params, reference)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
@pytest.mark.parametrize("space", ACTION_SPACES)
def test_model_parameters_action_space_and_nonzero_carry_continue(
    tmp_path, architecture, space
):
    from flax import nnx
    from tests.test_caps import make_model

    if architecture == "SharedActorCritic" and space == "Box":
        from jaxdem.rl.models import SharedActorCritic
        from jaxdem.rl.action_spaces import BoxSpace

        model = SharedActorCritic(
            observation_space_size=2,
            action_space_size=2,
            key=nnx.Rngs(10),
            architecture=[3, 5],
            activation=nnx.tanh,
            actor_sigma_head=True,
            action_space=BoxSpace(x_min=-2.0, x_max=3.0, width=0.7, eps=0.01),
        )
    else:
        model = make_model(architecture, space)
    observation = jnp.array([[0.3, -0.7], [0.6, 0.2]])
    model.reset(observation.shape)
    model(observation)
    if model.carry is not None:
        assert any(np.any(np.asarray(x) != 0) for x in jax.tree.leaves(model.carry))
    with jd.CheckpointModelWriter(tmp_path) as writer:
        writer.save(model, step=7)
    with jd.CheckpointModelLoader(tmp_path) as loader:
        restored = loader.load(step=7)
    assert type(restored) is type(model)
    assert type(restored.action_space) is type(model.action_space)
    assert json.dumps(restored.metadata, sort_keys=True) == json.dumps(
        model.metadata, sort_keys=True
    )
    expected_state, actual_state = nnx.state(model), nnx.state(restored)
    assert jax.tree.structure(expected_state) == jax.tree.structure(actual_state)
    for expected, actual in zip(
        jax.tree.leaves(expected_state), jax.tree.leaves(actual_state), strict=True
    ):
        assert expected.shape == actual.shape and expected.dtype == actual.dtype
        np.testing.assert_array_equal(actual, expected)
    expected_pi, expected_value = model(observation + 0.1)
    actual_pi, actual_value = restored(observation + 0.1)
    key = jax.random.key(52)
    np.testing.assert_allclose(
        actual_pi.sample(seed=key), expected_pi.sample(seed=key), rtol=2e-6, atol=1e-7
    )
    if space == "discrete":
        np.testing.assert_allclose(
            actual_pi.probs, expected_pi.probs, rtol=2e-6, atol=1e-7
        )
    else:
        latent = (
            expected_pi.distribution
            if hasattr(expected_pi, "distribution")
            else expected_pi
        )
        actual_latent = (
            actual_pi.distribution if hasattr(actual_pi, "distribution") else actual_pi
        )
        np.testing.assert_allclose(actual_latent.loc, latent.loc, rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(
            actual_latent.scale_diag, latent.scale_diag, rtol=2e-6, atol=1e-7
        )
    probe = expected_pi.sample(seed=jax.random.key(19))
    np.testing.assert_allclose(
        actual_pi.log_prob(probe), expected_pi.log_prob(probe), rtol=2e-6, atol=1e-7
    )
    np.testing.assert_allclose(actual_value, expected_value, rtol=2e-6, atol=1e-7)
