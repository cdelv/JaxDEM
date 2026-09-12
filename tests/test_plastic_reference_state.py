# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Regression tests for plastic reference-state updates."""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.bonded_forces.deformable_particle import DeformableParticleModel
from jaxdem.utils.particle_creation import create_dp_container


def _plastic_system(kind: str) -> tuple[jd.State, jd.System, str]:
    vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    state = jd.State.create(
        pos=vertices,
        bond_id=jnp.broadcast_to(jnp.arange(4), (4, 4)),
        fixed=jnp.ones(4, dtype=bool),
    )
    model = create_dp_container(
        state,
        el=1.0,
        eb=1.0,
        plasticity_type=kind,
        tau_s=1.0,
    )
    state = dataclasses.replace(
        state, pos_c=state.pos_c.at[2].set(jnp.array([2.0, 1.5]))
    )
    system = jd.System.create(
        state=state,
        dt=0.1,
        bonded_force_model=model,
        collider_type="",
        rotation_integrator_type="",
    )
    field = "initial_bendings" if kind == "bending" else "initial_edge_lengths"
    return state, system, field


@pytest.mark.parametrize("kind", ["edge", "perimeter", "bending"])
def test_step_persists_each_plastic_reference_update(kind: str) -> None:
    state, system, field = _plastic_system(kind)
    model = system.bonded_force_model
    assert model is not None
    expected_model = model.update_reference_state(state.pos, state, system)

    _, stepped_system = jd.System.step(state, system)

    np.testing.assert_allclose(
        getattr(stepped_system.bonded_force_model, field),
        getattr(expected_model, field),
    )
    assert not np.allclose(getattr(expected_model, field), getattr(model, field))


@pytest.mark.parametrize("kind", ["edge", "perimeter", "bending"])
def test_force_and_energy_evaluations_do_not_advance_plasticity(kind: str) -> None:
    state, system, field = _plastic_system(kind)
    reference = np.asarray(getattr(system.bonded_force_model, field)).copy()

    for _ in range(3):
        DeformableParticleModel.compute_forces(state.pos, state, system)
        system.force_manager.compute_potential_energy(state, system)
        _, returned_system = system.force_manager.apply(state, system)
        np.testing.assert_array_equal(
            getattr(returned_system.bonded_force_model, field), reference
        )

    np.testing.assert_array_equal(getattr(system.bonded_force_model, field), reference)


def test_edge_reference_updates_once_per_step() -> None:
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [2.0, 0.0]]),
        fixed=jnp.ones(2, dtype=bool),
    )
    model = jd.BondedForceModel.create(
        "PlasticDeformableParticleModel",
        edges=jnp.array([[0, 1]]),
        initial_edge_lengths=jnp.array([1.0]),
        el=1.0,
        tau_s=1.0,
    )
    system = jd.System.create(
        state=state,
        dt=0.1,
        bonded_force_model=model,
        collider_type="",
        rotation_integrator_type="",
    )

    force, _ = DeformableParticleModel.compute_forces(state.pos, state, system)
    np.testing.assert_allclose(force, [[1.0, 0.0], [-1.0, 0.0]])
    np.testing.assert_allclose(model.initial_edge_lengths, [1.0])

    for expected in (1.1, 1.19, 1.271):
        state, system = jd.System.step(state, system)
        np.testing.assert_allclose(
            system.bonded_force_model.initial_edge_lengths, [expected]
        )


def test_batched_steps_update_each_reference_independently() -> None:
    state, system, _ = _plastic_system("edge")
    stretched_state = dataclasses.replace(
        state, pos_c=state.pos_c.at[2].set(jnp.array([3.0, 2.0]))
    )
    batched_state = jd.State.stack([state, stretched_state])
    batched_system = jd.System.stack([system, system])

    _, stepped_system = jd.System.step(batched_state, batched_system)

    expected = [
        system.bonded_force_model.update_reference_state(item.pos, item, system)
        for item in (state, stretched_state)
    ]
    np.testing.assert_allclose(
        stepped_system.bonded_force_model.initial_edge_lengths,
        jnp.stack([item.initial_edge_lengths for item in expected]),
    )


@pytest.mark.parametrize("kind", ["edge", "perimeter", "bending"])
def test_split_stepping_preserves_plastic_history(kind: str) -> None:
    state, system, field = _plastic_system(kind)

    full_state, full_system = jd.System.step(state, system, n=3)
    split_state, split_system = jd.System.step(state, system)
    split_state, split_system = jd.System.step(split_state, split_system, n=2)

    np.testing.assert_allclose(full_state.pos, split_state.pos)
    np.testing.assert_allclose(
        getattr(full_system.bonded_force_model, field),
        getattr(split_system.bonded_force_model, field),
    )


@pytest.mark.parametrize("kind", ["edge", "perimeter", "bending"])
def test_checkpoint_restart_preserves_plastic_history(tmp_path, kind: str) -> None:
    state, system, field = _plastic_system(kind)
    state, system = jd.System.step(state, system)

    checkpoint = tmp_path / kind
    with jd.CheckpointWriter(checkpoint) as writer:
        writer.save(state, system)
    restored_state, restored_system = jd.CheckpointLoader(checkpoint).load()

    continuous_state, continuous_system = jd.System.step(state, system, n=2)
    restored_state, restored_system = jd.System.step(
        restored_state, restored_system, n=2
    )
    np.testing.assert_allclose(continuous_state.pos, restored_state.pos)
    np.testing.assert_allclose(
        getattr(continuous_system.bonded_force_model, field),
        getattr(restored_system.bonded_force_model, field),
    )
