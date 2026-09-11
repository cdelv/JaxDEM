# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end contracts separating force observation from time evolution."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

import jaxdem as jd
from jaxdem.bonded_forces.deformable_particle import DeformableParticleModel
from jaxdem.utils.particle_creation import create_dp_container
from jaxdem.utils.packing_utils import scale_to_packing_fraction
from jaxdem.utils.thermal import compute_potential_energy, compute_temperature


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _RememberingSpring(jd.forces.SpringForce):
    def history_shape(self, dim):
        return (1,)

    @staticmethod
    def force(i, j, pos, state, system, history, *, advance_history=True):
        force, torque, _ = jd.forces.SpringForce.force(
            i, j, pos, state, system, history, advance_history=advance_history
        )
        return force, torque, history + advance_history


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _HistoryScaledSpring(_RememberingSpring):
    @staticmethod
    def force(i, j, pos, state, system, history, *, advance_history=True):
        force, torque, _ = jd.forces.SpringForce.force(
            i, j, pos, state, system, history, advance_history=advance_history
        )
        scale = history[..., 0, None]
        return force * scale, torque * scale, history + advance_history


def _history_system() -> tuple[jd.State, jd.System]:
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.3, 0.0]]),
        rad=jnp.full(2, 0.5),
        fixed=jnp.ones(2, dtype=bool),
    )
    system = jd.System.create(
        state=state,
        force_model=_RememberingSpring(),
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 2,
            "cutoff": 1.0,
            "skin": 0.1,
            "secondary_collider_type": "naive",
        },
    )
    return jd.System.step(state, system)


def test_evaluation_preserves_history_and_queued_load_until_step() -> None:
    state, system = _history_system()
    queued = jnp.array([[3.0, 4.0], [-2.0, 1.0]])
    system = system.force_manager.add_force(state, system, queued)
    history = np.asarray(system.collider.history).copy()

    evaluated, cached = jd.System.evaluate_forces(state, system)
    evaluated_again, cached_again = jd.System.evaluate_forces(state, cached)

    np.testing.assert_allclose(evaluated.force - state.force, queued)
    np.testing.assert_allclose(evaluated_again.force, evaluated.force)
    np.testing.assert_array_equal(cached.collider.history, history)
    np.testing.assert_array_equal(cached_again.collider.history, history)
    np.testing.assert_array_equal(cached.force_manager.external_force, queued)
    np.testing.assert_array_equal(system.force_manager.external_force, queued)
    assert int(cached.step_count) == int(system.step_count)
    assert float(cached.time) == float(system.time)

    _, stepped = jd.System.step(state, cached)
    valid = np.asarray(stepped.collider.neighbor_list) >= 0
    np.testing.assert_array_equal(
        np.asarray(stepped.collider.history)[valid], history[valid] + 1
    )
    np.testing.assert_array_equal(stepped.force_manager.external_force, 0.0)


def test_evaluation_remaps_history_after_modest_box_rescale() -> None:
    state, system = _history_system()
    old_neighbors = np.asarray(system.collider.neighbor_list)
    old_history = np.asarray(system.collider.history)
    remembered = {
        (i, int(j)): old_history[i, slot].copy()
        for i, row in enumerate(old_neighbors)
        for slot, j in enumerate(row)
        if j >= 0
    }
    packing_fraction = jd.utils.compute_packing_fraction(state, system)

    scaled_state, invalidated = scale_to_packing_fraction(
        state, system, packing_fraction * 1.05
    )
    _, evaluated = jd.System.evaluate_forces(scaled_state, invalidated)

    for i, row in enumerate(np.asarray(evaluated.collider.neighbor_list)):
        for slot, j in enumerate(row):
            if (i, int(j)) in remembered:
                np.testing.assert_array_equal(
                    np.asarray(evaluated.collider.history)[i, slot],
                    remembered[i, int(j)],
                )


def test_evaluation_updates_free_bounds_after_particle_teleport() -> None:
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [10.0, 0.0]]), rad=jnp.full(2, 0.5)
    )
    system = jd.System.create(
        state=state,
        force_model=_RememberingSpring(),
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 2,
            "cutoff": 1.0,
            "skin": 0.1,
            "secondary_collider_type": "naive",
        },
    )
    moved = dataclasses.replace(state, pos_c=state.pos_c.at[1, 0].set(0.3))
    history = np.asarray(system.collider.history).copy()

    evaluated, updated = jd.System.evaluate_forces(moved, system)
    naive_system = dataclasses.replace(system, collider=jd.Collider.create("naive"))
    expected, _ = jd.System.evaluate_forces(moved, naive_system)

    np.testing.assert_allclose(evaluated.force, expected.force)
    assert np.any(np.asarray(evaluated.force) != 0.0)
    np.testing.assert_array_equal(system.collider.history, history)
    np.testing.assert_array_equal(updated.collider.history, history)
    assert np.all(np.asarray(moved.pos) >= np.asarray(updated.domain.anchor))
    assert np.all(
        np.asarray(moved.pos)
        <= np.asarray(updated.domain.anchor + updated.domain.box_size)
    )


def test_pair_force_analysis_reads_current_history_without_advancing_it() -> None:
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.3, 0.0]]),
        rad=jnp.full(2, 0.5),
        fixed=jnp.ones(2, dtype=bool),
    )
    system = jd.System.create(
        state=state,
        force_model=_HistoryScaledSpring(),
        collider_type="NeighborList",
        collider_kw={"max_neighbors": 2, "cutoff": 1.0, "skin": 0.1},
    )
    state, system = jd.System.step(state, system)
    before = np.asarray(system.collider.history).copy()

    _, returned, pair_ids, forces = jd.utils.get_pair_forces_and_ids(
        state, system, cutoff=1.0, max_neighbors=2
    )
    valid = np.asarray(pair_ids[:, 1]) >= 0

    assert np.all(np.linalg.norm(np.asarray(forces)[valid], axis=-1) > 0.0)
    np.testing.assert_array_equal(system.collider.history, before)
    np.testing.assert_array_equal(returned.collider.history, before)


def _plastic_system() -> tuple[jd.State, jd.System]:
    vertices = jnp.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    state = jd.State.create(
        pos=vertices,
        bond_id=jnp.broadcast_to(jnp.arange(4), (4, 4)),
        fixed=jnp.ones(4, dtype=bool),
    )
    model = create_dp_container(
        state, el=1.0, eb=1.0, plasticity_type="edge", tau_s=1.0
    )
    state = dataclasses.replace(
        state, pos_c=state.pos_c.at[2].set(jnp.array([2.0, 1.5]))
    )
    return state, jd.System.create(
        state=state,
        dt=0.1,
        bonded_force_model=model,
        collider_type="",
        rotation_integrator_type="",
    )


def test_force_energy_and_gradient_evaluation_preserve_plastic_reference() -> None:
    state, system = _plastic_system()
    reference = np.asarray(system.bonded_force_model.initial_edge_lengths).copy()

    _, evaluated_system = jd.System.evaluate_forces(state, system)
    compute_potential_energy(state, system)

    def energy(pos_c):
        moved = dataclasses.replace(state, pos_c=pos_c)
        return DeformableParticleModel.compute_potential_energy(
            moved.pos, moved, system
        )

    gradient = jax.grad(energy)(state.pos_c)
    assert np.all(np.isfinite(gradient))
    np.testing.assert_array_equal(
        evaluated_system.bonded_force_model.initial_edge_lengths, reference
    )
    np.testing.assert_array_equal(
        system.bonded_force_model.initial_edge_lengths, reference
    )

    expected = system.bonded_force_model.update_reference_state(
        state.pos, state, system
    )
    _, stepped_system = jd.System.step(state, system)
    np.testing.assert_allclose(
        stepped_system.bonded_force_model.initial_edge_lengths,
        expected.initial_edge_lengths,
    )


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _TerminalRotationalKick(jd.RotationIntegrator):
    @staticmethod
    def step_after_force(state, system):
        state.ang_vel = jnp.full_like(state.ang_vel, 2.0)
        return state, system


def test_evaluation_does_not_trigger_end_of_step_thermostat() -> None:
    state = jd.State.create(pos=jnp.zeros((1, 2)), mass=jnp.ones(1))
    system = jd.System.create(
        state=state,
        dt=0.1,
        linear_integrator_type="verlet_rescaling",
        linear_integrator_kw={"temperature": 3.0, "can_rotate": True},
        rotation_integrator_type=None,
    )
    system = dataclasses.replace(system, rotation_integrator=_TerminalRotationalKick())

    evaluated, unchanged = jd.System.evaluate_forces(state, system)
    np.testing.assert_array_equal(evaluated.vel, state.vel)
    np.testing.assert_array_equal(evaluated.ang_vel, state.ang_vel)
    assert int(unchanged.step_count) == 0

    stepped, evolved = jd.System.step(state, system)
    np.testing.assert_allclose(
        compute_temperature(stepped, can_rotate=True, subtract_drift=False),
        3.0,
        rtol=1e-6,
    )
    assert int(evolved.step_count) == 1
