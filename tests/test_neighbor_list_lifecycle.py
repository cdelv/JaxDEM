# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
from dataclasses import dataclass, replace

import jax
import pytest

import jax.numpy as jnp
import numpy as np

import jaxdem as jd
from jaxdem.colliders._neighbor_cache import pair_sources, remap_history


def _neighbor_system(pos, *, cutoff=1.0, max_neighbors=4, domain_type="free"):
    state = jd.State.create(
        pos=jnp.asarray(pos, dtype=float), rad=jnp.full(len(pos), 0.2)
    )
    system = jd.System.create(
        state=state,
        domain_type=domain_type,
        domain_kw={"box_size": jnp.array([10.0, 10.0])},
        collider_type="NeighborList",
        collider_kw={
            "cutoff": cutoff,
            "skin": 0.2,
            "max_neighbors": max_neighbors,
            "secondary_collider_type": "naive",
        },
    )
    return state, system


def test_explicit_query_honors_cutoff_width_and_does_not_replace_force_cache():
    state, system = _neighbor_system([[0.0, 0.0], [0.4, 0.0], [2.0, 0.0]])
    state, queried_system, neighbors, overflow = system.collider.create_neighbor_list(
        state, system, 0.5, 2
    )

    assert neighbors.shape == (3, 2)
    assert 1 in np.asarray(neighbors[0])
    assert 2 not in np.asarray(neighbors[0])
    assert not bool(overflow)
    assert int(queried_system.collider.n_build_times) == 0


def test_zero_width_query_reports_qualifying_neighbors_as_overflow():
    state, system = _neighbor_system([[0.0, 0.0], [0.4, 0.0]])
    _, _, neighbors, overflow = system.collider.create_neighbor_list(
        state, system, 0.5, 0
    )
    assert neighbors.shape == (2, 0)
    assert bool(overflow)


def test_periodic_metric_change_rebuilds_cached_force_list():
    state, system = _neighbor_system(
        [[0.1, 0.0], [9.9, 0.0]], cutoff=0.5, domain_type="periodic"
    )
    state, system, _ = system.collider.compute_potential_energy(state, system)
    first_count = int(system.collider.n_build_times)
    system = replace(
        system, domain=replace(system.domain, box_size=jnp.array([11.0, 10.0]))
    )
    state, system, _ = system.collider.compute_potential_energy(state, system)
    assert int(system.collider.n_build_times) == first_count + 1


def test_zero_radius_capacity_estimate_is_finite():
    state = jd.State.create(pos=jnp.array([[0.0, 0.0], [1.0, 0.0]]), rad=jnp.zeros(2))
    collider = jd.colliders.NeighborList.Create(
        state, cutoff=1.0, secondary_collider_type="naive"
    )
    assert 0 <= collider.max_neighbors <= state.N


@pytest.mark.parametrize("capacity", [0, 1, 7])
def test_explicit_neighbor_capacity_is_exact(capacity):
    state = jd.State.create(pos=jnp.array([[0.0, 0.0], [1.0, 0.0]]))
    collider = jd.colliders.NeighborList.Create(
        state,
        cutoff=1.0,
        max_neighbors=capacity,
        number_density=1.0e9,
        safety_factor=1.0e9,
        secondary_collider_type="naive",
    )
    assert collider.max_neighbors == capacity
    assert collider.neighbor_list.shape == (state.N * capacity,)
    assert collider.history.shape == (state.N * capacity, 0)


@jax.tree_util.register_dataclass
@jd.ForceModel.register("remembering-spring-test")
@dataclass(slots=True)
class RememberingSpring(jd.forces.SpringForce):
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
class SeededSpring(jd.forces.SpringForce):
    def history_shape(self, dim):
        return (1,)

    def init_history(self, pair_shape, dim):
        return jnp.full(pair_shape + (1,), 7.0)

    @staticmethod
    def force(i, j, pos, state, system, history, *, advance_history=True):
        force, torque, _ = jd.forces.SpringForce.force(
            i, j, pos, state, system, history, advance_history=advance_history
        )
        scale = history[..., 0, None]
        return force * scale, torque * scale, history


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class TwoComponentSeededSpring(jd.forces.SpringForce):
    def history_shape(self, dim):
        return (2,)

    def init_history(self, pair_shape, dim):
        values = jnp.array([8.0, 9.0])
        return jnp.broadcast_to(values, pair_shape + (2,))


@pytest.mark.parametrize("law", [jd.forces.LawCombiner(), jd.forces.ForceRouter()])
def test_empty_composite_history_has_zero_width(law):
    assert law.init_history((2, 3), 2).shape == (2, 3, 0)


def test_nested_composite_history_preserves_child_slice_order():
    law = jd.forces.LawCombiner(
        laws=(
            jd.forces.SpringForce(),
            SeededSpring(),
            jd.forces.LawCombiner(laws=(TwoComponentSeededSpring(),)),
        )
    )
    history = law.init_history((1,), 2)
    np.testing.assert_array_equal(history, [[7.0, 8.0, 9.0]])


@pytest.mark.parametrize(
    "law",
    [
        SeededSpring(),
        jd.forces.LawCombiner(laws=(SeededSpring(),)),
        jd.forces.ForceRouter.from_dict(1, {(0, 0): SeededSpring()}),
    ],
)
def test_composite_history_initializer_controls_physical_force(law):
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.3, 0.0]]), rad=jnp.full(2, 0.5)
    )
    system = jd.System.create(
        state=state,
        force_model=law,
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 2,
            "cutoff": 1.0,
            "skin": 0.1,
            "secondary_collider_type": "naive",
        },
    )
    initialized, system = jd.System.initialize(state, system)
    valid = np.asarray(system.collider.neighbor_list) >= 0
    np.testing.assert_array_equal(np.asarray(system.collider.history)[valid], 7.0)
    np.testing.assert_allclose(np.abs(initialized.force[:, 0]), 49_000.0)


@pytest.mark.parametrize("collider_type", ["CellList", "MultiCellList"])
def test_cached_collider_without_pair_history_rejects_stateful_force(collider_type):
    state = jd.State.create(pos=jnp.array([[0.0, 0.0], [0.3, 0.0]]))
    with pytest.raises(ValueError, match="cannot persist"):
        jd.System.create(
            state=state,
            collider_type=collider_type,
            force_model=RememberingSpring(),
        )


@pytest.mark.parametrize("composite", ["direct", "combiner", "router"])
def test_history_survives_evaluation_refresh_and_initial_preparation(composite):
    state, system = _neighbor_system([[0.0, 0.0], [0.3, 0.0]])
    state = replace(state, fixed=jnp.ones(2, dtype=bool))
    law = RememberingSpring()
    if composite == "combiner":
        law = jd.forces.LawCombiner(laws=(law, jd.forces.SpringForce()))
    elif composite == "router":
        law = jd.forces.ForceRouter(table=((law,),))
    system = jd.System.create(
        state=state,
        force_model=law,
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 3,
            "cutoff": 0.5,
            "skin": 0.1,
            "secondary_collider_type": "naive",
        },
    )
    # Initialization evaluates forces without advancing pair history.
    state, system = jd.System.initialize(state, system)
    np.testing.assert_array_equal(system.collider.history, 0.0)

    # One physical step advances each valid pair's history exactly once.
    state, system = jd.System.step(state, system)
    for history in jax.tree.leaves(system.collider.history):
        np.testing.assert_array_equal(
            np.asarray(history)[np.asarray(system.collider.neighbor_list) >= 0], 1.0
        )
    previous = system.collider.history
    previous_neighbors = system.collider.neighbor_list
    previous_sources = pair_sources(system.collider)
    moved = replace(state, pos_c=state.pos_c + 0.2)
    moved, evaluated, _ = system.collider.compute_potential_energy(moved, system)
    assert int(evaluated.collider.n_build_times) > int(system.collider.n_build_times)
    for before, after in zip(
        jax.tree.leaves(previous), jax.tree.leaves(evaluated.collider.history)
    ):
        expected = remap_history(
            before,
            previous_sources,
            previous_neighbors,
            pair_sources(evaluated.collider),
            evaluated.collider.neighbor_list,
            law.init_history(
                evaluated.collider.neighbor_list.shape, moved.pos.shape[-1]
            ),
        )
        np.testing.assert_array_equal(after, expected)
    refreshed = replace(
        evaluated,
        collider=jd.colliders.refresh_collider(moved, evaluated.collider, law),
    )
    _, refreshed = refreshed.collider.compute_force(
        moved, refreshed, advance_history=False
    )
    for before, after in zip(
        jax.tree.leaves(evaluated.collider.history),
        jax.tree.leaves(refreshed.collider.history),
    ):
        np.testing.assert_array_equal(after, before)


def test_composite_nonempty_history_checkpoint_continues_exactly(tmp_path):
    state, _ = _neighbor_system([[0.0, 0.0], [0.3, 0.0]])
    state = replace(state, fixed=jnp.ones(2, dtype=bool))
    law = jd.forces.LawCombiner(laws=(RememberingSpring(), jd.forces.SpringForce()))
    system = jd.System.create(
        state=state,
        force_model=law,
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 3,
            "cutoff": 0.5,
            "skin": 0.1,
            "secondary_collider_type": "naive",
        },
    )
    state, system = jd.System.step(state, system)
    assert bool(jnp.any(system.collider.history != 0))

    checkpoint_dir = tmp_path / "nonempty-history"
    with jd.CheckpointWriter(checkpoint_dir) as writer:
        writer.save(state, system)
    restored_state, restored_system = jd.CheckpointLoader(checkpoint_dir).load()

    np.testing.assert_array_equal(
        restored_system.collider.history, system.collider.history
    )
    next_state, next_system = jd.System.step(state, system)
    restored_next_state, restored_next_system = jd.System.step(
        restored_state, restored_system
    )
    for expected, actual in zip(
        jax.tree.leaves((next_state, next_system)),
        jax.tree.leaves((restored_next_state, restored_next_system)),
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)


def test_refresh_grows_history_capacity_without_losing_pair_memory():
    state, system = _neighbor_system([[0.0, 0.0], [0.3, 0.0]])
    law = RememberingSpring()
    system = jd.System.create(
        state=state,
        force_model=law,
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 1,
            "cutoff": 0.5,
            "skin": 0.1,
            "secondary_collider_type": "naive",
        },
    )
    state, system = system.collider.compute_force(state, system)
    resized = jd.colliders.refresh_collider(
        state, replace(system.collider, max_neighbors=4), law
    )
    assert resized.history.shape == (8, 1)
    _, system = resized.compute_force(
        state, replace(system, collider=resized), advance_history=False
    )
    valid = np.asarray(system.collider.neighbor_list) >= 0
    np.testing.assert_array_equal(np.asarray(system.collider.history)[valid], 1.0)
    with pytest.raises(ValueError, match="reset_history"):
        jd.colliders.refresh_collider(
            state, replace(system.collider, max_neighbors=1), law
        )


def test_refresh_growth_uses_force_history_initializer_for_new_slots():
    state, system = _neighbor_system([[0.0, 0.0], [0.3, 0.0]], max_neighbors=1)
    law = SeededSpring()
    system = jd.System.create(
        state=state,
        force_model=law,
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 1,
            "cutoff": 0.5,
            "skin": 0.1,
            "secondary_collider_type": "naive",
        },
    )
    old = replace(system.collider, history=system.collider.history * 3.0)
    resized = jd.colliders.refresh_collider(state, replace(old, max_neighbors=3), law)
    np.testing.assert_array_equal(resized.history[:2], 21.0)
    np.testing.assert_array_equal(resized.history[2:], 7.0)


def test_particle_count_change_requires_explicit_nonempty_history_reset():
    state, system = _neighbor_system([[0.0, 0.0], [0.3, 0.0]])
    law = SeededSpring()
    system = jd.System.create(
        state=state,
        force_model=law,
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 2,
            "cutoff": 0.5,
            "skin": 0.1,
            "secondary_collider_type": "naive",
        },
    )
    larger = jd.State.create(pos=jnp.array([[0.0, 0.0], [0.3, 0.0], [3.0, 0.0]]))
    with pytest.raises(ValueError, match="particle count.*reset_history=True"):
        jd.colliders.refresh_collider(larger, system.collider, law)
    reset = jd.colliders.refresh_collider(
        larger, system.collider, law, reset_history=True
    )
    np.testing.assert_array_equal(reset.history, 7.0)


def test_new_neighbor_uses_force_history_initializer_after_rebuild():
    far = jd.State.create(pos=jnp.array([[0.0, 0.0], [3.0, 0.0]]), rad=jnp.full(2, 0.5))
    law = SeededSpring()
    system = jd.System.create(
        state=far,
        force_model=law,
        collider_type="NeighborList",
        collider_kw={
            "max_neighbors": 2,
            "cutoff": 1.0,
            "skin": 0.1,
            "secondary_collider_type": "naive",
        },
    )
    _, system = jd.System.initialize(far, system)
    close = replace(far, pos_c=jnp.array([[0.0, 0.0], [0.3, 0.0]]))
    evaluated, system = system.collider.compute_force(
        close, system, advance_history=False
    )
    valid = np.asarray(system.collider.neighbor_list) >= 0
    np.testing.assert_array_equal(np.asarray(system.collider.history)[valid], 7.0)
    np.testing.assert_allclose(np.abs(evaluated.force[:, 0]), 49_000.0)


def test_refresh_shrinks_stateless_history_with_neighbor_capacity():
    state, system = _neighbor_system([[0.0, 0.0], [0.3, 0.0]], max_neighbors=4)
    resized = jd.colliders.refresh_collider(
        state, replace(system.collider, max_neighbors=1), system.force_model
    )
    assert resized.neighbor_list.shape == (2,)
    assert resized.history.shape == (2, 0)


def test_refresh_rejects_incompatible_force_history_without_reset():
    state, system = _neighbor_system([[0.0, 0.0], [0.3, 0.0]])
    with pytest.raises(ValueError, match="reset_history=True"):
        jd.colliders.refresh_collider(state, system.collider, RememberingSpring())


@pytest.mark.parametrize("collider_type", ["CellList", "MultiCellList", "NeighborList"])
def test_zero_radius_default_grid_remains_a_valid_geometric_query(collider_type):
    state = jd.State.create(pos=jnp.array([[0.0, 0.0], [0.2, 0.0]]), rad=jnp.zeros(2))
    kw = {"cutoff": 0.0, "skin": 0.0} if collider_type == "NeighborList" else {}
    system = jd.System.create(state=state, collider_type=collider_type, collider_kw=kw)
    _, _, neighbors, overflow = system.collider.create_neighbor_list(
        state, system, 0.3, 1
    )
    np.testing.assert_array_equal(neighbors, [[1], [0]])
    assert not bool(overflow)


@pytest.mark.parametrize("collider_type", ["CellList", "MultiCellList"])
@pytest.mark.parametrize(
    "kw", [{"cell_size": 0.0}, {"cell_size": float("nan")}, {"search_range": 0}]
)
def test_grid_constructor_rejects_invalid_search_geometry(collider_type, kw):
    state = jd.State.create(pos=jnp.array([[0.0, 0.0]]))
    with pytest.raises(ValueError):
        jd.Collider.create(collider_type, state=state, **kw)
