# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check neighbor-cache rebuilds, resizing, query isolation, and contact memory."""

import jax
import jax.numpy as jnp

from dataclasses import dataclass, replace

import pytest
import numpy as np

import jaxdem as jd


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


def _pair_histories(collider):
    pairs = {
        (i, int(collider.neighbor_list[k])): np.asarray(collider.history[k]).copy()
        for i in range(len(collider.row_offsets) - 1)
        for k in range(int(collider.row_offsets[i]), int(collider.row_offsets[i + 1]))
    }
    assert len(pairs) == int(collider.row_offsets[-1])
    return pairs


def test_explicit_query_honors_cutoff_width_and_does_not_replace_force_cache():
    state, system = _neighbor_system([[0.0, 0.0], [0.4, 0.0], [2.0, 0.0]])
    state, system = jd.System.initialize(state, system)
    before = [np.asarray(x).copy() for x in jax.tree.leaves(system.collider)]
    state, queried_system, neighbors, overflow = system.collider.create_neighbor_list(
        state, system, 0.5, 2
    )

    assert neighbors.shape == (3, 2)
    np.testing.assert_array_equal(neighbors, [[1, -1], [0, -1], [-1, -1]])
    assert not bool(overflow)
    for expected, actual in zip(
        before, jax.tree.leaves(queried_system.collider), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)
    _, wide_system, wide, overflow = system.collider.create_neighbor_list(
        state, system, 2.1, 2
    )
    np.testing.assert_array_equal(np.sort(wide, axis=-1), [[1, 2], [0, 2], [0, 1]])
    assert not overflow
    for expected, actual in zip(
        before, jax.tree.leaves(wide_system.collider), strict=True
    ):
        np.testing.assert_array_equal(actual, expected)


def test_periodic_metric_change_rebuilds_cached_force_list():
    state, system = _neighbor_system(
        [[0.1, 0.0], [9.9, 0.0]], cutoff=0.5, domain_type="periodic"
    )
    state, system, energy = system.collider.compute_potential_energy(state, system)
    np.testing.assert_allclose(energy, 0.5 * 1e4 * 0.2**2, rtol=2e-6)
    assert _pair_histories(system.collider).keys() == {(0, 1), (1, 0)}
    first_count = int(system.collider.n_build_times)
    system = replace(
        system, domain=replace(system.domain, box_size=jnp.array([11.0, 10.0]))
    )
    state, system, energy = system.collider.compute_potential_energy(state, system)
    np.testing.assert_array_equal(energy, 0.0)
    assert not _pair_histories(system.collider)
    evaluated, _ = jd.System.evaluate_forces(state, system)
    np.testing.assert_array_equal(evaluated.force, 0.0)
    assert int(system.collider.n_build_times) == first_count + 1


def test_neighbor_cache_rebuilds_when_skin_increases():
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [1.6, 0.0]]), rad=jnp.full(2, 0.5)
    )
    system = jd.System.create(
        state=state,
        force_model_type="spring",
        collider_type="NeighborList",
        collider_kw={"cutoff": 1.0, "skin": 0.1, "max_neighbors": 4},
    )
    state, system = system.collider.compute_force(state, system)
    first_build = system.collider.n_build_times

    moved = replace(state, pos_c=jnp.array([[0.31, 0.0], [1.29, 0.0]]))
    system = replace(system, collider=replace(system.collider, skin=jnp.asarray(1.0)))
    actual, system = system.collider.compute_force(moved, system)
    naive = replace(system, collider=jd.Collider.create("naive"))
    expected, _ = naive.collider.compute_force(moved, naive)

    assert int(system.collider.n_build_times) == int(first_build) + 1
    assert bool(jnp.allclose(actual.force, expected.force))
    assert bool(jnp.any(actual.force != 0))


def test_neighbor_cache_rebuilds_when_lees_edwards_axes_change():
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0, 0.1], [0.0, 4.0, 9.9]]), rad=jnp.ones(2)
    )
    box = jnp.full(3, 10.0)
    system = jd.System.create(
        state=state,
        force_model_type="spring",
        collider_type="NeighborList",
        collider_kw={"cutoff": 1.0, "skin": 0.2, "max_neighbors": 4},
        domain_type="LeesEdwards",
        domain_kw={"box_size": box, "gamma": 0.4, "alpha": 0, "beta": 1},
    )
    state, system = system.collider.compute_force(state, system)
    first_build = system.collider.n_build_times

    new_domain = jd.Domain.create(
        "LeesEdwards", dim=3, box_size=box, gamma=0.4, alpha=1, beta=2
    )
    system = replace(system, domain=new_domain)
    actual, system = system.collider.compute_force(state, system)
    naive = replace(system, collider=jd.Collider.create("naive"))
    expected, _ = naive.collider.compute_force(state, naive)

    assert int(system.collider.n_build_times) == int(first_build) + 1
    assert bool(jnp.allclose(actual.force, expected.force))
    assert bool(jnp.any(actual.force != 0))


def test_neighbor_capacity_defaults_and_explicit_overrides():
    zero_rad = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [1.0, 0.0]]), rad=jnp.zeros(2)
    )
    auto = jd.colliders.NeighborList.Create(
        zero_rad, cutoff=1.0, secondary_collider_type="naive"
    )
    assert 0 <= auto.max_neighbors <= zero_rad.N

    state = jd.State.create(pos=jnp.array([[0.0, 0.0], [1.0, 0.0]]))
    for capacity in (0, 1, 7):
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


def test_naive_collider_stores_dense_history_and_advances_it():
    state = jd.State.create(pos=jnp.array([[0.0, 0.0], [10.0, 0.0], [30.0, 0.0]]))
    system = jd.System.create(
        state=state, force_model=RememberingSpring(), collider_type="naive"
    )
    assert system.collider.history_pair_shape(state.pos.shape) == (3, 3)
    assert system.collider.history.shape == (3, 3, 1)
    np.testing.assert_array_equal(system.collider.history, 0.0)

    # Initialization is read-only.
    state, system = jd.System.initialize(state, system)
    np.testing.assert_array_equal(system.collider.history, 0.0)

    off_diagonal = ~np.eye(3, dtype=bool)
    state, system = system.collider.compute_force(state, system)
    history = np.asarray(system.collider.history)[..., 0]
    np.testing.assert_array_equal(history[off_diagonal], 1.0)
    np.testing.assert_array_equal(history[~off_diagonal], 0.0)

    # Every pair keeps its slot no matter how far particles move.
    moved = replace(
        state,
        pos_c=state.pos_c + jnp.array([[100.0, 0.0], [-200.0, 0.0], [300.0, 0.0]]),
    )
    _, system = system.collider.compute_force(moved, system)
    history = np.asarray(system.collider.history)[..., 0]
    np.testing.assert_array_equal(history[off_diagonal], 2.0)
    assert system.collider.invalidate() is system.collider

    query = jnp.array([[1, 2], [0, -1], [-1, -1]])
    queried = system.collider.get_history(moved, system, query)
    np.testing.assert_array_equal(queried[..., 0], [[2.0, 2.0], [2.0, 0.0], [0.0, 0.0]])


def test_refresh_naive_history_for_new_particle_count():
    state = jd.State.create(pos=jnp.zeros((2, 2)))
    law = SeededSpring()
    system = jd.System.create(state=state, force_model=law, collider_type="naive")
    np.testing.assert_array_equal(system.collider.history, 7.0)
    larger = jd.State.create(pos=jnp.zeros((4, 2)))
    refreshed = jd.colliders.refresh_collider(larger, system.collider, law)
    assert refreshed.history.shape == (4, 4, 1)
    np.testing.assert_array_equal(refreshed.history, 7.0)
    same = jd.colliders.refresh_collider(state, system.collider, law)
    assert same is system.collider


def test_naive_history_checkpoint_continues_exactly(tmp_path):
    pytest.importorskip("orbax.checkpoint")
    state = replace(
        jd.State.create(pos=jnp.array([[0.0, 0.0], [0.3, 0.0]])),
        fixed=jnp.ones(2, dtype=bool),
    )
    system = jd.System.create(
        state=state, force_model=RememberingSpring(), collider_type="naive"
    )
    state, system = jd.System.initialize(state, system)
    state, system = jd.System.step(state, system)

    checkpoint_dir = tmp_path / "naive-history"
    with jd.CheckpointWriter(checkpoint_dir) as writer:
        writer.save(state, system)
    restored_state, restored_system = jd.CheckpointLoader(checkpoint_dir).load()

    np.testing.assert_array_equal(
        restored_system.collider.history, system.collider.history
    )
    expected = jd.System.step(state, system)
    actual = jd.System.step(restored_state, restored_system)
    for expected_leaf, actual_leaf in zip(
        jax.tree.leaves(expected), jax.tree.leaves(actual), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


@jax.tree_util.register_dataclass
@jd.ForceModel.register("remembering-spring-test")
@dataclass(slots=True)
class RememberingSpring(jd.forces.SpringForce):
    def history_shape(self, dim):
        return (1,)

    @staticmethod
    def force(
        i, j, pos, state, system, history, *, advance_history=True, quasistatic=False
    ):
        force, torque, _ = jd.forces.SpringForce.force(
            i, j, pos, state, system, history
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
    def force(
        i, j, pos, state, system, history, *, advance_history=True, quasistatic=False
    ):
        force, torque, _ = jd.forces.SpringForce.force(
            i, j, pos, state, system, history
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


def test_composite_history_layout_for_empty_and_nested_laws():
    assert jd.forces.LawCombiner().init_history((2, 3), 2).shape == (2, 3, 0)
    assert jd.forces.ForceRouter().init_history((2, 3), 2).shape == (2, 3, 0)

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


@pytest.mark.parametrize("composite", ["combiner", "router"])
def test_history_survives_evaluation_refresh_and_initial_preparation(composite):
    state, _ = _neighbor_system([[0.0, 0.0], [0.3, 0.0]])
    state = replace(state, fixed=jnp.ones(2, dtype=bool))
    law = RememberingSpring()
    if composite == "combiner":
        law = jd.forces.LawCombiner(laws=(law, jd.forces.SpringForce()))
    else:
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
    system = replace(
        system,
        collider=replace(
            system.collider,
            history=system.collider.history.at[:2, 0].set(jnp.array([11.0, 29.0])),
        ),
    )
    previous = _pair_histories(system.collider)
    moved = replace(state, pos_c=state.pos_c + 0.2)
    moved, evaluated, _ = system.collider.compute_potential_energy(moved, system)
    assert int(evaluated.collider.n_build_times) > int(system.collider.n_build_times)
    actual = _pair_histories(evaluated.collider)
    assert actual.keys() == previous.keys()
    for pair in previous:
        np.testing.assert_array_equal(actual[pair], previous[pair])
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
    pytest.importorskip("orbax.checkpoint")
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


def test_refresh_collider_history_growth_and_reset_rules():
    state, _ = _neighbor_system([[0.0, 0.0], [0.3, 0.0]], max_neighbors=1)
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
    old = replace(system.collider, history=jnp.array([[21.0], [43.0]]))
    resized = jd.colliders.refresh_collider(state, replace(old, max_neighbors=3), law)
    np.testing.assert_array_equal(resized.history[:2], [[21.0], [43.0]])
    np.testing.assert_array_equal(resized.history[2:], 7.0)

    larger = jd.State.create(pos=jnp.array([[0.0, 0.0], [0.3, 0.0], [3.0, 0.0]]))
    with pytest.raises(ValueError, match="particle count.*reset_history=True"):
        jd.colliders.refresh_collider(larger, resized, law)
    reset = jd.colliders.refresh_collider(larger, resized, law, reset_history=True)
    np.testing.assert_array_equal(reset.history, 7.0)


def test_refresh_collider_stateless_shrink_and_incompatible_force_rules():
    state, system = _neighbor_system([[0.0, 0.0], [0.3, 0.0]], max_neighbors=4)
    resized = jd.colliders.refresh_collider(
        state, replace(system.collider, max_neighbors=1), system.force_model
    )
    assert resized.neighbor_list.shape == (2,)
    assert resized.history.shape == (2, 0)

    with pytest.raises(ValueError, match="reset_history=True"):
        jd.colliders.refresh_collider(state, system.collider, RememberingSpring())


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


def test_grid_constructor_rejects_invalid_search_geometry():
    state = jd.State.create(pos=jnp.array([[0.0, 0.0]]))
    for collider_type in ("CellList", "MultiCellList"):
        for kw in (
            {"cell_size": 0.0},
            {"cell_size": float("nan")},
            {"search_range": 0},
        ):
            with pytest.raises(ValueError):
                jd.Collider.create(collider_type, state=state, **kw)


def _moving_pair(vel, check_every, *, secondary_collider_type="naive"):
    # Two disks approach head-on; dt = 0.01 by default.
    state = jd.State.create(
        pos=jnp.asarray([[3.0, 5.0], [4.0, 5.0], [7.0, 5.0]]),
        rad=jnp.full(3, 0.4),
        vel=jnp.asarray([[vel, 0.0], [-vel, 0.0], [0.0, 0.0]]),
    )
    system = jd.System.create(
        state=state,
        domain_type="periodic",
        domain_kw={"box_size": jnp.array([10.0, 10.0])},
        collider_type="NeighborList",
        collider_kw={
            "cutoff": 0.8,
            "skin": 0.2,
            "max_neighbors": 4,
            "secondary_collider_type": secondary_collider_type,
            "check_every": check_every,
        },
    )
    return jd.System.initialize(state, system)


def _invalidate_on_step_two(state, system):
    should_edit = system.step_count == 2
    rad = jnp.where(should_edit, jnp.full_like(state.rad, 2.0), state.rad)
    collider = jax.lax.cond(
        should_edit,
        lambda c: c.invalidate(),
        lambda c: c,
        system.collider,
    )
    return replace(state, rad=rad), replace(system, collider=collider)


def test_check_every_matches_per_step_checks_and_flags_spent_skin():
    ref_state, _ = jd.System.step(*_moving_pair(1.0, 1), n=50)
    for use_pallas in (True, False):
        with jd.colliders.pallas_enabled(use_pallas):
            state, system = jd.System.step(
                *_moving_pair(1.0, 4, secondary_collider_type="CellList"), n=50
            )
            np.testing.assert_allclose(state.pos, ref_state.pos, rtol=0, atol=1e-6)
            assert not bool(system.search_overflow)
            assert system.collider.check_every == 4
            assert system.collider._check_horizon == 0
    # A window longer than the skin allows must be reported, not silently missed.
    _, system = jd.System.step(*_moving_pair(1.0, 30), n=60)
    assert bool(system.search_overflow)


@pytest.mark.parametrize("n", [0, 1, 2, 5])
def test_check_every_edge_step_counts(n):
    ref_state, _ = jd.System.step(*_moving_pair(1.0, 1), n=n)
    state, system = jd.System.step(*_moving_pair(1.0, 4), n=n)
    np.testing.assert_allclose(state.pos, ref_state.pos, rtol=0, atol=1e-6)
    assert not bool(system.search_overflow)
    assert system.collider._check_horizon == 0


def test_check_every_batched_vmap_and_jvp_debug_nans():
    s0, y0 = _moving_pair(1.0, 4)
    s1, y1 = _moving_pair(0.5, 4)
    out_state, out_sys = jd.System.step(
        jd.State.stack([s0, s1]), jd.System.stack([y0, y1]), n=20
    )
    r0, _ = jd.System.step(*_moving_pair(1.0, 1), n=20)
    r1, _ = jd.System.step(*_moving_pair(0.5, 1), n=20)
    np.testing.assert_allclose(out_state.pos[0], r0.pos, rtol=0, atol=1e-6)
    np.testing.assert_allclose(out_state.pos[1], r1.pos, rtol=0, atol=1e-6)
    assert not bool(jnp.any(out_sys.search_overflow))
    assert out_sys.collider._check_horizon == 0

    # Zero velocity / zero force must not produce NaN tangents in _motion_bound.
    st_rest, sy_rest = _moving_pair(0.0, 4)
    tangent = jnp.ones_like(st_rest.pos_c) * 1e-2
    with jax.debug_nans(True):
        _, dpos = jax.jvp(
            lambda pc: (
                jd.System.step(replace(st_rest, pos_c=pc), sy_rest, n=4)[0].pos_c
            ),
            (st_rest.pos_c,),
            (tangent,),
        )
    assert bool(jnp.all(jnp.isfinite(dpos)))


def test_check_every_flags_deferred_invalidation_and_survives_checkpoint(tmp_path):
    state = jd.State.create(
        pos=jnp.asarray([[0.0, 0.0], [2.5, 0.0]]), rad=jnp.full(2, 0.4)
    )
    system = jd.System.create(
        state=state,
        domain_type="periodic",
        domain_kw={"box_size": jnp.array([10.0, 10.0])},
        collider_type="NeighborList",
        collider_kw={
            "cutoff": 0.8,
            "skin": 0.2,
            "max_neighbors": 4,
            "secondary_collider_type": "naive",
            "check_every": 4,
        },
        user_pre_step_actions=_invalidate_on_step_two,
    )
    st0, sy0 = jd.System.initialize(state, system)
    _, sy1 = jd.System.step(st0, sy0, n=3)
    assert bool(sy1.search_overflow)

    pytest.importorskip("orbax.checkpoint")
    s_ckpt, y_ckpt = _moving_pair(1.0, 4)
    ckpt_dir = tmp_path / "check-every"
    with jd.CheckpointWriter(ckpt_dir) as writer:
        writer.save(s_ckpt, y_ckpt)
    r_state, r_sys = jd.CheckpointLoader(ckpt_dir).load()
    assert r_sys.collider.check_every == 4
    assert r_sys.collider._check_horizon == 0
    s_exp, _ = jd.System.step(s_ckpt, y_ckpt, n=10)
    s_act, _ = jd.System.step(r_state, r_sys, n=10)
    np.testing.assert_allclose(s_act.pos, s_exp.pos, rtol=0, atol=1e-6)
