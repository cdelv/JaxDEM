# SPDX-License-Identifier: BSD-3-Clause
"""Particle removal preserves the physical memory of surviving contacts."""

from dataclasses import dataclass, replace
from itertools import pairwise

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.colliders._neighbor_cache import pair_sources
from jaxdem.utils.contacts import remove_rattlers


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class MatrixHistorySpring(jd.forces.SpringForce):
    def history_shape(self, dim):
        return (2, 2)

    def init_history(self, pair_shape, dim):
        return jnp.broadcast_to(
            jnp.arange(7.0, 11.0).reshape(2, 2), pair_shape + (2, 2)
        )

    @staticmethod
    def force(i, j, pos, state, system, history, *, advance_history=True):
        force, torque, _ = jd.forces.SpringForce.force(
            i, j, pos, state, system, history, advance_history=advance_history
        )
        return force * history[0, 0], torque, history + advance_history


def _system(dim=2, backend="naive", law=None, *, clumps=False, initialized=True):
    pos = jnp.array(
        [[-0.8, 0.0], [0.0, 0.0], [0.0, -0.8], [0.8, 0.0], [0.4, 0.6], [5.0, 5.0]]
    )
    kwargs = {}
    if clumps:
        pos = jnp.repeat(pos[:4], 2, axis=0)
        kwargs = {
            "clump_id": jnp.repeat(jnp.arange(4), 2),
            "pos_p": jnp.tile(jnp.array([[0.0, -0.1], [0.0, 0.1]]), (4, 1)),
        }
    pos = jnp.pad(pos, ((0, 0), (0, dim - 2)))
    state = jd.State.create(
        pos=pos,
        vel=jnp.zeros_like(pos).at[:, 1].set(jnp.arange(len(pos)) * 0.03),
        rad=jnp.full(len(pos), 0.5),
        **kwargs,
    )
    material = jd.Material.create(
        "elasticfrict", density=1.0, young=100.0, poisson=0.0, e=1.0, mu=10.0, mu_r=0.0
    )
    system = jd.System.create(
        state=state,
        force_model=jd.forces.CundallStrackForce() if law is None else law,
        mat_table=jd.MaterialTable.from_materials([material]),
        dt=0.01,
        collider_type="NeighborList",
        collider_kw={
            "cutoff": 1.0,
            "skin": 0.1,
            "max_neighbors": state.N,
            "secondary_collider_type": backend,
        },
    )
    if not initialized:
        return state, system
    state, system = jd.System.initialize(state, system)
    state, system = system.collider.compute_force(state, system)
    state, system = jd.System.evaluate_forces(state, system)
    # Change cache slot order without changing the pair-to-history association.
    col = system.collider
    order = np.arange(col.neighbor_list.size)
    offsets = np.asarray(col.row_offsets)
    for start, end in pairwise(offsets):
        order[start:end] = order[start:end][::-1]
    system = replace(
        system,
        collider=replace(
            col, neighbor_list=col.neighbor_list[order], history=col.history[order]
        ),
    )
    return state, system


def _pairs(state, system):
    col = system.collider
    valid = col.neighbor_list >= 0
    src, dst = pair_sources(col)[valid], col.neighbor_list[valid]
    history = col.history[valid]
    force, torque, _ = jax.vmap(
        lambda i, j, h: system.force_model.force(
            i, j, state.pos, state, system, h, advance_history=False
        )
    )(src, dst, history)
    return {
        (int(i), int(j)): (np.asarray(h), np.asarray(f), np.asarray(t))
        for i, j, h, f, t in zip(src, dst, history, force, torque)
    }


def _assert_preserved(before, state, system, kept):
    remap = {int(old): new for new, old in enumerate(kept)}
    expected = {
        (remap[i], remap[j]): values
        for (i, j), values in before.items()
        if i in remap and j in remap
    }
    actual = _pairs(state, system)
    assert actual.keys() == expected.keys()
    for pair in expected:
        np.testing.assert_array_equal(actual[pair][0], expected[pair][0])
        np.testing.assert_allclose(
            actual[pair][1], expected[pair][1], atol=1e-6, rtol=1e-6
        )
        np.testing.assert_allclose(
            actual[pair][2], expected[pair][2], atol=1e-6, rtol=1e-6
        )
    col = system.collider
    padding = col.neighbor_list < 0
    initialized = system.force_model.init_history(col.neighbor_list.shape, state.dim)
    np.testing.assert_array_equal(col.history[padding], initialized[padding])
    return actual


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("backend", ["naive", "CellList", "MultiCellList"])
@pytest.mark.parametrize("removed", [[], [0, 2, 5]])
def test_cundall_strack_history_and_pair_forces_survive_removal(dim, backend, removed):
    state, system = _system(dim, backend)
    before = _pairs(state, system)
    assert any(np.any(h[:dim] != 0.0) for h, _, _ in before.values())
    kept = np.flatnonzero(~np.isin(state.clump_id, removed))

    reduced, updated = remove_rattlers(state, system, jnp.asarray(removed, dtype=int))

    updated.check_overflow()
    _assert_preserved(before, reduced, updated, kept)
    np.testing.assert_array_equal(reduced.pos, state.pos[kept])
    np.testing.assert_array_equal(reduced.vel, state.vel[kept])
    np.testing.assert_array_equal(reduced.ang_vel, state.ang_vel[kept])
    assert updated.time == system.time and updated.step_count == system.step_count
    observed, rebuilt = jd.System.evaluate_forces(
        reduced, replace(updated, collider=updated.collider.invalidate())
    )
    _assert_preserved(before, observed, rebuilt, kept)


@pytest.mark.parametrize("composition", ["matrix", "combiner", "router"])
def test_nonvector_and_composite_history_survives_removal(composition):
    law = MatrixHistorySpring()
    if composition == "combiner":
        law = jd.LawCombiner(laws=(jd.forces.CundallStrackForce(), law))
    elif composition == "router":
        law = jd.ForceRouter.from_dict(1, {(0, 0): law})
    state, system = _system(law=law)
    before = _pairs(state, system)

    reduced, updated = remove_rattlers(state, system, jnp.array([0, 2, 5]))

    _assert_preserved(before, reduced, updated, [1, 3, 4])


def test_removing_whole_clumps_preserves_member_pairs_and_body_forces():
    state, system = _system(clumps=True)
    before = _pairs(state, system)

    reduced, updated = remove_rattlers(state, system, jnp.array([0, 2]))

    actual = _assert_preserved(before, reduced, updated, [2, 3, 6, 7])
    np.testing.assert_array_equal(reduced.clump_id, [0, 0, 1, 1])
    expected = np.zeros_like(reduced.force)
    for (i, _), (_, force, _) in actual.items():
        expected[np.asarray(reduced.clump_id) == reduced.clump_id[i]] += force
    np.testing.assert_allclose(reduced.force, expected, atol=1e-6, rtol=1e-6)


def test_new_pairs_receive_force_law_initial_history():
    law = MatrixHistorySpring()
    state, system = _system(law=law)
    before = _pairs(state, system)
    moved = replace(state, pos_c=state.pos_c.at[5].set(jnp.array([0.4, -0.6])))
    kept = [1, 3, 4, 5]

    reduced, updated = remove_rattlers(moved, system, jnp.array([0, 2]))

    actual = _pairs(reduced, updated)
    assert any(i == 3 or j == 3 for i, j in actual)
    for (i, j), (history, _, _) in actual.items():
        old_pair = (kept[i], kept[j])
        expected = (
            before[old_pair][0] if old_pair in before else law.init_history((), 2)
        )
        np.testing.assert_array_equal(history, expected)


def test_removing_every_particle_returns_empty_history():
    state, system = _system()

    reduced, updated = remove_rattlers(state, system, jnp.arange(state.N))

    assert reduced.N == 0
    assert updated.collider.history.shape == (0, 4)
    updated.check_overflow()


def test_removal_before_first_cache_build_initializes_history():
    law = MatrixHistorySpring()
    state, system = _system(law=law, initialized=False)

    reduced, updated = remove_rattlers(state, system, jnp.array([0, 2, 5]))

    actual = _pairs(reduced, updated)
    assert actual
    for history, _, _ in actual.values():
        np.testing.assert_array_equal(history, law.init_history((), 2))


def test_stateless_force_removal_preserves_pair_forces():
    state, system = _system(law=jd.forces.SpringForce())
    before = _pairs(state, system)

    reduced, updated = remove_rattlers(state, system, jnp.array([0, 2, 5]))

    _assert_preserved(before, reduced, updated, [1, 3, 4])
    assert updated.collider.history.shape == (reduced.N * state.N, 0)


def test_removal_reports_overflow_when_reduced_pool_cannot_hold_surviving_pairs():
    state = jd.State.create(
        pos=jnp.array(
            [[0.0, 0.0], [0.5, 0.0], [0.25, 0.4], [4.0, 0.0], [8.0, 0.0], [12.0, 0.0]]
        ),
        rad=jnp.full(6, 0.5),
    )
    system = jd.System.create(
        state=state,
        force_model=MatrixHistorySpring(),
        collider_type="NeighborList",
        collider_kw={
            "cutoff": 1.0,
            "max_neighbors": 1,
            "secondary_collider_type": "naive",
        },
    )
    state, system = jd.System.initialize(state, system)
    state, system = system.collider.compute_force(state, system)
    system.check_overflow()
    before = _pairs(state, system)

    reduced, updated = remove_rattlers(state, system, jnp.array([3, 4, 5]))

    assert updated.search_overflow and updated.collider.overflow
    with pytest.raises(RuntimeError, match="overflow"):
        updated.check_overflow()
    actual = _pairs(reduced, updated)
    assert len(actual) == 3
    for pair, (history, _, _) in actual.items():
        np.testing.assert_array_equal(history, before[pair][0])
