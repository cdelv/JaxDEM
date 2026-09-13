# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import jaxdem as jd
import jaxdem.colliders.cell_list as cell_list_module
import jaxdem.colliders.multi_cell_list as multi_cell_list_module
from jaxdem.colliders._partition import _cell_starts


@pytest.fixture(
    params=[("CellList", cell_list_module), ("MultiCellList", multi_cell_list_module)]
)
def grid_system(request):
    collider_type, collider_module = request.param
    pos = jnp.asarray([[0.0, 0.0], [0.3, 0.0], [0.8, 0.0], [1.1, 0.0], [2.5, 0.0]])
    state = jd.State.create(pos=pos, rad=jnp.full((pos.shape[0],), 0.1))
    system = jd.System.create(
        state=state,
        collider_type=collider_type,
        collider_kw={"cell_size": 0.5},
    )
    return state, system, collider_module


def _neighbor_sets(neighbors):
    return [set(row[row >= 0].tolist()) for row in neighbors]


def _searchsorted_starts(sorted_hashes, queries):
    return jnp.searchsorted(sorted_hashes, queries, side="left", method="scan_unrolled")


def _reference_per_stencil_row(hashes, stencil, valid, capacity):
    candidates = []
    for target in stencil.tolist():
        candidates.extend(
            idx
            for idx, cell_hash in enumerate(hashes.tolist())
            if cell_hash == target and valid[idx]
        )
    row = (candidates[:capacity] + [-1] * capacity)[:capacity]
    return jnp.asarray(row, dtype=int), len(candidates) > capacity


@pytest.mark.parametrize("capacity", [0, 4, 6, 8])
def test_direct_row_matches_per_stencil_reference_with_holes(capacity):
    hash_dtype = jnp.uint64 if jax.config.jax_enable_x64 else jnp.uint32
    sentinel = jnp.asarray(jnp.iinfo(hash_dtype).max, dtype=hash_dtype)
    hashes = jnp.asarray([0, 0, 2, 2, 2, 2, 2, 2, 2, 5], dtype=hash_dtype)
    # The sentinel represents a periodic duplicate removed from the stencil;
    # hash 3 is an ordinary empty cell.
    stencil = jnp.asarray([2, sentinel, 3, 0, 5], dtype=hash_dtype)
    starts = _searchsorted_starts(hashes, stencil)
    valid = (True, False, True, False, True, False, True, False, True, True)

    row_body = cell_list_module._make_direct_row_body(
        hashes, hashes.shape[0], capacity, lambda k: jnp.asarray(valid)[k]
    )
    actual, actual_overflow = row_body(stencil, starts)
    expected, expected_overflow = _reference_per_stencil_row(
        hashes, stencil, valid, capacity
    )

    assert bool(jnp.array_equal(actual, expected))
    assert bool(actual_overflow) == expected_overflow


def test_cell_starts_dense_duplicates_absent_and_high_uint_queries():
    hash_dtype = jnp.uint64 if jax.config.jax_enable_x64 else jnp.uint32
    sentinel = jnp.asarray(jnp.iinfo(hash_dtype).max, dtype=hash_dtype)
    hashes = jnp.asarray([0, 1, 1, 5], dtype=hash_dtype)
    queries = jnp.asarray([0, 1, 2, 5, sentinel], dtype=hash_dtype)

    assert bool(
        jnp.array_equal(_cell_starts(hashes, queries), jnp.asarray([0, 1, 4, 3, 4]))
    )


def test_cell_starts_sparse_and_empty_inputs():
    hash_dtype = jnp.uint64 if jax.config.jax_enable_x64 else jnp.uint32
    hashes = jnp.asarray([0, 100], dtype=hash_dtype)
    queries = jnp.asarray([0, 1, 100, 101], dtype=hash_dtype)

    assert bool(
        jnp.array_equal(
            _cell_starts(hashes, queries), _searchsorted_starts(hashes, queries)
        )
    )
    assert bool(
        jnp.array_equal(
            _cell_starts(jnp.asarray([], dtype=hash_dtype), queries),
            jnp.zeros(queries.shape, dtype=jnp.int32),
        )
    )


def test_cell_starts_mixed_dense_and_sparse_vmap():
    hash_dtype = jnp.uint64 if jax.config.jax_enable_x64 else jnp.uint32
    hashes = jnp.asarray([[0, 1, 2], [0, 1, 100]], dtype=hash_dtype)
    queries = jnp.asarray([[0, 2, 3], [0, 2, 100]], dtype=hash_dtype)
    expected = jax.vmap(_searchsorted_starts)(hashes, queries)

    assert bool(jnp.array_equal(jax.vmap(_cell_starts)(hashes, queries), expected))


def test_grid_outputs_match_forced_searchsorted_fallback(grid_system, monkeypatch):
    state, system, collider_module = grid_system
    dense_force, _ = system.collider.compute_force(state, system)
    _, _, dense_energy = system.collider.compute_potential_energy(state, system)
    _, _, dense_neighbors, dense_overflow = system.collider.create_neighbor_list(
        state, system, 1.0, 4
    )
    dense_cross, dense_cross_overflow = system.collider.create_cross_neighbor_list(
        state.pos[:3], state.pos[2:], system, 1.0, 4
    )

    monkeypatch.setattr(collider_module, "_cell_starts", _searchsorted_starts)
    jax.clear_caches()
    sparse_force, _ = system.collider.compute_force(state, system)
    _, _, sparse_energy = system.collider.compute_potential_energy(state, system)
    _, _, sparse_neighbors, sparse_overflow = system.collider.create_neighbor_list(
        state, system, 1.0, 4
    )
    sparse_cross, sparse_cross_overflow = system.collider.create_cross_neighbor_list(
        state.pos[:3], state.pos[2:], system, 1.0, 4
    )

    assert bool(jnp.array_equal(sparse_force.force, dense_force.force))
    assert bool(jnp.array_equal(sparse_force.torque, dense_force.torque))
    assert bool(jnp.array_equal(sparse_energy, dense_energy))
    assert bool(jnp.array_equal(sparse_neighbors, dense_neighbors))
    assert bool(sparse_overflow) == bool(dense_overflow)
    assert bool(jnp.array_equal(sparse_cross, dense_cross))
    assert bool(sparse_cross_overflow) == bool(dense_cross_overflow)


def _same_neighbors(state, system, batch_size, monkeypatch):
    collider_module = (
        multi_cell_list_module
        if system.collider.__class__.__name__ == "DynamicMultiCellList"
        else cell_list_module
    )
    monkeypatch.setattr(collider_module, "NEIGHBOR_QUERY_BATCH_SIZE", batch_size)
    jax.clear_caches()
    return system.collider.create_neighbor_list(state, system, 1.0, 4)[2:]


def _cross_neighbors(pos_a, pos_b, system, collider_module, batch_size, monkeypatch):
    monkeypatch.setattr(collider_module, "NEIGHBOR_QUERY_BATCH_SIZE", batch_size)
    jax.clear_caches()
    return system.collider.create_cross_neighbor_list(pos_a, pos_b, system, 1.0, 4)


def test_same_neighbor_rows_are_exact_across_query_batch_boundary(
    grid_system, monkeypatch
):
    state, system, _ = grid_system
    expected, expected_overflow = _same_neighbors(state, system, state.N, monkeypatch)
    actual, actual_overflow = _same_neighbors(state, system, 2, monkeypatch)
    _, _, naive, naive_overflow = jd.Collider.create("Naive").create_neighbor_list(
        state, system, 1.0, 4
    )

    assert bool(jnp.array_equal(actual, expected))
    assert bool(actual_overflow) == bool(expected_overflow)
    assert _neighbor_sets(actual) == _neighbor_sets(naive)
    assert bool(actual_overflow) == bool(naive_overflow)


def test_cross_neighbor_rows_are_exact_across_query_batch_boundary(
    grid_system, monkeypatch
):
    _, system, collider_module = grid_system
    pos_a = jnp.asarray([[0.1, 0.0], [0.6, 0.0], [1.0, 0.0], [2.0, 0.0], [2.6, 0.0]])
    pos_b = jnp.asarray([[0.0, 0.0], [0.5, 0.0], [0.9, 0.0], [2.5, 0.0]])
    expected, expected_overflow = _cross_neighbors(
        pos_a, pos_b, system, collider_module, pos_a.shape[0], monkeypatch
    )
    actual, actual_overflow = _cross_neighbors(
        pos_a, pos_b, system, collider_module, 2, monkeypatch
    )
    naive, naive_overflow = jd.Collider.create("Naive").create_cross_neighbor_list(
        pos_a, pos_b, system, 1.0, 4
    )

    assert bool(jnp.array_equal(actual, expected))
    assert bool(actual_overflow) == bool(expected_overflow)
    assert _neighbor_sets(actual) == _neighbor_sets(naive)
    assert bool(actual_overflow) == bool(naive_overflow)


def test_zero_width_and_empty_same_query(grid_system):
    state, system, _ = grid_system
    _, _, neighbors, overflow = system.collider.create_neighbor_list(
        state, system, 1.0, 0
    )
    assert neighbors.shape == (state.N, 0)
    assert bool(overflow)

    cross_neighbors, cross_overflow = system.collider.create_cross_neighbor_list(
        state.pos[:2], state.pos[1:3], system, 1.0, 0
    )
    assert cross_neighbors.shape == (2, 0)
    assert bool(cross_overflow)

    empty_state = jd.State.create(
        pos=jnp.empty((0, state.dim)), rad=jnp.empty((0,), dtype=state.rad.dtype)
    )
    _, _, empty_neighbors, empty_overflow = system.collider.create_neighbor_list(
        empty_state, system, 1.0, 4
    )
    assert empty_neighbors.shape == (0, 4)
    assert not bool(empty_overflow)


def test_single_particle_same_and_cross_queries(grid_system, monkeypatch):
    _, system, collider_module = grid_system
    pos = jnp.asarray([[0.0, 0.0]])
    state = jd.State.create(pos=pos, rad=jnp.asarray([0.1]))
    monkeypatch.setattr(collider_module, "NEIGHBOR_QUERY_BATCH_SIZE", 2)
    jax.clear_caches()

    _, _, same, same_overflow = system.collider.create_neighbor_list(
        state, system, 1.0, 1
    )
    cross, cross_overflow = system.collider.create_cross_neighbor_list(
        pos, pos, system, 1.0, 1
    )

    assert bool(jnp.array_equal(same, jnp.asarray([[-1]])))
    assert not bool(same_overflow)
    assert bool(jnp.array_equal(cross, jnp.asarray([[0]])))
    assert not bool(cross_overflow)


def test_real_overflow_is_reduced_across_query_batches(grid_system, monkeypatch):
    _, system, collider_module = grid_system
    pos = jnp.stack((jnp.arange(5, dtype=float) * 0.1, jnp.zeros(5)), axis=-1)
    state = jd.State.create(pos=pos, rad=jnp.full((5,), 0.1))
    monkeypatch.setattr(collider_module, "NEIGHBOR_QUERY_BATCH_SIZE", 2)
    jax.clear_caches()

    _, _, same, same_overflow = system.collider.create_neighbor_list(
        state, system, 1.0, 1
    )
    cross, cross_overflow = system.collider.create_cross_neighbor_list(
        pos, pos, system, 1.0, 1
    )
    _, _, naive_same, _ = jd.Collider.create("Naive").create_neighbor_list(
        state, system, 1.0, 5
    )
    naive_cross, _ = jd.Collider.create("Naive").create_cross_neighbor_list(
        pos, pos, system, 1.0, 5
    )

    assert bool(same_overflow)
    assert bool(cross_overflow)
    for actual, reference in zip(_neighbor_sets(same), _neighbor_sets(naive_same)):
        assert actual <= reference
    for actual, reference in zip(_neighbor_sets(cross), _neighbor_sets(naive_cross)):
        assert actual <= reference


def test_same_and_cross_overflow_rows_preserve_first_k_order(grid_system):
    state, system, _ = grid_system
    _, _, full_same, _ = system.collider.create_neighbor_list(
        state, system, 1.0, state.N
    )
    _, _, short_same, same_overflow = system.collider.create_neighbor_list(
        state, system, 1.0, 2
    )

    pos_a = state.pos[:3]
    pos_b = state.pos
    full_cross, _ = system.collider.create_cross_neighbor_list(
        pos_a, pos_b, system, 1.0, pos_b.shape[0]
    )
    short_cross, cross_overflow = system.collider.create_cross_neighbor_list(
        pos_a, pos_b, system, 1.0, 2
    )

    assert bool(jnp.array_equal(short_same, full_same[:, :2]))
    assert bool(jnp.array_equal(short_cross, full_cross[:, :2]))
    assert bool(same_overflow)
    assert bool(cross_overflow)
