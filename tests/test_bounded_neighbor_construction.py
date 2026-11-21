# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check bounded neighbor construction, traversal batching, and cell lookups."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp

import numpy as np
import pytest

import jaxdem as jd
import jaxdem.colliders.cell_list as cell_list_module
import jaxdem.colliders.multi_cell_list as multi_cell_list_module
from jaxdem.colliders._cell_lookup import _cell_starts
from tests.test_sparse_neighbors import case


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


def test_cell_starts_dense_sparse_empty_and_vmap():
    hash_dtype = jnp.uint64 if jax.config.jax_enable_x64 else jnp.uint32
    sentinel = jnp.asarray(jnp.iinfo(hash_dtype).max, dtype=hash_dtype)

    dense_hashes = jnp.asarray([0, 1, 1, 5], dtype=hash_dtype)
    dense_queries = jnp.asarray([0, 1, 2, 5, sentinel], dtype=hash_dtype)
    assert bool(
        jnp.array_equal(
            _cell_starts(dense_hashes, dense_queries),
            jnp.asarray([0, 1, 4, 3, 4]),
        )
    )

    sparse_hashes = jnp.asarray([0, 100], dtype=hash_dtype)
    sparse_queries = jnp.asarray([0, 1, 100, 101], dtype=hash_dtype)
    assert bool(
        jnp.array_equal(
            _cell_starts(sparse_hashes, sparse_queries),
            _searchsorted_starts(sparse_hashes, sparse_queries),
        )
    )
    assert bool(
        jnp.array_equal(
            _cell_starts(jnp.asarray([], dtype=hash_dtype), sparse_queries),
            jnp.zeros(sparse_queries.shape, dtype=jnp.int32),
        )
    )

    vmap_hashes = jnp.asarray([[0, 1, 2], [0, 1, 100]], dtype=hash_dtype)
    vmap_queries = jnp.asarray([[0, 2, 3], [0, 2, 100]], dtype=hash_dtype)
    expected = jax.vmap(_searchsorted_starts)(vmap_hashes, vmap_queries)
    assert bool(
        jnp.array_equal(jax.vmap(_cell_starts)(vmap_hashes, vmap_queries), expected)
    )


def test_same_and_cross_neighbor_rows_are_exact_across_query_batch_boundary(
    grid_system, monkeypatch
):
    state, system, collider_module = grid_system
    pos_a = jnp.asarray([[0.1, 0.0], [0.6, 0.0], [1.0, 0.0], [2.0, 0.0], [2.6, 0.0]])
    pos_b = jnp.asarray([[0.0, 0.0], [0.5, 0.0], [0.9, 0.0], [2.5, 0.0]])
    naive_collider = jd.Collider.create("Naive")

    try:
        with monkeypatch.context() as mp:
            mp.setattr(collider_module, "NEIGHBOR_QUERY_BATCH_SIZE", state.N)
            jax.clear_caches()
            exp_same, exp_same_ov = system.collider.create_neighbor_list(
                state, system, 1.0, 4
            )[2:]
            exp_cross, exp_cross_ov = system.collider.create_cross_neighbor_list(
                pos_a, pos_b, system, 1.0, 4
            )
        with monkeypatch.context() as mp:
            mp.setattr(collider_module, "NEIGHBOR_QUERY_BATCH_SIZE", 2)
            jax.clear_caches()
            act_same, act_same_ov = system.collider.create_neighbor_list(
                state, system, 1.0, 4
            )[2:]
            act_cross, act_cross_ov = system.collider.create_cross_neighbor_list(
                pos_a, pos_b, system, 1.0, 4
            )
    finally:
        jax.clear_caches()

    _, _, naive_same, naive_same_ov = naive_collider.create_neighbor_list(
        state, system, 1.0, 4
    )
    naive_cross, naive_cross_ov = naive_collider.create_cross_neighbor_list(
        pos_a, pos_b, system, 1.0, 4
    )

    assert bool(jnp.array_equal(act_same, exp_same))
    assert bool(act_same_ov) == bool(exp_same_ov) == bool(naive_same_ov)
    assert _neighbor_sets(act_same) == _neighbor_sets(naive_same)

    assert bool(jnp.array_equal(act_cross, exp_cross))
    assert bool(act_cross_ov) == bool(exp_cross_ov) == bool(naive_cross_ov)
    assert _neighbor_sets(act_cross) == _neighbor_sets(naive_cross)


def test_zero_width_empty_and_single_particle_queries(grid_system):
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

    single_pos = jnp.asarray([[0.0, 0.0]])
    single_state = jd.State.create(pos=single_pos, rad=jnp.asarray([0.1]))
    _, _, same, same_overflow = system.collider.create_neighbor_list(
        single_state, system, 1.0, 1
    )
    cross, cross_overflow = system.collider.create_cross_neighbor_list(
        single_pos, single_pos, system, 1.0, 1
    )
    assert bool(jnp.array_equal(same, jnp.asarray([[-1]])))
    assert not bool(same_overflow)
    assert bool(jnp.array_equal(cross, jnp.asarray([[0]])))
    assert not bool(cross_overflow)

    for n in (0, 1):
        s_case, y_case = case(pos=np.zeros((n, 2)), capacity=0)
        y_case = replace(
            y_case,
            collider=jd.Collider.create(system.collider.type_name, state=s_case),
        )
        s_case, y_case = jd.System.initialize(s_case, y_case)
        s_case, y_case = jd.System.step(s_case, y_case, n=2)
        np.testing.assert_array_equal(s_case.force, 0.0)
        assert y_case.collider.compute_potential_energy(s_case, y_case)[2] == 0
        y_case.check_overflow()


@pytest.mark.parametrize("crowded_start", [0, 8])
def test_real_overflow_is_reduced_across_query_batches(
    grid_system, monkeypatch, crowded_start
):
    _, system, collider_module = grid_system
    # Only the first or partial final batch overflows; all others fit.
    pos = jnp.stack((jnp.arange(11, dtype=float) * 10, jnp.zeros(11)), axis=-1)
    pos = pos.at[crowded_start : crowded_start + 3, 0].set(
        10 * crowded_start + jnp.array([0.0, 0.1, 0.2])
    )
    state = jd.State.create(pos=pos, rad=jnp.full((11,), 0.1))
    try:
        with monkeypatch.context() as mp:
            mp.setattr(collider_module, "NEIGHBOR_QUERY_BATCH_SIZE", 4)
            jax.clear_caches()
            _, _, same, same_overflow = system.collider.create_neighbor_list(
                state, system, 1.0, 1
            )
            cross, cross_overflow = system.collider.create_cross_neighbor_list(
                pos, pos, system, 1.0, 1
            )
    finally:
        jax.clear_caches()

    assert bool(same_overflow)
    assert bool(cross_overflow)
    distances = np.linalg.norm(
        np.asarray(pos)[:, None] - np.asarray(pos)[None, :], axis=-1
    )
    for output, include_self in [(same, False), (cross, True)]:
        for i, row in enumerate(np.asarray(output)):
            wanted = set(
                np.flatnonzero(
                    (distances[i] <= 1.0) & (include_self | (np.arange(11) != i))
                )
            )
            valid = row[row >= 0]
            assert len(valid) == min(1, len(wanted))
            assert set(valid) <= wanted
            assert np.all(row[len(valid) :] == -1)


@pytest.mark.parametrize(
    "collider,domain",
    [("CellList", "periodic"), ("MultiCellList", "lees_edwards")],
)
def test_cell_batches_match_all_pairs_and_vmap(collider, domain, monkeypatch):
    pos = np.random.default_rng(45).uniform(0.0, 3.8, (11, 3))
    s, y = case(pos=pos, domain=domain)
    y = replace(y, collider=jd.Collider.create(collider, state=s, cell_size=1.15))
    ref = replace(y, collider=jd.Collider.create("naive"))
    try:
        with monkeypatch.context() as mp:
            mp.setattr(cell_list_module, "PAIR_TRAVERSAL_BATCH_SIZE", 4)
            mp.setattr(multi_cell_list_module, "PAIR_TRAVERSAL_BATCH_SIZE", 4)
            jax.clear_caches()
            got, gy = jd.System.initialize(s, y)
            want, wy = jd.System.initialize(s, ref)
            np.testing.assert_allclose(got.force, want.force, rtol=2e-5, atol=0.01)
            energy = lambda s, y: y.collider.compute_potential_energy(s, y)[2]
            np.testing.assert_allclose(
                energy(got, gy), energy(want, wy), rtol=2e-5, atol=0.01
            )
            other = replace(s, pos_c=s.pos_c * 0.7)
            states = jd.State.stack([s, other])
            systems = jd.System.stack([y, y])
            batched, by = jd.System.initialize(states, systems)
            batched, by = jd.System.step(batched, by, n=2)
    finally:
        jax.clear_caches()

    for i, one in enumerate([s, other]):
        expected, ey = jd.System.initialize(one, ref)
        expected, ey = jd.System.step(expected, ey, n=2)
        np.testing.assert_allclose(batched.pos[i], expected.pos, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(batched.vel[i], expected.vel, rtol=2e-5, atol=2e-6)
    assert not jnp.any(by.collider.overflow)
