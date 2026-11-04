# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Reject incomplete static searches at force, query, and host boundaries."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.colliders._neighbor_cache import build_pairs, count_pairs


def _case(backend, tiles, *, neighbor=False, clump_id=None, sparse=False):
    n = 5 if clump_id is None else len(clump_id)
    spacing = 2.0 if sparse else 0.05
    pos = jnp.column_stack((2.0 + spacing * jnp.arange(n), jnp.full(n, 2.0)))
    state = jd.State.create(pos=pos, rad=jnp.full(n, 0.5), clump_id=clump_id)
    options = {"cell_size": 1.0, "tile_capacity": tiles}
    if clump_id is not None:
        options["clump_aware"] = True
    if neighbor:
        collider_type = "NeighborList"
        options = {
            "cutoff": 1.0,
            "skin": 0.1,
            "max_neighbors": n - 1,
            "secondary_collider_type": backend,
            "secondary_collider_kw": options,
        }
    else:
        collider_type = backend
    system = jd.System.create(
        state=state,
        dt=1e-6,
        collider_type=collider_type,
        collider_kw=options,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(2, 32.0)},
    )
    return state, system


def _pairs(neighbors, offsets):
    return {
        (i, int(j))
        for i in range(len(offsets) - 1)
        for j in np.asarray(neighbors)[int(offsets[i]) : int(offsets[i + 1])]
    }


@pytest.mark.parametrize("backend", ["CellList", "MultiCellList"])
@pytest.mark.parametrize("tiles", [0, 9, 10])
def test_static_tile_overflow_reaches_force_energy_count_pack_and_query(backend, tiles):
    state, system = _case(backend, tiles)
    force, forced = system.collider.compute_force(state, system)
    _, energized, energy = system.collider.compute_potential_energy(state, system)
    counts, count_overflow = count_pairs(state, system, 1.0)
    neighbors, offsets, pack_overflow = build_pairs(state, system, 1.0, 20)
    _, _, rows, query_overflow = system.collider.create_neighbor_list(
        state, system, 1.0, 4
    )
    cross, cross_overflow = system.collider.create_cross_neighbor_list(
        state.pos, state.pos, system, 1.0, 5
    )
    overflow = tiles < 10
    for flag in (
        forced.collider.overflow,
        energized.collider.overflow,
        count_overflow,
        pack_overflow,
        query_overflow,
        cross_overflow,
    ):
        assert bool(flag) == overflow
    assert np.isfinite(force.force).all()
    assert np.isfinite(energy)
    if overflow:
        with pytest.raises(RuntimeError, match="overflow"):
            forced.check_overflow()
        with pytest.raises(RuntimeError, match="overflow"):
            energized.check_overflow()
    else:
        forced.check_overflow()
        energized.check_overflow()
        np.testing.assert_array_equal(counts, 4)
        expected = {(i, j) for i in range(5) for j in range(5) if i != j}
        assert _pairs(neighbors, offsets) == expected
        assert {(i, int(j)) for i, row in enumerate(rows) for j in row} == expected
        assert all(set(np.asarray(row)) == set(range(5)) for row in cross)
        reference = replace(system, collider=jd.Collider.create("naive"))
        ref_force, _ = reference.collider.compute_force(state, reference)
        _, _, ref_energy = reference.collider.compute_potential_energy(state, reference)
        np.testing.assert_allclose(force.force, ref_force.force, rtol=2e-5, atol=0.01)
        np.testing.assert_allclose(energy, ref_energy, rtol=2e-5, atol=0.01)


@pytest.mark.parametrize("backend", ["CellList", "MultiCellList"])
@pytest.mark.parametrize("capacity", [0, 19, 20])
def test_static_pair_pool_overflow_is_independent_of_tile_capacity(backend, capacity):
    state, system = _case(backend, 10)
    counts, overflow = count_pairs(state, system, 1.0)
    assert not bool(overflow)
    np.testing.assert_array_equal(counts, 4)
    neighbors, offsets, overflow = build_pairs(state, system, 1.0, capacity)
    assert bool(overflow) == (capacity < 20)
    assert neighbors.shape == (capacity,)
    assert int(offsets[-1]) == min(capacity, 20)
    assert np.all(np.diff(offsets) >= 0)
    assert np.isin(np.asarray(neighbors), np.arange(state.N)).all()
    if capacity == 20:
        assert _pairs(neighbors, offsets) == {
            (i, j) for i in range(5) for j in range(5) if i != j
        }
    _, _, rows, row_overflow = system.collider.create_neighbor_list(
        state, system, 1.0, 3
    )
    assert bool(row_overflow)
    assert rows.shape == (5, 3)


@pytest.mark.parametrize("backend", ["CellList", "MultiCellList"])
@pytest.mark.parametrize("neighbor", [False, True])
@pytest.mark.parametrize("batch", [False, True])
def test_static_transient_overflow_remains_sticky_after_valid_steps(
    backend, neighbor, batch
):
    sparse, system = _case(backend, 5, neighbor=neighbor, sparse=True)
    dense, _ = _case(backend, 5, neighbor=neighbor)
    if batch:
        state = jd.State.stack([sparse, dense, sparse])
        system = jd.System.stack([system, system, system])
        expected = [False, True, False]
    else:
        state = dense
        expected = True
    state, system = jd.System.initialize(state, system)
    np.testing.assert_array_equal(system.collider.overflow, expected)
    np.testing.assert_array_equal(system.search_overflow, expected)
    with pytest.raises(RuntimeError, match="overflow"):
        system.check_overflow()

    state = replace(
        state,
        pos_c=jnp.broadcast_to(sparse.pos_c, state.pos_c.shape),
        force=jnp.zeros_like(state.force),
        torque=jnp.zeros_like(state.torque),
        vel=jnp.zeros_like(state.vel),
    )
    state, system = jd.System.step(state, system, n=2)
    np.testing.assert_array_equal(system.collider.overflow, False)
    np.testing.assert_array_equal(system.search_overflow, expected)
    _, system = jd.System.evaluate_forces(state, system)
    np.testing.assert_array_equal(system.collider.overflow, False)
    np.testing.assert_array_equal(system.search_overflow, expected)
    with pytest.raises(RuntimeError, match="overflow"):
        system.check_overflow()


@pytest.mark.parametrize("backend", ["CellList", "MultiCellList"])
def test_static_count_pack_and_query_keep_overflow_per_replica(backend, monkeypatch):
    from jaxdem.colliders import _static_cell

    monkeypatch.setattr(_static_cell, "_TILE_BATCH_SIZE", 4)
    jax.clear_caches()
    sparse, system = _case(backend, 5, sparse=True)
    dense, _ = _case(backend, 5)
    states = jd.State.stack([sparse, dense, sparse])
    systems = jd.System.stack([system, system, system])
    counts, overflow = jax.jit(jax.vmap(lambda s, y: count_pairs(s, y, 1.0)))(
        states, systems
    )
    np.testing.assert_array_equal(overflow, [False, True, False])
    np.testing.assert_array_equal(counts[[0, 2], :], 0)
    _, _, overflow = jax.jit(jax.vmap(lambda s, y: build_pairs(s, y, 1.0, 20)))(
        states, systems
    )
    np.testing.assert_array_equal(overflow, [False, True, False])
    query = lambda s, y: y.collider.create_neighbor_list(s, y, 1.0, 4)[2:]
    _, overflow = jax.jit(jax.vmap(query))(states, systems)
    np.testing.assert_array_equal(overflow, [False, True, False])


@pytest.mark.parametrize("backend", ["CellList", "MultiCellList"])
@pytest.mark.parametrize("neighbor", [False, True])
def test_static_hash_overflow_reaches_host_and_query_flags(backend, neighbor):
    state, system = _case(backend, 100, neighbor=neighbor)
    box = jnp.full(2, 1e20)
    system = replace(
        system, domain=replace(system.domain, box_size=box, inv_box_size=1 / box)
    )
    _, forced = jd.System.initialize(state, system)
    assert bool(forced.collider.overflow)
    assert bool(forced.search_overflow)
    with pytest.raises(RuntimeError, match="overflow"):
        forced.check_overflow()
    inner = (
        replace(system, collider=system.collider.secondary_collider)
        if neighbor
        else system
    )
    _, count_overflow = count_pairs(state, inner, 1.0)
    _, _, pack_overflow = build_pairs(state, inner, 1.0, 20)
    _, _, _, query_overflow = system.collider.create_neighbor_list(
        state, system, 1.0, 4
    )
    assert bool(count_overflow)
    assert bool(pack_overflow)
    assert bool(query_overflow)


@pytest.mark.parametrize("backend", ["CellList", "MultiCellList"])
@pytest.mark.parametrize("external", [False, True])
def test_static_clump_exclusion_preserves_overflow_detection(backend, external):
    labels = [0] * 10 + ([10, 11] if external else [])
    state, system = _case(backend, 0, clump_id=jnp.asarray(labels))
    _, forced = jd.System.initialize(state, system)
    counts, overflow = count_pairs(state, system, 1.0)
    _, _, pair_overflow = build_pairs(state, system, 1.0, 100)
    assert bool(forced.collider.overflow) == external
    assert bool(overflow) == external
    assert bool(pair_overflow) == external
    if not external:
        forced.check_overflow()
        np.testing.assert_array_equal(counts, 0)
    else:
        with pytest.raises(RuntimeError, match="overflow"):
            forced.check_overflow()
        collider = replace(system.collider, tile_capacity=16)
        system = replace(system, collider=collider)
        force, complete = jd.System.initialize(state, system)
        complete.check_overflow()
        counts, overflow = count_pairs(state, system, 1.0)
        assert not bool(overflow)
        np.testing.assert_array_equal(counts, [2] * 10 + [11, 11])
        reference = replace(system, collider=jd.Collider.create("naive"))
        reference_force, _ = jd.System.initialize(state, reference)
        np.testing.assert_allclose(
            force.force, reference_force.force, rtol=2e-5, atol=0.01
        )
