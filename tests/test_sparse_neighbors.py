# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check pooled sparse neighbor capacity, contact history, and force equivalence."""

import jax
import jax.numpy as jnp

from dataclasses import dataclass, replace

import numpy as np
import pytest

import jaxdem as jd
from jaxdem.colliders._neighbor_cache import pair_sources, remap_history


def case(capacity=3, backend="CellList", domain="free", pos=None, force=None):
    if pos is None:
        pos = [[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [8.0, 0.0], [16.0, 0.0], [32.0, 0.0]]
    state = jd.State.create(pos=jnp.asarray(pos), rad=jnp.full(len(pos), 0.5))
    system = jd.System.create(
        state=state,
        force_model=force,
        dt=1e-5,
        collider_type="NeighborList",
        collider_kw={
            "cutoff": 1.0,
            "skin": 0.1,
            "max_neighbors": capacity,
            "secondary_collider_type": backend,
        },
        domain_type=domain,
        domain_kw={"box_size": jnp.full(state.dim, 4.0)},
    )
    return state, system


@pytest.mark.parametrize("backend", ["CellList", "MultiCellList", "naive"])
def test_shared_capacity_has_no_per_particle_limit(backend):
    state, system = case(capacity=1, backend=backend)
    state, system = system.collider.compute_force(state, system)
    col = system.collider
    assert not col.overflow
    system.check_overflow()
    assert col.neighbor_list.shape == (6,)
    np.testing.assert_array_equal(col.row_offsets, [0, 2, 4, 6, 6, 6, 6])
    for i, wanted in enumerate(({1, 2}, {0, 2}, {0, 1})):
        assert (
            set(
                np.asarray(
                    col.neighbor_list[col.row_offsets[i] : col.row_offsets[i + 1]]
                )
            )
            == wanted
        )


def test_force_cache_uses_pair_specific_search_radii_after_invalidation():
    state = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [1.0, 0.0], [1.35, 0.0]]),
        rad=jnp.array([1.0, 0.1, 0.1]),
    )
    system = jd.System.create(
        state=state,
        collider_type="NeighborList",
        collider_kw={
            "cutoff": 2.0,
            "skin": 0.05,
            "max_neighbors": 3,
            "secondary_collider_type": "naive",
        },
    )
    actual, system = jd.System.initialize(state, system)
    np.testing.assert_array_equal(system.collider.row_offsets, [0, 1, 2, 2])
    np.testing.assert_array_equal(system.collider.neighbor_list[:2], [1, 0])
    reference, _ = jd.System.initialize(
        state, replace(system, collider=jd.Collider.create("naive"))
    )
    np.testing.assert_allclose(actual.force, reference.force)

    previous_builds = int(system.collider.n_build_times)
    expanded_radii = jnp.array([1.0, 0.1, 0.4])
    expanded = replace(state, rad=expanded_radii, _rad=expanded_radii)
    invalidated = replace(system, collider=system.collider.invalidate())
    _, rebuilt = invalidated.collider.compute_force(expanded, invalidated)
    assert int(rebuilt.collider.n_build_times) == previous_builds + 1
    np.testing.assert_array_equal(rebuilt.collider.row_offsets, [0, 2, 4, 6])


@pytest.mark.parametrize("capacity,n", [(0, 3), (1, 3), (2, 3), (8, 3), (1, 5)])
def test_global_overflow_and_padding(capacity, n):
    pos = [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]] + [[10.0 * i, 0.0] for i in range(3, n)]
    state, system = case(capacity=capacity, pos=pos)
    state, system = system.collider.compute_force(state, system)
    col = system.collider
    assert bool(col.overflow) == (n * capacity < 6)
    assert col.row_offsets[-1] == min(6, n * capacity)
    assert jnp.all(col.neighbor_list[col.row_offsets[-1] :] == -1)
    assert jnp.all(pair_sources(col)[col.row_offsets[-1] :] == state.N)
    assert jnp.all(jnp.isfinite(state.force))
    if col.overflow:
        with pytest.raises(RuntimeError, match="overflow"):
            system.check_overflow()
    else:
        system.check_overflow()


@pytest.mark.parametrize("backend", ["CellList", "MultiCellList", "naive"])
def test_default_sparse_overflow_tracks_total_pairs_through_rebuilds(backend):
    # Six slots: the triangle fills them exactly despite two neighbors per row.
    s, y = case(capacity=1, backend=backend)
    col = jd.colliders.NeighborList.Create(
        s, cutoff=1.0, skin=0.1, max_neighbors=1, secondary_collider_type=backend
    )
    y = jd.System.create(state=s, collider=col)
    s, y = jd.System.initialize(s, y)
    y.check_overflow()
    assert y.collider.row_offsets[-1] == 6

    # A fourth particle joins the triangle: 12 directed pairs exceed six slots.
    crowded = replace(s, pos_c=s.pos_c.at[3].set(jnp.array([0.3, 0.3])))
    _, full = y.collider.compute_force(crowded, y)
    assert full.collider.overflow
    assert full.collider.row_offsets[-1] == 6
    with pytest.raises(RuntimeError, match="overflow"):
        full.check_overflow()
    _, stepped = jd.System.step(crowded, y)
    assert stepped.search_overflow
    with pytest.raises(RuntimeError, match="overflow"):
        stepped.check_overflow()

    # Rebuilding after separation must clear the global overflow flag.
    _, recovered = full.collider.compute_force(
        s, replace(full, collider=full.collider.invalidate())
    )
    recovered.check_overflow()
    assert recovered.collider.row_offsets[-1] == 6


@pytest.mark.parametrize("backend", ["CellList", "MultiCellList", "naive"])
@pytest.mark.parametrize("domain", ["free", "periodic", "lees_edwards"])
def test_force_energy_and_trajectory_match_all_pairs(backend, domain):
    pos = np.random.default_rng(3).uniform(0.0, 3.8, (12, 3))
    s, y = case(capacity=12, backend=backend, domain=domain, pos=pos)
    ds, dy = s, replace(y, collider=jd.Collider.create("naive"))
    s, y = jd.System.initialize(s, y)
    ds, dy = jd.System.initialize(ds, dy)
    assert not y.collider.overflow
    np.testing.assert_allclose(s.force, ds.force, rtol=2e-5, atol=0.01)
    _, _, e = y.collider.compute_potential_energy(s, y)
    _, _, de = dy.collider.compute_potential_energy(ds, dy)
    np.testing.assert_allclose(e, de, rtol=2e-5, atol=0.01)
    s, y = jd.System.step(s, y, n=3)
    ds, dy = jd.System.step(ds, dy, n=3)
    np.testing.assert_allclose(s.pos, ds.pos, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(s.vel, ds.vel, rtol=2e-5, atol=1e-6)


@jax.tree_util.register_dataclass
@jd.ForceModel.register("sparse-remember-test")
@dataclass(slots=True)
class Remember(jd.forces.SpringForce):
    def history_shape(self, dim):
        return (2, 2)

    def init_history(self, pair_shape, dim):
        return jnp.full(pair_shape + (2, 2), 7.0)

    @staticmethod
    def force(
        i, j, pos, state, system, history, *, advance_history=True, quasistatic=False
    ):
        f, t, _ = jd.forces.SpringForce.force(i, j, pos, state, system, history)
        return f, t, history + advance_history


def test_history_survives_rebuild_resize_and_query():
    state, system = case(force=Remember())
    _, system = jd.System.evaluate_forces(state, system)
    col = system.collider
    neighbors, history = (
        np.asarray(col.neighbor_list).copy(),
        np.asarray(col.history).copy(),
    )
    remembered = {}
    for i in range(state.N):
        start, end = map(int, col.row_offsets[i : i + 2])
        for slot in range(start, end):
            j = int(neighbors[slot])
            history[slot] = 100 * i + 10 * j + np.arange(4).reshape(2, 2)
            remembered[i, j] = history[slot].copy()
        # Reorder both buffers within a row before asking the rebuild to remap pairs.
        neighbors[start:end] = neighbors[start:end][::-1]
        history[start:end] = history[start:end][::-1]
    assert remembered.keys() == {(i, j) for i in [0, 1, 2] for j in [0, 1, 2] if i != j}
    system = replace(
        system,
        collider=replace(
            col, neighbor_list=jnp.asarray(neighbors), history=jnp.asarray(history)
        ).invalidate(),
    )
    state = replace(
        state,
        pos_c=state.pos_c.at[2]
        .set(jnp.array([12.0, 4.0]))
        .at[3]
        .set(jnp.array([0.25, -0.25])),
    )
    state, system = jd.System.evaluate_forces(state, system)
    wanted = {(i, j) for i in [0, 1, 3] for j in [0, 1, 3] if i != j}
    expected = {pair: remembered.get(pair, np.full((2, 2), 7.0)) for pair in wanted}

    def check_pairs(collider, increment=0):
        assert int(collider.row_offsets[-1]) == len(wanted)
        found = {}
        for i in range(state.N):
            for slot in range(
                int(collider.row_offsets[i]), int(collider.row_offsets[i + 1])
            ):
                found[i, int(collider.neighbor_list[slot])] = np.asarray(
                    collider.history[slot]
                )
        assert found.keys() == wanted
        for pair in wanted:
            np.testing.assert_array_equal(found[pair], expected[pair] + increment)

    check_pairs(system.collider)
    _, _, query, overflow = system.collider.create_neighbor_list(state, system, 0.6, 3)
    assert not overflow
    assert {
        (i, int(j)) for i, row in enumerate(np.asarray(query)) for j in row if j >= 0
    } == wanted
    assert np.count_nonzero(np.asarray(query) >= 0) == len(wanted)
    queried = np.asarray(system.collider.get_history(state, system, query))
    for i, row in enumerate(np.asarray(query)):
        for slot, j in enumerate(row):
            np.testing.assert_array_equal(
                queried[i, slot], expected.get((i, int(j)), np.full((2, 2), 7.0))
            )
    grown = jd.colliders.refresh_collider(
        state, replace(system.collider, max_neighbors=4), system.force_model
    )
    system = replace(system, collider=grown)
    _, system = system.collider.compute_force(state, system)
    check_pairs(system.collider, increment=1)
    with pytest.raises(ValueError, match="reset_history"):
        jd.colliders.refresh_collider(
            state, replace(system.collider, max_neighbors=1), system.force_model
        )


def test_pair_remap_handles_scalar_history_and_reordered_pairs():
    actual = remap_history(
        jnp.array([10.0, 20.0, 30.0, 40.0]),
        jnp.array([0, 0, 1, 4]),
        jnp.array([2, 1, 0, -1]),
        jnp.array([1, 0, 0, 2, 4]),
        jnp.array([0, 1, 2, 0, -1]),
        jnp.full(5, 7.0),
    )
    np.testing.assert_array_equal(actual, [30.0, 20.0, 10.0, 7.0, 7.0])


def test_total_capacity_overflow_does_not_wrap_int32():
    from jaxdem.colliders._neighbor_cache import _capacity_offsets

    offsets, overflow = _capacity_offsets(
        jnp.full(4, 1_500_000_000, dtype=jnp.int32), 100
    )
    assert overflow
    np.testing.assert_array_equal(offsets, [0, 100, 100, 100, 100])


def test_dense_cluster_borrows_capacity_from_dilute_particles():
    pos = np.zeros((72, 2))
    pos[:12, 0] = np.linspace(0.0, 0.4, 12)
    pos[12:, 0] = 10 + np.arange(60) * 3
    s, y = case(capacity=2, pos=pos)
    s, y = y.collider.compute_force(s, y)
    assert not y.collider.overflow
    assert y.collider.row_offsets[-1] == 12 * 11
    np.testing.assert_array_equal(np.diff(y.collider.row_offsets)[:12], 11)
    ds, dy = s, replace(y, collider=jd.Collider.create("naive"))
    ds, dy = dy.collider.compute_force(ds, dy)
    np.testing.assert_allclose(s.force, ds.force, rtol=2e-5, atol=0.01)


def test_system_rejects_cache_for_different_particle_count():
    _, y = case()
    other = jd.State.create(pos=jnp.zeros((2, 2)))
    with pytest.raises(ValueError, match="history has shape"):
        jd.System.create(state=other, collider=y.collider)


def test_empty_and_no_neighbors():
    s, y = case(capacity=0, backend="naive", pos=np.empty((0, 2)))
    s, y = y.collider.compute_force(s, y)
    assert s.force.shape == (0, 2)
    assert not y.collider.overflow
    s, y = case(capacity=2, pos=[[0.0, 0.0], [5.0, 0.0]])
    s, y = y.collider.compute_force(s, y)
    np.testing.assert_array_equal(s.force, 0.0)
    assert not y.collider.overflow


def test_sparse_checkpoint_roundtrip(tmp_path):
    pytest.importorskip("orbax.checkpoint")
    s, y = case(force=Remember())
    s, y = jd.System.initialize(s, y)
    with jd.CheckpointWriter(tmp_path / "checkpoint") as writer:
        writer.save(s, y)
    rs, ry = jd.CheckpointLoader(tmp_path / "checkpoint").load()
    s, y = jd.System.step(s, y, n=2)
    rs, ry = jd.System.step(rs, ry, n=2)
    np.testing.assert_allclose(s.pos, rs.pos)
    np.testing.assert_allclose(y.collider.history, ry.collider.history)


@pytest.mark.parametrize("n", [0, 1, 5])
def test_row_energy_gradients_match_all_pairs(n):
    pos = np.random.default_rng(19).uniform(0.0, 0.7, (n, 2))
    s, y = case(capacity=n, backend="naive", domain="periodic", pos=pos)
    s, y = jd.System.initialize(s, y)
    reference = replace(y, collider=jd.Collider.create("naive"))

    def observable(pos, system):
        moved = replace(s, pos_c=pos)
        return system.collider.compute_potential_energy(moved, system)[2]

    row = lambda p: observable(p, y)
    direct = lambda p: observable(p, reference) if n else jnp.sum(p * 0.0)
    np.testing.assert_allclose(
        jax.grad(row)(s.pos_c), jax.grad(direct)(s.pos_c), rtol=2e-5, atol=0.01
    )
    direction = jnp.arange(s.pos_c.size, dtype=s.pos_c.dtype).reshape(s.pos_c.shape)
    np.testing.assert_allclose(
        jax.jvp(row, (s.pos_c,), (direction,))[1],
        jax.jvp(direct, (s.pos_c,), (direction,))[1],
        rtol=2e-5,
        atol=0.01,
    )


@pytest.mark.parametrize("n", [3, 9, 17])
def test_stateless_rows_match_all_pairs_for_uneven_degrees(n):
    # Empty, partially filled, and several full rows in one cache.
    pos = np.zeros((n + 2, 3))
    pos[:n, 0] = np.linspace(0.0, 0.7, n)
    pos[n:, 0] = [10.0, 20.0]
    s, y = case(capacity=n, pos=pos)
    rs, ry = jd.System.initialize(s, y)
    ss, sy = jd.System.initialize(s, replace(y, collider=jd.Collider.create("naive")))
    np.testing.assert_allclose(rs.force, ss.force, rtol=2e-5, atol=0.01)
    np.testing.assert_allclose(rs.torque, ss.torque, rtol=2e-5, atol=0.01)
    assert not ry.collider.overflow and not sy.collider.overflow


def test_friction_history_and_torque_survive_unnecessary_rebuild():
    from tests.test_cundall_strack_history import _system

    s = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.8, 0.0], [0.0, 0.8]]),
        rad=jnp.full(3, 0.5),
        vel=jnp.array([[0.0, 0.1], [0.1, 0.0], [0.0, -0.1]]),
    )
    base = _system(s, dt=0.001)
    systems = []
    for rebuild in (False, True):
        col = jd.colliders.NeighborList.Create(
            s,
            cutoff=1.0,
            skin=0.1,
            max_neighbors=3,
        )
        y = jd.System.create(
            state=s,
            collider=col,
            force_model=base.force_model,
            mat_table=base.mat_table,
            dt=0.001,
        )
        st, y = jd.System.initialize(s, y)
        st, y = jd.System.step(st, y, n=3)
        if rebuild:
            y = replace(y, collider=y.collider.invalidate())
        st, y = jd.System.step(st, y, n=2)
        _, _, query, _ = y.collider.create_neighbor_list(st, y, 1.1, 3)
        systems.append((st, y.collider.get_history(st, y, query)))
    np.testing.assert_allclose(
        systems[0][0].force, systems[1][0].force, rtol=2e-5, atol=1e-6
    )
    np.testing.assert_allclose(
        systems[0][0].torque, systems[1][0].torque, rtol=2e-5, atol=1e-6
    )
    # Both paths query the same final pairs and must preserve pair memory.
    np.testing.assert_allclose(systems[0][1], systems[1][1], rtol=2e-5, atol=1e-6)
