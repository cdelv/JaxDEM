# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
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
    def force(i, j, pos, state, system, history, *, advance_history=True):
        f, t, _ = jd.forces.SpringForce.force(
            i, j, pos, state, system, history, advance_history=advance_history
        )
        return f, t, history + advance_history


def test_history_survives_rebuild_resize_and_query():
    state, system = case(force=Remember())
    state, system = system.collider.compute_force(state, system)
    col = system.collider
    assert col.history.shape == (18, 2, 2)
    np.testing.assert_array_equal(col.history[:6], 8.0)
    np.testing.assert_array_equal(col.history[6:], 7.0)
    original = col.history
    system = replace(system, collider=col.invalidate())
    state, system = system.collider.evaluate_force(state, system)
    np.testing.assert_array_equal(system.collider.history, original)
    _, _, query, _ = system.collider.create_neighbor_list(state, system, 0.6, 3)
    queried = system.collider.get_history(state, system, query)
    np.testing.assert_array_equal(queried[query >= 0], 8.0)
    np.testing.assert_array_equal(queried[query < 0], 7.0)
    grown = replace(system.collider, max_neighbors=4)
    grown = jd.colliders.refresh_collider(state, grown, system.force_model)
    assert grown.neighbor_list.shape == (24,)
    system = replace(system, collider=grown)
    state, system = system.collider.compute_force(state, system)
    np.testing.assert_array_equal(system.collider.history[:6], 9.0)
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


def test_vmap_sparse_caches():
    s, y = case()
    bs, by = jax.tree.map(lambda x: jnp.stack((x, x)), (s, y))
    bs, by = jax.jit(jax.vmap(jd.System.initialize))(bs, by)
    assert by.collider.neighbor_list.shape == (2, 18)
    np.testing.assert_allclose(bs.force[0], bs.force[1])


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


@pytest.mark.parametrize("advance", [False, True])
def test_row_history_updates_match_pair_reference(advance):
    from tests.test_neighbor_force_blocks import _pair_reference

    s, y = case(force=Remember())
    s, y = y.collider.compute_force(s, y)
    f, t, h = _pair_reference(s, y, advance)
    actual, updated = y.collider.compute_force(s, y, advance_history=advance)
    np.testing.assert_allclose(actual.force, f, rtol=2e-5, atol=0.01)
    np.testing.assert_allclose(actual.torque, t, rtol=2e-5, atol=0.01)
    np.testing.assert_array_equal(updated.collider.history, h)


def test_csr_sources_with_empty_rows_and_padding():
    s, y = case(pos=[[5.0, 0.0], [0.0, 0.0], [10.0, 0.0], [0.5, 0.0]])
    s, y = jd.System.initialize(s, y)
    np.testing.assert_array_equal(y.collider.row_offsets, [0, 0, 1, 1, 2])
    np.testing.assert_array_equal(pair_sources(y.collider), [1, 3] + [4] * 10)


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
def test_stateless_rows_match_all_pairs_for_uneven_degrees(n, monkeypatch):
    from jaxdem.colliders import _neighbor_cache

    # Exercise multiple complete batches and a partial batch with a small case.
    monkeypatch.setattr(_neighbor_cache, "_ROW_BATCH_SIZE", 4)
    # Empty, partially filled, and several full force blocks in one cache.
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
