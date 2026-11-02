# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Size automatic static pools from the actual search and reference state."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd

BACKENDS = ["CellList", "MultiCellList"]


def _backend(system):
    return getattr(system.collider, "secondary_collider", system.collider)


def _state(n=32, *, dense=False, dim=2, clumps=None):
    if dense:
        pos = np.full((n, dim), 2.0)
        pos[:, 0] += np.arange(n) * 0.01
    else:
        side = max(8, int(np.ceil(n ** (1 / dim))))
        grid = np.indices((side,) * dim).reshape(dim, -1).T[:n]
        pos = 2.0 + 3.0 * grid
    return jd.State.create(pos=jnp.asarray(pos), rad=jnp.full(n, 0.5), clump_id=clumps)


def _system(state, backend, *, neighbor=False, options=None, **kwargs):
    options = {"cell_size": 1.0, "static": True} | (options or {})
    if neighbor:
        collider_type = "NeighborList"
        options = {
            "cutoff": 1.0,
            "skin": 0.1,
            "max_neighbors": state.N,
            "secondary_collider_type": backend,
            "secondary_collider_kw": options,
        }
    else:
        collider_type = backend
    return jd.System.create(
        state=state,
        collider_type=collider_type,
        collider_kw=options,
        dt=1e-6,
        **kwargs,
    )


def _compare_all_pairs(state, system):
    force, updated = jd.System.initialize(state, system)
    updated.check_overflow()
    if state.N == 0:
        assert force.force.shape == (0, state.dim)
        _, energy_system, energy = system.collider.compute_potential_energy(
            state, system
        )
        energy_system.check_overflow()
        assert float(energy) == 0.0
        return updated
    reference = replace(system, collider=jd.Collider.create("naive"))
    expected, _ = jd.System.initialize(state, reference)
    np.testing.assert_allclose(force.force, expected.force, rtol=2e-5, atol=0.01)
    np.testing.assert_allclose(force.torque, expected.torque, rtol=2e-5, atol=0.01)
    _, energy_system, energy = system.collider.compute_potential_energy(state, system)
    energy_system.check_overflow()
    _, _, expected_energy = reference.collider.compute_potential_energy(
        state, reference
    )
    np.testing.assert_allclose(energy, expected_energy, rtol=2e-5, atol=0.01)
    return updated


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("neighbor", [False, True])
@pytest.mark.parametrize("geometry", ["free", "periodic", "anchored", "sheared"])
def test_automatic_capacity_uses_complete_search_geometry(backend, neighbor, geometry):
    pos = np.random.default_rng(853).uniform(-5.0, 12.0, (24, 2))
    state = jd.State.create(pos=jnp.asarray(pos), rad=jnp.full(24, 0.5))
    domain_type = "free" if geometry == "free" else "periodic"
    domain_kw = {"box_size": jnp.array([6.0, 7.0])}
    if geometry in {"anchored", "sheared"}:
        domain_kw["anchor"] = jnp.array([1.0, -1.0])
    if geometry == "sheared":
        domain_type = "leesedwards"
        domain_kw["gamma"] = 0.37
    system = _system(
        state, backend, neighbor=neighbor, domain_type=domain_type, domain_kw=domain_kw
    )
    assert _backend(system)._auto_capacity
    assert 0 < _backend(system).tile_capacity <= state.N**2
    _compare_all_pairs(state, system)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("neighbor", [False, True])
def test_automatic_capacity_covers_unwrapped_dense_3d_reference(backend, neighbor):
    n = 64
    grid = np.indices((4, 4, 4)).reshape(3, -1).T
    state = jd.State.create(pos=jnp.asarray((grid + 0.5) * 3), rad=jnp.ones(n))
    length = (n * (4 * np.pi / 3) / 0.5) ** (1 / 3)
    system = _system(
        state,
        backend,
        neighbor=neighbor,
        options={"cell_size": 2.1 if neighbor else 2.0},
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(3, length)},
    )
    provisional = jd.Collider.create(
        backend,
        state=state,
        static=True,
        cell_size=2.1 if neighbor else 2.0,
    )
    assert _backend(system).tile_capacity < provisional.tile_capacity
    assert _backend(system).tile_capacity <= n**2
    _compare_all_pairs(state, system)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("neighbor", [False, True])
@pytest.mark.parametrize("law", ["lennardjones", "wca"])
def test_automatic_capacity_uses_force_search_extent(backend, neighbor, law):
    grid = np.indices((8, 8)).reshape(2, -1).T
    distance = 1.08
    state = jd.State.create(pos=jnp.asarray(2 + distance * grid), rad=jnp.full(64, 0.5))
    materials = jd.MaterialTable.from_materials(
        [jd.Material.create("lj", density=1.0, epsilon=1.0)]
    )
    system = _system(
        state,
        backend,
        neighbor=neighbor,
        force_model_type=law,
        mat_table=materials,
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(2, 32.0)},
    )
    assert np.max(system.force_model.search_radii(state, system)) > 0.5
    _compare_all_pairs(state, system)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("neighbor", [False, True])
@pytest.mark.parametrize("external", [False, True])
def test_automatic_capacity_respects_same_clump_exclusion(backend, neighbor, external):
    n = 32
    clumps = jnp.zeros(n, dtype=int)
    if external:
        clumps = clumps.at[-2:].set(jnp.array([n - 2, n - 1]))
    state = _state(n, dense=True, clumps=clumps)
    system = _system(state, backend, neighbor=neighbor, options={"clump_aware": True})
    capacity = _backend(system).tile_capacity
    assert capacity > 0 if external else capacity == 0
    assert capacity <= (4 * n - 6 if external else 0)
    _compare_all_pairs(state, system)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", [0, 1, 2])
def test_automatic_capacity_handles_empty_and_tiny_states(backend, n):
    state = _state(n)
    system = _system(state, backend)
    assert 0 <= _backend(system).tile_capacity <= n**2
    assert (_backend(system).tile_capacity == 0) == (n == 0)
    _compare_all_pairs(state, system)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("neighbor", [False, True])
def test_explicit_capacity_and_dynamic_traversal_remain_explicit(backend, neighbor):
    state = _state(dense=True)
    for capacity in [0, 7, 1024]:
        system = _system(
            state, backend, neighbor=neighbor, options={"tile_capacity": capacity}
        )
        assert _backend(system).tile_capacity == capacity
        assert not _backend(system)._auto_capacity
    dynamic = _system(state, backend, neighbor=neighbor, options={"static": False})
    assert _backend(dynamic).tile_capacity is None
    assert not _backend(dynamic)._auto_capacity


@pytest.mark.parametrize("backend", BACKENDS)
def test_prebuilt_automatic_collider_is_sized_for_system_reference(backend):
    state = _state(dense=True)
    collider = jd.Collider.create(backend, state=state, cell_size=1.0, static=True)
    assert collider._auto_capacity
    system = jd.System.create(state=state, collider=collider)
    assert system.collider.tile_capacity < collider.tile_capacity
    assert system.collider._auto_capacity
    _compare_all_pairs(state, system)
    sparse = _state()
    rebuilt = jd.System.create(state=sparse, collider=system.collider)
    assert rebuilt.collider.tile_capacity < system.collider.tile_capacity
    _compare_all_pairs(sparse, rebuilt)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("neighbor", [False, True])
def test_automatic_batch_capacity_uses_largest_reference_and_remains_static(
    backend, neighbor
):
    sparse = _state()
    dense = _state(dense=True)
    pairs = [(s, _system(s, backend, neighbor=neighbor)) for s in [sparse, dense]]
    capacities = [_backend(y).tile_capacity for _, y in pairs]
    assert capacities[0] < capacities[1]
    states = jd.State.stack([s for s, _ in pairs])
    systems = jd.System.stack([y for _, y in pairs])
    assert _backend(systems).tile_capacity == max(capacities)
    assert _backend(systems)._auto_capacity
    states, systems = jd.System.initialize(states, systems)
    systems.check_overflow()
    states, systems = jd.System.step(states, systems, n=2)
    systems.check_overflow()
    assert _backend(systems).tile_capacity == max(capacities)
    for state, system in zip(jd.State.unstack(states), jd.System.unstack(systems)):
        _compare_all_pairs(state, system)


@pytest.mark.parametrize("backend", BACKENDS)
def test_explicit_pools_are_not_resized_to_stack_with_automatic_pools(backend):
    state = _state()
    auto = _system(state, backend)
    capacity = auto.collider.tile_capacity
    explicit = _system(state, backend, options={"tile_capacity": capacity - 1})
    with pytest.raises(ValueError):
        jd.System.stack([auto, explicit])
    assert explicit.collider.tile_capacity == capacity - 1
    matching = _system(state, backend, options={"tile_capacity": capacity})
    stacked = jd.System.stack([auto, matching])
    assert stacked.collider.tile_capacity == capacity
    assert not stacked.collider._auto_capacity


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("neighbor", [False, True])
def test_automatic_pool_stays_fixed_and_reports_later_extreme_clustering(
    backend, neighbor
):
    sparse = _state(128)
    dense = _state(128, dense=True)
    system = _system(sparse, backend, neighbor=neighbor)
    capacity = _backend(system).tile_capacity
    _, valid = jd.System.initialize(sparse, system)
    valid.check_overflow()
    _, failed = jd.System.initialize(dense, system)
    assert _backend(failed).tile_capacity == capacity
    assert bool(failed.collider.overflow)
    assert bool(failed.search_overflow)
    with pytest.raises(RuntimeError, match="overflow"):
        failed.check_overflow()


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("neighbor", [False, True])
def test_refresh_preserves_automatic_choice_without_resizing_the_pool(
    backend, neighbor
):
    dense = _state(dense=True)
    sparse = _state()
    system = _system(dense, backend, neighbor=neighbor)
    capacity = _backend(system).tile_capacity
    refreshed = jd.colliders.refresh_collider(
        sparse, system.collider, system.force_model
    )
    refreshed_backend = getattr(refreshed, "secondary_collider", refreshed)
    assert refreshed_backend._auto_capacity
    assert refreshed_backend.tile_capacity == capacity
    recreated = jd.System.create(state=sparse, collider=refreshed)
    assert _backend(recreated)._auto_capacity
    assert _backend(recreated).tile_capacity < capacity
    _compare_all_pairs(sparse, recreated)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("neighbor", [False, True])
def test_factory_metadata_preserves_automatic_choice(backend, neighbor):
    state = _state(dense=True)
    system = _system(state, backend, neighbor=neighbor)
    metadata = dict(system.collider.metadata)
    if neighbor:
        secondary = metadata.pop("secondary_collider")
        assert secondary["kw"]["_auto_capacity"]
        metadata["secondary_collider_type"] = secondary["type"]
        metadata["secondary_collider_kw"] = secondary["kw"]
    else:
        assert metadata["_auto_capacity"]
    collider = jd.Collider.create(system.collider.type_name, state=state, **metadata)
    restored = jd.System.create(state=state, collider=collider)
    assert _backend(restored)._auto_capacity
    assert _backend(restored).tile_capacity == _backend(system).tile_capacity
    _compare_all_pairs(state, restored)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("neighbor", [False, True])
def test_earlier_schema3_hdf5_retains_capacity_as_explicit_budget(
    backend, neighbor, tmp_path
):
    h5py = pytest.importorskip("h5py")
    from jaxdem.utils import h5

    state = _state(dense=True)
    system = _system(state, backend, neighbor=neighbor)
    state, system = jd.System.initialize(state, system)
    path = tmp_path / "automatic.h5"
    h5.save((state, system), str(path))
    restored_state, restored_system = h5.load(str(path))
    assert _backend(restored_system)._auto_capacity
    assert _backend(restored_system).tile_capacity == _backend(system).tile_capacity
    np.testing.assert_array_equal(restored_state.pos, state.pos)
    with h5py.File(path, "r+") as file:
        group = file["root/1/collider"]
        if neighbor:
            group = group["secondary_collider"]
        del group["_auto_capacity"]
    with pytest.warns(RuntimeWarning, match="_auto_capacity"):
        restored_state, restored_system = h5.load(str(path))
    assert not _backend(restored_system)._auto_capacity
    assert _backend(restored_system).tile_capacity == _backend(system).tile_capacity
    _compare_all_pairs(restored_state, restored_system)
    recreated = jd.System.create(state=_state(), collider=restored_system.collider)
    assert _backend(recreated).tile_capacity == _backend(system).tile_capacity
    assert not _backend(recreated)._auto_capacity


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("neighbor", [False, True])
def test_large_mixed_radius_reference_does_not_fail_on_provisional_integer_limit(
    backend, neighbor
):
    state = _state(4096, dim=3)
    state = jd.State.create(pos=state.pos, rad=jnp.full(4096, 0.001).at[0].set(1.0))
    options = {"static": True, "cell_size": 0.5}
    if neighbor:
        options = {
            "cutoff": 2.0,
            "skin": 0.1,
            "max_neighbors": 8,
            "secondary_collider_type": backend,
            "secondary_collider_kw": options,
        }
    system = jd.System.create(
        state=state,
        collider_type="NeighborList" if neighbor else backend,
        collider_kw=options,
    )
    index_limit = (1 << (63 if jax.config.jax_enable_x64 else 31)) - 1
    assert _backend(system)._auto_capacity
    assert 0 < _backend(system).tile_capacity < index_limit
    assert _backend(system).tile_capacity <= state.N**2
