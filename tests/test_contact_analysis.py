# SPDX-License-Identifier: BSD-3-Clause
"""Contact analysis follows simulation searches without diagnostic capacities."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.colliders._neighbor_cache import pair_sources
from jaxdem.utils import contacts as contact
from jaxdem.utils.linalg import cross


BACKENDS = ("naive", "CellList", "MultiCellList", "neighbor-cell", "neighbor-multi")


def make_system(state, backend="naive", *, capacity=None, domain="periodic"):
    options = {}
    if backend.startswith("neighbor-"):
        options = {
            "max_neighbors": state.N if capacity is None else capacity,
            "cutoff": 2 * float(jnp.max(state.rad, initial=0.0)),
            "skin": 0.05,
            "secondary_collider_type": "CellList"
            if backend == "neighbor-cell"
            else "MultiCellList",
        }
        backend = "NeighborList"
    material = jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)
    table = jd.MaterialTable.from_materials(
        [material], matcher=jd.MaterialMatchmaker.create("harmonic")
    )
    return jd.System.create(
        state=state,
        collider_type=backend,
        collider_kw=options,
        domain_type=domain,
        domain_kw={
            "box_size": jnp.full(state.dim, 8.0),
            **({"gamma": 0.3} if domain == "leesedwards" else {}),
        },
        mat_table=table,
        dt=0.01,
    )


def snapshot(dim=2):
    pos = np.array([[0.1, 0.2], [7.8, 0.25], [2.0, 2.0], [2.6, 2.0], [5.0, 5.0]])
    pos = np.pad(pos, ((0, 0), (0, dim - 2)))
    return jd.State.create(pos=pos, rad=jnp.full(5, 0.4))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dim,domain", [(2, "periodic"), (3, "leesedwards")])
def test_direct_reductions_and_contacts_match_force_and_virial(backend, dim, domain):
    state = snapshot(dim)
    system = make_system(state, backend, domain=domain)
    _, system, data = contact.get_contacts(state, system)
    assert data.pair_ids.shape == (4, 2)
    np.testing.assert_array_equal(data.pair_ids, [[0, 1], [1, 0], [2, 3], [3, 2]])
    assert np.all(np.linalg.norm(data.forces, axis=1) > 0)

    reference, _ = jd.colliders.NaiveSimulator.compute_force(
        replace(state), replace(system, collider=jd.Collider.create("naive"))
    )
    totals = np.zeros((state.N, dim))
    np.add.at(totals, np.asarray(data.pair_ids)[:, 0], np.asarray(data.forces))
    np.testing.assert_allclose(totals, reference.force, rtol=2e-5, atol=2e-6)
    dr = np.asarray(data.displacements)
    ids = np.asarray(data.pair_ids)
    keep = ids[:, 0] < ids[:, 1]
    expected = np.einsum("ni,nj->ij", dr[keep], np.asarray(data.forces)[keep]) / 8**dim

    _, _, stress = jax.jit(contact.compute_contact_stress_tensor)(state, system)
    _, _, reused = contact.compute_contact_stress_tensor(state, system, contacts=data)
    _, _, pressure = jax.jit(contact.compute_contact_pressure)(state, system)
    _, _, counts = jax.jit(contact.count_sphere_contacts)(state, system)
    np.testing.assert_allclose(stress, expected, rtol=2e-5, atol=1e-8)
    np.testing.assert_allclose(reused, expected, rtol=2e-5, atol=1e-8)
    np.testing.assert_allclose(pressure, np.trace(expected) / dim, rtol=2e-5)
    np.testing.assert_array_equal(counts, [1, 1, 1, 1, 0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_diagnostics_never_request_a_dense_neighbor_query(backend, monkeypatch):
    state = snapshot()
    system = make_system(state, backend)

    def forbidden(*args, **kwargs):
        raise AssertionError("Contact analysis must not construct a dense query.")

    for cls in (
        jd.colliders.NaiveSimulator,
        jd.colliders.DynamicCellList,
        jd.colliders.DynamicMultiCellList,
        jd.colliders.NeighborList,
    ):
        monkeypatch.setattr(cls, "create_neighbor_list", forbidden)
    _, system, data = contact.get_contacts(state, system)
    assert data.pair_ids.shape[0] == 4
    monkeypatch.setattr(contact, "get_contacts", forbidden)
    monkeypatch.setattr(jd.forces.SpringForce, "energy", forbidden)
    _, _, pressure = contact.compute_contact_pressure(state, system)
    _, _, counts = contact.count_vertex_contacts(state, system)
    assert pressure > 0
    np.testing.assert_array_equal(counts, [1, 1, 1, 1, 0])


@pytest.mark.parametrize("backend", BACKENDS)
def test_directed_bond_exclusions_match_the_configured_force_traversal(backend):
    state = jd.State.create(
        pos=[[1.0, 1.0], [1.6, 1.0], [1.3, 1.5]],
        rad=jnp.full(3, 0.5),
        bond_id=[[1], [-1], [-1]],
    )
    system = make_system(state, backend)
    _, system, data = contact.get_contacts(state, system)
    reference, _ = system.collider.compute_force(replace(state), system)
    totals = np.zeros((state.N, state.dim))
    np.add.at(totals, np.asarray(data.pair_ids)[:, 0], np.asarray(data.forces))
    np.testing.assert_allclose(totals, reference.force, rtol=2e-5, atol=1e-6)
    _, _, direct = contact.count_sphere_contacts(state, system)
    _, _, collected = contact.count_sphere_contacts(state, system, contacts=data)
    np.testing.assert_array_equal(direct, collected)
    _, _, direct = contact.compute_contact_pressure(state, system)
    _, _, collected = contact.compute_contact_pressure(state, system, contacts=data)
    np.testing.assert_allclose(direct, collected, rtol=2e-5, atol=1e-8)


def test_pooled_cache_supports_more_than_one_hundred_contacts_in_one_row():
    grid = (
        np.stack(np.meshgrid(np.arange(11), np.arange(11)), axis=-1).reshape(-1, 2)
        * 0.02
        + 1
    )
    isolated = np.column_stack((5.0 + 2 * np.arange(20), np.full(20, 4.0)))
    state = jd.State.create(
        pos=np.concatenate((grid, isolated)), rad=jnp.full(141, 0.5)
    )
    system = make_system(state, "neighbor-cell", capacity=103)
    box = jnp.full(2, 100.0)
    system = replace(
        system, domain=replace(system.domain, box_size=box, inv_box_size=1 / box)
    )
    _, system, data = contact.get_contacts(state, system)
    assert not bool(system.collider.overflow)
    assert data.pair_ids.shape == (121 * 120, 2)
    builds = int(system.collider.n_build_times)
    _, system, counts = contact.count_sphere_contacts(state, system)
    assert int(system.collider.n_build_times) == builds
    np.testing.assert_array_equal(counts[:121], np.full(121, 120))
    np.testing.assert_array_equal(counts[121:], 0)
    _, returned, repeated = contact.get_contacts(state, system)
    np.testing.assert_array_equal(repeated.pair_ids, data.pair_ids)
    assert int(returned.collider.n_build_times) == builds


def frictional_system(dim):
    centers = np.pad(
        np.array([[0.5, 1.0], [0.5, 1.0], [1.3, 1.0]]), ((0, 0), (0, dim - 2))
    )
    offsets = np.pad(
        np.array([[0.0, -0.2], [0.0, 0.2], [0.0, 0.0]]), ((0, 0), (0, dim - 2))
    )
    velocity = np.zeros((3, dim))
    velocity[2, 1] = 0.2
    omega = np.zeros((3, 1 if dim == 2 else 3))
    omega[:2, -1] = 0.1
    state = jd.State.create(
        pos=centers,
        pos_p=offsets,
        rad=jnp.full(3, 0.5),
        clump_id=jnp.array([0, 0, 1]),
        vel=velocity,
        ang_vel=omega,
    )
    material = jd.Material.create(
        "elasticfrict", young=100.0, poisson=0.0, density=1.0, e=1.0, mu=10.0, mu_r=0.3
    )
    system = jd.System.create(
        state=state,
        force_model_type="cundallstrack",
        mat_table=jd.MaterialTable.from_materials([material]),
        dt=0.01,
        collider_type="NeighborList",
        collider_kw={"cutoff": 1.0, "skin": 0.05, "max_neighbors": 4},
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(dim, 8.0)},
    )
    state, system = jd.System.initialize(state, system)
    return system.collider.compute_force(state, system)


def history_by_pair(system):
    col = system.collider
    valid = np.asarray(col.neighbor_list) >= 0
    return {
        (int(i), int(j)): np.asarray(h)
        for i, j, h in zip(
            np.asarray(pair_sources(col))[valid],
            np.asarray(col.neighbor_list)[valid],
            np.asarray(col.history)[valid],
            strict=True,
        )
    }


@pytest.mark.parametrize("dim", [2, 3])
def test_history_and_force_law_torques_survive_rebuild_and_repeated_analysis(dim):
    state, system = frictional_system(dim)
    before = history_by_pair(system)
    moved = replace(state, pos_c=state.pos_c.at[2, 0].add(0.06))
    original = [np.asarray(x).copy() for x in jax.tree.leaves((moved, system))]
    _, updated, data = contact.get_contacts(moved, system)
    assert int(updated.collider.n_build_times) == int(system.collider.n_build_times) + 1
    after = history_by_pair(updated)
    assert before.keys() == after.keys()
    for pair in before:
        np.testing.assert_array_equal(before[pair], after[pair])
    expected_force, expected_torque = [], []
    for i, j in np.asarray(data.pair_ids):
        f, t, _ = system.force_model.force(
            i,
            j,
            moved.pos,
            moved,
            updated,
            jnp.asarray(before[(i, j)]),
            advance_history=False,
        )
        expected_force.append(f)
        expected_torque.append(t + cross(moved._pos_p_rot[i], f))
    np.testing.assert_allclose(data.forces, expected_force, rtol=2e-5, atol=1e-6)
    np.testing.assert_allclose(data.torques, expected_torque, rtol=2e-5, atol=1e-6)
    for _ in range(2):
        _, updated, _ = contact.compute_contact_pressure(moved, updated)
        _, updated, _ = contact.count_sphere_contacts(moved, updated)
        _, updated, _ = contact.get_contacts(moved, updated)
    for pair, history in history_by_pair(updated).items():
        np.testing.assert_array_equal(history, before[pair])
    for old, new in zip(original, jax.tree.leaves((moved, system)), strict=True):
        np.testing.assert_array_equal(old, new)


def test_reusing_contact_data_performs_no_search_or_force_evaluation(monkeypatch):
    state = snapshot()
    system = make_system(state)
    _, system, data = contact.get_contacts(state, system)

    def forbidden(*args, **kwargs):
        raise AssertionError("A supplied contact snapshot must be reused.")

    monkeypatch.setattr(contact, "get_contacts", forbidden)
    monkeypatch.setattr(contact, "_reduce_contacts", forbidden)
    _, _, pressure = contact.compute_contact_pressure(state, system, contacts=data)
    _, _, counts = contact.count_clump_contacts(state, system, contacts=data)
    _, _, groups = contact.get_group_contacts(state, system, contacts=data)
    _, _, rattlers, non_rattlers = contact.get_sphere_rattler_ids(
        state, system, contacts=data, zc=1
    )
    assert pressure > 0
    np.testing.assert_array_equal(counts, [1, 1, 1, 1, 0])
    assert groups.pair_ids.shape == (4, 2)
    np.testing.assert_array_equal(rattlers, [4])
    np.testing.assert_array_equal(non_rattlers, [0, 1, 2, 3])


def test_group_contacts_preserve_cancelling_interactions():
    world = jnp.array([[0.0, 0.0], [2.0, 1.0], [0.8, 0.0], [1.2, 1.0]])
    centers = jnp.array([[1.0, 0.4], [1.0, 0.4], [1.0, 0.6], [1.0, 0.6]])
    state = jd.State.create(
        pos=centers,
        pos_p=world - centers,
        rad=jnp.full(4, 0.5),
        clump_id=jnp.array([0, 0, 1, 1]),
    )
    system = make_system(state)
    _, _, groups = contact.get_group_contacts(state, system)
    np.testing.assert_array_equal(groups.pair_ids, [[0, 1], [1, 0]])
    np.testing.assert_allclose(groups.forces, 0, atol=1e-6)
    np.testing.assert_array_equal(groups.sphere_counts, [[2, 2], [2, 2]])
    np.testing.assert_array_equal(groups.contact_counts, [2, 2])
    _, _, vertices = contact.count_vertex_contacts(state, system)
    _, _, clumps = contact.count_clump_contacts(state, system)
    np.testing.assert_array_equal(vertices, [2, 2])
    np.testing.assert_array_equal(clumps, [1, 1])


def test_bond_components_and_sparse_group_labels_use_periodic_centroids():
    state = jd.State.create(
        pos=[[7.8, 1.0], [0.2, 1.0], [0.9, 1.0], [1.3, 1.0]],
        rad=jnp.full(4, 0.5),
        bond_id=[[1], [0], [3], [2]],
    )
    system = make_system(state)
    _, system, data = contact.get_contacts(state, system)
    _, _, groups = contact.get_group_contacts(
        state, system, group_by="bond_id", contacts=data
    )
    np.testing.assert_array_equal(groups.group_ids, [0, 1])
    np.testing.assert_array_equal(groups.pair_ids, [[0, 1], [1, 0]])
    np.testing.assert_allclose(groups.friction, 0, atol=1e-6)
    np.testing.assert_array_equal(groups.sphere_counts, [[1, 1], [1, 1]])
    _, _, labeled = contact.get_group_contacts(
        state, system, group_by=jnp.array([42, 42, 9001, 9001]), contacts=data
    )
    np.testing.assert_array_equal(labeled.group_ids, [42, 9001])
    np.testing.assert_array_equal(labeled.pair_ids, [[42, 9001], [9001, 42]])
    assert labeled.forces.shape == (2, 2)


def test_clump_rank_uses_contact_torques():
    state = jd.State.create(
        pos=np.column_stack((np.arange(4), np.zeros(4))), rad=jnp.full(4, 0.5)
    )
    system = make_system(state)
    pairs = jnp.asarray([(i, j) for i in range(4) for j in range(4) if i != j])
    forces = jnp.tile(jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]), (4, 1))
    torques = jnp.tile(jnp.array([[0.0], [0.0], [1.0]]), (4, 1))
    data = contact.ContactData(pairs, forces, torques, jnp.zeros_like(forces))
    _, _, rattlers, non_rattlers = contact.get_clump_rattler_ids(
        state, system, zc=3, check_contact_rank=True, contacts=data
    )
    assert rattlers.size == 0
    np.testing.assert_array_equal(non_rattlers, np.arange(4))
    with pytest.warns(UserWarning, match="No valid particles remain"):
        _, _, rattlers, non_rattlers = contact.get_clump_rattler_ids(
            state,
            system,
            zc=3,
            check_contact_rank=True,
            contacts=data._replace(torques=jnp.zeros_like(torques)),
        )
    np.testing.assert_array_equal(rattlers, np.arange(4))
    assert non_rattlers.size == 0


@pytest.mark.parametrize("backend", BACKENDS)
def test_empty_configurations(backend):
    state = jd.State.create(dim=2)
    system = make_system(state, backend)
    _, _, data = contact.get_contacts(state, system)
    assert data.pair_ids.shape == (0, 2)
    for fn in (
        contact.count_sphere_contacts,
        contact.count_vertex_contacts,
        contact.count_clump_contacts,
    ):
        _, _, result = fn(state, system)
        assert result.shape == (0,)
    _, _, pressure = contact.compute_contact_pressure(state, system)
    assert pressure == 0
    _, _, groups = contact.get_group_contacts(state, system)
    assert groups.pair_ids.shape == (0, 2)
    _, _, rattlers, non_rattlers = contact.get_clump_rattler_ids(state, system)
    assert rattlers.size == non_rattlers.size == 0


def test_incomplete_collider_fails_on_host_and_marks_compiled_results():
    state = snapshot()
    system = make_system(state, "neighbor-cell", capacity=0)
    for fn in (
        contact.get_contacts,
        contact.compute_contact_pressure,
        contact.count_sphere_contacts,
    ):
        with pytest.raises(ValueError, match="configured collider overflowed"):
            fn(state, system)
    _, returned, pressure = jax.jit(contact.compute_contact_pressure)(state, system)
    assert bool(returned.search_overflow)
    assert jnp.isnan(pressure)
    _, returned, counts = jax.jit(contact.count_sphere_contacts)(state, system)
    assert bool(returned.search_overflow)
    np.testing.assert_array_equal(counts, -1)


def test_contact_pressure_uses_explicit_volume():
    state = snapshot()
    system = make_system(state)
    _, _, default = contact.compute_contact_pressure(state, system)
    _, _, pressure = contact.compute_contact_pressure(state, system, volume=2.0)
    np.testing.assert_allclose(pressure * 2.0, default * 8.0**state.dim)


@pytest.mark.parametrize(
    "labels",
    [jnp.array([0.0] * 5), jnp.zeros((5, 1), dtype=int), jnp.array([-1, 0, 0, 0, 0])],
)
def test_invalid_group_labels_are_rejected(labels):
    state = snapshot()
    system = make_system(state)
    with pytest.raises(ValueError, match="Group labels"):
        contact.get_group_contacts(state, system, group_by=labels)
