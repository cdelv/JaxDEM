# SPDX-License-Identifier: BSD-3-Clause
"""Hessian candidates and assembly agree with the configured pair energy."""

from dataclasses import dataclass, field, replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils import (
    clump_non_bonded_hessian,
    get_contacts,
    non_bonded_hessian,
    pair_non_bonded_hessian,
)
from jaxdem.utils.quaternion import Quaternion


BACKENDS = ("naive", "CellList", "MultiCellList", "neighbor-cell", "neighbor-multi")


def make_system(
    state, backend, *, domain="periodic", law=None, capacity=None, material=None
):
    options = {}
    if backend.startswith("neighbor-"):
        options = {
            "max_neighbors": state.N if capacity is None else capacity,
            "cutoff": 2 * float(jnp.max(state.rad, initial=0.0)),
            "skin": 0.1,
            "secondary_collider_type": "CellList"
            if backend == "neighbor-cell"
            else "MultiCellList",
        }
        backend = "NeighborList"
    if material is None:
        material = jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)
    return jd.System.create(
        state=state,
        force_model=jd.forces.SpringForce() if law is None else law,
        mat_table=jd.MaterialTable.from_materials(
            [material], matcher=jd.MaterialMatchmaker.create("harmonic")
        ),
        collider_type=backend,
        collider_kw=options,
        domain_type=domain,
        domain_kw={
            "box_size": jnp.full(state.dim, 8.0),
            **({"gamma": 0.3} if domain == "leesedwards" else {}),
        },
    )


def all_pairs_energy(pos, state, system):
    indices = jnp.arange(state.N)
    energies = jax.vmap(
        lambda i: jax.vmap(
            lambda j: system.force_model.energy(i, j, pos, state, system)
        )(indices)
    )(indices)
    valid = jax.vmap(
        lambda i: jd.colliders.valid_interaction_mask(
            state.clump_id[i],
            state.clump_id,
            state.bond_id[i],
            indices,
            system.interact_same_bond_id,
        )
    )(indices)
    return 0.5 * jnp.sum(jnp.where(valid, energies, 0.0))


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("dim,domain", [(2, "periodic"), (3, "leesedwards")])
def test_hessians_match_total_energy_without_dense_queries(
    backend, dim, domain, monkeypatch
):
    pos = np.array([[0.1, 0.2], [7.8, 0.25], [2.0, 2.0], [2.6, 2.1], [5.0, 5.0]])
    if domain == "leesedwards":
        pos[:2] = [[1.0, 0.1], [3.4, 7.8]]
    pos = np.pad(pos, ((0, 0), (0, dim - 2)))
    state = jd.State.create(pos=pos, rad=jnp.full(5, 0.4))
    system = make_system(state, backend, domain=domain)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Hessian analysis must not request a dense neighbor query."
        )

    for collider in (
        jd.colliders.NaiveSimulator,
        jd.colliders.DynamicCellList,
        jd.colliders.DynamicMultiCellList,
        jd.colliders.NeighborList,
    ):
        monkeypatch.setattr(collider, "create_neighbor_list", forbidden)
    _, system, pairs, blocks = pair_non_bonded_hessian(state, system)
    np.testing.assert_array_equal(pairs, [[0, 1], [2, 3]])
    assert blocks.shape == (2, 2 * dim, 2 * dim)
    _, returned, matrix = non_bonded_hessian(state, system)
    if backend.startswith("neighbor-"):
        assert (
            int(returned.collider.n_build_times)
            == int(system.collider.n_build_times)
            == 1
        )

    expected = jax.hessian(lambda p: all_pairs_energy(p, state, system))(state.pos)
    expected = expected.reshape(state.N * dim, state.N * dim)
    np.testing.assert_allclose(matrix, expected, rtol=3e-5, atol=2e-6)
    assembled = np.zeros(matrix.shape)
    for pair, block in zip(np.asarray(pairs), np.asarray(blocks), strict=True):
        rows = (pair[:, None] * dim + np.arange(dim)).reshape(-1)
        assembled[np.ix_(rows, rows)] += block
    np.testing.assert_allclose(assembled, matrix, rtol=3e-5, atol=2e-6)


@pytest.mark.parametrize("backend", ["CellList", "neighbor-multi"])
@pytest.mark.parametrize("composition", ["single", "combiner", "router"])
def test_hessian_follows_per_particle_force_reach(backend, composition):
    state = jd.State.create(
        pos=[[1.0, 1.0], [2.1, 1.0], [3.4, 1.0], [6.0, 6.0]],
        rad=jnp.array([0.2, 0.2, 0.2, 0.4]),
    )
    law = jd.ForceModel.create("lennardjones", cutoff_ratio=3.0)
    if composition == "combiner":
        law = jd.LawCombiner(laws=(law, jd.ForceModel.create("wca")))
    elif composition == "router":
        law = jd.ForceRouter.from_dict(1, {(0, 0): law})
    system = make_system(
        state,
        backend,
        law=law,
        material=jd.Material.create("lj", density=1.0, epsilon=1.0),
    )
    _, system, pairs, blocks = pair_non_bonded_hessian(state, system)
    np.testing.assert_array_equal(pairs, [[0, 1]])
    assert float(jnp.linalg.norm(blocks)) > 0
    _, _, matrix = non_bonded_hessian(state, system)
    expected = jax.hessian(lambda p: all_pairs_energy(p, state, system))(state.pos)
    np.testing.assert_allclose(matrix, expected.reshape(8, 8), rtol=3e-5, atol=2e-6)


@jax.tree_util.register_dataclass
@dataclass
class WellEnergy(jd.ForceModel):
    equilibrium_distance: jax.Array = field(default_factory=lambda: jnp.asarray(1.0))

    def search_radii(self, state, system):
        return 2 * state.rad

    @staticmethod
    def energy(i, j, pos, state, system):
        dr = system.domain.displacement(pos[i], pos[j], system)
        r2 = jnp.sum(dr * dr)
        displacement = r2 - system.force_model.equilibrium_distance**2
        return jnp.where((i != j) & (r2 < 4.0), displacement**2 / 8, 0.0)

    @staticmethod
    def force(i, j, pos, state, system, history, *, advance_history=True):
        dr = system.domain.displacement(pos[i], pos[j], system)
        r2 = jnp.sum(dr * dr)
        displacement = r2 - system.force_model.equilibrium_distance**2
        force = jnp.where((i != j) & (r2 < 4.0), -0.5 * displacement, 0.0) * dr
        return force, jnp.zeros_like(state.torque[i]), history


@pytest.mark.parametrize("backend", BACKENDS)
def test_zero_force_and_zero_energy_pair_retains_stiffness(backend):
    state = jd.State.create(pos=[[1.0, 1.0], [2.0, 1.0]], rad=jnp.full(2, 0.5))
    system = make_system(state, backend, law=WellEnergy())
    _, system, contacts = get_contacts(state, system)
    assert contacts.pair_ids.shape == (0, 2)
    assert all_pairs_energy(state.pos, state, system) == 0
    _, system, pairs, blocks = pair_non_bonded_hessian(state, system)
    np.testing.assert_array_equal(pairs, [[0, 1]])
    expected = np.zeros((4, 4))
    expected[np.ix_([0, 2], [0, 2])] = [[1, -1], [-1, 1]]
    np.testing.assert_allclose(blocks[0], expected, atol=1e-6)
    _, _, matrix = non_bonded_hessian(state, system)
    np.testing.assert_allclose(matrix, expected, atol=1e-6)


@pytest.mark.parametrize("backend", BACKENDS)
def test_directed_exclusions_match_collider_energy(backend):
    state = jd.State.create(
        pos=[[1.0, 1.0], [1.6, 1.0], [1.3, 1.5]],
        rad=jnp.full(3, 0.5),
        bond_id=[[1], [-1], [-1]],
    )
    system = make_system(state, backend)
    _, system, matrix = non_bonded_hessian(state, system)

    def energy(pos):
        _, _, value = system.collider.compute_potential_energy(
            replace(state, pos_c=pos), system
        )
        return value

    expected = jax.jacfwd(jax.jacfwd(energy))(state.pos).reshape(6, 6)
    np.testing.assert_allclose(matrix, expected, rtol=3e-5, atol=2e-6)


def rigid_state(dim):
    centers = np.array([[0.1, 0.2], [7.6, 0.3], [3.0, 3.0]])
    centers = np.pad(centers, ((0, 0), (0, dim - 2)))
    offsets = np.array([[0.15, 0.12], [-0.15, -0.12]])
    offsets = np.pad(offsets, ((0, 0), (0, dim - 2)))
    if dim == 3:
        offsets[:, 2] = [0.1, -0.1]
    axis = np.array([0.0, 0.0, 1.0]) if dim == 2 else np.array([0.2, 0.3, 0.4])
    axis /= np.linalg.norm(axis)
    return jd.State.create(
        pos=np.repeat(centers, 2, axis=0),
        pos_p=np.tile(offsets, (3, 1)),
        rad=jnp.full(6, 0.35),
        clump_id=jnp.repeat(jnp.arange(3), 2),
        q=Quaternion.create(
            w=jnp.full((6, 1), np.cos(0.3)),
            xyz=jnp.tile(jnp.asarray(axis * np.sin(0.3)), (6, 1)),
        ),
    )


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("domain", ["periodic", "leesedwards"])
def test_clump_hessian_matches_exact_rotations_and_scaling(dim, domain):
    state = rigid_state(dim)
    system = make_system(state, "neighbor-multi", domain=domain)
    _, system, matrix = clump_non_bonded_hessian(state, system)
    dof = dim + (1 if dim == 2 else 3)

    def rotation(omega):
        if dim == 2:
            c, s = jnp.cos(omega[0]), jnp.sin(omega[0])
            return jnp.array([[c, -s], [s, c]])
        from jax.scipy.linalg import expm

        x, y, z = omega
        skew = jnp.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        return expm(skew)

    def energy(q):
        q = q.reshape(3, dof)
        matrices = jax.vmap(rotation)(q[:, dim:])
        offset = jnp.einsum("nij,nj->ni", matrices[state.clump_id], state._pos_p_rot)
        pos = state.pos_c + q[state.clump_id, :dim] + offset
        return all_pairs_energy(pos, state, system)

    expected = jax.hessian(energy)(jnp.zeros(3 * dof))
    np.testing.assert_allclose(matrix, expected, rtol=4e-5, atol=3e-6)
    assert np.max(np.abs(np.asarray(matrix)[dim:dof])) > 1e-3
    scales = jnp.array([0.5, 0.7, 1.0])
    _, _, scaled = clump_non_bonded_hessian(state, system, rotation_scale=scales)
    factors = np.ones((3, dof))
    factors[:, dim:] = 1 / np.asarray(scales)[:, None]
    factors = factors.reshape(-1)
    np.testing.assert_allclose(
        scaled, expected * factors[:, None] * factors[None, :], rtol=4e-5, atol=4e-6
    )


@pytest.mark.parametrize("dim", [2, 3])
def test_hessian_rebuild_preserves_contact_history(dim, monkeypatch):
    from .test_contact_analysis import frictional_system, history_by_pair

    state, system = frictional_system(dim)
    moved = replace(state, pos_c=state.pos_c.at[2, 0].add(0.06))
    before = history_by_pair(system)
    original = [np.asarray(x).copy() for x in jax.tree.leaves((moved, system))]

    def forbidden(*args, **kwargs):
        raise AssertionError("Energy Hessians must not evaluate contact forces.")

    monkeypatch.setattr(jd.forces.CundallStrackForce, "force", forbidden)
    _, updated, matrix = non_bonded_hessian(moved, system)
    assert int(updated.collider.n_build_times) == int(system.collider.n_build_times) + 1
    _, returned, _ = clump_non_bonded_hessian(moved, updated)
    assert int(returned.collider.n_build_times) == int(updated.collider.n_build_times)
    after = history_by_pair(returned)
    assert before.keys() == after.keys()
    for pair in before:
        np.testing.assert_array_equal(before[pair], after[pair])
    expected = jax.hessian(lambda p: all_pairs_energy(p, moved, updated))(moved.pos)
    np.testing.assert_allclose(
        matrix, expected.reshape(matrix.shape), rtol=3e-5, atol=1e-5
    )
    for old, new in zip(original, jax.tree.leaves((moved, system)), strict=True):
        np.testing.assert_array_equal(old, new)


def test_pooled_cache_with_more_than_one_hundred_neighbors_matches_analytical_hessian():
    grid = np.stack(np.meshgrid(np.arange(11), np.arange(11)), axis=-1).reshape(-1, 2)
    grid = grid * 0.02 + 1
    isolated = np.column_stack((5.0 + 2 * np.arange(20), np.full(20, 4.0)))
    state = jd.State.create(
        pos=np.concatenate((grid, isolated)), rad=jnp.full(141, 0.5)
    )
    system = make_system(state, "neighbor-cell", capacity=103)
    box = jnp.full(2, 100.0)
    system = replace(
        system, domain=replace(system.domain, box_size=box, inv_box_size=1 / box)
    )
    _, system, matrix = non_bonded_hessian(state, system)
    assert not bool(system.collider.overflow)
    assert int(jnp.max(jnp.diff(system.collider.row_offsets))) == 120
    expected = np.zeros((282, 282))
    positions = np.asarray(state.pos)
    for i in range(121):
        for j in range(i + 1, 121):
            dr = positions[i] - positions[j]
            distance = np.linalg.norm(dr)
            nn = np.outer(dr, dr) / distance**2
            block = nn - (1 - distance) / distance * (np.eye(2) - nn)
            rows_i, rows_j = 2 * i + np.arange(2), 2 * j + np.arange(2)
            expected[np.ix_(rows_i, rows_i)] += block
            expected[np.ix_(rows_j, rows_j)] += block
            expected[np.ix_(rows_i, rows_j)] -= block
            expected[np.ix_(rows_j, rows_i)] -= block
    np.testing.assert_allclose(matrix, expected, rtol=4e-5, atol=2e-3)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("n", [0, 1])
def test_empty_pair_sets_return_zero_hessians(backend, n):
    state = jd.State.create(pos=jnp.zeros((n, 2)), rad=jnp.full(n, 0.5), dim=2)
    system = make_system(state, backend)
    _, _, pairs, blocks = pair_non_bonded_hessian(state, system)
    assert pairs.shape == (0, 2)
    assert blocks.shape == (0, 4, 4)
    _, _, matrix = non_bonded_hessian(state, system)
    np.testing.assert_array_equal(matrix, np.zeros((n * 2, n * 2)))
    _, _, matrix = clump_non_bonded_hessian(state, system)
    np.testing.assert_array_equal(matrix, np.zeros((n * 3, n * 3)))


def test_overflow_is_rejected_before_differentiation(monkeypatch):
    state = jd.State.create(pos=[[0.0, 0.0], [0.5, 0.0]], rad=jnp.full(2, 0.5))
    system = make_system(state, "neighbor-cell", capacity=0)

    def forbidden(*args, **kwargs):
        raise AssertionError("An incomplete search must fail before energy evaluation.")

    monkeypatch.setattr(jd.forces.SpringForce, "energy", forbidden)
    for function in (
        pair_non_bonded_hessian,
        non_bonded_hessian,
        clump_non_bonded_hessian,
    ):
        with pytest.raises(ValueError, match="configured collider overflowed"):
            function(state, system)


@pytest.mark.parametrize("scale", [[1.0], [1.0, 0.0, 1.0], [1.0, float("nan"), 1.0]])
def test_invalid_rotation_scales_are_rejected(scale):
    state = rigid_state(2)
    system = make_system(state, "naive")
    with pytest.raises(ValueError, match="rotation_scale"):
        clump_non_bonded_hessian(state, system, rotation_scale=jnp.asarray(scale))
