# SPDX-License-Identifier: BSD-3-Clause
"""Packing analysis preserves geometry, history, and aligned pair statistics."""

from dataclasses import dataclass, field, fields, replace
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.utils import PackingData, analyze_packing
from jaxdem.utils import contacts as contact


def make_system(state, backend="NeighborList", *, domain="periodic", law=None):
    material = jd.Material.create("elastic", young=1.0, poisson=0.5, density=1.0)
    return jd.System.create(
        state=state,
        collider_type=backend,
        collider_kw={"max_neighbors": state.N, "skin": 0.1, "cutoff": 1.0}
        if backend == "NeighborList"
        else {},
        domain_type=domain,
        domain_kw={
            "box_size": jnp.full(state.dim, 8.0),
            **({"gamma": 0.3} if domain == "leesedwards" else {}),
        },
        force_model=jd.forces.SpringForce() if law is None else law,
        mat_table=jd.MaterialTable.from_materials(
            [material], matcher=jd.MaterialMatchmaker.create("harmonic")
        ),
    )


def square_with_rattler():
    return jd.State.create(
        pos=[[6.0, 6.0], [1.0, 1.0], [1.6, 1.0], [1.0, 1.6], [1.6, 1.6]],
        rad=jnp.full(5, 0.5),
        clump_id=[3, 0, 4, 1, 2],
    )


def assert_packed(network, n):
    pairs, offsets = np.asarray(network.pair_ids), np.asarray(network.row_offsets)
    assert offsets.shape == (n + 1,)
    assert offsets[0] == 0
    assert offsets[-1] == len(pairs)
    assert np.all(np.diff(offsets) >= 0)
    for i in range(n):
        np.testing.assert_array_equal(pairs[offsets[i] : offsets[i + 1], 0], i)
    for item in fields(network):
        value = getattr(network, item.name)
        if item.name != "row_offsets" and value is not None:
            assert value.shape[0] == len(pairs)


@pytest.mark.parametrize(
    "backend", ["Naive", "CellList", "MultiCellList", "NeighborList"]
)
def test_full_and_non_rattler_summaries_and_species(backend):
    state = square_with_rattler()
    system = make_system(state, backend)
    species = np.array([5, 8, 5, 99, 8])
    result = analyze_packing(state, system, clump_species_ids=species)
    assert isinstance(result.full, PackingData)
    assert isinstance(result.non_rattlers, PackingData)
    np.testing.assert_array_equal(result.rattler_ids, [3])
    np.testing.assert_array_equal(result.non_rattler_ids, [0, 1, 2, 4])
    assert result.rattler_proportion == 0.2
    full, reduced = result.full, result.non_rattlers
    np.testing.assert_array_equal(full.original_sphere_ids, np.arange(5))
    np.testing.assert_array_equal(reduced.original_sphere_ids, [1, 2, 3, 4])
    np.testing.assert_array_equal(reduced.original_clump_ids, [0, 1, 2, 4])
    np.testing.assert_array_equal(reduced.state.clump_id, [0, 3, 1, 2])
    np.testing.assert_allclose(reduced.state.pos, state.pos[1:])
    np.testing.assert_array_equal(full.sphere_contact_counts, [0, 3, 3, 3, 3])
    np.testing.assert_array_equal(full.vertex_contact_counts, [3, 3, 3, 0, 3])
    np.testing.assert_array_equal(reduced.vertex_contact_counts, [3, 3, 3, 3])
    assert full.mean_clump_contacts == pytest.approx(12 / 5)
    assert reduced.mean_vertex_contacts == 3
    assert reduced.isostatic_coordination == 3
    assert reduced.satisfies_isostatic_count
    assert not full.satisfies_isostatic_count
    for packing in (full, reduced):
        assert packing.coordinates == "sphere"
        n_coordinates = packing.state.N * packing.state.dim
        assert packing.hessian.shape == (n_coordinates, n_coordinates)
        assert packing.eigenvalues.shape == (n_coordinates,)
        assert packing.eigenvectors.shape == (n_coordinates, n_coordinates)
        assert isinstance(packing.zero_mode_count, int)
        assert isinstance(packing.negative_mode_count, int)
        assert_packed(packing.sphere_contacts, packing.state.N)
        assert_packed(packing.clump_contacts, len(packing.original_clump_ids))
        groups = packing.clump_contacts
        np.testing.assert_allclose(groups.mu, 0, atol=1e-6)
        np.testing.assert_array_equal(groups.sphere_counts, np.ones((12, 2)))
        np.testing.assert_array_equal(groups.contact_counts, np.ones(12))
        original_ids = np.asarray(packing.original_clump_ids)[
            np.asarray(groups.pair_ids)
        ]
        np.testing.assert_array_equal(groups.pair_species, species[original_ids])
        sphere_clumps = np.asarray(packing.state.clump_id)[
            np.asarray(packing.sphere_contacts.pair_ids)
        ]
        original_ids = np.asarray(packing.original_clump_ids)[sphere_clumps]
        np.testing.assert_array_equal(
            packing.sphere_contacts.pair_species, species[original_ids]
        )
        expected_force, _ = jd.System.evaluate_forces(packing.state, packing.system)
        np.testing.assert_allclose(
            packing.state.force, expected_force.force, rtol=2e-5, atol=2e-6
        )
    diagonal = math.sqrt(2) * 0.6
    energy = 4 * 0.5 * 0.4**2 + (1 - diagonal) ** 2
    pressure = (4 * 0.6 * 0.4 + 2 * diagonal * (1 - diagonal)) / (2 * 64)
    effective_area = 4 * math.pi * ((1.2 + diagonal) / 6) ** 2
    np.testing.assert_allclose(full.potential_energy, energy, rtol=2e-5)
    np.testing.assert_allclose(reduced.potential_energy, energy, rtol=2e-5)
    np.testing.assert_allclose(full.pressure, pressure, rtol=2e-5)
    np.testing.assert_allclose(reduced.pressure, pressure, rtol=2e-5)
    np.testing.assert_allclose(full.packing_fraction, 5 * math.pi / 4 / 64, rtol=2e-6)
    np.testing.assert_allclose(reduced.packing_fraction, math.pi / 64, rtol=2e-6)
    np.testing.assert_allclose(
        full.effective_packing_fraction, (effective_area + math.pi / 4) / 64, rtol=2e-6
    )
    np.testing.assert_allclose(
        reduced.effective_packing_fraction, effective_area / 64, rtol=2e-6
    )


def test_contacts_are_collected_once_per_configuration(monkeypatch):
    state = square_with_rattler()
    system = make_system(state)
    calls = []
    collect = contact.get_contacts

    def counted(state, system):
        calls.append(state.N)
        return collect(state, system)

    def forbidden(*args, **kwargs):
        raise AssertionError("Analysis must reuse the configured collider candidates.")

    monkeypatch.setattr(contact, "get_contacts", counted)
    monkeypatch.setattr(jd.colliders.NeighborList, "create_neighbor_list", forbidden)
    analyze_packing(state, system)
    assert calls == [5, 4]


def test_full_spectra_and_coordination_do_not_claim_stability():
    state = square_with_rattler()
    result = analyze_packing(state, make_system(state))
    for data in (result.full, result.non_rattlers):
        h = np.asarray(data.hessian)
        assert h.shape == (data.state.N * 2, data.state.N * 2)
        np.testing.assert_allclose(data.eigenvalues, np.linalg.eigvalsh(h), atol=2e-6)
        np.testing.assert_allclose(
            h @ data.eigenvectors, data.eigenvectors * data.eigenvalues, atol=3e-6
        )
        np.testing.assert_allclose(
            data.eigenvectors.T @ data.eigenvectors, np.eye(h.shape[0]), atol=2e-6
        )
    assert result.non_rattlers.satisfies_isostatic_count
    assert result.non_rattlers.negative_mode_count > 0
    assert result.full.zero_mode_count == result.non_rattlers.zero_mode_count + 2


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("n", [0, 1, 2])
def test_empty_and_all_rattler_packings(dim, n):
    pos = np.zeros((n, dim))
    if n == 2:
        pos[1, 0] = 3
    state = jd.State.create(pos=pos, rad=jnp.full(n, 0.5), dim=dim)
    result = analyze_packing(
        state,
        make_system(state),
        clump_species_ids=np.arange(n),
    )
    empty = result.non_rattlers
    assert result.rattler_proportion == (1.0 if n else 0.0)
    assert empty.state.N == 0
    assert empty.sphere_contacts.pair_ids.shape == (0, 2)
    assert empty.clump_contacts.pair_species.shape == (0, 2)
    assert empty.hessian.shape == empty.eigenvectors.shape == (0, 0)
    assert empty.eigenvalues.shape == (0,)
    assert empty.zero_mode_count == empty.negative_mode_count == 0
    assert np.isnan(empty.mean_vertex_contacts)
    assert not empty.satisfies_isostatic_count
    assert float(empty.packing_fraction) == float(empty.effective_packing_fraction) == 0
    assert result.full.zero_mode_count == n * dim
    np.testing.assert_allclose(
        result.full.packing_fraction, result.full.effective_packing_fraction
    )


@pytest.mark.parametrize("dim,domain", [(2, "periodic"), (3, "leesedwards")])
def test_periodic_geometry_and_dimension_of_effective_volume(dim, domain):
    pos = [[0.1, 1.0], [7.8, 1.0]] if dim == 2 else [[1.0, 0.1, 0.0], [3.4, 7.8, 0.0]]
    state = jd.State.create(pos=pos, rad=jnp.full(2, 0.5))
    result = analyze_packing(state, make_system(state, domain=domain), zc=1)
    assert result.full is result.non_rattlers
    data = result.full
    np.testing.assert_allclose(
        jnp.linalg.norm(data.clump_contacts.displacements, axis=1), 0.3, atol=1e-6
    )
    np.testing.assert_allclose(data.sphere_contacts.overlap, 0.7, atol=1e-6)
    expected = 2 * math.pi ** (dim / 2) / math.gamma(dim / 2 + 1) * 0.15**dim / 8**dim
    np.testing.assert_allclose(data.effective_packing_fraction, expected, rtol=5e-6)
    assert data.sphere_contacts.pair_species is None


@pytest.mark.parametrize("dim", [2, 3])
def test_clump_hessians_map_rotation_scales_and_preserve_moments(dim):
    from .test_hessian_candidates import rigid_state

    state = rigid_state(dim)
    system = make_system(state)
    result = analyze_packing(
        state,
        system,
        zc=1,
        rotation_scale=[0.5, 0.7, 0.9],
        clump_species_ids=[8, 12, 99],
    )
    assert result.full.coordinates == "clump"
    np.testing.assert_array_equal(result.rattler_ids, [2])
    for data, scales in (
        (result.full, [0.5, 0.7, 0.9]),
        (result.non_rattlers, [0.5, 0.7]),
    ):
        _, _, unscaled = jd.utils.clump_non_bonded_hessian(data.state, data.system)
        dof = dim + (1 if dim == 2 else 3)
        factors = np.ones((len(scales), dof))
        factors[:, dim:] /= np.asarray(scales)[:, None]
        factors = factors.ravel()
        np.testing.assert_allclose(
            data.hessian,
            unscaled * factors[:, None] * factors[None, :],
            rtol=4e-5,
            atol=3e-6,
        )
        spheres, groups = data.sphere_contacts, data.clump_contacts
        for row, (i, j) in enumerate(np.asarray(groups.pair_ids)):
            sphere_pairs = np.asarray(data.state.clump_id)[np.asarray(spheres.pair_ids)]
            mask = np.all(sphere_pairs == [i, j], axis=1)
            np.testing.assert_allclose(
                groups.forces[row], spheres.forces[mask].sum(axis=0), atol=2e-6
            )
            np.testing.assert_allclose(
                groups.torques[row], spheres.torques[mask].sum(axis=0), atol=2e-6
            )
    assert np.max(np.abs(np.asarray(result.full.clump_contacts.torques))) > 0


@pytest.mark.parametrize("dim", [2, 3])
def test_history_and_queued_loads_survive_analysis_and_removal(dim, tmp_path):
    from .test_contact_analysis import history_by_pair

    pos = np.pad(
        np.array([[1.0, 1.0], [1.0, 1.0], [1.8, 1.0], [1.8, 1.0], [6.0, 6.0]]),
        ((0, 0), (0, dim - 2)),
    )
    offset = np.pad(
        np.array([[0.0, -0.2], [0.0, 0.2], [0.0, -0.2], [0.0, 0.2], [0.0, 0.0]]),
        ((0, 0), (0, dim - 2)),
    )
    state = jd.State.create(
        pos=pos,
        pos_p=offset,
        rad=jnp.full(5, 0.5),
        clump_id=[0, 0, 1, 1, 2],
        vel=jnp.zeros((5, dim)).at[2:4, 1].set(0.2),
    )
    material = jd.Material.create(
        "elasticfrict", young=100.0, poisson=0.0, density=1.0, e=1.0, mu=10.0, mu_r=0.3
    )
    system = jd.System.create(
        state=state,
        collider_type="NeighborList",
        collider_kw={"max_neighbors": 5, "skin": 0.1, "cutoff": 1.0},
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(dim, 8.0)},
        force_model_type="cundallstrack",
        mat_table=jd.MaterialTable.from_materials([material]),
        dt=0.01,
    )
    state, system = jd.System.initialize(state, system)
    state, system = system.collider.compute_force(state, system)
    system = replace(
        system,
        force_manager=replace(
            system.force_manager,
            external_force=jnp.ones_like(state.force),
            external_torque=jnp.ones_like(state.torque),
        ),
    )
    before = [np.asarray(x).copy() for x in jax.tree.leaves((state, system))]
    history = history_by_pair(system)
    assert any(np.any(value != 0) for value in history.values())
    result = analyze_packing(state, system, zc=4)
    path = str(tmp_path / "analysis.h5")
    jd.utils.h5.save(result, path)
    result = jd.utils.h5.load(path)
    np.testing.assert_array_equal(result.rattler_ids, [2])
    for data in (result.full, result.non_rattlers):
        current = history_by_pair(data.system)
        assert current.keys() == history.keys()
        for key, value in current.items():
            np.testing.assert_array_equal(value, history[key])
        np.testing.assert_array_equal(data.system.force_manager.external_force, 1)
        np.testing.assert_array_equal(data.system.force_manager.external_torque, 1)
        evaluated, _ = jd.System.evaluate_forces(data.state, data.system)
        np.testing.assert_allclose(
            data.state.force, evaluated.force, rtol=3e-5, atol=3e-5
        )
        np.testing.assert_allclose(
            data.state.torque, evaluated.torque, rtol=3e-5, atol=3e-5
        )
    for old, new in zip(before, jax.tree.leaves((state, system)), strict=True):
        np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize("with_species", [False, True])
@pytest.mark.parametrize("case", ["spheres", "clumps", "all_rattlers", "empty"])
def test_h5_analysis_round_trip(case, with_species, tmp_path):
    if case == "clumps":
        from .test_hessian_candidates import rigid_state

        state = rigid_state(2)
    elif case == "spheres":
        state = square_with_rattler()
    else:
        n = 1 if case == "all_rattlers" else 0
        state = jd.State.create(pos=np.zeros((n, 2)), rad=np.full(n, 0.5))
    species = np.arange(np.unique(state.clump_id).size) if with_species else None
    result = analyze_packing(state, make_system(state), clump_species_ids=species, zc=1)
    path = str(tmp_path / "analysis.h5")
    jd.utils.h5.save(result, path)
    restored = jd.utils.h5.load(path)
    assert type(restored) is type(result)
    np.testing.assert_array_equal(restored.rattler_ids, result.rattler_ids)
    np.testing.assert_array_equal(restored.non_rattler_ids, result.non_rattler_ids)
    assert restored.rattler_proportion == result.rattler_proportion
    for old, new in (
        (result.full, restored.full),
        (result.non_rattlers, restored.non_rattlers),
    ):
        assert type(new) is type(old)
        for item in fields(old):
            before, after = getattr(old, item.name), getattr(new, item.name)
            if item.name in ("state", "system"):
                assert type(after) is type(before)
                if item.name == "state":
                    for a, b in zip(
                        jax.tree.leaves(before), jax.tree.leaves(after), strict=True
                    ):
                        np.testing.assert_array_equal(a, b)
                else:
                    assert type(after.domain) is type(before.domain)
                    assert type(after.collider) is type(before.collider)
                    assert after.collider.max_neighbors == before.collider.max_neighbors
                continue
            if item.name in ("sphere_contacts", "clump_contacts"):
                assert type(after) is type(before)
                for network_field in fields(before):
                    a, b = (
                        getattr(before, network_field.name),
                        getattr(after, network_field.name),
                    )
                    if a is None:
                        assert b is None
                    else:
                        np.testing.assert_array_equal(a, b)
            else:
                if not isinstance(before, jax.Array):
                    assert type(after) is type(before)
                np.testing.assert_array_equal(before, after)
        if new.state.N:
            evaluated, _ = jd.System.evaluate_forces(new.state, new.system)
            np.testing.assert_allclose(evaluated.force, new.state.force, atol=2e-6)
            np.testing.assert_allclose(evaluated.torque, new.state.torque, atol=2e-6)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize(
    "force,coincident,expected",
    [
        ([2, 1], False, 0.5),
        ([0, 2], False, np.inf),
        ([0, 0], False, 0),
        ([2, 0], True, np.nan),
    ],
)
def test_group_friction_and_intrinsic_moments_have_no_validity_flag(
    dim, force, coincident, expected
):
    pos = np.zeros((2, dim))
    pos[1, 0] = 0 if coincident else 1
    state = jd.State.create(pos=pos, rad=jnp.full(2, 0.5))
    system = make_system(state)
    force = np.pad(force, (0, dim - 2))
    pairs = jnp.array([[0, 1], [1, 0]])
    data = contact.ContactData(
        pairs,
        jnp.asarray([force, -force]),
        jnp.ones((2, 1 if dim == 2 else 3)),
        jnp.asarray([pos[0] - pos[1], pos[1] - pos[0]]),
    )
    _, _, groups = contact.get_group_contacts(state, system, contacts=data)
    np.testing.assert_array_equal(groups.pair_ids, pairs)
    np.testing.assert_allclose(groups.friction, expected, equal_nan=True)
    np.testing.assert_array_equal(groups.torques, 1)
    assert "friction_valid" not in groups._fields


@pytest.mark.parametrize("dim", [2, 3])
def test_group_moments_shift_from_member_coms_to_periodic_centroid(dim):
    pos = np.pad(np.array([[7.8, 1.0], [0.2, 1.0], [0.8, 1.0]]), ((0, 0), (0, dim - 2)))
    state = jd.State.create(pos=pos, rad=jnp.full(3, 0.5))
    force = jnp.zeros((2, dim)).at[:, 1].set(jnp.array([1.0, -1.0]))
    torque = jnp.zeros((2, 1 if dim == 2 else 3)).at[:, -1].set(0.3)
    system = make_system(state)
    pairs = jnp.array([[0, 2], [2, 0]])
    data = contact.ContactData(
        pairs,
        force,
        torque,
        system.domain.displacement(
            state.pos[pairs[:, 0]], state.pos[pairs[:, 1]], system
        ),
    )
    _, _, groups = contact.get_group_contacts(
        state, system, group_by=jnp.array([7, 7, 8]), contacts=data
    )
    np.testing.assert_allclose(groups.torques[:, -1], [0.1, 0.3], atol=1e-6)
    np.testing.assert_allclose(groups.displacements[:, 0], [-0.8, 0.8], atol=1e-6)


@jax.tree_util.register_dataclass
@dataclass
class ConstantPairEnergy(jd.ForceModel):
    pair_energy_offset: jax.Array = field(default_factory=lambda: jnp.asarray(2.0))

    def search_radii(self, state, system):
        return state.rad

    @staticmethod
    def force(i, j, pos, state, system, history, *, advance_history=True):
        return jnp.zeros_like(state.force[i]), jnp.zeros_like(state.torque[i]), history

    @staticmethod
    def energy(i, j, pos, state, system):
        dr = system.domain.displacement(pos[i], pos[j], system)
        return jnp.where(
            (i != j) & (jnp.sum(dr * dr) < (state.rad[i] + state.rad[j]) ** 2),
            system.force_model.pair_energy_offset,
            0.0,
        )


def test_pair_energy_is_evaluated_even_when_all_contact_forces_are_zero():
    state = jd.State.create(pos=[[1.0, 1.0], [1.5, 1.0]], rad=jnp.full(2, 0.5))
    result = analyze_packing(state, make_system(state, law=ConstantPairEnergy()))
    assert result.full.sphere_contacts.pair_ids.shape == (0, 2)
    assert result.full.potential_energy == 2


def test_gapped_clump_ids_map_species_and_hessian_without_empty_body_modes():
    state = jd.State.create(
        pos=[[1.0, 1.0], [1.8, 1.0], [1.0, 1.2], [1.8, 1.2]], rad=jnp.full(4, 0.5)
    )
    state = replace(
        state,
        clump_id=jnp.array([1, 3, 1, 3]),
        pos_c=jnp.array([[1.0, 1.1], [1.8, 1.1], [1.0, 1.1], [1.8, 1.1]]),
        pos_p=jnp.array([[0.0, -0.1], [0.0, -0.1], [0.0, 0.1], [0.0, 0.1]]),
    )
    result = analyze_packing(state, make_system(state), zc=1, clump_species_ids=[8, 9])
    data = result.full
    assert result.non_rattlers is data
    np.testing.assert_array_equal(data.original_clump_ids, [-1, 1, -1, 3])
    np.testing.assert_array_equal(data.clump_contacts.pair_ids, [[1, 3], [3, 1]])
    np.testing.assert_array_equal(data.clump_contacts.pair_species, [[8, 9], [9, 8]])
    assert data.hessian.shape == (6, 6)
    assert data.mean_clump_contacts == 1
    assert_packed(data.clump_contacts, 4)


@pytest.mark.parametrize(
    "options",
    [
        {"clump_species_ids": [1, 2]},
        {"clump_species_ids": [1.0, 2.0, 3.0, 4.0, 5.0]},
        {"coordinates": "bad"},
        {"rotation_scale": [1] * 5},
        {"global_modes": -1},
        {"zero_mode_rel_gap": np.nan},
        {"zero_mode_atol": -1},
    ],
)
def test_invalid_analysis_options(options):
    state = square_with_rattler()
    with pytest.raises(ValueError):
        analyze_packing(state, make_system(state), **options)


def test_overflow_fails_before_analysis():
    state = square_with_rattler()
    system = jd.System.create(
        state=state,
        collider_type="NeighborList",
        collider_kw={"max_neighbors": 0, "cutoff": 1.0},
        domain_type="periodic",
        domain_kw={"box_size": jnp.full(2, 8.0)},
    )
    with pytest.raises(ValueError, match="overflow"):
        analyze_packing(state, system)
