# SPDX-License-Identifier: BSD-3-Clause
"""Capacity measurements agree with geometric candidates and pooled builds."""

import math
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from jaxdem.colliders._neighbor_cache import check_and_rebuild
from jaxdem.utils import estimate_neighbor_capacity, measure_neighbor_candidates


def _system(
    dim=2,
    *,
    clumps=False,
    history=False,
    domain="periodic",
    backend="CellList",
    n_bodies=6,
):
    rng = np.random.default_rng(8)
    centers = rng.uniform(0, 4, (n_bodies, dim))
    kwargs = {}
    if clumps:
        centers = np.repeat(centers, 3, axis=0)
        offsets = np.zeros((3, dim))
        offsets[0, 0], offsets[1, 0] = -0.15, 0.15
        kwargs = {
            "pos_p": np.tile(offsets, (n_bodies, 1)),
            "clump_id": np.repeat(np.arange(n_bodies), 3),
        }
    state = jd.State.create(pos=centers, rad=np.full(len(centers), 0.2), **kwargs)
    system_kwargs = {}
    if history:
        material = jd.Material.create(
            "elasticfrict",
            young=100.0,
            poisson=0.0,
            density=1.0,
            e=1.0,
            mu=0.5,
            mu_r=0.0,
        )
        system_kwargs = {
            "force_model_type": "cundallstrack",
            "mat_table": jd.MaterialTable.from_materials([material]),
        }
    system = jd.System.create(
        state=state,
        domain_type=domain,
        domain_kw={
            "box_size": jnp.full(dim, 4.0),
            **({"gamma": 0.37} if domain == "leesedwards" else {}),
        },
        collider_type="NeighborList",
        collider_kw={
            "cutoff": 0.5,
            "skin": 0.2,
            "max_neighbors": state.N if history else 0,
            "secondary_collider_type": backend,
        },
        **system_kwargs,
    )
    return state, system


def _reference_counts(state, system, cutoff):
    pos = np.asarray(state.pos)
    box = np.asarray(system.domain.box_size)
    dr = pos[:, None, :] - pos[None, :, :]
    if system.domain.periodic:
        if hasattr(system.domain, "gamma"):
            a, b = system.domain.alpha, system.domain.beta
            images = np.round(dr[..., b] / box[b])
            dr[..., a] -= images * float(system.domain.gamma) * box[b]
        dr -= box * np.round(dr / box)
    valid = np.sum(dr * dr, axis=-1) <= cutoff**2
    valid &= np.asarray(state.clump_id)[:, None] != np.asarray(state.clump_id)[None, :]
    if not bool(system.interact_same_bond_id):
        bonds = np.asarray(state.bond_id)
        for src in range(state.N):
            for dst in range(state.N):
                if src in bonds[dst]:
                    valid[src, dst] = False
    return valid.sum(axis=1)


@pytest.mark.parametrize("dim", [2, 3])
@pytest.mark.parametrize("domain", ["periodic", "leesedwards"])
@pytest.mark.parametrize("backend", ["naive", "CellList", "MultiCellList"])
def test_counts_match_geometry_and_cache_without_trial_capacity(dim, domain, backend):
    state, system = _system(
        dim, clumps=True, domain=domain, backend=backend, n_bodies=32
    )
    # Directed bond exclusions also exercise counts whose sum is not even.
    bonds = jnp.full((state.N, 1), -1).at[0, 0].set(3)
    state = replace(state, bond_id=bonds)
    stats = measure_neighbor_candidates(state, system, cutoff=1.2)
    expected = _reference_counts(state, system, stats.list_cutoff)
    np.testing.assert_array_equal(stats.degree, expected)
    assert stats.directed_pair_count == int(expected.sum())
    assert stats.max_degree == int(expected.max())
    assert stats.mean_degree == pytest.approx(expected.mean())

    collider = jd.Collider.create(
        "NeighborList",
        state=state,
        cutoff=1.2,
        skin=0.2,
        max_neighbors=math.ceil(stats.mean_degree),
        secondary_collider_type=backend,
    )
    built = check_and_rebuild(state, replace(system, collider=collider))
    np.testing.assert_array_equal(jnp.diff(built.collider.row_offsets), stats.degree)
    assert not bool(built.collider.overflow)


def test_skin_search_radius_and_bond_settings():
    state = jd.State.create(pos=[[0.0, 0.0], [0.65, 0.0]], rad=[0.2, 0.2])
    system = jd.System.create(state=state)
    assert measure_neighbor_candidates(state, system, skin=0).directed_pair_count == 0
    assert measure_neighbor_candidates(state, system, skin=0.3).directed_pair_count == 2
    assert measure_neighbor_candidates(
        state, system, skin_fraction=1
    ).list_cutoff == pytest.approx(0.8)
    # Search radii can exceed physical radii even with a small configured cutoff.
    state = replace(state, _rad=jnp.full(2, 0.4))
    stats = measure_neighbor_candidates(state, system, cutoff=0.1, skin=0)
    assert stats.list_cutoff == pytest.approx(0.8)
    assert stats.directed_pair_count == 2
    state = replace(state, bond_id=jnp.array([[1], [0]]))
    assert measure_neighbor_candidates(state, system).directed_pair_count == 0
    assert (
        measure_neighbor_candidates(
            state, replace(system, interact_same_bond_id=True)
        ).directed_pair_count
        == 2
    )


def test_periodic_seam_and_skin_only_pairs():
    state = jd.State.create(pos=[[0.05, 0.0], [1.8, 0.0]], rad=[0.1, 0.1])
    system = jd.System.create(
        state=state, domain_type="periodic", domain_kw={"box_size": jnp.full(2, 2.0)}
    )
    assert measure_neighbor_candidates(state, system, skin=0).directed_pair_count == 0
    stats = measure_neighbor_candidates(state, system, skin=0.1)
    collider = jd.Collider.create(
        "NeighborList", state=state, cutoff=0.2, skin=0.1, max_neighbors=1
    )
    built = check_and_rebuild(state, replace(system, collider=collider))
    np.testing.assert_array_equal(
        stats.degree,
        [1, 1],
        err_msg=f"Cached builder counts: {np.asarray(jnp.diff(built.collider.row_offsets))}",
    )
    np.testing.assert_array_equal(jnp.diff(built.collider.row_offsets), [1, 1])


@pytest.mark.parametrize("dim", [2, 3])
def test_empty_and_single_sphere_counts(dim):
    state = jd.State.create(dim=dim)
    system = jd.System.create(state=state)
    stats = measure_neighbor_candidates(state, system)
    assert stats.degree.shape == (0,)
    assert stats.directed_pair_count == stats.max_degree == stats.mean_degree == 0
    state = jd.State.create(pos=jnp.zeros((1, dim)), rad=jnp.ones(1))
    stats = measure_neighbor_candidates(state, jd.System.create(state=state))
    np.testing.assert_array_equal(stats.degree, [0])


@pytest.mark.parametrize("dim", [2, 3])
def test_probe_geometry_convergence_capacity_and_input_history(dim, monkeypatch):
    state, system = _system(dim, clumps=True, history=True)
    system = check_and_rebuild(state, system)
    # Seed physical contact history without advancing time.
    collider = replace(
        system.collider,
        history=jnp.full_like(
            system.force_model.init_history(system.collider.neighbor_list.shape, dim),
            0.125,
        ),
    )
    system = replace(system, collider=collider)

    def unexpected_evaluation(*args, **kwargs):
        raise AssertionError("Sizing must not evaluate the physical contact law.")

    monkeypatch.setattr(jd.forces.CundallStrackForce, "force", unexpected_evaluation)
    monkeypatch.setattr(jd.forces.CundallStrackForce, "energy", unexpected_evaluation)
    before = [np.array(x) for x in jax.tree.leaves((state, system))]
    result = estimate_neighbor_capacity(
        state,
        system,
        0.25,
        max_steps=10_000,
        force_tol=1e-7,
        orientation_samples=3,
        safety_factor=1.5,
        extra_neighbors=2,
        round_to=4,
    )
    after = jax.tree.leaves((state, system))
    for old, new in zip(before, after, strict=True):
        np.testing.assert_array_equal(old, new)
    assert bool(result.relaxation_info.converged)
    assert float(result.relaxation_info.force_max) <= 1e-7
    np.testing.assert_allclose(result.bounding_radii, np.full(6, 0.35), rtol=1e-6)
    volume = math.pi ** (dim / 2) / math.gamma(dim / 2 + 1)
    phi = volume * np.sum(result.bounding_radii**dim) / np.prod(result.box_size)
    assert phi == pytest.approx(0.25, rel=1e-6)
    assert len(result.samples) == 3
    assert result.max_neighbors % 4 == result.query_max_neighbors % 4 == 0
    largest_mean = max(s.mean_degree for s in result.samples)
    largest_max = max(s.max_degree for s in result.samples)
    assert (
        result.max_neighbors
        == math.ceil(max(1.5 * largest_mean, largest_mean + 2) / 4) * 4
    )
    assert (
        result.query_max_neighbors
        == math.ceil(max(1.5 * largest_max, largest_max + 2) / 4) * 4
    )


def test_single_sphere_species_need_one_orientation_sample():
    state, system = _system()
    state = replace(
        state,
        rad=jnp.array([0.2, 0.2, 0.2, 0.28, 0.28, 0.28]),
        _rad=jnp.array([0.2, 0.2, 0.2, 0.28, 0.28, 0.28]),
    )
    result = estimate_neighbor_capacity(state, system, 0.2, max_steps=10_000)
    assert len(result.samples) == 1
    np.testing.assert_allclose(result.bounding_radii, state.rad)
    assert bool(result.relaxation_info.converged)


def test_failed_relaxation_does_not_produce_a_recommendation():
    state, system = _system()
    with pytest.raises(RuntimeError, match="MAX_STEPS.*0 steps"):
        estimate_neighbor_capacity(state, system, 1.1, max_steps=0)


def test_pooled_count_allows_a_degree_larger_than_the_average_budget():
    state = jd.State.create(
        pos=[
            [0, 0],
            [0.8, 0],
            [-0.8, 0],
            [0, 0.8],
            [0, -0.8],
            [10, 0],
            [20, 0],
            [30, 0],
        ],
        rad=jnp.full(8, 0.5),
    )
    system = jd.System.create(state=state)
    stats = measure_neighbor_candidates(state, system, skin=0)
    np.testing.assert_array_equal(stats.degree, [4, 1, 1, 1, 1, 0, 0, 0])
    assert stats.mean_degree == 1
    collider = jd.Collider.create(
        "NeighborList", state=state, cutoff=1.0, skin=0, max_neighbors=1
    )
    built = check_and_rebuild(state, replace(system, collider=collider))
    assert not bool(built.collider.overflow)
    assert int(built.collider.row_offsets[-1]) == 8


def test_probe_preserves_shear_aspect_ratio_and_length_scale():
    state, system = _system(domain="leesedwards", clumps=True)
    box = jnp.array([4.0, 6.0])
    system = replace(
        system, domain=replace(system.domain, box_size=box, inv_box_size=1 / box)
    )
    first = estimate_neighbor_capacity(state, system, 0.3, orientation_samples=2)
    length = 1e-4
    anchor = jnp.array([0.0001, -0.0002])
    scaled = replace(
        state,
        pos_c=state.pos_c * length + anchor,
        pos_p=state.pos_p * length,
        rad=state.rad * length,
        _rad=state._rad * length,
    )
    scaled_system = replace(
        system,
        domain=replace(
            system.domain,
            box_size=box * length,
            inv_box_size=1 / (box * length),
            anchor=anchor,
        ),
        collider=replace(
            system.collider,
            cutoff=system.collider.cutoff * length,
            skin=system.collider.skin * length,
        ),
    )
    second = estimate_neighbor_capacity(
        scaled, scaled_system, 0.3, orientation_samples=2
    )
    np.testing.assert_allclose(second.box_size / length, first.box_size, rtol=1e-6)
    assert first.box_size[1] / first.box_size[0] == pytest.approx(1.5)
    assert second.max_neighbors == first.max_neighbors
    for original, scaled_stats in zip(first.samples, second.samples, strict=True):
        np.testing.assert_array_equal(scaled_stats.degree, original.degree)


def test_coincident_analogue_centers_are_separated_before_relaxation():
    state, system = _system()
    state = replace(state, pos_c=jnp.zeros_like(state.pos_c))
    result = estimate_neighbor_capacity(state, system, 0.2, max_steps=10_000)
    assert result.relaxation_steps > 0
    assert bool(result.relaxation_info.converged)


def test_hash_overflow_rejects_incomplete_counts():
    state, system = _system(n_bodies=96)
    huge_box = jnp.full(2, 1e20)
    system = replace(
        system,
        domain=replace(system.domain, box_size=huge_box, inv_box_size=1 / huge_box),
    )
    with pytest.raises(RuntimeError, match="Spatial hash overflow"):
        measure_neighbor_candidates(state, system)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"packing_fraction": 0},
        {"packing_fraction": np.nan},
        {"safety_factor": 0.9},
        {"safety_factor": np.inf},
        {"orientation_samples": 0},
        {"orientation_samples": 1.5},
        {"max_steps": -1},
        {"max_steps": True},
        {"extra_neighbors": -1},
        {"round_to": 0},
        {"skin": -1},
        {"skin": 0.1, "skin_fraction": 0.1},
        {"cutoff": np.inf},
        {"force_tol": np.nan},
    ],
)
def test_invalid_estimator_options(kwargs):
    state, system = _system()
    with pytest.raises(ValueError):
        estimate_neighbor_capacity(
            state, system, **({"packing_fraction": 0.5} | kwargs)
        )


def test_rejects_nonperiodic_or_empty_probes():
    state = jd.State.create(pos=[[0.0, 0.0]], rad=[1.0])
    with pytest.raises(ValueError, match="periodic"):
        estimate_neighbor_capacity(state, jd.System.create(state=state), 0.5)
    state = jd.State.create(dim=2)
    with pytest.raises(ValueError, match="At least one"):
        estimate_neighbor_capacity(state, jd.System.create(state=state), 0.5)
