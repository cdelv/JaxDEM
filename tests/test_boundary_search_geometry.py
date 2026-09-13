# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jdem
from jaxdem.colliders import DynamicCellList, DynamicMultiCellList
from jaxdem.colliders._partition import _grid_params
from jaxdem.domains import Domain, LeesEdwardsDomain, SearchGeometry


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _UndeclaredSearchDomain(Domain):
    pass


@jax.tree_util.register_dataclass
@dataclass(slots=True)
class _OrthogonalSearchDomain(Domain):
    search_geometry = SearchGeometry.ORTHOGONAL


def set_up_spheres(
    dim: int,
    n: int = 64,
    domain_type: str = "periodic",
    collider_type: str = "naive",
):
    """Build a small sphere system without changing global JAX configuration."""
    spacing = 1.2
    radius = 0.5
    n_per_axis_value = int(n ** (1 / dim))
    state = jdem.utils.grid_state(
        n_per_axis=(n_per_axis_value,) * dim,
        spacing=spacing,
        radius_range=(radius, radius),
        vel_range=[-1.0, 1.0],
        seed=42,
    )
    collider_kw = (
        {"state": state}
        if collider_type.lower()
        in {
            "celllist",
            "multicelllist",
        }
        else {}
    )
    material_table = jdem.MaterialTable.from_materials(
        [jdem.Material.create("elastic", density=0.27, young=1.0e3, poisson=0.3)]
    )
    system = jdem.System.create(
        state.shape,
        domain_type=domain_type,
        domain_kw={
            "box_size": (spacing * n_per_axis_value,) * dim,
            "anchor": (-radius,) * dim,
        },
        collider_type=collider_type,
        collider_kw=collider_kw,
        dt=0.001,
        mat_table=material_table,
    )
    return state, system


def test_free_domain_apply_reduces_per_particle_radii_to_domain_geometry():
    state, system = set_up_spheres(2, n=4, domain_type="free")
    state, system = system.domain.apply(state, system)
    expected_min = jnp.min(state.pos - state.rad[:, None], axis=0)
    expected_max = jnp.max(state.pos + state.rad[:, None], axis=0)
    assert system.domain.anchor.shape == (state.dim,)
    assert system.domain.box_size.shape == (state.dim,)
    np.testing.assert_allclose(system.domain.anchor, expected_min)
    np.testing.assert_allclose(system.domain.box_size, expected_max - expected_min)


def test_hashed_collider_requires_custom_domain_search_opt_in():
    state = jdem.State.create(pos=jnp.zeros((2, 2)), rad=jnp.full(2, 0.5))
    undeclared = _UndeclaredSearchDomain.Create(dim=2)
    collider = jdem.Collider.create("CellList", state=state, cell_size=1.0)

    with pytest.raises(ValueError, match="does not declare a search geometry"):
        collider.validate_domain(undeclared)
    with pytest.raises(ValueError, match="does not declare a search geometry"):
        jdem.System.create(state=state, collider=collider, domain=undeclared)

    opted_in = _OrthogonalSearchDomain.Create(dim=2)
    system = jdem.System.create(state=state, collider=collider, domain=opted_in)
    assert system.domain is opted_in


@pytest.mark.parametrize(
    "domain_type", ["free", "periodic", "reflect", "reflectsphere", "leesedwards"]
)
def test_builtin_domains_match_cell_search_capabilities(domain_type):
    state = jdem.State.create(pos=jnp.zeros((2, 2)), rad=jnp.full(2, 0.5))
    domain = jdem.Domain.create(domain_type, dim=2)
    collider = jdem.Collider.create("CellList", state=state, cell_size=1.0)
    collider.validate_domain(domain)


def test_grid_hash_dtype_and_preoverflow_guard():
    hash_dtype = jnp.uint64 if jax.config.jax_enable_x64 else jnp.uint32
    side = 2**32 if jax.config.jax_enable_x64 else 2**16
    dims, strides, _, overflow = _grid_params(
        jnp.asarray([side, side], dtype=float), jnp.asarray(1.0), False
    )
    assert dims.dtype == hash_dtype
    assert strides.dtype == hash_dtype
    assert bool(overflow)


@pytest.mark.parametrize("x64,bits", [("0", 32), ("1", 64)])
def test_unsigned_hash_boundary_in_fresh_process(x64, bits):
    code = f"""
import jax
import jax.numpy as jnp
import numpy as np
from jaxdem.colliders._partition import _grid_params
from jaxdem.colliders.cell_list import _get_spatial_partition
from jaxdem.domains import PeriodicDomain
from typing import NamedTuple
assert jax.config.jax_enable_x64 is {x64 == '1'}
dtype = jnp.uint{bits}
safe = 2 ** ({bits} // 2) - 1
unsafe = 2 ** ({bits} // 2)
for side, expected in [(safe, False), (unsafe, True)]:
    dims, strides, _, overflow = _grid_params(
        jnp.asarray([float(side), float(side)]), jnp.asarray(1.0), False)
    assert dims.dtype == dtype and strides.dtype == dtype
    assert bool(overflow) is expected
if {x64 == '0'}:
    class TinySystem(NamedTuple):
        domain: object
    domain = PeriodicDomain.Create(1, box_size=jnp.asarray([float(2**31 + 1024)]))
    system = TinySystem(domain)
    _, hashes, stencil, overflow = _get_spatial_partition(
        jnp.asarray([[float(2**31 + 7)]]), system, jnp.asarray(1.0),
        jnp.asarray([[-1], [0], [1]], dtype=jnp.int32), jnp.asarray([0], dtype=jnp.int32))
    assert hashes.dtype == jnp.uint32 and stencil.dtype == jnp.uint32
    assert not bool(overflow)
    sentinel = jnp.asarray(np.iinfo(np.uint32).max, dtype=jnp.uint32)
    assert bool(jnp.all(stencil != sentinel))
"""
    env = os.environ.copy()
    env["JAX_ENABLE_X64"] = x64
    env["JAX_PLATFORMS"] = "cpu"
    root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


@pytest.mark.parametrize("collider_cls", [DynamicCellList, DynamicMultiCellList])
@pytest.mark.parametrize("alpha,beta", [(0, 1), (1, 0)])
def test_fractional_le_cross_image_stencil(collider_cls, alpha, beta):
    state, base = set_up_spheres(2, n=4, collider_type="CellList")
    domain = LeesEdwardsDomain.Create(
        2,
        box_size=jnp.asarray([4.0, 4.0]),
        anchor=jnp.zeros(2),
        gamma=0.375,
        alpha=alpha,
        beta=beta,
    )
    collider = collider_cls.Create(
        state, cell_size=1.0, search_range=1, box_size=domain.box_size
    )
    system = replace(base, domain=domain, collider=collider)

    query = np.asarray([0.2, 0.1])
    image_neighbor = np.asarray([1.7, 3.9])
    if alpha == 1:
        query = query[::-1].copy()
        image_neighbor = image_neighbor[::-1].copy()

    nl, overflow = collider.create_cross_neighbor_list(
        jnp.asarray(query[None]),
        jnp.asarray(image_neighbor[None]),
        system,
        cutoff=0.3,
        max_neighbors=1,
    )
    np.testing.assert_array_equal(nl, [[0]])
    assert not bool(overflow)


@pytest.mark.parametrize("collider_cls", [DynamicCellList, DynamicMultiCellList])
def test_3d_unwrapped_tiny_box_matches_independent_image_oracle(collider_cls):
    state, base = set_up_spheres(3, n=8, collider_type="CellList")
    domain = LeesEdwardsDomain.Create(
        3,
        box_size=jnp.asarray([1.2, 1.0, 1.4]),
        anchor=jnp.asarray([-0.3, 0.2, -0.5]),
        gamma=0.37,
        alpha=2,
        beta=0,
    )
    collider = collider_cls.Create(state, cell_size=0.45, search_range=1)
    system = replace(base, domain=domain, collider=collider)
    query = np.asarray([[-2.72, 0.41, -1.83], [1.44, -1.78, 2.31]])
    database = np.asarray([[2.02, 0.43, -0.08], [-1.18, 2.23, 1.77], [0.1, 0.7, 0.1]])
    cutoff = 0.43  # below half every box length: unique-image regime

    expected = []
    box = np.asarray(domain.box_size)
    for qa in query:
        row = []
        for j, qb in enumerate(database):
            distances = []
            for nb in range(-4, 5):
                for n1 in range(-4, 5):
                    for n2 in range(-4, 5):
                        image = qb + np.asarray([nb * box[0], n1 * box[1], n2 * box[2]])
                        image[2] += nb * float(domain.gamma) * box[0]
                        distances.append(np.linalg.norm(qa - image))
            if min(distances) <= cutoff:
                row.append(j)
        expected.append(row)

    nl, overflow = collider.create_cross_neighbor_list(
        jnp.asarray(query), jnp.asarray(database), system, cutoff, 3
    )
    assert not bool(overflow)
    for actual, wanted in zip(np.asarray(nl), expected, strict=True):
        assert set(actual[actual >= 0]) == set(wanted)


@pytest.mark.parametrize("collider_cls", [DynamicCellList, DynamicMultiCellList])
@pytest.mark.parametrize(
    "dim,alpha,beta,gamma",
    [(2, 0, 1, -0.43), (2, 1, 0, 0.31), (3, 0, 2, -0.27), (3, 2, 1, 0.38)],
)
def test_random_unwrapped_le_search_matches_physical_images(
    collider_cls, dim, alpha, beta, gamma
):
    """Compare hashing with physical replicas, independently of displacement."""
    rng = np.random.default_rng(9182 + 100 * dim + 10 * alpha + beta)
    box = np.asarray([1.1, 1.35, 0.95][:dim])
    anchor = np.asarray([-0.4, 0.2, -0.15][:dim])
    state, base = set_up_spheres(dim, n=8, collider_type="CellList")
    domain = LeesEdwardsDomain.Create(
        dim,
        box_size=jnp.asarray(box),
        anchor=jnp.asarray(anchor),
        gamma=gamma,
        alpha=alpha,
        beta=beta,
    )
    collider = collider_cls.Create(state, cell_size=0.31, search_range=2)
    system = replace(base, domain=domain, collider=collider)
    query = anchor + rng.random((7, dim)) * box
    database = anchor + rng.random((9, dim)) * box
    query_images = rng.integers(-3, 4, size=query.shape)
    database_images = rng.integers(-3, 4, size=database.shape)
    query += query_images * box
    database += database_images * box
    query[:, alpha] += query_images[:, beta] * gamma * box[beta]
    database[:, alpha] += database_images[:, beta] * gamma * box[beta]
    cutoff = 0.28

    expected = []
    for qa in query:
        row = []
        for j, qb in enumerate(database):
            nearest = np.inf
            for image_index in np.ndindex(*([17] * dim)):
                images = np.asarray(image_index) - 8
                image = qb + images * box
                image[alpha] += images[beta] * gamma * box[beta]
                nearest = min(nearest, np.linalg.norm(qa - image))
            if nearest <= cutoff:
                row.append(j)
        expected.append(row)

    nl, overflow = collider.create_cross_neighbor_list(
        jnp.asarray(query), jnp.asarray(database), system, cutoff, len(database)
    )
    assert not bool(overflow)
    for actual, wanted in zip(np.asarray(nl), expected, strict=True):
        assert set(actual[actual >= 0]) == set(wanted)


@pytest.mark.parametrize("collider_type", ["naive", "CellList", "MultiCellList"])
def test_zero_capacity_reports_real_overflow(collider_type):
    state, system = set_up_spheres(2, n=4, collider_type=collider_type)
    _, _, nl, overflow = system.collider.create_neighbor_list(
        state, system, cutoff=10.0, max_neighbors=0
    )
    assert nl.shape == (state.N, 0)
    assert bool(overflow)


def test_neighbor_cache_rebuilds_when_skin_increases():
    state = jdem.State.create(
        pos=jnp.array([[0.0, 0.0], [1.6, 0.0]]), rad=jnp.full(2, 0.5)
    )
    system = jdem.System.create(
        state=state,
        force_model_type="spring",
        collider_type="NeighborList",
        collider_kw={"cutoff": 1.0, "skin": 0.1, "max_neighbors": 4},
    )
    state, system = system.collider.compute_force(state, system)
    first_build = system.collider.n_build_times

    moved = replace(state, pos_c=jnp.array([[0.31, 0.0], [1.29, 0.0]]))
    system = replace(system, collider=replace(system.collider, skin=jnp.asarray(1.0)))
    actual, system = system.collider.compute_force(moved, system)
    naive = replace(system, collider=jdem.Collider.create("naive"))
    expected, _ = naive.collider.compute_force(moved, naive)

    assert int(system.collider.n_build_times) == int(first_build) + 1
    assert bool(jnp.allclose(actual.force, expected.force))
    assert bool(jnp.any(actual.force != 0))


def test_neighbor_cache_rebuilds_when_lees_edwards_axes_change():
    state = jdem.State.create(
        pos=jnp.array([[0.0, 0.0, 0.1], [0.0, 4.0, 9.9]]), rad=jnp.ones(2)
    )
    box = jnp.full(3, 10.0)
    system = jdem.System.create(
        state=state,
        force_model_type="spring",
        collider_type="NeighborList",
        collider_kw={"cutoff": 1.0, "skin": 0.2, "max_neighbors": 4},
        domain_type="LeesEdwards",
        domain_kw={"box_size": box, "gamma": 0.4, "alpha": 0, "beta": 1},
    )
    state, system = system.collider.compute_force(state, system)
    first_build = system.collider.n_build_times

    new_domain = jdem.Domain.create(
        "LeesEdwards", dim=3, box_size=box, gamma=0.4, alpha=1, beta=2
    )
    system = replace(system, domain=new_domain)
    actual, system = system.collider.compute_force(state, system)
    naive = replace(system, collider=jdem.Collider.create("naive"))
    expected, _ = naive.collider.compute_force(state, naive)

    assert int(system.collider.n_build_times) == int(first_build) + 1
    assert bool(jnp.allclose(actual.force, expected.force))
    assert bool(jnp.any(actual.force != 0))


def test_lees_edwards_partition_accepts_empty_queries():
    """The expanded shear stencil must have an explicit width for zero rows."""
    from jaxdem.colliders.cell_list import _get_spatial_partition

    state = jdem.State.create(pos=jnp.array([[0.1, 0.1]]), rad=jnp.array([0.1]))
    system = jdem.System.create(
        state=state,
        domain_type="leesedwards",
        domain_kw={"box_size": jnp.array([2.0, 2.0]), "gamma": 0.25},
    )
    _, _, stencil, overflow = _get_spatial_partition(
        state.pos[:0],
        system,
        jnp.asarray(1.0),
        jnp.array([[0, 0], [2, 0], [0, 0]]),
        jnp.empty((0,), dtype=int),
    )
    assert stencil.shape == (0, 6)
    assert not bool(overflow)
