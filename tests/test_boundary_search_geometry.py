# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check hashed neighbor searches against boundary-domain geometry."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from dataclasses import dataclass, replace
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import jaxdem as jdem
from jaxdem.colliders import DynamicCellList, DynamicMultiCellList
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
        if collider_type.lower() in {"celllist", "multicelllist"}
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
    state = jdem.State.create(
        pos=jnp.array([[0.0, 0.2], [1.2, -0.3], [0.4, 2.0], [2.5, 1.0]]),
        rad=jnp.array([0.1, 0.7, 0.25, 0.85]),
    )
    system = jdem.System.create(state=state, domain_type="free")
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

    for domain_type in ("free", "periodic", "reflect", "reflectsphere", "leesedwards"):
        collider.validate_domain(jdem.Domain.create(domain_type, dim=2))


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
assert jax.config.jax_enable_x64 is {x64 == "1"}
dtype = jnp.uint{bits}
safe = 2 ** ({bits} // 2) - 1
unsafe = 2 ** ({bits} // 2)
for side, expected in [(safe, False), (unsafe, True)]:
    dims, strides, _, overflow = _grid_params(
        jnp.asarray([float(side), float(side)]), jnp.asarray(1.0), False)
    assert dims.dtype == dtype and strides.dtype == dtype
    assert bool(overflow) is expected
if {x64 == "0"}:
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


@pytest.mark.parametrize(
    "collider_cls,dim,alpha,beta,gamma,search_range",
    [
        (DynamicCellList, 2, 0, 1, -0.43, 1),
        (DynamicMultiCellList, 2, 1, 0, 0.31, 2),
        (DynamicCellList, 3, 0, 2, -0.27, 2),
        (DynamicMultiCellList, 3, 2, 1, 0.38, 1),
    ],
)
def test_random_unwrapped_le_search_matches_physical_images(
    collider_cls, dim, alpha, beta, gamma, search_range
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
    collider = collider_cls.Create(state, cell_size=0.31, search_range=search_range)
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

    axes = [np.arange(-8, 9)] * dim
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dim)
    shifts = grid * box
    shifts[:, alpha] += grid[:, beta] * gamma * box[beta]
    diff = query[:, None, None, :] - (
        database[None, :, None, :] + shifts[None, None, :, :]
    )
    nearest = np.sqrt(np.min(np.sum(diff * diff, axis=-1), axis=-1))
    expected = [np.flatnonzero(row <= cutoff).tolist() for row in nearest]

    nl, overflow = collider.create_cross_neighbor_list(
        jnp.asarray(query), jnp.asarray(database), system, cutoff, len(database)
    )
    assert not bool(overflow)
    for actual, wanted in zip(np.asarray(nl), expected, strict=True):
        np.testing.assert_array_equal(np.sort(actual[actual >= 0]), sorted(wanted))
        assert np.all(actual[len(wanted) :] == -1)


def test_lees_edwards_partition_accepts_empty_queries():
    """The expanded shear stencil must handle zero query rows."""
    state = jdem.State.create(pos=jnp.array([[0.1, 0.1]]), rad=jnp.array([0.1]))
    system = jdem.System.create(
        state=state,
        collider_type="CellList",
        collider_kw={"cell_size": 1.0},
        domain_type="leesedwards",
        domain_kw={"box_size": jnp.array([2.0, 2.0]), "gamma": 0.25},
    )
    nl, overflow = system.collider.create_cross_neighbor_list(
        state.pos[:0], state.pos, system, cutoff=0.5, max_neighbors=2
    )
    assert nl.shape == (0, 2)
    assert not bool(overflow)
