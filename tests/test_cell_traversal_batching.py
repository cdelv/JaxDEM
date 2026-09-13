# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Traversal batching must preserve forces, energy and shared capacity."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from tests.test_sparse_neighbors import case


@pytest.mark.parametrize("collider", ["CellList", "MultiCellList"])
@pytest.mark.parametrize("domain", ["free", "periodic", "lees_edwards"])
def test_cell_batches_match_all_pairs_and_vmap(collider, domain, monkeypatch):
    from jaxdem.colliders import cell_list, multi_cell_list

    monkeypatch.setattr(cell_list, "PAIR_TRAVERSAL_BATCH_SIZE", 4)
    monkeypatch.setattr(multi_cell_list, "PAIR_TRAVERSAL_BATCH_SIZE", 4)
    pos = np.random.default_rng(45).uniform(0.0, 3.8, (11, 3))
    s, y = case(pos=pos, domain=domain)
    y = replace(y, collider=jd.Collider.create(collider, state=s, cell_size=1.15))
    ref = replace(y, collider=jd.Collider.create("naive"))
    got, gy = jd.System.initialize(s, y)
    want, wy = jd.System.initialize(s, ref)
    np.testing.assert_allclose(got.force, want.force, rtol=2e-5, atol=0.01)
    energy = lambda s, y: y.collider.compute_potential_energy(s, y)[2]
    np.testing.assert_allclose(energy(got, gy), energy(want, wy), rtol=2e-5, atol=0.01)
    # Include a replica with different occupancy and no shared physical history.
    other = replace(s, pos_c=s.pos_c * 0.7)
    states = jd.State.stack([s, other])
    systems = jd.System.stack([y, y])
    batched, by = jd.System.initialize(states, systems)
    batched, by = jd.System.step(batched, by, n=2)
    for i, one in enumerate([s, other]):
        expected, ey = jd.System.initialize(one, ref)
        expected, ey = jd.System.step(expected, ey, n=2)
        np.testing.assert_allclose(batched.pos[i], expected.pos, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(batched.vel[i], expected.vel, rtol=2e-5, atol=2e-6)
    assert not jnp.any(by.collider.overflow)


@pytest.mark.parametrize("n", [6, 7])
def test_search_batches_share_one_pool(n, monkeypatch):
    from jaxdem.colliders import _neighbor_cache

    monkeypatch.setattr(_neighbor_cache, "_SEARCH_BATCH_SIZE", 2)
    pos = np.zeros((n, 2))
    pos[:, 0] = np.arange(n) * 3
    pos[:3, 0] = [0.0, 0.2, 0.4]
    s, y = case(capacity=1, pos=pos)
    s, y = jd.System.initialize(s, y)
    assert not y.collider.overflow
    np.testing.assert_array_equal(jnp.diff(y.collider.row_offsets)[:3], [2, 2, 2])
    assert y.collider.row_offsets[-1] == 6
    # The first two-particle batch needs four slots, borrowing the unused pool.
    assert y.collider.row_offsets[2] > 2 * y.collider.max_neighbors


@pytest.mark.parametrize("collider", ["CellList", "MultiCellList"])
@pytest.mark.parametrize("n", [0, 1])
def test_cell_force_and_energy_handle_empty_or_single_particle(collider, n):
    s, y = case(pos=np.zeros((n, 2)), capacity=0)
    y = replace(y, collider=jd.Collider.create(collider, state=s))
    s, y = jd.System.initialize(s, y)
    s, y = jd.System.step(s, y, n=2)
    np.testing.assert_array_equal(s.force, 0.0)
    assert y.collider.compute_potential_energy(s, y)[2] == 0
    y.check_overflow()
