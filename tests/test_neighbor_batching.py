# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Pooled-cache batching must agree with independent simulations."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jaxdem as jd
from tests.test_sparse_neighbors import Remember, case


def assert_tree_close(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    leaves, _ = jax.tree_util.tree_flatten_with_path(actual)
    for (path, a), b in zip(leaves, jax.tree.leaves(expected), strict=True):
        name = jax.tree_util.keystr(path)
        # Symmetric contacts cancel O(1e4) forces; float32 fusion changes the
        # residual by about one ULP. Keep positions and history checks strict.
        force_field = name.endswith((".force", ".torque"))
        atol = 1e-3 if force_field and not jax.config.jax_enable_x64 else 2e-6
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=atol, err_msg=name)


def stack(pairs):
    return jd.State.stack([s for s, _ in pairs]), jd.System.stack([y for _, y in pairs])


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("history", [False, True])
def test_batched_lifecycle_matches_independent_runs(batch_size, history, monkeypatch):
    from jaxdem.colliders import _neighbor_cache

    monkeypatch.setattr(_neighbor_cache, "_ROW_BATCH_SIZE", 4)
    monkeypatch.setattr(_neighbor_cache, "_SEARCH_BATCH_SIZE", 4)
    pairs = []
    for degree in [3, 0, 5][:batch_size]:
        pos = np.zeros((9, 2))
        pos[:, 0] = np.arange(9) * 3
        pos[:degree, 0] = np.arange(degree) * 0.15
        pairs.append(case(capacity=4, pos=pos, force=Remember() if history else None))
    expected = [jd.System.initialize(s, y) for s, y in pairs]
    actual = jd.System.initialize(*stack(pairs))
    assert_tree_close(actual, stack(expected))

    # Different replicas rebuild on different steps, including an empty cache.
    expected = [
        (
            s,
            replace(
                y, collider=replace(y.collider, invalidated=jnp.asarray(i % 2 == 0))
            ),
        )
        for i, (s, y) in enumerate(expected)
    ]
    actual = (
        actual[0],
        replace(
            actual[1],
            collider=replace(
                actual[1].collider, invalidated=jnp.arange(batch_size) % 2 == 0
            ),
        ),
    )
    assert_tree_close(
        jd.System.evaluate_forces(*actual),
        stack([jd.System.evaluate_forces(s, y) for s, y in expected]),
    )
    actual = jd.System.step(*actual, n=2)
    expected = [jd.System.step(s, y, n=2) for s, y in expected]
    assert_tree_close(actual, stack(expected))

    energy = lambda s, y: y.collider.compute_potential_energy(s, y)[2]
    np.testing.assert_allclose(
        jax.jit(jax.vmap(energy))(*actual),
        jnp.stack([energy(s, y) for s, y in expected]),
        rtol=2e-5,
    )


def test_batched_overflow_is_per_replica_and_sticky():
    pairs = []
    for occupied in (3, 4, 0):
        pos = np.zeros((6, 2))
        pos[:, 0] = np.arange(6) * 3
        pos[:occupied, 0] = np.arange(occupied) * 0.1
        pairs.append(case(capacity=1, pos=pos))
    s, y = jd.System.initialize(*stack(pairs))
    np.testing.assert_array_equal(y.collider.row_offsets[:, -1], [6, 6, 0])
    np.testing.assert_array_equal(y.collider.overflow, [False, True, False])
    with pytest.raises(RuntimeError, match="overflow"):
        y.check_overflow()
    # Remove every contact. The cache recovers, while the system retains the failure.
    s = replace(s, pos_c=jnp.broadcast_to(pairs[2][0].pos_c, s.pos_c.shape))
    s, y = jd.System.step(s, y)
    np.testing.assert_array_equal(y.collider.overflow, False)
    np.testing.assert_array_equal(y.search_overflow, [False, True, False])


@pytest.mark.parametrize("advance", [False, True])
def test_nested_vmap_force_and_energy_gradients(advance):
    s, y = case(force=Remember())
    s, y = jd.System.initialize(s, y)
    positions = jnp.stack([s.pos_c, s.pos_c * 1.01, s.pos_c * 0.99])
    positions = jnp.stack([positions, positions + 0.05])

    def observable(pos):
        moved = replace(s, pos_c=pos)
        result, updated = y.collider.compute_force(moved, y, advance_history=advance)
        energy = y.collider.compute_potential_energy(moved, y)[2]
        weights = jnp.arange(result.force.size).reshape(result.force.shape)
        return energy + jnp.sum(result.force * weights) + updated.collider.history.sum()

    batched = jax.jit(jax.vmap(jax.vmap(jax.grad(observable))))(positions)
    independent = jnp.stack(
        [jnp.stack([jax.grad(observable)(p) for p in row]) for row in positions]
    )
    np.testing.assert_allclose(batched, independent, rtol=2e-5, atol=2e-5)
    direction = (
        jnp.arange(positions.size, dtype=positions.dtype).reshape(positions.shape)
        / positions.size
    )
    total = lambda p: jax.vmap(jax.vmap(observable))(p).sum()
    tangent = jax.jvp(total, (positions,), (direction,))[1]
    assert jnp.abs(tangent) > 1e-4
    np.testing.assert_allclose(
        tangent, jnp.sum(independent * direction), rtol=2e-5, atol=2e-5
    )


@pytest.mark.parametrize("backend", ["CellList", "MultiCellList", "naive"])
@pytest.mark.parametrize("capacity", [None, 0, 2])
def test_empty_simulation_initializes_and_steps(backend, capacity):
    s, y = case(capacity=capacity, backend=backend, pos=np.empty((0, 2)))
    s, y = jd.System.initialize(s, y)
    s, y = jd.System.step(s, y, n=2)
    y.check_overflow()
    assert s.pos.shape == (0, 2)
    assert y.collider.neighbor_list.shape == (0,)
    assert y.collider.compute_potential_energy(s, y)[2] == 0
    bs, by = jd.System.initialize(*stack([(s, y), (s, y)]))
    bs, by = jd.System.step(bs, by)
    assert bs.pos.shape == (2, 0, 2)
    by.check_overflow()


@pytest.mark.parametrize("capacity", [0, 2, 12])
def test_source_decoding_handles_empty_rows_overflow_and_vmap(capacity):
    from types import SimpleNamespace

    from jaxdem.colliders._neighbor_cache import pair_sources

    counts = np.asarray(
        [[0, 0, 0, 0, 0], [0, 2, 0, 1, 0], [1, 0, 0, 3, 0], [3, 2, 1, 2, 0]]
    )
    offsets = np.minimum(np.pad(counts.cumsum(axis=1), ((0, 0), (1, 0))), capacity)
    expected = []
    for row in counts:
        sources = np.repeat(np.arange(len(row)), row)[:capacity]
        expected.append(
            np.pad(sources, (0, capacity - len(sources)), constant_values=len(row))
        )

    def decode(offsets):
        return pair_sources(
            SimpleNamespace(
                row_offsets=offsets, neighbor_list=jnp.empty(capacity, dtype=int)
            )
        )

    actual = jax.jit(jax.vmap(decode))(jnp.asarray(offsets))
    np.testing.assert_array_equal(actual, expected)
