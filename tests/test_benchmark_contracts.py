# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Correctness gates for benchmark fixtures and timing inputs."""

import jax.numpy as jnp

import numpy as np
import pytest

import jaxdem as jd
from benchmarks.base import (
    create_clumps_state,
    create_deformable_state,
    create_mixed_state,
)
from benchmarks.run_benchmarks import benchmark_function


@pytest.mark.parametrize("n", [1, 2, 3, 7, 19])
def test_clump_fixture_has_coherent_body_properties(n):
    state = create_clumps_state(N=n, dim=3)
    for clump in np.unique(np.asarray(state.clump_id)):
        members = np.asarray(state.clump_id) == clump
        np.testing.assert_allclose(
            np.asarray(state.pos_p)[members].mean(axis=0), 0.0, atol=1e-7
        )
        assert np.unique(np.asarray(state.mass)[members]).size == 1
        assert np.unique(np.asarray(state.volume)[members]).size == 1
        offsets = np.asarray(state.pos_p)[members]
        count = members.sum()
        tensor = sum(
            (np.dot(r, r) * np.eye(3) - np.outer(r, r)) / count for r in offsets
        )
        tensor += 0.4 * 0.1**2 * np.eye(3)
        np.testing.assert_allclose(
            state.inertia[members],
            np.broadcast_to(np.diag(tensor), (count, 3)),
            rtol=2e-6,
        )
        np.testing.assert_allclose(
            state.volume[members], count * 4 * np.pi * 0.1**3 / 3, rtol=2e-6
        )


def test_deformable_fixture_uses_neighbor_indices():
    state = create_deformable_state(N=8, dim=2)
    expected = np.array(
        [[1, 5], [0, 2], [1, 3], [2, 4], [3, 5], [0, 4], [7, -1], [6, -1]]
    )
    np.testing.assert_array_equal(state.bond_id, expected)


def test_mixed_fixture_contains_all_thirds_and_preserves_properties():
    state = create_mixed_state(N=10, dim=3)
    assert state.N == 10
    assert np.any(np.asarray(state.pos_p) != 0.0)
    assert np.all(np.asarray(state.inertia) > 0.0)
    assert np.any(np.asarray(state.bond_id) >= 0)
    np.testing.assert_array_equal(
        np.sort(state.bond_id[6:], axis=-1), [[7, 9], [6, 8], [7, 9], [6, 8]]
    )
    assert len(np.unique(state.clump_id[:3])) == 3
    assert len(np.unique(state.clump_id[3:6])) == 1
    assert len(np.unique(state.clump_id[6:])) == 4
    np.testing.assert_array_equal(state.pos_p[:3], np.zeros((3, 3)))
    np.testing.assert_array_equal(state.pos_p[6:], np.zeros((4, 3)))


def test_benchmark_function_rejects_nonfinite_output():
    with pytest.raises(ValueError, match="non-finite"):
        benchmark_function(lambda: jnp.array(jnp.nan), (), {}, repeat=1)


def test_benchmark_function_rejects_query_overflow():
    def overflowing_query():
        return None, None, jnp.zeros((1, 1), dtype=int), jnp.array(True)

    with pytest.raises(RuntimeError, match="overflow"):
        benchmark_function(overflowing_query, (), {}, repeat=1)


@pytest.mark.parametrize("batched", [False, True])
def test_release_workload_initializes_the_known_pair_force(monkeypatch, batched):
    from benchmarks import release_workloads

    pair = jd.State.create(
        pos=jnp.array([[0.0, 0.0], [0.15, 0.0]]), rad=jnp.full(2, 0.1)
    )
    monkeypatch.setattr(
        release_workloads, "create_spheres_state", lambda *args, **kw: pair
    )
    state, system = release_workloads.build_workload("spheres", size=2, batched=batched)
    stiffness = float(np.asarray(system.mat_table.young_eff).ravel()[0])
    expected = stiffness * np.array([[-0.05, 0.0], [0.05, 0.0]])
    np.testing.assert_allclose(
        state.force, expected[None] if batched else expected, rtol=2e-6
    )
    np.testing.assert_array_equal(
        system.step_count, np.zeros((1,) if batched else (), dtype=int)
    )
