# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Correctness gates for benchmark fixtures and timing inputs."""

import jax.numpy as jnp
import numpy as np
import pytest

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
        assert np.all(np.asarray(state.inertia)[members] > 0.0)


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


def test_benchmark_function_rejects_nonfinite_output():
    with pytest.raises(ValueError, match="non-finite"):
        benchmark_function(lambda: jnp.array(jnp.nan), (), {}, repeat=1)


def test_benchmark_function_rejects_query_overflow():
    def overflowing_query():
        return None, None, jnp.zeros((1, 1), dtype=int), jnp.array(True)

    with pytest.raises(RuntimeError, match="overflow"):
        benchmark_function(overflowing_query, (), {}, repeat=1)
