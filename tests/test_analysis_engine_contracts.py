# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check chunked analysis shapes, pair budgets, and bounded compilation caches."""

from __future__ import annotations

import jax.numpy as jnp

import numpy as np
import pytest

from jaxdem.analysis import (
    BinSpec,
    LagBinsExact,
    clear_jit_cache,
    evaluate_binned,
)
from jaxdem.analysis import engine


def displacement(arrays, i, j, *, scale=1.0):
    return scale * (arrays["x"][j] - arrays["x"][i])


class EmptyBins(BinSpec):
    def num_bins(self):
        return 2

    def weight(self, b):
        return 0

    def iter_tuples(self, b):
        return iter(())


class OverBudgetBins(BinSpec):
    iterated = False

    def num_bins(self):
        return 1

    def weight(self, b):
        return 1_000_000

    def iter_tuples(self, b):
        self.iterated = True
        raise AssertionError("budget must reject before enumeration")


@pytest.mark.parametrize("jit", [False, True])
@pytest.mark.parametrize("chunk_size", [None, 7])
def test_chunked_analysis_matches_quadratic_displacements(jit, chunk_size):
    # For x(t)=t^2, the displacement sum at lag l is l*(T-l)*(T-1).
    frames, lags = 23, np.array([1, 3, 8])
    arrays = {"x": jnp.arange(frames, dtype=float) ** 2}
    bins = LagBinsExact(frames, lags)

    def kernel(values, i, j):
        delta = displacement(values, i, j)
        return {"delta": delta, "vector": jnp.stack([delta, -2 * delta], axis=-1)}

    result = evaluate_binned(kernel, arrays, bins, jit=jit, chunk_size=chunk_size)
    counts = frames - lags
    sums = lags * counts * (frames - 1)
    np.testing.assert_array_equal(result.counts, counts)
    np.testing.assert_allclose(result.sums["delta"], sums)
    np.testing.assert_allclose(result.mean["delta"], lags * (frames - 1))
    np.testing.assert_allclose(
        result.sums["vector"], np.stack([sums, -2 * sums], axis=-1)
    )
    assert result.pairs.pair_i.size == counts.sum()


def test_empty_pairs_preserve_kernel_output_tree_and_bin_shapes():
    arrays = {"x": jnp.arange(3.0)}
    result = evaluate_binned(
        lambda values, i, j: {"delta": values["x"][j] - values["x"][i]},
        arrays,
        EmptyBins(3),
    )

    assert result.sums["delta"].shape == (2,)
    assert result.mean["delta"].shape == (2,)
    np.testing.assert_array_equal(result.counts, np.zeros(2, dtype=int))
    assert bool(jnp.all(jnp.isnan(result.mean["delta"])))


def test_analysis_jit_cache_is_bounded_and_explicitly_clearable():
    clear_jit_cache()
    arrays = {"x": jnp.arange(2.0)}
    bins = LagBinsExact(2, [1])
    for scale in range(engine._JIT_CACHE_MAXSIZE + 5):
        result = evaluate_binned(
            displacement, arrays, bins, kernel_kwargs={"scale": scale}
        )
        np.testing.assert_allclose(result.sums, [scale])
        np.testing.assert_allclose(result.mean, [scale])
    assert len(engine._JIT_CACHE) == engine._JIT_CACHE_MAXSIZE
    np.testing.assert_allclose(
        evaluate_binned(displacement, arrays, bins, kernel_kwargs={"scale": 1}).sums,
        [1],
    )
    clear_jit_cache()
    assert not engine._JIT_CACHE


def test_max_pairs_rejects_estimate_before_allocation_or_enumeration():
    bins = OverBudgetBins(2)
    with np.testing.assert_raises_regex(ValueError, "exceeding max_pairs=100"):
        evaluate_binned(displacement, {"x": jnp.arange(2.0)}, bins, max_pairs=100)
    assert not bins.iterated

    class UnderestimatedBins(BinSpec):
        def num_bins(self):
            return 1

        def weight(self, b):
            return 1

        def iter_tuples(self, b):
            yield from [(0, 1), (0, 2), (1, 2)]

    def forbidden(*args):
        raise AssertionError("pair budget must reject before running a kernel")

    with pytest.raises(ValueError, match="more than max_pairs=2"):
        evaluate_binned(
            forbidden, {"x": jnp.arange(3.0)}, UnderestimatedBins(3), max_pairs=2
        )
