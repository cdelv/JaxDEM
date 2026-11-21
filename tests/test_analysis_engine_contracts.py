# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Check chunked analysis shapes, timestep bins, pair budgets, and JIT cache bounds."""

from __future__ import annotations

import jax.numpy as jnp

import numpy as np
import pytest

from jaxdem.analysis import (
    BinSpec,
    LagBinsExact,
    TimeBins,
    build_pairs,
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


def test_analysis_jit_cache_is_bounded_and_explicitly_clearable(monkeypatch):
    clear_jit_cache()
    monkeypatch.setattr(engine, "_JIT_CACHE_MAXSIZE", 2)
    arrays = {"x": jnp.arange(2.0)}
    bins = LagBinsExact(2, [1])
    try:
        for scale in range(4):
            result = evaluate_binned(
                displacement, arrays, bins, kernel_kwargs={"scale": scale}
            )
            np.testing.assert_allclose(result.sums, [scale])
            np.testing.assert_allclose(result.mean, [scale])
        assert len(engine._JIT_CACHE) == 2
        np.testing.assert_allclose(
            evaluate_binned(displacement, arrays, bins, kernel_kwargs={"scale": 1}).sums,
            [1],
        )
    finally:
        clear_jit_cache()
    assert not engine._JIT_CACHE


def test_max_pairs_rejects_estimate_before_allocation_or_enumeration():
    bins = OverBudgetBins(2)
    with pytest.raises(ValueError, match="exceeding max_pairs=100"):
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


def test_time_bins_accept_singleton_and_empty_timestep_arrays():
    singleton = TimeBins.from_source({"timestep": np.asarray([7])})
    assert singleton.values().tolist() == [7]
    assert list(singleton.iter_tuples(0)) == [[0]]

    empty = TimeBins(0, timestep=np.asarray([], dtype=int))
    assert empty.num_bins() == 0
    assert empty.values().size == 0


def test_time_bins_reject_ambiguous_timestep_labels():
    for labels, message in (
        ([0, 2, 1, 3], "strictly increasing"),
        ([0, 1, 1, 3], "strictly increasing"),
        ([0.0, 1.5, 3.0], "integers"),
    ):
        with pytest.raises(ValueError, match=message):
            TimeBins(len(labels), timestep=np.asarray(labels))
    with pytest.raises(ValueError, match="integers"):
        TimeBins.from_source({"timestep": np.asarray([0.0, 1.5, 3.0])})


def test_lag_bins_preserve_nonuniform_integer_pairs():
    labels = np.asarray([0, 2, 5, 9])
    bins = LagBinsExact(4, [3, 4, 5, 7, 9], timestep=labels)
    pairs = build_pairs(bins)

    actual = list(
        zip(pairs.bin_id.tolist(), pairs.pair_i.tolist(), pairs.pair_j.tolist())
    )
    assert actual == [
        (0, 1, 2),
        (1, 2, 3),
        (2, 0, 2),
        (3, 1, 3),
        (4, 0, 3),
    ]
