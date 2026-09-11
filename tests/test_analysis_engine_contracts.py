from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jaxdem.analysis import BinSpec, TimeBins, clear_jit_cache, evaluate_binned
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


def test_chunked_analysis_matches_single_shot_for_many_pairs():
    arrays = {"x": jnp.arange(2001, dtype=float)}
    bins = TimeBins(2001)

    single = evaluate_binned(displacement, arrays, bins, jit=False)
    chunked = evaluate_binned(displacement, arrays, bins, jit=False, chunk_size=37)

    np.testing.assert_array_equal(chunked.counts, single.counts)
    np.testing.assert_allclose(chunked.sums, single.sums)
    np.testing.assert_allclose(chunked.mean, single.mean)
    assert chunked.pairs.pair_i.shape == (2001,)


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
    bins = TimeBins(2)
    for scale in range(engine._JIT_CACHE_MAXSIZE + 5):
        evaluate_binned(displacement, arrays, bins, kernel_kwargs={"scale": scale})
    assert len(engine._JIT_CACHE) == engine._JIT_CACHE_MAXSIZE
    clear_jit_cache()
    assert not engine._JIT_CACHE


def test_max_pairs_rejects_estimate_before_allocation_or_enumeration():
    bins = OverBudgetBins(2)
    with np.testing.assert_raises_regex(ValueError, "exceeding max_pairs=100"):
        evaluate_binned(displacement, {"x": jnp.arange(2.0)}, bins, max_pairs=100)
    assert not bins.iterated
