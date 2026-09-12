"""Contracts for physical timestep labels used by analysis bins."""

from __future__ import annotations

import numpy as np
import pytest

from jaxdem.analysis import LagBinsExact, TimeBins, build_pairs


def test_time_bins_accept_singleton_and_empty_timestep_arrays():
    singleton = TimeBins.from_source({"timestep": np.asarray([7])})
    assert singleton.values().tolist() == [7]
    assert list(singleton.iter_tuples(0)) == [[0]]

    empty = TimeBins(0, timestep=np.asarray([], dtype=int))
    assert empty.num_bins() == 0
    assert empty.values().size == 0


@pytest.mark.parametrize(
    "labels, message",
    [
        ([0, 2, 1, 3], "strictly increasing"),
        ([0, 1, 1, 3], "strictly increasing"),
        ([0.0, 1.5, 3.0], "integers"),
    ],
)
def test_time_bins_reject_ambiguous_timestep_labels(labels, message):
    with pytest.raises(ValueError, match=message):
        TimeBins(len(labels), timestep=np.asarray(labels))


def test_from_source_does_not_truncate_fractional_timestep_labels():
    source = {"timestep": np.asarray([0.0, 1.5, 3.0])}
    with pytest.raises(ValueError, match="integers"):
        TimeBins.from_source(source)


def test_lag_bins_preserve_nonuniform_integer_pairs():
    labels = np.asarray([0, 2, 5, 9])
    bins = LagBinsExact(4, [3, 4, 5, 7, 9], timestep=labels)
    pairs = build_pairs(bins)

    actual = list(
        zip(
            pairs.bin_id.tolist(),
            pairs.pair_i.tolist(),
            pairs.pair_j.tolist(),
        )
    )
    assert actual == [
        (0, 1, 2),
        (1, 2, 3),
        (2, 0, 2),
        (3, 1, 3),
        (4, 0, 3),
    ]
