# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project - https://github.com/cdelv/JaxDEM
"""Index same-clump exclusions without changing the original cell order."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp


class ClumpIndex(NamedTuple):
    """Clump-grouped primary ranks, signed gaps, and N + 1 group offsets."""

    ranks: jax.Array
    gaps: jax.Array
    offsets: jax.Array


@jax.jit
def build_clump_index(perm: jax.Array, clump_id: jax.Array) -> ClumpIndex:
    """Group primary sorted ranks by clump, with stable order inside each group.

    ``perm`` maps primary sorted ranks to original particle indices. Clump IDs
    must lie in ``[0, N)``. Missing labels are valid. Equal labels denote one
    clump, including when their original members are not contiguous.
    """
    n = perm.size
    if n == 0:
        empty = jnp.empty_like(perm)
        return ClumpIndex(empty, empty, jnp.zeros_like(perm, shape=(1,)))
    labels = clump_id[perm]
    _, ranks = jax.lax.sort(
        (labels, jnp.arange(n, dtype=perm.dtype)), num_keys=1, is_stable=True
    )
    counts = jnp.zeros_like(perm).at[labels].add(1)
    offsets = jnp.concatenate((jnp.zeros_like(perm, shape=(1,)), jnp.cumsum(counts)))
    gaps = ranks - jnp.arange(n, dtype=ranks.dtype)
    return ClumpIndex(ranks, gaps, offsets)


@partial(jax.jit, static_argnames=("right",))
def _bounded_search(
    values: jax.Array,
    target: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    *,
    right: bool,
) -> jax.Array:
    """Search sorted slices with fixed depth and broadcast bounds."""
    target, lo, hi = jnp.broadcast_arrays(target, lo, hi)
    n = values.size
    if n == 0:
        return lo

    def step(
        _: int, bounds: tuple[jax.Array, jax.Array]
    ) -> tuple[jax.Array, jax.Array]:
        lower, upper = bounds
        mid = lower + (upper - lower) // 2
        value = values[jnp.minimum(mid, n - 1)]
        open_range = lower < upper
        less = value <= target if right else value < target
        advance = open_range & less
        return (
            jnp.where(advance, mid + 1, lower),
            jnp.where(open_range & ~less, mid, upper),
        )

    # The first step joins variance from bounds, targets, and indexed values.
    bounds = step(0, (lo, hi))
    return jax.lax.fori_loop(1, n.bit_length(), step, bounds)[0]


@jax.jit
def clump_exclusion_ranges(
    index: ClumpIndex,
    clump_id: jax.Array,
    starts: jax.Array,
    ends: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Find each query clump's auxiliary slice inside each target cell.

    ``starts`` and ``ends`` have one final stencil axis beyond ``clump_id``.
    Returned bounds index ``index.ranks`` and ``index.gaps``. Their difference
    counts excluded members before tile allocation.
    """
    labels = jnp.broadcast_to(clump_id[..., None], starts.shape)
    if index.ranks.size == 0:
        empty = jnp.zeros_like(starts, dtype=index.offsets.dtype)
        return empty, empty
    lower, upper = index.offsets[labels], index.offsets[labels + 1]
    a = _bounded_search(index.ranks, starts, lower, upper, right=False)
    b = _bounded_search(index.ranks, ends, lower, upper, right=False)
    return a, b


@jax.jit
def select_clump_complement(
    gaps: jax.Array,
    lo: jax.Array,
    hi: jax.Array,
    cell_start: jax.Array,
    eligible_rank: jax.Array,
) -> jax.Array:
    """Map eligible ranks to primary sorted ranks in original member order.

    ``lo`` starts the excluded clump's slice inside the cell. ``hi`` can end
    that slice or the entire clump group. Eligible ranks must be nonnegative.
    Callers must mask padded ranks and clamp later gathers to database bounds.
    """
    target = cell_start + eligible_rank - lo
    excluded_end = _bounded_search(gaps, target, lo, hi, right=True)
    return cell_start + eligible_rank + excluded_end - lo
