# SPDX-License-Identifier: BSD-3-Clause
# Part of the JaxDEM project – https://github.com/cdelv/JaxDEM
"""
Utility functions to assign radius disperisty.
"""

from __future__ import annotations

import warnings
from typing import Sequence
import jax
import numpy as np
import jax.numpy as jnp

def allocate_counts(N: int, count_ratios: Sequence[float], *, ensure_each_size_nonzero: bool = True) -> np.ndarray:
    """
    Convert population fractions into integer counts that sum exactly to N.

    Uses a "largest remainder" method (Hamilton apportionment).

    If ensure_each_size_nonzero is True, enforces count >= 1 for each species
    (requires N >= num_species).
    """
    if N <= 0:
        raise ValueError(f"N must be positive; got {N}.")

    ratios = np.asarray(count_ratios, dtype=float)
    if ratios.ndim != 1 or ratios.size == 0:
        raise ValueError("count_ratios must be a 1D non-empty array-like.")
    if not np.isfinite(ratios).all():
        raise ValueError("count_ratios contains non-finite values.")
    if np.any(ratios < 0):
        raise ValueError("count_ratios must be non-negative.")

    k = int(ratios.size)

    # Normalize (keeping behavior stable if user passes percentages that don't sum to 1).
    total = float(ratios.sum())
    if total <= 0:
        raise ValueError("count_ratios must sum to a positive value.")
    ratios = ratios / total

    if ensure_each_size_nonzero:
        if N < k:
            raise ValueError(f"Cannot give each of {k} sizes at least 1 particle with N={N}.")
        if np.any(ratios == 0):
            warnings.warn(
                "ensure_each_size_nonzero=True but some count_ratios are 0; "
                "those species will still be forced to have count=1.",
                RuntimeWarning,
            )
        # Start with one particle per species, apportion the remaining N-k.
        base = np.ones(k, dtype=int)
        N_rem = N - k
        raw = N_rem * ratios
    else:
        base = np.zeros(k, dtype=int)
        raw = N * ratios

    floors = np.floor(raw).astype(int)
    frac = raw - floors
    counts = base + floors

    remaining = int(N - counts.sum())
    if remaining > 0:
        # Give +1 to the species with largest fractional parts.
        order = np.argsort(-frac)  # descending
        counts[order[:remaining]] += 1
    elif remaining < 0:
        # Remove -1 from the smallest fractional parts, but never drop below base.
        order = np.argsort(frac)  # ascending
        to_remove = -remaining
        for idx in order:
            if to_remove == 0:
                break
            if counts[idx] > base[idx]:
                counts[idx] -= 1
                to_remove -= 1
        if to_remove != 0:
            raise RuntimeError("Failed to apportion counts without violating nonzero constraint.")

    # Final safety: exact sum and integer non-negative.
    if counts.sum() != N:
        raise RuntimeError(f"Internal error: counts sum {counts.sum()} != N {N}.")
    if np.any(counts < 0):
        raise RuntimeError("Internal error: negative counts produced.")
    if ensure_each_size_nonzero and np.any(counts < 1):
        raise RuntimeError("Internal error: nonzero constraint violated.")
    return counts

def get_polydisperse_radii(
    N: int,
    count_ratios: Sequence[float] = (0.5, 0.5),
    size_ratios: Sequence[float] = (1.0, 1.4),
    small_radius: float = 0.5,
    ensure_size_nonzero: bool = False,
) -> jax.Array:
    """
    Construct a polydisperse set of particle radii from population and size ratios.

    Parameters
    ----------
    N : int
        Total number of particles.
    count_ratios : array-like of float
        Population fractions for each size class (will be normalized to sum to 1).
    size_ratios : array-like of float
        Radius multipliers for each size class, relative to the smallest size.
        For example, size_ratios=[1.0, 1.4] means the large particles have radius 1.4x the
        small particles (before normalization by min(size_ratios)).
    small_radius : float
        The absolute radius corresponding to the smallest size class (must be > 0).
    ensure_size_nonzero : bool
        If True, enforce that each size class has at least one particle (requires N >= number of sizes).

    Returns
    -------
    jax.Array
        1D array of length N containing the radii for each particle.
    """
    count_ratios = np.asarray(count_ratios)
    size_ratios = np.asarray(size_ratios)
    assert len(count_ratios) == len(size_ratios), f"Got inconsistent sizes for count_ratios ({len(count_ratios)}) and size_ratios ({len(size_ratios)})"
    count_ratios = count_ratios / np.sum(count_ratios)

    assert np.all(np.isfinite(size_ratios))
    assert np.all(size_ratios > 0), "size_ratios must be positive (multiples of small_radius)."
    assert np.isfinite(small_radius) and small_radius > 0, "small_radius must be positive."
    size_ratios = size_ratios / np.min(size_ratios)

    counts = allocate_counts(N, count_ratios, ensure_each_size_nonzero=ensure_size_nonzero)
    sizes = small_radius * size_ratios

    # Warn if finite-size rounding makes the achieved fractions differ from requested.
    achieved = counts / N
    if not np.all(np.isclose(achieved, count_ratios)):
        print(f'Warning: cannot achieve exact count ratio ({count_ratios}) - got ({achieved})')

    return jnp.array(np.concatenate([np.ones(c) * s for c, s in zip(counts, sizes)]))
